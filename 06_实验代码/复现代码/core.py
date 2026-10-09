"""Shared data protocol, models and training loop (reimplementation of the manuscript's protocol)."""
import math, time, json, os
import numpy as np
import torch
import torch.nn as nn

EXCLUDE = ['ctx_cycles', 'ctx_rpt', 'ctx_soc_coul', 'ctx_soc_dev']


def load_data(name):
    z = np.load(f'data_{name}.npz', allow_pickle=False)
    return {k: z[k] for k in z.files}


def cv_folds(cells, n_folds=5, seed=42):
    cells = np.unique(cells)
    perm = np.random.default_rng(seed).permutation(cells)
    tests = np.array_split(perm, n_folds)
    out = []
    for f, te in enumerate(tests):
        rest = np.sort(np.setdiff1d(cells, te))
        va = rest[f::5]
        tr = np.setdiff1d(rest, va)
        out.append((tr, va, np.sort(te)))
    return out


def logo_folds(cells, groups):
    out = []
    ug = np.unique(groups)
    cell_group = {c: g for c, g in zip(cells, groups)}
    allc = np.unique(cells)
    for i, g in enumerate(ug):
        te = np.array([c for c in allc if cell_group[c] == g])
        rest = np.sort(np.setdiff1d(allc, te))
        va = rest[i % 5::5]
        tr = np.setdiff1d(rest, va)
        out.append((tr, va, te))
    return out


def rankdata_cols(X):
    # average ranks per column, NaN ignored (ranked as NaN)
    R = np.full(X.shape, np.nan)
    for j in range(X.shape[1]):
        x = X[:, j]
        m = np.isfinite(x)
        xs = x[m]
        order = np.argsort(xs, kind='mergesort')
        ranks = np.empty(xs.size)
        sx = xs[order]
        # average ties
        i = 0
        r = np.arange(1, xs.size + 1, dtype=float)
        uniq, idx, cnt = np.unique(sx, return_index=True, return_counts=True)
        avg = idx + (cnt + 1) / 2.0
        ranks_sorted = np.repeat(avg, cnt)
        ranks[order] = ranks_sorted
        R[m, j] = ranks
    return R


def spearman_select(F, y, names, K, thr=0.98):
    cand = [j for j, n in enumerate(names) if n not in EXCLUDE]
    X = F[:, cand]
    R = rankdata_cols(X)
    ry = rankdata_cols(y[:, None])[:, 0]
    rho = np.full(len(cand), np.nan)
    for j in range(len(cand)):
        m = np.isfinite(R[:, j])
        a = R[m, j] - R[m, j].mean(); b = ry[m] - ry[m].mean()
        den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        rho[j] = (a * b).sum() / den if den > 0 else np.nan
    order = [j for j in np.argsort(-np.abs(np.nan_to_num(rho, nan=-1))) if np.isfinite(rho[j])]
    Rf = np.where(np.isfinite(R), R, np.nanmean(R, 0))
    Rc = Rf - Rf.mean(0)
    Rn = Rc / np.maximum(np.sqrt((Rc ** 2).sum(0)), 1e-12)
    sel = []
    for j in order:
        if len(sel) >= K:
            break
        if sel and np.max(np.abs(Rn[:, sel].T @ Rn[:, j])) >= thr:
            continue
        sel.append(j)
    if len(sel) < K:
        for j in order:
            if len(sel) >= K:
                break
            if j not in sel:
                sel.append(j)
    return [cand[j] for j in sel], {names[cand[j]]: float(rho[j]) for j in range(len(cand))}


class Prep:
    """Train-fold-only imputation/scaling for handcrafted features, waveform, tau and label."""

    def __init__(self, d, tr_idx, K=50, use_tau=True):
        names = list(d['names'])
        self.names = names
        self.sel, self.rho = spearman_select(d['F'][tr_idx], d['y'][tr_idx], names, K)
        Ftr = d['F'][tr_idx][:, self.sel]
        self.med = np.nanmedian(Ftr, 0)
        Ftr = np.where(np.isfinite(Ftr), Ftr, self.med)
        self.mu = Ftr.mean(0); self.sd = Ftr.std(0); self.sd[self.sd < 1e-12] = 1.0
        dv = d['V'][tr_idx] - d['V'][tr_idx][:, :1]
        self.wmu = dv.mean(0); self.wsd = dv.std(0); self.wsd[self.wsd < 1e-12] = 1.0
        cum = d['cum'][tr_idx]
        self.nmu, self.nsd = cum.mean(), cum.std()
        yy = d['y'][tr_idx] / 100
        self.ymu, self.ysd = yy.mean(), yy.std()
        self.dcir_j = names.index('ctx_dcir')
        dc = d['F'][tr_idx][:, self.dcir_j]
        self.dmu, self.dsd = np.nanmean(dc), np.nanstd(dc)

    def transform(self, d, idx, V=None, cum=None, F=None):
        Fall = d['F'][idx] if F is None else F
        Fs = Fall[:, self.sel]
        Fs = np.where(np.isfinite(Fs), Fs, self.med)
        h = (Fs - self.mu) / self.sd
        VV = d['V'][idx] if V is None else V
        w = ((VV - VV[:, :1]) - self.wmu) / self.wsd
        cc = d['cum'][idx] if cum is None else cum
        tau = (cc - self.nmu) / self.nsd
        Fd = Fall[:, self.dcir_j]
        dcir = (np.where(np.isfinite(Fd), Fd, self.dmu) - self.dmu) / self.dsd
        y = (d['y'][idx] / 100 - self.ymu) / self.ysd
        return dict(h=h.astype(np.float32), w=w.astype(np.float32), tau=tau.astype(np.float32),
                    dcir=dcir.astype(np.float32), y=y.astype(np.float32))

    def inv(self, ys):
        return 100 * (self.ysd * ys + self.ymu)


# ----------------------------------------------------------------------------- models
class MLPBackbone(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class PatchTST(nn.Module):
    def __init__(self, L=101, P=9, S=4, D=32, heads=4, ff=64, flatten=False, revin=False):
        super().__init__()
        self.P, self.S = P, S
        self.M = (L - P) // S + 1
        self.flatten, self.revin = flatten, revin
        if revin:
            self.aff_w = nn.Parameter(torch.ones(1)); self.aff_b = nn.Parameter(torch.zeros(1))
        self.embed = nn.Linear(P, D)
        self.pos = nn.Parameter(torch.empty(self.M, D).uniform_(-0.02, 0.02))
        self.layer = nn.TransformerEncoderLayer(D, heads, ff, dropout=0.2, activation='gelu',
                                                batch_first=True, norm_first=False)
        self.norm = nn.LayerNorm(D)
        din = D * self.M if flatten else D
        self.proj = nn.Sequential(nn.Linear(din, 32), nn.ReLU(), nn.Dropout(0.2))

    def forward(self, w):
        if self.revin:
            mu = w.mean(1, keepdim=True); sd = w.std(1, keepdim=True) + 1e-5
            w = (w - mu) / sd * self.aff_w + self.aff_b
        p = w.unfold(1, self.P, self.S)  # (B, M, P)
        z = self.embed(p) + self.pos
        z = self.layer(z)
        z = self.norm(z)
        z = z.flatten(1) if self.flatten else z.mean(1)
        return self.proj(z)


class Head(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, z):
        return self.net(z).squeeze(-1)


class DualBranch(nn.Module):
    """mode: 'R0' (backbone+wave), 'A2' (backbone only), 'A1' (wave + dcir + tau)."""

    def __init__(self, K, mode='R0', use_tau=True, patch=(9, 4), flatten=False, revin=False):
        super().__init__()
        self.mode, self.use_tau = mode, use_tau
        din = K + (1 if use_tau else 0)
        if mode in ('R0', 'A2'):
            self.Eh = MLPBackbone(din)
        if mode in ('R0', 'A1'):
            self.Ev = PatchTST(P=patch[0], S=patch[1], flatten=flatten, revin=revin)
        hin = {'R0': 64, 'A2': 32, 'A1': 32 + 1 + (1 if use_tau else 0)}[mode]
        self.g = Head(hin)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, h, w, tau, dcir):
        parts = []
        if self.mode in ('R0', 'A1'):
            parts.append(self.Ev(w))
        if self.mode in ('R0', 'A2'):
            hh = torch.cat([h, tau[:, None]], 1) if self.use_tau else h
            parts.append(self.Eh(hh))
        if self.mode == 'A1':
            parts.append(dcir[:, None])
            if self.use_tau:
                parts.append(tau[:, None])
        return self.g(torch.cat(parts, 1))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------------------------------------------------------- metrics
def cell_rmse(pred, y, cell, rpt):
    """Six-condition aggregation per cell-RPT, RMSE per cell. Returns dict cell->rmse."""
    key = cell.astype(np.int64) * 10000 + rpt.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    pm = np.bincount(inv, pred) / np.bincount(inv)
    ym = np.bincount(inv, y) / np.bincount(inv)
    kc = uk // 10000
    out = {}
    for c in np.unique(kc):
        m = kc == c
        out[int(c)] = float(np.sqrt(np.mean((pm[m] - ym[m]) ** 2)))
    return out


def cell_mae(pred, y, cell, rpt):
    key = cell.astype(np.int64) * 10000 + rpt.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    pm = np.bincount(inv, pred) / np.bincount(inv)
    ym = np.bincount(inv, y) / np.bincount(inv)
    kc = uk // 10000
    return {int(c): float(np.mean(np.abs(pm[kc == c] - ym[kc == c]))) for c in np.unique(kc)}


# ----------------------------------------------------------------------------- training
def train_nn(d, tr, va, te, seed, mode='R0', K=50, use_tau=True, lam=0.1, patch=(9, 4),
             flatten=False, revin=False, raw_voltage=False, max_epochs=100, verbose=False,
             return_model=False):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    cell = d['cell']
    itr = np.nonzero(np.isin(cell, tr))[0]; iva = np.nonzero(np.isin(cell, va))[0]
    ite = np.nonzero(np.isin(cell, te))[0]
    pp = Prep(d, itr, K=K)
    if raw_voltage:
        pp.wmu = d['V'][itr].mean(0); pp.wsd = d['V'][itr].std(0)
    def tf(idx):
        out = pp.transform(d, idx)
        if raw_voltage:
            out['w'] = ((d['V'][idx] - pp.wmu) / pp.wsd).astype(np.float32)
        return {k: torch.from_numpy(v) for k, v in out.items()}
    Ttr, Tva, Tte = tf(itr), tf(iva), tf(ite)
    model = DualBranch(K, mode, use_tau, patch, flatten, revin)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    gen = torch.Generator().manual_seed(seed + 10000)
    n = itr.size
    best, best_state, stall, best_ep = np.inf, None, 0, 0
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for b in range(0, n, 256):
            idx = perm[b:b + 256]
            h, w, y, dc = Ttr['h'][idx], Ttr['w'][idx], Ttr['y'][idx], Ttr['dcir'][idx]
            tau = Ttr['tau'][idx].clone().requires_grad_(lam > 0 and use_tau)
            yh = model(h, w, tau, dc)
            loss = torch.mean((yh - y) ** 2)
            if lam > 0 and use_tau:
                g, = torch.autograd.grad(yh.sum(), tau, create_graph=True)
                loss = loss + lam * torch.mean(torch.clamp(g, min=0) ** 2)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = pp.inv(model(Tva['h'], Tva['w'], Tva['tau'], Tva['dcir']).numpy())
        vr = np.mean(list(cell_rmse(pv, d['y'][iva], cell[iva], d['rpt'][iva]).values()))
        sch.step(vr)
        if vr < best - 1e-12:
            best, stall, best_ep = vr, 0, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stall += 1
        if verbose:
            print(ep, round(vr, 4), f'{time.time() - t0:.1f}s', flush=True)
        if stall >= 15:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pt = pp.inv(model(Tte['h'], Tte['w'], Tte['tau'], Tte['dcir']).numpy())
    res = dict(pred=pt, idx=ite, val_rmse=best, best_epoch=best_ep, epochs=ep,
               time=time.time() - t0, n_params=n_params(model), sel=[pp.names[j] for j in pp.sel])
    if return_model:
        res['model'] = model; res['prep'] = pp
    return res
