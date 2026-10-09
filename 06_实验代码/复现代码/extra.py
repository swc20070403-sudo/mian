"""Post-hoc analyses on the 5-fold models: noise/cycle-count robustness, pulse-count aggregation,
GBDT+R0 averaging, waveform attribution, and computational cost."""
import os, sys, json, time, pickle, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
torch.set_num_threads(1)
from core import load_data, cv_folds, DualBranch, cell_rmse, n_params
from features import waveform_features

D = load_data('UConn-ILCC-LFP')
FOLDS = cv_folds(D['cell'])
NAMES = list(D['names'])
SEEDS = [0, 1, 2]


def load_run(mode, f, s):
    z = np.load(f'results/cv/{mode}_f{f}_s{s}.npz')
    m = DualBranch(50, mode)
    m.load_state_dict(torch.load(f'results/cv/{mode}_f{f}_s{s}.pt'))
    m.eval()
    return z, m


def features_from_V(V, cum, idx):
    Fw, _ = waveform_features(V)
    F = D['F'][idx].copy()
    F[:, :143] = Fw
    F[:, NAMES.index('ctx_dcir')] = np.abs(V[:, 40] - V[:, 30]) / (1.2 - 0.24)
    F[:, NAMES.index('ctx_cycles')] = cum
    return F


def predict_nn(z, m, F, V, cum):
    sel = z['sel_idx']
    Fs = F[:, sel]; Fs = np.where(np.isfinite(Fs), Fs, z['med'])
    h = ((Fs - z['mu']) / z['sd']).astype(np.float32)
    w = (((V - V[:, :1]) - z['wmu']) / z['wsd']).astype(np.float32)
    tau = ((cum - z['nmu']) / z['nsd']).astype(np.float32)
    dc = np.zeros(len(V), np.float32)
    with torch.no_grad():
        ys = m(torch.from_numpy(h), torch.from_numpy(w), torch.from_numpy(tau), torch.from_numpy(dc)).numpy()
    return 100 * (z['ysd'] * ys + z['ymu'])


def predict_gbdt(f, F, cum):
    z = np.load(f'results/cv/GBDT_f{f}.npz')
    with open(f'results/cv/GBDT_f{f}.pkl', 'rb') as fh:
        g = pickle.load(fh)
    # rebuild the same preprocessing as in gbdt_job (train-fold statistics)
    from core import Prep
    tr = FOLDS[f][0]
    itr = np.nonzero(np.isin(D['cell'], tr))[0]
    pp = Prep(D, itr, K=50)
    Fs = F[:, pp.sel]; Fs = np.where(np.isfinite(Fs), Fs, pp.med)
    h = (Fs - pp.mu) / pp.sd
    tau = (cum - pp.nmu) / pp.nsd
    return g.predict(np.c_[h, tau])


def macro(pred, idx):
    r = cell_rmse(pred, D['y'][idx], D['cell'][idx], D['rpt'][idx])
    return r


def robustness():
    out = {}
    settings = [('clean', 0.0, 1.0)] + [(f'noise{s}', s, 1.0) for s in [0.5, 1.0, 2.0, 5.0]] + \
               [('tau+10', 0.0, 1.1), ('tau-10', 0.0, 0.9)]
    for name, sig, tfac in settings:
        cellres = {k: {} for k in ['R0', 'A2', 'GBDT']}
        for f in range(5):
            te = FOLDS[f][2]
            idx = np.nonzero(np.isin(D['cell'], te))[0]
            rng = np.random.default_rng(1000 + f)
            V = D['V'][idx] + rng.normal(0, sig / 1000, D['V'][idx].shape) if sig > 0 else D['V'][idx]
            cum = D['cum'][idx] * tfac
            F = features_from_V(V, cum, idx) if sig > 0 else D['F'][idx].copy()
            if sig == 0:
                F[:, NAMES.index('ctx_cycles')] = cum
            for mode in ['R0', 'A2']:
                for s in SEEDS:
                    z, m = load_run(mode, f, s)
                    p = predict_nn(z, m, F, V, cum)
                    for c, v in macro(p, idx).items():
                        cellres[mode].setdefault(c, []).append(v)
            p = predict_gbdt(f, F, cum)
            for c, v in macro(p, idx).items():
                cellres['GBDT'].setdefault(c, []).append(v)
        out[name] = {k: {int(c): float(np.mean(v)) for c, v in d.items()} for k, d in cellres.items()}
        print(name, {k: round(np.mean(list(v.values())), 4) for k, v in out[name].items()}, flush=True)
    json.dump(out, open('results/robustness.json', 'w'))


def pulse_count_and_ensemble():
    # record-level predictions per seed
    preds = {m: {s: np.full(D['y'].size, np.nan) for s in SEEDS} for m in ['R0', 'A2']}
    for m in ['R0', 'A2']:
        for f in range(5):
            for s in SEEDS:
                z = np.load(f'results/cv/{m}_f{f}_s{s}.npz'); preds[m][s][z['idx']] = z['pred']
    g = np.full(D['y'].size, np.nan)
    for f in range(5):
        z = np.load(f'results/cv/GBDT_f{f}.npz'); g[z['idx']] = z['pred']
    cond = np.array([f'{p}{s}' for p, s in zip(D['ptype'], D['soc'])])
    conds = sorted(set(cond))
    res = {}
    for m in ['R0', 'A2', 'GBDT']:
        res[m] = {}
        for k in range(1, 7):
            vals = []
            for combo in itertools.combinations(conds, k):
                msk = np.isin(cond, combo)
                ss = SEEDS if m != 'GBDT' else [0]
                cr = {}
                for s in ss:
                    p = preds[m][s] if m != 'GBDT' else g
                    for c, v in cell_rmse(p[msk], D['y'][msk], D['cell'][msk], D['rpt'][msk]).items():
                        cr.setdefault(c, []).append(v)
                vals.append(np.mean([np.mean(v) for v in cr.values()]))
            res[m][k] = [float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))]
        print(m, {k: [round(x, 3) for x in v] for k, v in res[m].items()}, flush=True)
    # ensemble GBDT + R0
    ens = {}
    for wgt in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cr = {}
        for s in SEEDS:
            p = wgt * preds['R0'][s] + (1 - wgt) * g
            for c, v in cell_rmse(p, D['y'], D['cell'], D['rpt']).items():
                cr.setdefault(c, []).append(v)
        ens[wgt] = {int(c): float(np.mean(v)) for c, v in cr.items()}
        print('ens w_R0=', wgt, round(np.mean(list(ens[wgt].values())), 4), flush=True)
    json.dump({'pulses': res, 'ensemble': {str(k): v for k, v in ens.items()}}, open('results/pulses_ensemble.json', 'w'))


def attribution(steps=32):
    """Integrated gradients wrt standardized waveform; segment occlusion (replace by train mean = 0)."""
    ig_all = []; occ = {k: {} for k in ['S1', 'S2', 'S3', 'none']}
    segs = {'S1': slice(1, 31), 'S2': slice(31, 41), 'S3': slice(41, 101)}
    cond_ig = {}
    for f in range(5):
        te = FOLDS[f][2]
        idx = np.nonzero(np.isin(D['cell'], te))[0]
        for s in SEEDS:
            z, m = load_run('R0', f, s)
            F = D['F'][idx]; V = D['V'][idx]; cum = D['cum'][idx]
            sel = z['sel_idx']
            Fs = np.where(np.isfinite(F[:, sel]), F[:, sel], z['med'])
            h = torch.from_numpy(((Fs - z['mu']) / z['sd']).astype(np.float32))
            w = torch.from_numpy((((V - V[:, :1]) - z['wmu']) / z['wsd']).astype(np.float32))
            tau = torch.from_numpy(((cum - z['nmu']) / z['nsd']).astype(np.float32))
            dc = torch.zeros(len(idx))
            base = torch.zeros_like(w)
            tot = torch.zeros_like(w)
            for a in (np.arange(steps) + 0.5) / steps:
                x = (base + a * (w - base)).requires_grad_(True)
                y = m(h, x, tau, dc)
                g, = torch.autograd.grad(y.sum(), x)
                tot += g
            ig = ((w - base) * tot / steps).detach().numpy() * z['ysd'] * 100  # in SOH pp
            ig_all.append(np.abs(ig))
            for c in ['chg20', 'chg50', 'chg90', 'dchg20', 'dchg50', 'dchg90']:
                msk = np.array([f'{p}{q}' == c for p, q in zip(D['ptype'][idx], D['soc'][idx])])
                cond_ig.setdefault(c, []).append(np.abs(ig[msk]).mean(0))
            with torch.no_grad():
                for k in ['none', 'S1', 'S2', 'S3']:
                    ww = w.clone()
                    if k != 'none':
                        ww[:, segs[k]] = 0.0
                    p = 100 * (z['ysd'] * m(h, ww, tau, dc).numpy() + z['ymu'])
                    for c, v in macro(p, idx).items():
                        occ[k].setdefault(c, []).append(v)
    ig_mean = np.concatenate(ig_all, 0).mean(0)
    occ_cell = {k: {int(c): float(np.mean(v)) for c, v in d.items()} for k, d in occ.items()}
    for k in occ_cell:
        print('occlusion', k, round(np.mean(list(occ_cell[k].values())), 4))
    np.savez('results/attribution.npz', ig_mean=ig_mean,
             cond=np.array(list(cond_ig.keys())), cond_ig=np.array([np.mean(v, 0) for v in cond_ig.values()]))
    json.dump(occ_cell, open('results/occlusion.json', 'w'))


def cost():
    from torch.utils.flop_counter import FlopCounterMode
    out = {}
    for mode in ['R0', 'A2']:
        m = DualBranch(50, mode).eval()
        h = torch.randn(1, 50); w = torch.randn(1, 101); tau = torch.randn(1); dc = torch.zeros(1)
        with FlopCounterMode(display=False) as fc:
            m(h, w, tau, dc)
        flops = fc.get_total_flops()
        with torch.no_grad():
            for _ in range(200):
                m(h, w, tau, dc)
            t = time.perf_counter(); n = 3000
            for _ in range(n):
                m(h, w, tau, dc)
            lat = (time.perf_counter() - t) / n * 1e3
        out[mode] = dict(params=n_params(m), flops=flops, latency_ms=lat, size_kB=n_params(m) * 4 / 1024)
    # handcrafted feature extraction latency (single record, Python reference implementation)
    V = D['V'][:1]
    for _ in range(5):
        waveform_features(V)
    t = time.perf_counter(); n = 200
    for i in range(n):
        waveform_features(D['V'][i:i + 1])
    out['features_ms'] = (time.perf_counter() - t) / n * 1e3
    # GBDT size and latency
    with open('results/cv/GBDT_f0.pkl', 'rb') as fh:
        g = pickle.load(fh)
    nodes = sum(len(p[0].nodes) for p in g._predictors)
    x = np.random.randn(1, 51)
    for _ in range(20):
        g.predict(x)
    t = time.perf_counter(); n = 300
    for _ in range(n):
        g.predict(x)
    out['GBDT'] = dict(trees=len(g._predictors), nodes=int(nodes), latency_ms=(time.perf_counter() - t) / n * 1e3,
                       hp=str(np.load('results/cv/GBDT_f0.npz')['hp']))
    print(json.dumps(out, indent=1))
    json.dump(out, open('results/cost.json', 'w'))


if __name__ == '__main__':
    what = sys.argv[1:]
    if 'pulses' in what: pulse_count_and_ensemble()
    if 'attr' in what: attribution()
    if 'cost' in what: cost()
    if 'robust' in what: robustness()
