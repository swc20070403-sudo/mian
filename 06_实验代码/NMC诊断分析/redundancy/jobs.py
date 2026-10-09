"""Redundancy diagnostics: how much SOH information does dV carry beyond the 50 selected features + tau?
Usage: python jobs.py <part> [nproc]   part in {a,b,c,d,all}
Each job (dataset, fold, part) writes out/<ds>_<part>_f<fold>.npz (or .json) and is skipped if it exists."""
import os, sys, json, time, itertools
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np

WORK = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/work'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
sys.path.insert(0, WORK)
DS = {'LFP': ('UConn-ILCC-LFP', 'cv'), 'NMC': ('UConn-ILCC-NMC', 'nmc')}
ALPHAS = np.logspace(-4, 4, 9)
# small grid: lr 0.1; leaves {15,31} x min_samples_leaf {20,100} x max_iter {300,1000} (warm start 300->1000)
GRID = list(itertools.product([15, 31], [20, 100]))
ITERS = [300, 1000]


def load(ds):
    z = np.load(os.path.join(WORK, f'data_{DS[ds][0]}.npz'), allow_pickle=False)
    return {k: z[k] for k in z.files}


def cond_onehot(d, idx):
    s = np.searchsorted([20, 50, 90], d['soc'][idx])
    c = s * 2 + (d['ptype'][idx] == 'dchg').astype(int)
    return np.eye(6, dtype=np.float32)[c]


def cond_id(d, idx):
    return np.searchsorted([20, 50, 90], d['soc'][idx]) * 2 + (d['ptype'][idx] == 'dchg').astype(int)


def split(d, f):
    from core import cv_folds
    tr, va, te = cv_folds(d['cell'])[f]
    c = d['cell']
    return (np.nonzero(np.isin(c, tr))[0], np.nonzero(np.isin(c, va))[0], np.nonzero(np.isin(c, te))[0])


def macro(pred, d, idx):
    from core import cell_rmse
    return float(np.mean(list(cell_rmse(pred, d['y'][idx], d['cell'][idx], d['rpt'][idx]).values())))


def gbdt_grid(Xtr, ytr, Xva, score):
    """score(pred_va) -> lower is better. Returns (best_score, (leaves, msl, iters), val scores dict)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    best = (np.inf, None); allv = {}
    for leaves, msl in GRID:
        m = HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=leaves, min_samples_leaf=msl,
                                          max_iter=ITERS[0], early_stopping=False, random_state=0)
        for it in ITERS:
            m.set_params(max_iter=it, warm_start=True)
            m.fit(Xtr, ytr)
            v = score(m.predict(Xva))
            allv[f'{leaves}_{msl}_{it}'] = v
            if v < best[0]:
                best = (v, (leaves, msl, it))
    return best[0], best[1], allv


def gbdt_fit(X, y, hp):
    from sklearn.ensemble import HistGradientBoostingRegressor
    leaves, msl, it = hp
    return HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=leaves, min_samples_leaf=msl,
                                         max_iter=it, early_stopping=False, random_state=0).fit(X, y)


def ridge_sel(Xtr, ytr, Xva, score):
    from sklearn.linear_model import Ridge
    best = (np.inf, None)
    for a in ALPHAS:
        v = score(Ridge(alpha=a).fit(Xtr, ytr).predict(Xva))
        if v < best[0]:
            best = (v, a)
    return best


def feats(d, pp, idx):
    o = pp.transform(d, idx)
    return dict(h=o['h'], tau=o['tau'][:, None], w=o['w'], cond=cond_onehot(d, idx))


def vabs(d, itr, idx):
    mu = d['V'][itr].mean(0); sd = d['V'][itr].std(0); sd[sd < 1e-12] = 1
    return ((d['V'][idx] - mu) / sd).astype(np.float32)


def oof(ds, d, mode):
    ex = DS[ds][1]
    pred = np.zeros(d['y'].size); cnt = np.zeros(d['y'].size)
    for f in range(5):
        for s in range(3):
            z = np.load(os.path.join(WORK, 'results', ex, f'{mode}_f{f}_s{s}.npz'))
            pred[z['idx']] += z['pred']; cnt[z['idx']] += 1
    assert np.all(cnt == 3)
    return pred / 3


def job(args):
    ds, f, part = args
    import torch
    torch.set_num_threads(1)
    from core import Prep
    path = os.path.join(OUT, f'{ds}_{part}_f{f}.npz')
    if os.path.exists(path):
        return path + ' (exists)'
    t0 = time.time()
    d = load(ds)
    itr, iva, ite = split(d, f)
    pp = Prep(d, itr, K=50)
    Ftr, Fva, Fte = feats(d, pp, itr), feats(d, pp, iva), feats(d, pp, ite)
    Vtr, Vva, Vte = vabs(d, itr, itr), vabs(d, itr, iva), vabs(d, itr, ite)
    for F, V in ((Ftr, Vtr), (Fva, Vva), (Fte, Vte)):
        F['vabs'] = V
    y = d['y']
    sc_va = lambda p: macro(p, d, iva)
    out = dict(idx=ite)
    meta = {}

    def X(F, keys):
        return np.concatenate([F[k] for k in keys], 1)

    if part == 'a' or part == 'b':
        if part == 'a':
            sets = {'w_cond': ['w', 'cond'], 'w_cond_tau': ['w', 'cond', 'tau'], 'vabs_cond': ['vabs', 'cond'],
                    'h_tau': ['h', 'tau']}
            for nm, keys in sets.items():
                v, a = ridge_sel(X(Ftr, keys), y[itr], X(Fva, keys), sc_va)
                from sklearn.linear_model import Ridge
                out[f'ridge_{nm}'] = Ridge(alpha=a).fit(X(Ftr, keys), y[itr]).predict(X(Fte, keys))
                meta[f'ridge_{nm}'] = dict(val=v, alpha=float(a))
            gsets = {'w_cond': ['w', 'cond'], 'w_cond_tau': ['w', 'cond', 'tau'], 'vabs_cond': ['vabs', 'cond']}
        else:
            gsets = {'h_tau': ['h', 'tau'], 'h_tau_w': ['h', 'tau', 'w']}
        for nm, keys in gsets.items():
            v, hp, allv = gbdt_grid(X(Ftr, keys), y[itr], X(Fva, keys), sc_va)
            m = gbdt_fit(X(Ftr, keys), y[itr], hp)
            out[f'gbdt_{nm}'] = m.predict(X(Fte, keys))
            meta[f'gbdt_{nm}'] = dict(val=v, hp=list(hp))

    elif part == 'c':
        from sklearn.linear_model import Ridge
        wtr, wva, wte = Ftr['w'].astype(np.float64), Fva['w'].astype(np.float64), Fte['w'].astype(np.float64)
        mu = wtr.mean(0)
        U, S, Vt = np.linalg.svd(wtr - mu, full_matrices=False)
        ev = S ** 2 / (S ** 2).sum()
        K = 10
        P = Vt[:K].T
        str_, sva, ste = (wtr - mu) @ P, (wva - mu) @ P, (wte - mu) @ P
        Htr, Hva, Hte = X(Ftr, ['h', 'tau']), X(Fva, ['h', 'tau']), X(Fte, ['h', 'tau'])
        ctr, cte = cond_id(d, itr), cond_id(d, ite)
        res = {'ev': ev[:K].tolist(), 'ev_cum10': float(ev[:K].sum())}
        preds = {'ridge': np.zeros_like(ste), 'gbdt': np.zeros_like(ste), 'cond': np.zeros_like(ste)}
        hps = {'ridge': [], 'gbdt': []}
        for k in range(K):
            mse = lambda p, kk=k: float(np.mean((p - sva[:, kk]) ** 2))
            v, a = ridge_sel(Htr, str_[:, k], Hva, mse)
            preds['ridge'][:, k] = Ridge(alpha=a).fit(Htr, str_[:, k]).predict(Hte); hps['ridge'].append(float(a))
            v, hp, _ = gbdt_grid(Htr, str_[:, k], Hva, mse)
            preds['gbdt'][:, k] = gbdt_fit(Htr, str_[:, k], hp).predict(Hte); hps['gbdt'].append(list(hp))
            cm = np.array([str_[ctr == c, k].mean() for c in range(6)])
            preds['cond'][:, k] = cm[cte]
        sst = ((ste - ste.mean(0)) ** 2).sum(0)
        sst_w = ((wte - wte.mean(0)) ** 2).sum()
        for nm, p in preds.items():
            sse = ((ste - p) ** 2).sum(0)
            res[f'{nm}_r2_pc'] = (1 - sse / sst).tolist()
            res[f'{nm}_r2_top10'] = float(1 - sse.sum() / sst.sum())
            what = mu + p @ P.T
            res[f'{nm}_r2_w'] = float(1 - ((wte - what) ** 2).sum() / sst_w)
            res[f'{nm}_sse'] = sse.tolist()
        res['sst'] = sst.tolist(); res['sst_w'] = float(sst_w)
        # within-condition: share of the condition-mean residual SSE removed by [h,tau]
        for nm in ['ridge', 'gbdt']:
            res[f'{nm}_r2_within_top10'] = float(1 - np.sum(res[f'{nm}_sse']) / np.sum(res['cond_sse']))
            res[f'{nm}_r2_within_pc'] = (1 - np.array(res[f'{nm}_sse']) / np.array(res['cond_sse'])).tolist()
        res['hps'] = hps
        res['time'] = time.time() - t0
        os.makedirs(OUT, exist_ok=True)
        with open(path.replace('.npz', '.json'), 'w') as fh:
            json.dump(res, fh)
        np.savez_compressed(path, idx=ite, **{f'pc_{k}': v for k, v in preds.items()}, pc_true=ste)
        return path

    elif part == 'd':
        base = {'A2': oof(ds, d, 'A2'), 'R0': oof(ds, d, 'R0')}
        itv = np.concatenate([itr, iva])
        Ftv = {k: np.concatenate([Ftr[k], Fva[k]]) for k in Ftr}
        sets = {'A2': {'w_cond': ['w', 'cond'], 'h_tau': ['h', 'tau'], 'h_tau_w': ['h', 'tau', 'w'],
                       'vabs_cond': ['vabs', 'cond'], 'h_tau_cond': ['h', 'tau', 'cond']},
                'R0': {'h_tau': ['h', 'tau'], 'h_tau_w': ['h', 'tau', 'w']}}
        for bm, ss in sets.items():
            b = base[bm]
            r = y - b
            out[f'base_{bm}'] = b[ite]
            sc = lambda p, bb=b: macro(bb[iva] + p, d, iva)
            meta[f'base_{bm}_val'] = macro(b[iva], d, iva)
            for nm, keys in ss.items():
                v, hp, allv = gbdt_grid(X(Ftr, keys), r[itr], X(Fva, keys), sc)
                m = gbdt_fit(X(Ftv, keys), r[itv], hp)   # refit on all four other folds' cells
                out[f'{bm}_{nm}'] = b[ite] + m.predict(X(Fte, keys))
                meta[f'{bm}_{nm}'] = dict(val=v, hp=list(hp))
    out['time'] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(path, **out)
    with open(path.replace('.npz', '.json'), 'w') as fh:
        json.dump(meta, fh)
    return f'{path} {time.time() - t0:.0f}s'


if __name__ == '__main__':
    part = sys.argv[1]; nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    parts = ['a', 'b', 'c', 'd'] if part == 'all' else part.split(',')
    dss = sys.argv[3].split(',') if len(sys.argv) > 3 else ['NMC', 'LFP']
    jobs = [(ds, f, p) for p in parts for ds in dss for f in range(5)]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(OUT, f'{j[0]}_{j[2]}_f{j[1]}.npz'))]
    from multiprocessing import Pool
    t = time.time()
    with Pool(nproc) as p:
        for i, r in enumerate(p.imap_unordered(job, jobs)):
            print(f'[{i + 1}/{len(jobs)}] {r} {time.time() - t:.0f}s', flush=True)
