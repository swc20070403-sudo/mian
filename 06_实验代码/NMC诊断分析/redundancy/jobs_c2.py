"""Part c2 (refinement of c): the global PCA of w is dominated by pulse direction (PC1 ~99% of variance).
Here w is first centred by its training-fold mean per pulse condition (SOC x direction), then PCA is fitted on the
training cells; each test PC score is predicted from [h, tau, cond] with ridge and HistGBDT (val-selected)."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jobs import *


def job_c2(args):
    ds, f = args
    import torch
    torch.set_num_threads(1)
    from core import Prep
    from sklearn.linear_model import Ridge
    path = os.path.join(OUT, f'{ds}_c2_f{f}.json')
    if os.path.exists(path):
        return path + ' (exists)'
    t0 = time.time()
    d = load(ds)
    itr, iva, ite = split(d, f)
    pp = Prep(d, itr, K=50)
    Ftr, Fva, Fte = feats(d, pp, itr), feats(d, pp, iva), feats(d, pp, ite)
    ctr, cva, cte = cond_id(d, itr), cond_id(d, iva), cond_id(d, ite)
    W = {}
    cm = np.stack([Ftr['w'][ctr == c].astype(np.float64).mean(0) for c in range(6)])
    for nm, F, c in (('tr', Ftr, ctr), ('va', Fva, cva), ('te', Fte, cte)):
        W[nm] = F['w'].astype(np.float64) - cm[c]
    mu = W['tr'].mean(0)
    _, S, Vt = np.linalg.svd(W['tr'] - mu, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    K = 10
    P = Vt[:K].T
    s = {k: (v - mu) @ P for k, v in W.items()}
    X = lambda F: np.concatenate([F['h'], F['tau'], F['cond']], 1)
    Htr, Hva, Hte = X(Ftr), X(Fva), X(Fte)
    res = {'ev': ev[:K].tolist(), 'ev_cum10': float(ev[:K].sum()),
           'within_cond_var_share_of_total_w': float(((W['tr'] - mu) ** 2).sum() /
                                                     ((Ftr['w'] - Ftr['w'].mean(0)) ** 2).sum())}
    preds = {'ridge': np.zeros_like(s['te']), 'gbdt': np.zeros_like(s['te'])}
    for k in range(K):
        mse = lambda p, kk=k: float(np.mean((p - s['va'][:, kk]) ** 2))
        v, a = ridge_sel(Htr, s['tr'][:, k], Hva, mse)
        preds['ridge'][:, k] = Ridge(alpha=a).fit(Htr, s['tr'][:, k]).predict(Hte)
        v, hp, _ = gbdt_grid(Htr, s['tr'][:, k], Hva, mse)
        preds['gbdt'][:, k] = gbdt_fit(Htr, s['tr'][:, k], hp).predict(Hte)
    ste = s['te']
    sst = ((ste - ste.mean(0)) ** 2).sum(0)
    sst_w = ((W['te'] - W['te'].mean(0)) ** 2).sum()
    res['sst'] = sst.tolist(); res['sst_w'] = float(sst_w)
    for nm, p in preds.items():
        sse = ((ste - p) ** 2).sum(0)
        res[f'{nm}_sse'] = sse.tolist()
        res[f'{nm}_r2_pc'] = (1 - sse / sst).tolist()
        res[f'{nm}_r2_top10'] = float(1 - sse.sum() / sst.sum())
        what = mu + p @ P.T
        res[f'{nm}_r2_w'] = float(1 - ((W['te'] - what) ** 2).sum() / sst_w)
    res['time'] = time.time() - t0
    with open(path, 'w') as fh:
        json.dump(res, fh)
    return f'{path} {time.time() - t0:.0f}s'


if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [(ds, f) for ds in ['NMC', 'LFP'] for f in range(5)]
    t = time.time()
    with Pool(2) as p:
        for i, r in enumerate(p.imap_unordered(job_c2, jobs)):
            print(f'[{i + 1}/{len(jobs)}] {r} {time.time() - t:.0f}s', flush=True)
