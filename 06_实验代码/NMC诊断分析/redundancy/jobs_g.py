"""Extra (part g): does adding the absolute voltage level V0 (= V[:,0], removed by dV referencing) to the waveform
help a strong learner?  GBDT SOH on [w,cond,V0] and [w,cond,tau,V0]; A2-residual model on [w,cond,V0]."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jobs import *


def job_g(args):
    ds, f = args
    import torch
    torch.set_num_threads(1)
    from core import Prep
    path = os.path.join(OUT, f'{ds}_g_f{f}.npz')
    if os.path.exists(path):
        return path + ' (exists)'
    t0 = time.time()
    d = load(ds)
    itr, iva, ite = split(d, f)
    pp = Prep(d, itr, K=50)
    Ftr, Fva, Fte = feats(d, pp, itr), feats(d, pp, iva), feats(d, pp, ite)
    m0, s0 = d['V'][itr, 0].mean(), d['V'][itr, 0].std()
    for F, idx in ((Ftr, itr), (Fva, iva), (Fte, ite)):
        F['v0'] = ((d['V'][idx, :1] - m0) / s0).astype(np.float32)
    itv = np.concatenate([itr, iva])
    Ftv = {k: np.concatenate([Ftr[k], Fva[k]]) for k in Ftr}
    X = lambda F, keys: np.concatenate([F[k] for k in keys], 1)
    y = d['y']
    out = dict(idx=ite); meta = {}
    sc_va = lambda p: macro(p, d, iva)
    for nm, keys in {'w_cond_v0': ['w', 'cond', 'v0'], 'w_cond_tau_v0': ['w', 'cond', 'tau', 'v0']}.items():
        v, hp, _ = gbdt_grid(X(Ftr, keys), y[itr], X(Fva, keys), sc_va)
        out[f'gbdt_{nm}'] = gbdt_fit(X(Ftr, keys), y[itr], hp).predict(X(Fte, keys))
        meta[f'gbdt_{nm}'] = dict(val=v, hp=list(hp))
    b = oof(ds, d, 'A2'); r = y - b
    sc = lambda p: macro(b[iva] + p, d, iva)
    keys = ['w', 'cond', 'v0']
    v, hp, _ = gbdt_grid(X(Ftr, keys), r[itr], X(Fva, keys), sc)
    out['A2_w_cond_v0'] = b[ite] + gbdt_fit(X(Ftv, keys), r[itv], hp).predict(X(Fte, keys))
    meta['A2_w_cond_v0'] = dict(val=v, hp=list(hp))
    np.savez_compressed(path, **out)
    with open(path.replace('.npz', '.json'), 'w') as fh:
        json.dump(meta, fh)
    return f'{path} {time.time() - t0:.0f}s'


if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [(ds, f) for ds in ['NMC', 'LFP'] for f in range(5)]
    t = time.time()
    with Pool(2) as p:
        for i, r in enumerate(p.imap_unordered(job_g, jobs)):
            print(f'[{i + 1}/{len(jobs)}] {r} {time.time() - t:.0f}s', flush=True)
