"""Extra (part e): localise what the A2 residual depends on: tau only, tau+cond, h only (no tau), w+cond+tau.
Same protocol as part d (select hp on fold's val cells, refit on train+val cells, predict test fold)."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jobs import *


def job_e(args):
    ds, f = args
    import torch
    torch.set_num_threads(1)
    from core import Prep
    path = os.path.join(OUT, f'{ds}_e_f{f}.npz')
    if os.path.exists(path):
        return path + ' (exists)'
    t0 = time.time()
    d = load(ds)
    itr, iva, ite = split(d, f)
    pp = Prep(d, itr, K=50)
    Ftr, Fva, Fte = feats(d, pp, itr), feats(d, pp, iva), feats(d, pp, ite)
    itv = np.concatenate([itr, iva])
    Ftv = {k: np.concatenate([Ftr[k], Fva[k]]) for k in Ftr}
    X = lambda F, keys: np.concatenate([F[k] for k in keys], 1)
    y = d['y']
    b = oof(ds, d, 'A2'); r = y - b
    sc = lambda p: macro(b[iva] + p, d, iva)
    out = dict(idx=ite, base_A2=b[ite]); meta = {}
    for nm, keys in {'tau': ['tau'], 'tau_cond': ['tau', 'cond'], 'h': ['h'], 'w_cond_tau': ['w', 'cond', 'tau']}.items():
        v, hp, _ = gbdt_grid(X(Ftr, keys), r[itr], X(Fva, keys), sc)
        m = gbdt_fit(X(Ftv, keys), r[itv], hp)
        out[f'A2_{nm}'] = b[ite] + m.predict(X(Fte, keys))
        meta[f'A2_{nm}'] = dict(val=v, hp=list(hp))
    np.savez_compressed(path, **out)
    with open(path.replace('.npz', '.json'), 'w') as fh:
        json.dump(meta, fh)
    return f'{path} {time.time() - t0:.0f}s'


def job_f(args):
    """Extra (part f): waveform compressed to its top-10 training PCs: GBDT SOH [h,tau,pc] vs [h,tau] (from part b),
    and A2-residual model [h,tau,pc] and [pc,cond]."""
    ds, f = args
    import torch
    torch.set_num_threads(1)
    from core import Prep
    path = os.path.join(OUT, f'{ds}_f_f{f}.npz')
    if os.path.exists(path):
        return path + ' (exists)'
    t0 = time.time()
    d = load(ds)
    itr, iva, ite = split(d, f)
    pp = Prep(d, itr, K=50)
    Ftr, Fva, Fte = feats(d, pp, itr), feats(d, pp, iva), feats(d, pp, ite)
    mu = Ftr['w'].mean(0)
    _, _, Vt = np.linalg.svd(Ftr['w'] - mu, full_matrices=False)
    P = Vt[:10].T
    for F in (Ftr, Fva, Fte):
        F['pc'] = ((F['w'] - mu) @ P).astype(np.float32)
    itv = np.concatenate([itr, iva])
    Ftv = {k: np.concatenate([Ftr[k], Fva[k]]) for k in Ftr}
    X = lambda F, keys: np.concatenate([F[k] for k in keys], 1)
    y = d['y']
    out = dict(idx=ite); meta = {}
    sc_va = lambda p: macro(p, d, iva)
    v, hp, _ = gbdt_grid(X(Ftr, ['h', 'tau', 'pc']), y[itr], X(Fva, ['h', 'tau', 'pc']), sc_va)
    out['gbdt_h_tau_pc'] = gbdt_fit(X(Ftr, ['h', 'tau', 'pc']), y[itr], hp).predict(X(Fte, ['h', 'tau', 'pc']))
    meta['gbdt_h_tau_pc'] = dict(val=v, hp=list(hp))
    b = oof(ds, d, 'A2'); r = y - b
    sc = lambda p: macro(b[iva] + p, d, iva)
    for nm, keys in {'h_tau_pc': ['h', 'tau', 'pc'], 'pc_cond': ['pc', 'cond']}.items():
        v, hp, _ = gbdt_grid(X(Ftr, keys), r[itr], X(Fva, keys), sc)
        out[f'A2_{nm}'] = b[ite] + gbdt_fit(X(Ftv, keys), r[itv], hp).predict(X(Fte, keys))
        meta[f'A2_{nm}'] = dict(val=v, hp=list(hp))
    np.savez_compressed(path, **out)
    with open(path.replace('.npz', '.json'), 'w') as fh:
        json.dump(meta, fh)
    return f'{path} {time.time() - t0:.0f}s'


def run1(a):
    return a[0](a[1])


if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [(fn, (ds, f)) for fn in (job_e, job_f) for ds in ['NMC', 'LFP'] for f in range(5)]
    t = time.time()
    with Pool(2) as p:
        for i, r in enumerate(p.imap_unordered(run1, jobs)):
            print(f'[{i + 1}/{len(jobs)}] {r} {time.time() - t:.0f}s', flush=True)
