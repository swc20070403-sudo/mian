"""Resumable job runner: GBDT feature-set tests (tasks 2, 3c, 4) and NN tests (task 3d).
Usage: python jobs.py <budget_seconds> [which]"""
import os, sys, time, json
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
from common import *

T0 = time.time()
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 400


def prep_excl(d, itr, extra):
    old = core.EXCLUDE
    core.EXCLUDE = list(old) + list(extra)
    try:
        pp = core.Prep(d, itr, K=50)
    finally:
        core.EXCLUDE = old
    return pp


def build(d, setname, itr):
    """returns X (all records), column names"""
    names = list(d['names'])
    V = d['V'].astype(float); dV = V - V[:, :1]
    tau = tau_std(d, itr)[:, None]; oh = onehot(d)
    ohn = [f'cond{k}' for k in range(6)]
    dcir = d['F'][:, names.index('ctx_dcir')][:, None]
    a31 = np.abs(V[:, 31] - V[:, 30])[:, None]; a40 = np.abs(V[:, 40] - V[:, 30])[:, None]
    tn = [f't{t}' for t in range(1, 101)]
    if setname == 'tau': return np.c_[tau, oh], ['tau'] + ohn
    if setname == 'V0': return np.c_[V[:, :1], tau, oh], ['glob_V0', 'tau'] + ohn
    if setname == 'amp': return np.c_[a31, a40, dcir, tau, oh], ['a31', 'a40', 'ctx_dcir', 'tau'] + ohn
    if setname == 'amp_V0': return np.c_[a31, a40, dcir, V[:, :1], tau, oh], ['a31', 'a40', 'ctx_dcir', 'glob_V0', 'tau'] + ohn
    if setname == 'shape':
        return np.c_[dV[:, 1:] / np.abs(V[:, 40] - V[:, 0])[:, None], tau, oh], tn + ['tau'] + ohn
    if setname == 'dV': return np.c_[dV[:, 1:], tau, oh], tn + ['tau'] + ohn
    if setname == 'absV': return np.c_[V, tau, oh], ['V0'] + tn + ['tau'] + ohn
    if setname == 'dV_V0': return np.c_[dV[:, 1:], V[:, :1], tau, oh], tn + ['glob_V0', 'tau'] + ohn
    if setname == 'abs':
        j = [names.index(n) for n in ABS]
        return np.c_[d['F'][:, j], tau, oh], ABS + ['tau'] + ohn
    if setname.startswith('K50'):
        extra = {'K50': [], 'K50noabs': ABS, 'K50noabswav': ABS + WAV, 'K50noabs_V0': ABS, 'K50_dV': []}[setname]
        pp = prep_excl(d, itr, extra)
        o = pp.transform(d, np.arange(d['y'].size))
        cols = [names[j] for j in pp.sel]
        X = np.c_[o['h'], o['tau']]; cols = cols + ['tau']
        if setname == 'K50noabs_V0':
            X = np.c_[X, V[:, :1]]; cols = cols + ['glob_V0']
        if setname == 'K50_dV':
            X = np.c_[X, dV[:, 1:]]; cols = cols + tn
        return X, cols
    raise ValueError(setname)


def perm_groups(cols):
    g = {}
    for k, c in enumerate(cols):
        g.setdefault(family(c), []).append(k)
    for k, c in enumerate(cols):
        if family(c) in ('seg-level', 'seg-shape'): g.setdefault('seg(all)', []).append(k)
        if c in ABS: g.setdefault('level(all)', []).append(k)
    return g


def gbdt(args):
    ds, setname, f = args
    path = f'{OUT}/gbdt/{ds}_{setname}_f{f}.npz'
    if os.path.exists(path): return path + ' exists'
    if time.time() - T0 > BUDGET: return path + ' skipped(budget)'
    t = time.time()
    d = load(ds)
    itr, iva, ite = split_idx(d, f)
    X, cols = build(d, setname, itr)
    r = fit_gbdt(X[itr], d['y'][itr], X[iva], d, iva, X[ite], return_model=setname in ('K50', 'K50noabs'))
    extra = {}
    if 'predict' in r:
        base = macro(r['pred'], d, ite)
        rng = np.random.default_rng(f)
        imp = {}
        for g, ks in perm_groups(cols).items():
            vals = []
            for rep in range(5):
                Xp = X[ite].copy(); pi = rng.permutation(ite.size)
                Xp[:, ks] = Xp[pi][:, ks]
                vals.append(macro(r['predict'](Xp), d, ite) - base)
            imp[g] = (float(np.mean(vals)), float(np.std(vals)), len(ks))
        extra['perm'] = json.dumps(imp)
    os.makedirs(f'{OUT}/gbdt', exist_ok=True)
    np.savez_compressed(path, pred=r['pred'], idx=ite, val=r['val'], hp=np.array(r['hp']), cols=np.array(cols), **extra)
    return f'{path} {time.time() - t:.0f}s hp={r["hp"]}'


def nn(args):
    ds, tag, f, seed = args
    path = f'{OUT}/nn/{ds}_{tag}_f{f}_s{seed}.npz'
    if os.path.exists(path): return path + ' exists'
    if time.time() - T0 > BUDGET: return path + ' skipped(budget)'
    import torch
    torch.set_num_threads(1)
    d = load(ds)
    tr, va, te = cv_folds(d['cell'])[f]
    mode = 'A2' if tag.startswith('A2') else 'R0'
    raw = 'raw' in tag
    old = core.EXCLUDE
    if 'noabs' in tag: core.EXCLUDE = list(old) + ABS
    try:
        r = core.train_nn(d, tr, va, te, seed, mode=mode, raw_voltage=raw)
    finally:
        core.EXCLUDE = old
    os.makedirs(f'{OUT}/nn', exist_ok=True)
    np.savez_compressed(path, pred=r['pred'], idx=r['idx'], val_rmse=r['val_rmse'], best_epoch=r['best_epoch'],
                        epochs=r['epochs'], time=r['time'], sel=np.array(r['sel']))
    return f'{path} {r["time"]:.0f}s ep={r["epochs"]} val={r["val_rmse"]:.3f}'


def run_job(j):
    try:
        return nn(j[1:]) if j[0] == 'nn' else gbdt(j[1:])
    except Exception as e:
        import traceback
        return f'ERROR {j}: {traceback.format_exc()}'


if __name__ == '__main__':
    which = sys.argv[2] if len(sys.argv) > 2 else 'all'
    jobs = []
    if which in ('all', 'nn'):
        for tag in ['R0raw', 'A2noabs', 'R0noabs', 'R0rawnoabs']:
            jobs += [('nn', 'NMC', tag, f, 0) for f in range(5)]
    if which in ('all', 'nn12'):
        for tag in ['A2noabs', 'R0noabs', 'R0raw', 'R0rawnoabs']:
            jobs += [('nn', 'NMC', tag, f, s) for s in (1, 2) for f in range(5)]
    if which in ('all', 'kdv'):
        jobs += [('gbdt', ds, 'K50_dV', f) for ds in ('NMC', 'LFP') for f in range(5)]
    if which in ('all', 'gbdt', 'gbdt_nmc'):
        for s in ['K50', 'K50noabs', 'dV', 'shape', 'absV', 'dV_V0', 'K50noabswav', 'K50noabs_V0', 'abs', 'amp', 'amp_V0', 'tau']:
            jobs += [('gbdt', 'NMC', s, f) for f in range(5)]
    if which in ('all', 'gbdt', 'gbdt_lfp'):
        for s in ['K50', 'K50noabs', 'dV', 'shape', 'absV', 'dV_V0', 'K50noabswav', 'K50noabs_V0', 'abs', 'amp', 'amp_V0', 'tau']:
            jobs += [('gbdt', 'LFP', s, f) for f in range(5)]
    if which == 'v0':
        jobs += [('gbdt', ds, 'V0', f) for ds in ('NMC', 'LFP') for f in range(5)]
    if which == 'nn_lfp':
        jobs += [('nn', 'LFP', 'R0raw', f, 0) for f in range(5)]
    todo = [j for j in jobs if not os.path.exists(
        f'{OUT}/nn/{j[1]}_{j[2]}_f{j[3]}_s{j[4]}.npz' if j[0] == 'nn' else f'{OUT}/gbdt/{j[1]}_{j[2]}_f{j[3]}.npz')]
    print(f'{len(todo)} jobs to do', flush=True)
    from multiprocessing import Pool
    with Pool(2) as p:
        for r in p.imap(run_job, todo, chunksize=1):
            print(f'[{time.time() - T0:.0f}s] {r}', flush=True)
