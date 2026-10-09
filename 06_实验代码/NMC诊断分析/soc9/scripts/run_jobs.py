"""Train R0/A2 (seeds 0,1) and GBDT (reduced grid) on the 9-SOC NMC dataset; 2 worker processes; skip existing."""
import os, sys, time, itertools
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
W = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/work'
D = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/nmcdiag/soc9'
sys.path.insert(0, W)
os.chdir(D)
import numpy as np
OUT = D + '/runs'
NAME = 'UConn-ILCC-NMC'


def nn_job(args):
    f, seed, mode = args
    import torch
    torch.set_num_threads(1)
    from core import load_data, cv_folds, train_nn
    path = f'{OUT}/{mode}_f{f}_s{seed}.npz'
    if os.path.exists(path):
        return path + ' (exists)'
    d = load_data(NAME)
    tr, va, te = cv_folds(d['cell'])[f]
    r = train_nn(d, tr, va, te, seed, mode=mode)
    np.savez_compressed(path + '.tmp.npz', pred=r['pred'], idx=r['idx'], val_rmse=r['val_rmse'], epochs=r['epochs'],
                        best_epoch=r['best_epoch'], time=r['time'], n_params=r['n_params'], sel=np.array(r['sel']))
    os.replace(path + '.tmp.npz', path)
    return f"{path} t={r['time']:.0f}s ep={r['epochs']} best={r['best_epoch']} val={r['val_rmse']:.3f}"


def gbdt_job(f):
    path = f'{OUT}/GBDT_f{f}.npz'
    if os.path.exists(path):
        return path + ' (exists)'
    from sklearn.ensemble import HistGradientBoostingRegressor
    from core import load_data, cv_folds, Prep, cell_rmse
    d = load_data(NAME)
    tr, va, te = cv_folds(d['cell'])[f]
    cell = d['cell']
    itr = np.nonzero(np.isin(cell, tr))[0]; iva = np.nonzero(np.isin(cell, va))[0]; ite = np.nonzero(np.isin(cell, te))[0]
    pp = Prep(d, itr, K=50)
    X = lambda idx: (lambda o: np.c_[o['h'], o['tau']])(pp.transform(d, idx))
    Xtr, Xva, Xte = X(itr), X(iva), X(ite)
    best = (np.inf, None, None); allv = []
    for lr, leaves, msl, it in itertools.product([0.1], [31], [20, 100], [300, 1000]):
        m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl,
                                          max_iter=it, early_stopping=False, random_state=0)
        m.fit(Xtr, d['y'][itr])
        v = np.mean(list(cell_rmse(m.predict(Xva), d['y'][iva], cell[iva], d['rpt'][iva]).values()))
        allv.append((lr, leaves, msl, it, v))
        if v < best[0]:
            best = (v, m, (lr, leaves, msl, it))
    np.savez_compressed(path + '.tmp.npz', pred=best[1].predict(Xte), idx=ite, val_rmse=best[0], hp=np.array(best[2]),
                        grid=np.array(allv), sel=np.array([pp.names[j] for j in pp.sel]))
    os.replace(path + '.tmp.npz', path)
    return f'{path} hp={best[2]} val={best[0]:.3f}'


if __name__ == '__main__':
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1]
    t = time.time()
    with Pool(2) as p:
        if which in ('gbdt', 'all'):
            for r in p.imap_unordered(gbdt_job, range(5)):
                print(f'{time.time() - t:.0f}s {r}', flush=True)
        if which in ('nn', 'all'):
            # interleave R0 (slow) and A2 (fast)
            jobs = [(f, s, m) for s in (0, 1) for f in range(5) for m in ('R0', 'A2')]
            for r in p.imap_unordered(nn_job, jobs):
                print(f'{time.time() - t:.0f}s {r}', flush=True)
    print('DONE', flush=True)
