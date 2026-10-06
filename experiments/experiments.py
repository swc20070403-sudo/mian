"""Job runner for supplementary experiments. Usage: python experiments.py <exp> [nproc]"""
import os, sys, json, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = 'results'


def nn_job(args):
    exp, dname, fold_kind, f, seed, mode, kw = args
    import torch
    torch.set_num_threads(1)
    from core import load_data, cv_folds, logo_folds, train_nn
    path = f'{OUT}/{exp}/{mode}_f{f}_s{seed}.npz'
    if os.path.exists(path):
        return path
    d = load_data(dname)
    folds = cv_folds(d['cell']) if fold_kind == 'cv' else logo_folds(d['cell'], d['group'])
    tr, va, te = folds[f]
    r = train_nn(d, tr, va, te, seed, mode=kw.get('base', mode if mode in ('R0', 'A2', 'A1') else 'R0'),
                 return_model=True, **{k: v for k, v in kw.items() if k != 'base'})
    os.makedirs(f'{OUT}/{exp}', exist_ok=True)
    torch.save(r['model'].state_dict(), path.replace('.npz', '.pt'))
    pp = r['prep']
    np.savez_compressed(path, pred=r['pred'], idx=r['idx'], val_rmse=r['val_rmse'], epochs=r['epochs'],
                        best_epoch=r['best_epoch'], time=r['time'], n_params=r['n_params'],
                        sel=np.array(r['sel']), sel_idx=np.array(pp.sel),
                        med=pp.med, mu=pp.mu, sd=pp.sd, wmu=pp.wmu, wsd=pp.wsd,
                        nmu=pp.nmu, nsd=pp.nsd, ymu=pp.ymu, ysd=pp.ysd, dmu=pp.dmu, dsd=pp.dsd)
    return path


def gbdt_job(args):
    exp, dname, fold_kind, f = args
    path = f'{OUT}/{exp}/GBDT_f{f}.npz'
    if os.path.exists(path):
        return path
    from sklearn.ensemble import HistGradientBoostingRegressor
    from core import load_data, cv_folds, logo_folds, Prep, cell_rmse
    import pickle
    d = load_data(dname)
    folds = cv_folds(d['cell']) if fold_kind == 'cv' else logo_folds(d['cell'], d['group'])
    tr, va, te = folds[f]
    cell = d['cell']
    itr = np.nonzero(np.isin(cell, tr))[0]; iva = np.nonzero(np.isin(cell, va))[0]
    ite = np.nonzero(np.isin(cell, te))[0]
    pp = Prep(d, itr, K=50)
    X = lambda idx: (lambda o: np.c_[o['h'], o['tau']])(pp.transform(d, idx))
    Xtr, Xva, Xte = X(itr), X(iva), X(ite)
    best = (np.inf, None, None)
    for lr, leaves, msl, it in itertools.product([0.05, 0.1], [15, 31], [20, 100], [300, 1000]):
        m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl,
                                          max_iter=it, early_stopping=False, random_state=0)
        m.fit(Xtr, d['y'][itr])
        v = np.mean(list(cell_rmse(m.predict(Xva), d['y'][iva], cell[iva], d['rpt'][iva]).values()))
        if v < best[0]:
            best = (v, m, (lr, leaves, msl, it))
    os.makedirs(f'{OUT}/{exp}', exist_ok=True)
    with open(path.replace('.npz', '.pkl'), 'wb') as fh:
        pickle.dump(best[1], fh)
    np.savez_compressed(path, pred=best[1].predict(Xte), idx=ite, val_rmse=best[0], hp=np.array(best[2]),
                        sel_idx=np.array(pp.sel))
    return path


def svr_job(args):
    exp, dname, fold_kind, f = args
    path = f'{OUT}/{exp}/SVR_f{f}.npz'
    if os.path.exists(path):
        return path
    from sklearn.svm import SVR
    from core import load_data, cv_folds, logo_folds, Prep, cell_rmse
    d = load_data(dname)
    folds = cv_folds(d['cell']) if fold_kind == 'cv' else logo_folds(d['cell'], d['group'])
    tr, va, te = folds[f]
    cell = d['cell']
    itr = np.nonzero(np.isin(cell, tr))[0]; iva = np.nonzero(np.isin(cell, va))[0]
    ite = np.nonzero(np.isin(cell, te))[0]
    pp = Prep(d, itr, K=50)
    def X(idx):
        o = pp.transform(d, idx)
        return np.c_[o['h'], o['tau']], o['y']
    (Xtr, ytr), (Xva, _), (Xte, _) = X(itr), X(iva), X(ite)
    best = (np.inf, None, None)
    for C, eps in itertools.product([1, 10, 100], [0.05, 0.1]):
        m = SVR(C=C, epsilon=eps, gamma='scale', cache_size=1000).fit(Xtr, ytr)
        v = np.mean(list(cell_rmse(pp.inv(m.predict(Xva)), d['y'][iva], cell[iva], d['rpt'][iva]).values()))
        if v < best[0]:
            best = (v, m, (C, eps))
    os.makedirs(f'{OUT}/{exp}', exist_ok=True)
    np.savez_compressed(path, pred=pp.inv(best[1].predict(Xte)), idx=ite, val_rmse=best[0], hp=np.array(best[2]))
    return path


def base_job(args):
    exp, dname, f = args
    path = f'{OUT}/{exp}/BASE_f{f}.npz'
    if os.path.exists(path):
        return path
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
    from core import load_data, cv_folds, Prep, cell_rmse
    d = load_data(dname)
    tr, va, te = cv_folds(d['cell'])[f]
    cell = d['cell']
    itr = np.nonzero(np.isin(cell, tr))[0]; iva = np.nonzero(np.isin(cell, va))[0]; ite = np.nonzero(np.isin(cell, te))[0]
    pp = Prep(d, itr, K=50)
    def X(idx, kind):
        o = pp.transform(d, idx)
        if kind == 'full': return np.c_[o['h'], o['tau']]
        if kind == 'tau': return o['tau'][:, None]
        return o['h']
    ytr = d['y'][itr]
    vr = lambda p: np.mean(list(cell_rmse(p, d['y'][iva], cell[iva], d['rpt'][iva]).values()))
    out = {}
    for kind in ['full', 'tau', 'notau']:
        Xtr, Xva, Xte = X(itr, kind), X(iva, kind), X(ite, kind)
        # ridge
        best = min(((vr(Ridge(alpha=a).fit(Xtr, ytr).predict(Xva)), a) for a in np.logspace(-4, 3, 8)))
        out[f'Ridge_{kind}'] = Ridge(alpha=best[1]).fit(Xtr, ytr).predict(Xte)
        # random forest: select on seed 0, then average seeds 0-2 predictions per seed
        cand = []
        for mf, msl in itertools.product([0.33, 1.0], [1, 5, 20]):
            m = RandomForestRegressor(300, max_features=mf, min_samples_leaf=msl, random_state=0, n_jobs=1).fit(Xtr, ytr)
            cand.append((vr(m.predict(Xva)), mf, msl, m))
        b = min(cand, key=lambda c: c[0])
        out[f'RF_{kind}_s0'] = b[3].predict(Xte)
        for sd in (1, 2):
            out[f'RF_{kind}_s{sd}'] = RandomForestRegressor(300, max_features=b[1], min_samples_leaf=b[2], random_state=sd,
                                                          n_jobs=1).fit(Xtr, ytr).predict(Xte)
        del cand
        if kind != 'full':
            bestg = (np.inf, None)
            for lr, leaves, msl, it in itertools.product([0.05, 0.1], [15, 31], [20, 100], [300, 1000]):
                m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl,
                                                  max_iter=it, early_stopping=False, random_state=0).fit(Xtr, ytr)
                v = vr(m.predict(Xva))
                if v < bestg[0]: bestg = (v, m)
            out[f'GBDT_{kind}'] = bestg[1].predict(Xte)
    # GPR on full input
    Xtr, Xte = X(itr, 'full'), X(ite, 'full')
    ys = (ytr - ytr.mean()) / ytr.std()
    rng = np.random.default_rng(0); sub = rng.choice(len(itr), 2000, replace=False)
    k = ConstantKernel(1.0) * RBF(length_scale=np.ones(Xtr.shape[1])) + WhiteKernel(0.1)
    g = GaussianProcessRegressor(k, n_restarts_optimizer=2, random_state=0).fit(Xtr[sub], ys[sub])
    g2 = GaussianProcessRegressor(g.kernel_, optimizer=None).fit(Xtr, ys)
    out['GPR_full'] = g2.predict(Xte) * ytr.std() + ytr.mean()
    os.makedirs(f'{OUT}/{exp}', exist_ok=True)
    np.savez_compressed(path, idx=ite, **out)
    return path


def base_light_job(args):
    exp, dname, f = args
    path = f'{OUT}/{exp}/BASEL_f{f}.npz'
    if os.path.exists(path):
        return path
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from core import load_data, cv_folds, Prep, cell_rmse
    d = load_data(dname)
    tr, va, te = cv_folds(d['cell'])[f]
    cell = d['cell']
    itr = np.nonzero(np.isin(cell, tr))[0]; iva = np.nonzero(np.isin(cell, va))[0]; ite = np.nonzero(np.isin(cell, te))[0]
    pp = Prep(d, itr, K=50)
    def X(idx, kind):
        o = pp.transform(d, idx)
        return {'full': np.c_[o['h'], o['tau']], 'tau': o['tau'][:, None], 'notau': o['h']}[kind]
    ytr = d['y'][itr]
    vr = lambda p: np.mean(list(cell_rmse(p, d['y'][iva], cell[iva], d['rpt'][iva]).values()))
    out = {}
    for kind in ['full', 'tau', 'notau']:
        Xtr, Xva, Xte = X(itr, kind), X(iva, kind), X(ite, kind)
        best = min(((vr(Ridge(alpha=a).fit(Xtr, ytr).predict(Xva)), a) for a in np.logspace(-4, 3, 8)))
        out[f'Ridge_{kind}'] = Ridge(alpha=best[1]).fit(Xtr, ytr).predict(Xte)
        if kind == 'tau':
            cand = []
            for mf, msl in itertools.product([0.33, 1.0], [1, 5, 20]):
                m = RandomForestRegressor(300, max_features=mf, min_samples_leaf=msl, random_state=0, n_jobs=1).fit(Xtr, ytr)
                cand.append((vr(m.predict(Xva)), mf, msl, m))
            b = min(cand, key=lambda c: c[0])
            out['RF_tau_s0'] = b[3].predict(Xte)
            for sd in (1, 2):
                out[f'RF_tau_s{sd}'] = RandomForestRegressor(300, max_features=b[1], min_samples_leaf=b[2],
                                                            random_state=sd, n_jobs=1).fit(Xtr, ytr).predict(Xte)
        if kind != 'full':
            bestg = (np.inf, None)
            for lr, leaves, msl, it in itertools.product([0.05, 0.1], [15, 31], [20, 100], [300, 1000]):
                m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl,
                                                  max_iter=it, early_stopping=False, random_state=0).fit(Xtr, ytr)
                v = vr(m.predict(Xva))
                if v < bestg[0]: bestg = (v, m)
            out[f'GBDT_{kind}'] = bestg[1].predict(Xte)
    os.makedirs(f'{OUT}/{exp}', exist_ok=True)
    np.savez_compressed(path, idx=ite, **out)
    return path


def run(jobs, fn, nproc):
    from multiprocessing import Pool
    t = time.time()
    with Pool(nproc) as p:
        for i, r in enumerate(p.imap_unordered(fn, jobs)):
            print(f'[{i + 1}/{len(jobs)}] {r} {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    exp = sys.argv[1]; nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    LFP, NMC = 'UConn-ILCC-LFP', 'UConn-ILCC-NMC'
    seeds = [0, 1, 2]
    if exp == 'cv':      # reproduction under the original 5-fold protocol
        jobs = [('cv', LFP, 'cv', f, s, m, {}) for m in ['R0', 'A2'] for f in range(5) for s in seeds]
        run([('cv', LFP, 'cv', f) for f in range(5)], gbdt_job, nproc)
        run(jobs, nn_job, nproc)
    elif exp == 'logo':  # unseen cycling conditions
        jobs = [('logo', LFP, 'logo', f, s, m, {}) for m in ['R0', 'A2'] for f in range(11) for s in seeds]
        run([('logo', LFP, 'logo', f) for f in range(11)], gbdt_job, nproc)
        run(jobs, nn_job, nproc)
    elif exp == 'nmc':   # independent NMC/Gr dataset, same pulse protocol
        jobs = [('nmc', NMC, 'cv', f, s, m, {}) for m in ['R0', 'A2', 'A1'] for f in range(5) for s in seeds]
        run([('nmc', NMC, 'cv', f) for f in range(5)], gbdt_job, nproc)
        run(jobs, nn_job, nproc)
    elif exp.startswith('noisy'):  # matched noise in training and test data
        sig = exp[5:]
        dn = f'UConn-ILCC-LFP-n{sig}'
        jobs = [(exp, dn, 'cv', f, s, m, {}) for m in ['R0', 'A2'] for f in range(5) for s in seeds]
        run([(exp, dn, 'cv', f) for f in range(5)], gbdt_job, nproc)
        run(jobs, nn_job, nproc)
    elif exp == 'baselight':
        run([('cv', LFP, f) for f in range(5)], base_light_job, nproc)
    elif exp == 'base':
        run([('cv', LFP, f) for f in range(5)], base_job, nproc)
    elif exp == 'svr':
        run([('cv', LFP, 'cv', f) for f in range(5)], svr_job, nproc)
