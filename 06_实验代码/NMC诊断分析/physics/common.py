import os, sys, itertools, numpy as np
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
W = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/work'
D = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/nmcdiag/physics'
OUT = D + '/out'
sys.path.insert(0, W)
import core
from core import cv_folds, cell_rmse, EXCLUDE
from stats import paired

DS = {'LFP': 'UConn-ILCC-LFP', 'NMC': 'UConn-ILCC-NMC'}

ABS = ([f'seg_{s}_{o}' for s in ('s1', 's2', 's3') for o in ('mean', 'rms', 'energy', 'auc', 'intercept')]
       + ['glob_mean', 'glob_auc', 'glob_V0', 'glob_V30', 'glob_V40', 'glob_V100']
       + [f'glob_q{q}' for q in (5, 10, 25, 50, 75, 90, 95)])
WAV = ['wav_cA4', 'wav_cD4', 'wav_cD3', 'wav_cD2', 'wav_cD1']


def load(ds):
    z = np.load(f'{W}/data_{DS[ds]}.npz', allow_pickle=False)
    return {k: z[k] for k in z.files}


def family(n):
    if n == 'tau': return 'tau'
    if n == 'ctx_dcir': return 'dcir'
    if n.startswith('ctx_'): return 'ctx'
    if n.startswith('seg_'): return 'seg-level' if n in ABS else 'seg-shape'
    if n.startswith('glob_'): return 'glob-level' if n in ABS else 'glob-shape'
    return n.split('_')[0]


def cond_id(d):
    return (np.searchsorted([20, 50, 90], d['soc']) * 2 + (d['ptype'] == 'chg')).astype(int)


def onehot(d):
    c = cond_id(d)
    return np.eye(6)[c]


def split_idx(d, f):
    tr, va, te = cv_folds(d['cell'])[f]
    cell = d['cell']
    return (np.nonzero(np.isin(cell, tr))[0], np.nonzero(np.isin(cell, va))[0], np.nonzero(np.isin(cell, te))[0])


def macro(pred, d, idx=None):
    if idx is None: idx = np.arange(d['y'].size)
    return float(np.mean(list(cell_rmse(pred, d['y'][idx], d['cell'][idx], d['rpt'][idx]).values())))


def fit_gbdt(Xtr, ytr, Xva, d, iva, Xte, return_model=False):
    """Paper grid lr{.05,.1} x leaves{15,31} x msl{20,100} x iters{300,1000}; 300 read from staged preds
    of the 1000-iter fit (identical: early_stopping=False, no subsampling)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    best = (np.inf, None, None)
    for lr, leaves, msl in itertools.product([0.05, 0.1], [15, 31], [20, 100]):
        m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl,
                                          max_iter=1000, early_stopping=False, random_state=0).fit(Xtr, ytr)
        for it, pv in enumerate(m.staged_predict(Xva), 1):
            if it in (300, 1000):
                v = macro(pv, d, iva)
                if v < best[0]:
                    best = (v, m, (lr, leaves, msl, it))
    m, hp = best[1], best[2]
    if hp[3] == 1000:
        pt = m.predict(Xte); pred_fn = m.predict
    else:
        def pred_fn(X):
            for it, p in enumerate(m.staged_predict(X), 1):
                if it == 300: return p
        pt = pred_fn(Xte)
    out = dict(pred=pt, val=best[0], hp=hp)
    if return_model: out['predict'] = pred_fn
    return out


def tau_std(d, itr):
    mu, sd = d['cum'][itr].mean(), d['cum'][itr].std()
    return (d['cum'] - mu) / sd


def per_cell(pred_full, d):
    r = cell_rmse(pred_full, d['y'], d['cell'], d['rpt'])
    cells = sorted(r)
    return np.array(cells), np.array([r[c] for c in cells])


def pstat(a, b):
    s = paired(a, b)
    return {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else int(v)) if k not in ('p',) else float(f'{v:.4g}') for k, v in s.items()}
