import sys, os, glob
W = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/work'
sys.path.insert(0, W)
import numpy as np
from core import cv_folds, cell_rmse
z9 = dict(np.load('/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/nmcdiag/soc9/data_UConn-ILCC-NMC.npz'))
z3 = dict(np.load(W + '/data_UConn-ILCC-NMC.npz'))
m = np.isin(z9['soc'], [20, 50, 90])
print('9soc n', len(z9['y']), '3soc-subset n', m.sum(), 'orig n', len(z3['y']))
for k in ['F', 'V', 'y', 'cell', 'rpt', 'soc', 'cum', 'group']:
    a, b = z9[k][m], z3[k]
    if a.dtype.kind == 'f':
        print(k, 'equal', np.allclose(a, b, equal_nan=True))
    else:
        print(k, 'equal', np.array_equal(a, b))
f9 = cv_folds(z9['cell']); f3 = cv_folds(z3['cell'])
print('folds equal', all(all(np.array_equal(a, b) for a, b in zip(x, y)) for x, y in zip(f9, f3)))
print('test cells per fold', [len(f[2]) for f in f9], 'train', [len(f[0]) for f in f9], 'val', [len(f[1]) for f in f9])
# reproduce headline numbers
for ds, dname, models in [('nmc', z3, ['R0', 'A2', 'A1']), ('cv', dict(np.load(W + '/data_UConn-ILCC-LFP.npz')), ['R0', 'A2'])]:
    d = dname
    for mo in models:
        per = []
        for s in range(3):
            pred = np.full(len(d['y']), np.nan)
            for f in range(5):
                r = np.load(f'{W}/results/{ds}/{mo}_f{f}_s{s}.npz')
                pred[r['idx']] = r['pred']
            per.append(cell_rmse(pred, d['y'], d['cell'], d['rpt']))
        cells = sorted(per[0])
        arr = np.mean([[p[c] for c in cells] for p in per], 0)
        print(ds, mo, round(arr.mean(), 3), ' seeds0-1 only:', round(np.mean([[p[c] for c in cells] for p in per[:2]], 0).mean(), 3))
    pred = np.full(len(d['y']), np.nan)
    for f in range(5):
        r = np.load(f'{W}/results/{ds}/GBDT_f{f}.npz'); pred[r['idx']] = r['pred']
        print('  GBDT hp fold', f, r['hp'])
    print(ds, 'GBDT', round(np.mean(list(cell_rmse(pred, d['y'], d['cell'], d['rpt']).values())), 3))
