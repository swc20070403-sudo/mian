import sys, os, json, itertools
os.environ['OMP_NUM_THREADS'] = '1'
W = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/work'
D = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/nmcdiag/soc9'
sys.path.insert(0, W)
import numpy as np
from core import cell_rmse, cv_folds, Prep
from stats import paired, holm
from sklearn.ensemble import HistGradientBoostingRegressor
d3 = dict(np.load(W + '/data_UConn-ILCC-NMC.npz'))
d9 = dict(np.load(D + '/data_UConn-ILCC-NMC.npz'))
os.makedirs(D + '/runs3', exist_ok=True)
# 1) reduced-grid GBDT on the 3-SOC dataset (matched grid)
for f in range(5):
    path = f'{D}/runs3/GBDT_f{f}.npz'
    if os.path.exists(path):
        continue
    tr, va, te = cv_folds(d3['cell'])[f]
    cell = d3['cell']
    itr = np.nonzero(np.isin(cell, tr))[0]; iva = np.nonzero(np.isin(cell, va))[0]; ite = np.nonzero(np.isin(cell, te))[0]
    pp = Prep(d3, itr, K=50)
    X = lambda idx: (lambda o: np.c_[o['h'], o['tau']])(pp.transform(d3, idx))
    Xtr, Xva, Xte = X(itr), X(iva), X(ite)
    best = (np.inf, None, None)
    for lr, leaves, msl, it in itertools.product([0.1], [31], [20, 100], [300, 1000]):
        m = HistGradientBoostingRegressor(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl, max_iter=it,
                                          early_stopping=False, random_state=0).fit(Xtr, d3['y'][itr])
        v = np.mean(list(cell_rmse(m.predict(Xva), d3['y'][iva], cell[iva], d3['rpt'][iva]).values()))
        if v < best[0]:
            best = (v, m, (lr, leaves, msl, it))
    np.savez_compressed(path, pred=best[1].predict(Xte), idx=ite, val_rmse=best[0], hp=np.array(best[2]))
    print('fold', f, best[2], round(best[0], 3), flush=True)

def preds(rdir, mo, n, seeds):
    out = []
    for s in seeds:
        p = np.full(n, np.nan)
        for f in range(5):
            r = np.load(f'{rdir}/{mo}_f{f}.npz' if s is None else f'{rdir}/{mo}_f{f}_s{s}.npz'); p[r['idx']] = r['pred']
        out.append(p)
    return out
def pc(p, d, m=None):
    m = np.ones(len(d['y']), bool) if m is None else m
    r = cell_rmse(p[m], d['y'][m], d['cell'][m], d['rpt'][m]); return np.array([r[c] for c in sorted(r)])
rnd = lambda r: {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)) for k, v in r.items()}
out = {}
g3r = pc(preds(D + '/runs3', 'GBDT', len(d3['y']), [None])[0], d3)
g9 = preds(D + '/runs', 'GBDT', len(d9['y']), [None])[0]
m3 = np.isin(d9['soc'], [20, 50, 90])
g9e3 = pc(g9, d9, m3)
out['GBDT_reducedgrid_3soc'] = round(float(g3r.mean()), 4)
out['GBDT_reducedgrid_9soc_eval3'] = round(float(g9e3.mean()), 4)
out['GBDT_9_minus_3_reducedgrid_eval3'] = rnd(paired(g9e3, g3r))
print('GBDT reduced grid: 3-SOC %.3f | 9-SOC eval on 20/50/90 %.3f | paired' % (g3r.mean(), g9e3.mean()), out['GBDT_9_minus_3_reducedgrid_eval3'])
# 2) seed noise reference: R0 s0 vs R0 s1, A2 s0 vs s1 (per cell, 9-SOC, all 18)
for mo in ['R0', 'A2']:
    P = preds(D + '/runs', mo, len(d9['y']), [0, 1])
    r = rnd(paired(pc(P[0], d9), pc(P[1], d9)))
    out[f'seed_noise_{mo}_s0_minus_s1_9soc'] = r
    print(f'{mo} seed0 - seed1 (9-SOC):', r)
# same-seed R0-A2 per seed
for s in [0, 1]:
    a = pc(preds(D + '/runs', 'R0', len(d9['y']), [s])[0], d9); b = pc(preds(D + '/runs', 'A2', len(d9['y']), [s])[0], d9)
    out[f'R0_minus_A2_seed{s}_9soc'] = rnd(paired(a, b)); print(f'seed {s} R0-A2 9-SOC:', out[f'R0_minus_A2_seed{s}_9soc'])
# 3) Holm over 18 SOC x direction tests
an = json.load(open(D + '/analysis.json'))
keys = list(an['per_soc_dir_soc9_train']); ps = [an['per_soc_dir_soc9_train'][k]['p'] for k in keys]
adj = holm(ps)
out['per_soc_dir_holm18'] = {k: round(float(a), 4) for k, a in zip(keys, adj)}
print('min Holm-18 p:', min(adj), keys[int(np.argmin(adj))])
# count of SOC x dir conditions with negative diff
diffs = [an['per_soc_dir_soc9_train'][k]['diff'] for k in keys]
out['n_socdir_negative'] = int(np.sum(np.array(diffs) < 0)); print('neg diffs', out['n_socdir_negative'], 'of 18; mean', np.mean(diffs))
# 4) validation RMSE means
for mo in ['R0', 'A2']:
    v9 = [float(np.load(f'{D}/runs/{mo}_f{f}_s{s}.npz')['val_rmse']) for f in range(5) for s in [0, 1]]
    v3 = [float(np.load(f'{W}/results/nmc/{mo}_f{f}_s{s}.npz')['val_rmse']) for f in range(5) for s in [0, 1]]
    b3 = [int(np.load(f'{W}/results/nmc/{mo}_f{f}_s{s}.npz')['best_epoch']) for f in range(5) for s in [0, 1]]
    b9 = [int(np.load(f'{D}/runs/{mo}_f{f}_s{s}.npz')['best_epoch']) for f in range(5) for s in [0, 1]]
    out[f'{mo}_val'] = dict(val9=round(np.mean(v9), 4), val3=round(np.mean(v3), 4), best_epoch9=np.mean(b9), best_epoch3=np.mean(b3))
    print(mo, out[f'{mo}_val'])
json.dump(out, open(D + '/extra.json', 'w'), indent=1)
