import sys, os, json
W = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/work'
D = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/nmcdiag/soc9'
sys.path.insert(0, W)
import numpy as np
from core import cell_rmse, cv_folds
from stats import paired, holm

d9 = dict(np.load(D + '/data_UConn-ILCC-NMC.npz'))
d3 = dict(np.load(W + '/data_UConn-ILCC-NMC.npz'))
SEEDS = [0, 1]


def load_preds(rdir, mo, n, seeds):
    out = []
    for s in (seeds if mo != 'GBDT' else [None]):
        pred = np.full(n, np.nan)
        for f in range(5):
            fn = f'{rdir}/{mo}_f{f}.npz' if mo == 'GBDT' else f'{rdir}/{mo}_f{f}_s{s}.npz'
            r = np.load(fn)
            pred[r['idx']] = r['pred']
        assert np.isfinite(pred).all()
        out.append(pred)
    return out


def per_cell(preds, d, mask=None):
    """seed-averaged per-cell RMSE (cell-RPT aggregation within mask). returns cells, arr"""
    m = np.ones(len(d['y']), bool) if mask is None else mask
    per = [cell_rmse(p[m], d['y'][m], d['cell'][m], d['rpt'][m]) for p in preds]
    cells = sorted(per[0])
    return np.array(cells), np.mean([[q[c] for c in cells] for q in per], 0)


def rec_rmse(preds, d, mask=None):
    m = np.ones(len(d['y']), bool) if mask is None else mask
    return float(np.mean([np.sqrt(np.mean((p[m] - d['y'][m]) ** 2)) for p in preds]))


def pstat(a, b):
    r = paired(a, b)
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)) for k, v in r.items()}


res = {}
P9 = {mo: load_preds(D + '/runs', mo, len(d9['y']), SEEDS) for mo in ['R0', 'A2', 'GBDT']}
P3 = {mo: load_preds(W + '/results/nmc', mo, len(d3['y']), SEEDS) for mo in ['R0', 'A2']}
P3all = {mo: load_preds(W + '/results/nmc', mo, len(d3['y']), [0, 1, 2]) for mo in ['R0', 'A2']}
P3['GBDT'] = load_preds(W + '/results/nmc', 'GBDT', len(d3['y']), None)
P3all['GBDT'] = P3['GBDT']

# ---- overall, 9-SOC training, evaluated on all 18 conditions per cell-RPT
def block(P, d, mask=None, label=''):
    out = {}
    pc = {}
    for mo in P:
        cells, a = per_cell(P[mo], d, mask)
        pc[mo] = a
        out[mo] = dict(cell_macro_rmse=round(float(a.mean()), 4), record_rmse=round(rec_rmse(P[mo], d, mask), 4))
        if mo != 'GBDT':
            out[mo]['per_seed_cell_macro'] = [round(float(per_cell([p], d, mask)[1].mean()), 4) for p in P[mo]]
    out['R0_minus_A2'] = pstat(pc['R0'], pc['A2'])
    out['GBDT_minus_A2'] = pstat(pc['GBDT'], pc['A2'])
    out['cells'] = cells.tolist()
    out['_pc'] = pc
    return out

o9 = block(P9, d9)
res['soc9_train_eval_all18'] = o9
# 9-SOC training evaluated only on 20/50/90 records (6 per cell-RPT): comparable to the 3-SOC protocol
m3in9 = np.isin(d9['soc'], [20, 50, 90])
res['soc9_train_eval_3soc'] = block(P9, d9, m3in9)
# 3-SOC training (saved), seeds 0-1 and 0-2
res['soc3_train_seeds01'] = block(P3, d3)
res['soc3_train_seeds012'] = block(P3all, d3)

# ---- per nominal SOC (aggregate chg+dchg within cell-RPT for that SOC only)
per_soc = {}
ps_list = []
for s in np.unique(d9['soc']):
    m = d9['soc'] == s
    b = block(P9, d9, m)
    row = dict(R0=b['R0']['cell_macro_rmse'], A2=b['A2']['cell_macro_rmse'], GBDT=b['GBDT']['cell_macro_rmse'],
               R0_minus_A2=b['R0_minus_A2'])
    if s in (20, 50, 90):
        m3 = d3['soc'] == s
        b3 = block(P3, d3, m3)
        b3a = block(P3all, d3, m3)
        row['soc3_train_seeds01'] = dict(R0=b3['R0']['cell_macro_rmse'], A2=b3['A2']['cell_macro_rmse'],
                                         GBDT=b3['GBDT']['cell_macro_rmse'], R0_minus_A2=b3['R0_minus_A2'])
        row['soc3_train_seeds012'] = dict(R0=b3a['R0']['cell_macro_rmse'], A2=b3a['A2']['cell_macro_rmse'],
                                          R0_minus_A2=b3a['R0_minus_A2'])
    per_soc[int(s)] = row
    ps_list.append(row['R0_minus_A2']['p'])
adj = holm(ps_list)
for (s, row), a in zip(per_soc.items(), adj):
    row['R0_minus_A2']['p_holm9'] = round(float(a), 4)
res['per_soc_soc9_train'] = per_soc

# ---- per SOC x direction (single record per cell-RPT)
psd = {}
for s in np.unique(d9['soc']):
    for p in ['chg', 'dchg']:
        m = (d9['soc'] == s) & (d9['ptype'] == p)
        b = block(P9, d9, m)
        psd[f'{s}_{p}'] = dict(R0=b['R0']['cell_macro_rmse'], A2=b['A2']['cell_macro_rmse'], GBDT=b['GBDT']['cell_macro_rmse'],
                               diff=b['R0_minus_A2']['diff'], p=b['R0_minus_A2']['p'], improved=b['R0_minus_A2']['improved'])
res['per_soc_dir_soc9_train'] = psd

# ---- 9-SOC vs 3-SOC training, per cell (eval on 20/50/90 only, seeds 0-1), paired
pc9 = res['soc9_train_eval_3soc']['_pc']; pc3 = res['soc3_train_seeds01']['_pc']
res['soc9_vs_soc3_training_eval3soc'] = {mo: pstat(pc9[mo], pc3[mo]) for mo in ['R0', 'A2', 'GBDT']}
# difference-in-differences: (R0-A2)_9 - (R0-A2)_3 per cell
res['did_R0A2_9_minus_3'] = pstat(pc9['R0'] - pc9['A2'], pc3['R0'] - pc3['A2'])

# ---- run diagnostics
diag = {}
abs_feats = ['glob_V0', 'glob_V30', 'glob_V40', 'glob_V100', 'glob_mean', 'glob_median', 'glob_min', 'glob_max']
for mo in ['R0', 'A2']:
    rows = []
    for f in range(5):
        for s in SEEDS:
            r = np.load(f'{D}/runs/{mo}_f{f}_s{s}.npz')
            rows.append(dict(fold=f, seed=s, time=round(float(r['time']), 1), epochs=int(r['epochs']),
                             best_epoch=int(r['best_epoch']), val_rmse=round(float(r['val_rmse']), 4)))
    diag[mo] = rows
g = []
for f in range(5):
    r = np.load(f'{D}/runs/GBDT_f{f}.npz')
    g.append(dict(fold=f, hp=r['hp'].tolist(), val_rmse=round(float(r['val_rmse']), 4)))
diag['GBDT'] = g
# selected features per fold (9-SOC), and level-type features
sel = {}
for f in range(5):
    s9 = list(np.load(f'{D}/runs/R0_f{f}_s0.npz')['sel'])
    s3 = list(np.load(f'{W}/results/nmc/R0_f{f}_s0.npz')['sel'])
    lvl = lambda L: [x for x in L if x.startswith('glob_V') or x.startswith('glob_q') or x.endswith('_intercept') or x in ('glob_mean', 'glob_median', 'glob_min', 'glob_max')]
    sel[f] = dict(n_overlap_with_3soc=len(set(s9) & set(s3)), level_feats_9soc=lvl(s9), level_feats_3soc=lvl(s3),
                  first10_9soc=s9[:10], first10_3soc=s3[:10])
diag['selected_features'] = sel
res['diagnostics'] = diag

for k in list(res):
    if isinstance(res[k], dict) and '_pc' in res[k]:
        res[k]['per_cell'] = {mo: [round(float(x), 4) for x in v] for mo, v in res[k].pop('_pc').items()}
json.dump(res, open(D + '/analysis.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

# ---- print summary
def fmt(b):
    r = b['R0_minus_A2']
    return (f"R0 {b['R0']['cell_macro_rmse']:.3f}  A2 {b['A2']['cell_macro_rmse']:.3f}  GBDT {b['GBDT']['cell_macro_rmse']:.3f} | "
            f"R0-A2 {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] p={r['p']:.3f} improved {r['improved']}/{r['n']} dz={r['dz']:+.3f}")
for k in ['soc9_train_eval_all18', 'soc9_train_eval_3soc', 'soc3_train_seeds01', 'soc3_train_seeds012']:
    print(f'{k:28s}', fmt(res[k]))
    print(' ' * 28, 'record RMSE: R0 %.3f A2 %.3f GBDT %.3f' % tuple(res[k][m]['record_rmse'] for m in ['R0', 'A2', 'GBDT']),
          '| per-seed R0', res[k]['R0']['per_seed_cell_macro'], 'A2', res[k]['A2']['per_seed_cell_macro'])
print('\nper nominal SOC (9-SOC training; cell-RPT agg of chg+dchg at that SOC):')
for s, row in per_soc.items():
    r = row['R0_minus_A2']
    line = (f"SOC {s:2d}: R0 {row['R0']:.3f} A2 {row['A2']:.3f} GBDT {row['GBDT']:.3f}  R0-A2 {r['diff']:+.3f} "
            f"[{r['lo']:+.3f},{r['hi']:+.3f}] p={r['p']:.3f} holm={r['p_holm9']:.3f} impr {r['improved']}/{r['n']}")
    if 'soc3_train_seeds01' in row:
        q = row['soc3_train_seeds01']; qa = row['soc3_train_seeds012']
        line += (f" || 3SOC-train s01: R0 {q['R0']:.3f} A2 {q['A2']:.3f} GBDT {q['GBDT']:.3f} R0-A2 {q['R0_minus_A2']['diff']:+.3f} p={q['R0_minus_A2']['p']:.3f}"
                 f" | s012: R0-A2 {qa['R0_minus_A2']['diff']:+.3f} p={qa['R0_minus_A2']['p']:.3f}")
    print(line)
print('\nper SOC x direction:')
for k, v in psd.items():
    print(f"{k:9s} R0 {v['R0']:.3f} A2 {v['A2']:.3f} GBDT {v['GBDT']:.3f} diff {v['diff']:+.3f} p={v['p']:.3f} impr {v['improved']}")
print('\n9-SOC vs 3-SOC training (eval 20/50/90, seeds 0-1), paired per cell (9 minus 3):')
for mo, r in res['soc9_vs_soc3_training_eval3soc'].items():
    print(f"  {mo}: {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] p={r['p']:.3f} improved {r['improved']}/{r['n']}")
r = res['did_R0A2_9_minus_3']
print(f"  DiD (R0-A2)_9 - (R0-A2)_3: {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] p={r['p']:.3f}")
print('\nrun diagnostics:')
for mo in ['R0', 'A2']:
    print(mo, 'mean time %.0fs, mean epochs %.1f, mean best_epoch %.1f' % (np.mean([x['time'] for x in diag[mo]]), np.mean([x['epochs'] for x in diag[mo]]), np.mean([x['best_epoch'] for x in diag[mo]])))
print('GBDT', g)
for f, v in sel.items():
    print(f, 'overlap', v['n_overlap_with_3soc'], 'level9', v['level_feats_9soc'], '| level3', v['level_feats_3soc'])
