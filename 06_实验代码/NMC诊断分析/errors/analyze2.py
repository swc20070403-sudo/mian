"""Supplementary error-structure diagnostics from saved OOF predictions (no training)."""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
from scipy import stats as ss

OUT = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/nmcdiag/errors'
sys.path.insert(0, OUT)
import importlib.util
spec = importlib.util.spec_from_file_location('an', f'{OUT}/analyze.py')
# reuse helpers without re-running the main analysis
src = open(f'{OUT}/analyze.py').read().split('res = {}')[0]
ns = {}
exec(compile(src, 'analyze_helpers', 'exec'), ns)
load, collect, cellmacro, pstats, R, DS, cv_folds = ns['load'], ns['collect'], ns['cellmacro'], ns['pstats'], ns['R'], ns['DS'], ns['cv_folds']

res = {}
for ds, (name, exp, models) in DS.items():
    d = load(name)
    P = {m: collect(d, exp, m) for m in models}
    cells = sorted(np.unique(d['cell']).tolist())
    y = d['y']
    r = res[ds] = {}
    key = d['cell'].astype(np.int64) * 10000 + d['rpt'].astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    ym = np.bincount(inv, y) / cnt
    kc = uk // 10000
    tau_k = np.bincount(inv, d['cum']) / cnt
    pm = {m: np.bincount(inv, np.mean(P[m], 0)) / cnt for m in models}
    # (a) first RPT (tau=0, SOH=100)
    first = tau_k == 0
    r['first_rpt'] = {m: dict(n=int(first.sum()), bias=R(np.mean(pm[m][first] - ym[first])),
                              rmse=R(np.sqrt(np.mean((pm[m][first] - ym[first]) ** 2))),
                              sse_share=R(np.sum((pm[m][first] - ym[first]) ** 2) / np.sum((pm[m] - ym) ** 2)))
                      for m in models}
    nofirst = {m: cellmacro(P[m], d, d['cum'] > 0) for m in models}
    r['macro_excluding_first_rpt'] = {m: R(np.mean(list(nofirst[m].values()))) for m in models}
    r['R0_vs_A2_excluding_first_rpt'] = pstats(nofirst['R0'], nofirst['A2'], cells)
    # (b) how much does the waveform branch change the output at all?
    eR = pm['R0'] - ym; eA = pm['A2'] - ym; dRA = pm['R0'] - pm['A2']
    r['R0_A2_output_difference'] = dict(
        residual_corr_cellrpt=R(np.corrcoef(eR, eA)[0, 1]),
        rms_pred_diff=R(np.sqrt(np.mean(dRA ** 2))), rms_resid_A2=R(np.sqrt(np.mean(eA ** 2))),
        ratio=R(np.sqrt(np.mean(dRA ** 2)) / np.sqrt(np.mean(eA ** 2))),
        # fraction of A2 residual variance "explained" by R0-A2 change (projection), sign: negative = R0 moves toward truth
        corr_predDiff_vs_minusA2resid=R(np.corrcoef(dRA, -eA)[0, 1]))
    # same per seed (single-seed difference vs seed-to-seed difference of A2)
    seedd = []
    for s in range(3):
        pR = np.bincount(inv, P['R0'][s]) / cnt; pA = np.bincount(inv, P['A2'][s]) / cnt
        pA2 = np.bincount(inv, P['A2'][(s + 1) % 3]) / cnt
        seedd.append((np.sqrt(np.mean((pR - pA) ** 2)), np.sqrt(np.mean((pA - pA2) ** 2))))
    seedd = np.array(seedd)
    r['R0_A2_output_difference']['rms_R0s_minus_A2s_single_seed'] = R(seedd[:, 0].mean())
    r['R0_A2_output_difference']['rms_A2s_minus_A2sprime_single_seed'] = R(seedd[:, 1].mean())
    # (c) error decomposition
    dec = {}
    for m in models:
        prec = np.mean(P[m], 0)
        e = prec - y
        ek = np.bincount(inv, e) / cnt  # shared across the 6 conditions
        within = e - ek[inv]
        # per-cell offset at cell-RPT level
        cb = {c: ek[kc == c].mean() for c in cells}
        off = np.array([cb[c] for c in kc])
        dec[m] = dict(record_mse=R(np.mean(e ** 2)), shared_cellrpt_share=R(np.mean(ek[inv] ** 2) / np.mean(e ** 2)),
                      condition_specific_share=R(np.mean(within ** 2) / np.mean(e ** 2)),
                      cellrpt_mse=R(np.mean(ek ** 2)), cell_offset_share_of_cellrpt_mse=R(np.mean(off ** 2) / np.mean(ek ** 2)),
                      mean_per_condition_cellmacro=None)
        conds = []
        for soc in [20, 50, 90]:
            for pt in ['chg', 'dchg']:
                msk = (d['soc'] == soc) & (d['ptype'] == pt)
                conds.append(np.mean(list(cellmacro(P[m], d, msk).values())))
        dec[m]['mean_per_condition_cellmacro'] = R(np.mean(conds))
        dec[m]['six_cond_cellmacro'] = R(np.mean(list(cellmacro(P[m], d).values())))
        dec[m]['averaging_gain_pct'] = R(100 * (1 - dec[m]['six_cond_cellmacro'] / np.mean(conds)))
    r['error_decomposition'] = dec
    # (d) residual vs tau and vs truth
    r['resid_vs_truth_spearman'] = {m: R(ss.spearmanr(ym, pm[m] - ym).correlation) for m in models}
    # (e) cross-fold linear recalibration (diagnostic): slope/intercept fitted on the OTHER folds' OOF record predictions
    folds = cv_folds(d['cell'])
    rec = {}
    for m in models:
        Pc = []
        for p in P[m]:
            q = p.copy()
            for f, (_, _, te) in enumerate(folds):
                mt = np.isin(d['cell'], te)
                b, a = np.polyfit(p[~mt], y[~mt], 1)
                q[mt] = a + b * p[mt]
            Pc.append(q)
        pcR = cellmacro(Pc, d)
        rec[m] = R(np.mean(list(pcR.values())))
        if m == 'R0':
            pcR0 = pcR
        if m == 'A2':
            pcA2 = pcR
    r['crossfold_linear_recalibrated_macro'] = rec
    r['crossfold_linear_recalibrated_R0_vs_A2'] = pstats(pcR0, pcA2, cells)
    # (f) error by life stage of each cell (tau / max tau per cell)
    life = np.array([tau_k[i] / tau_k[kc == kc[i]].max() for i in range(len(tau_k))])
    lb = []
    for lo, hi, lab in [(-1, 1e-9, '0'), (1e-9, 0.25, '(0,0.25)'), (0.25, 0.5, '[0.25,0.5)'), (0.5, 0.75, '[0.5,0.75)'), (0.75, 1.01, '[0.75,1]')]:
        s = (life >= lo) & (life < hi) if lab != '0' else life == 0
        lb.append(dict(life=lab, n=int(s.sum()), mean_soh=R(ym[s].mean()),
                       **{f'bias_{m}': R(np.mean(pm[m][s] - ym[s])) for m in models},
                       **{f'rmse_{m}': R(np.sqrt(np.mean((pm[m][s] - ym[s]) ** 2))) for m in models}))
    r['by_life_fraction'] = lb

json.dump(res, open(f'{OUT}/results2.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
