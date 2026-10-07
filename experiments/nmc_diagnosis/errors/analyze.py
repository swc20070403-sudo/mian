"""Error localisation diagnostic using saved out-of-fold predictions only (no training)."""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from scipy import stats as ss

W = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/work'
OUT = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/nmcdiag/errors'
sys.path.insert(0, W)
from core import cell_rmse, cv_folds
from stats import paired

DS = {'LFP': ('UConn-ILCC-LFP', 'cv', ['R0', 'A2', 'GBDT']),
      'NMC': ('UConn-ILCC-NMC', 'nmc', ['R0', 'A2', 'A1', 'GBDT'])}
BINS = [(-np.inf, 60, '<60'), (60, 70, '60-70'), (70, 80, '70-80'), (80, 90, '80-90'), (90, 95, '90-95'), (95, np.inf, '>=95')]
R = lambda x: None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 3)


def load(name):
    z = np.load(f'{W}/data_{name}.npz', allow_pickle=False)
    return {k: z[k] for k in z.files}


def collect(d, exp, mode):
    """returns list of per-seed record-level prediction arrays (GBDT: one)."""
    out = []
    seeds = [None] if mode == 'GBDT' else [0, 1, 2]
    for s in seeds:
        p = np.full(d['y'].size, np.nan)
        for f in range(5):
            fn = f'{W}/results/{exp}/{mode}_f{f}' + ('' if s is None else f'_s{s}') + '.npz'
            z = np.load(fn, allow_pickle=True); p[z['idx']] = z['pred']
        assert np.isfinite(p).all()
        out.append(p)
    return out


def cellmacro(preds, d, mask=None):
    """paper metric: per-seed cell RMSE (6-condition aggregation per cell-RPT), averaged over seeds.
    mask: record-level boolean subset (applied before aggregation). returns dict cell->rmse"""
    m = np.ones(d['y'].size, bool) if mask is None else mask
    acc = {}
    for p in preds:
        for c, v in cell_rmse(p[m], d['y'][m], d['cell'][m], d['rpt'][m]).items():
            acc.setdefault(c, []).append(v)
    return {c: float(np.mean(v)) for c, v in acc.items()}


def pstats(a, b, cells):
    """paired per-cell stats for dicts a, b over common cells"""
    cs = [c for c in cells if c in a and c in b]
    x = np.array([a[c] for c in cs]); y = np.array([b[c] for c in cs])
    r = paired(x, y)
    return dict(a=R(x.mean()), b=R(y.mean()), diff=R(r['diff']), lo=R(r['lo']), hi=R(r['hi']), p=R(r['p']),
                improved=r['improved'], n=r['n'], dz=R(r['dz']))


res = {}
cache = {}
for ds, (name, exp, models) in DS.items():
    d = load(name)
    P = {m: collect(d, exp, m) for m in models}
    cache[ds] = (d, P)
    cells = sorted(np.unique(d['cell']).tolist())
    r = res[ds] = {}
    # ---------------- 1. headline
    pc = {m: cellmacro(P[m], d) for m in models}
    r['headline'] = {m: R(np.mean(list(pc[m].values()))) for m in models}
    r['headline_R0_vs_A2'] = pstats(pc['R0'], pc['A2'], cells)
    # seed-noise reference: per-cell differences between seeds of the same model
    sn = {}
    for m in ['R0', 'A2']:
        per_seed = [cellmacro([P[m][s]], d) for s in range(3)]
        diffs = []
        for s1, s2 in [(0, 1), (0, 2), (1, 2)]:
            diffs.append(np.array([per_seed[s1][c] - per_seed[s2][c] for c in cells]))
        diffs = np.array(diffs)
        sn[m] = dict(mean_abs_macro_diff_between_seeds=R(np.mean(np.abs(diffs.mean(1)))),
                     sd_percell_diff_between_seeds=R(diffs.std()))
        sn[m]['macro_per_seed'] = [R(np.mean(list(ps.values()))) for ps in per_seed]
    dR = np.array([pc['R0'][c] - pc['A2'][c] for c in cells])
    sn['sd_percell_R0minusA2_seedavg'] = R(dR.std(ddof=1))
    r['seed_noise'] = sn
    # ---------------- cell-RPT aggregation (avg over 6 conditions and seeds)
    key = d['cell'].astype(np.int64) * 10000 + d['rpt'].astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    ym = np.bincount(inv, d['y']) / cnt
    kc = uk // 10000
    kgroup = np.bincount(inv, d['group']) / cnt
    pm = {m: np.bincount(inv, np.mean(P[m], 0)) / cnt for m in models}
    # ---------------- 2. bins, calibration, low-SOH SSE share
    bins = {}
    for m in models:
        e = pm[m] - ym
        rows = []
        for lo, hi, lab in BINS:
            s = (ym >= lo) & (ym < hi)
            rows.append(dict(bin=lab, n_pts=int(s.sum()), n_cells=int(np.unique(kc[s]).size),
                             rmse=R(np.sqrt(np.mean(e[s] ** 2))) if s.any() else None,
                             bias=R(np.mean(e[s])) if s.any() else None,
                             sse_share=R(np.sum(e[s] ** 2) / np.sum(e ** 2)) if s.any() else None))
        sl, ic, rr, _, _ = ss.linregress(ym, pm[m])
        hi72 = ym >= 72
        sl72 = ss.linregress(ym[hi72], pm[m][hi72]).slope
        bins[m] = dict(bins=rows, calib_slope=R(sl), calib_intercept=R(ic), calib_r=R(rr),
                       calib_slope_soh_ge72=R(sl72),
                       pooled_rmse_cellrpt=R(np.sqrt(np.mean(e ** 2))),
                       sse_share_soh_lt72=R(np.sum(e[~hi72] ** 2) / np.sum(e ** 2)),
                       frac_pts_soh_lt72=R(np.mean(~hi72)),
                       rmse_soh_ge72=R(np.sqrt(np.mean(e[hi72] ** 2))),
                       rmse_soh_lt72=R(np.sqrt(np.mean(e[~hi72] ** 2))) if (~hi72).any() else None,
                       bias_soh_ge72=R(np.mean(e[hi72])),
                       bias_soh_lt72=R(np.mean(e[~hi72])) if (~hi72).any() else None)
    r['by_soh_bin'] = bins
    # ---------------- 3. R0-A2 restricted to SOH >= 72 and < 72 (record mask from cell-RPT truth)
    yrec = d['y']
    for lab, msk in [('soh_ge72', yrec >= 72), ('soh_lt72', yrec < 72), ('soh_ge80', yrec >= 80), ('soh_lt80', yrec < 80)]:
        if not msk.any():
            continue
        sub = {m: cellmacro(P[m], d, msk) for m in models}
        r[f'R0_vs_A2_{lab}'] = pstats(sub['R0'], sub['A2'], cells)
        r[f'R0_vs_A2_{lab}']['macro'] = {m: R(np.mean(list(sub[m].values()))) for m in models}
        r[f'R0_vs_A2_{lab}']['GBDT_vs_A2'] = pstats(sub['GBDT'], sub['A2'], cells)
    # per pulse condition
    cond = {}
    for soc in [20, 50, 90]:
        for pt in ['chg', 'dchg']:
            msk = (d['soc'] == soc) & (d['ptype'] == pt)
            sub = {m: cellmacro(P[m], d, msk) for m in models}
            e = pstats(sub['R0'], sub['A2'], cells)
            e['macro'] = {m: R(np.mean(list(sub[m].values()))) for m in models}
            cond[f'{soc}_{pt}'] = e
    r['R0_vs_A2_by_condition'] = cond
    # per cycling group (headline per-cell diffs)
    cg = {c: int(d['group'][d['cell'] == c][0]) for c in cells}
    grp = {}
    for g in sorted(set(cg.values())):
        cs = [c for c in cells if cg[c] == g]
        x = np.array([pc['R0'][c] for c in cs]); y = np.array([pc['A2'][c] for c in cs])
        dd = x - y
        # exact sign-flip p over all 2^n sign patterns
        n = len(dd); obs = abs(dd.mean())
        signs = np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T
        pex = float(np.mean(np.abs((signs * dd).mean(1)) >= obs - 1e-12))
        grp[str(g)] = dict(n=n, R0=R(x.mean()), A2=R(y.mean()), GBDT=R(np.mean([pc['GBDT'][c] for c in cs])),
                           diff=R(dd.mean()), improved=int(np.sum(dd < 0)), p_exact_signflip=R(pex),
                           min_soh_mean=R(np.mean([yrec[d['cell'] == c].min() for c in cs])))
    kw = ss.kruskal(*[[pc['R0'][c] - pc['A2'][c] for c in cells if cg[c] == g] for g in sorted(set(cg.values()))])
    r['R0_vs_A2_by_group'] = dict(groups=grp, kruskal_H=R(kw.statistic), kruskal_p=R(kw.pvalue))
    # per fold
    folds = cv_folds(d['cell'])
    pf = {}
    for f, (_, _, te) in enumerate(folds):
        cs = [int(c) for c in te]
        pf[str(f)] = dict(n=len(cs), R0=R(np.mean([pc['R0'][c] for c in cs])), A2=R(np.mean([pc['A2'][c] for c in cs])),
                          GBDT=R(np.mean([pc['GBDT'][c] for c in cs])),
                          diff=R(np.mean([pc['R0'][c] - pc['A2'][c] for c in cs])),
                          improved=int(sum(pc['R0'][c] < pc['A2'][c] for c in cs)))
    r['R0_vs_A2_by_fold'] = pf
    # ---------------- 4. NN vs GBDT gap: by bin (SSE decomposition) and by cell
    gap = {}
    for m in [x for x in models if x != 'GBDT']:
        eN = (pm[m] - ym) ** 2; eG = (pm['GBDT'] - ym) ** 2
        tot = eN.sum() - eG.sum()
        rows = []
        for lo, hi, lab in BINS:
            s = (ym >= lo) & (ym < hi)
            if not s.any():
                continue
            rows.append(dict(bin=lab, n_pts=int(s.sum()), dSSE_share=R((eN[s].sum() - eG[s].sum()) / tot),
                             rmse_nn=R(np.sqrt(eN[s].mean())), rmse_gbdt=R(np.sqrt(eG[s].mean()))))
        dc = np.array([pc[m][c] - pc['GBDT'][c] for c in cells])
        order = np.argsort(-dc)
        cs_sorted = [cells[i] for i in order]
        top5 = order[:5]
        keep = np.setdiff1d(np.arange(len(cells)), top5)
        # cell-level gap contributions
        gap[m] = dict(by_bin=rows, macro_gap=R(dc.mean()), median_percell_gap=R(np.median(dc)),
                      n_cells_nn_worse=int(np.sum(dc > 0)), n_cells=len(cells),
                      share_of_gap_top5_cells=R(dc[top5].sum() / dc.sum()),
                      share_of_gap_top10_cells=R(dc[order[:10]].sum() / dc.sum()),
                      macro_gap_without_top5=R(dc[keep].mean()),
                      top5_cells=[dict(cell=int(cells[i]), group=cg[cells[i]], gap=R(dc[i]), nn=R(pc[m][cells[i]]),
                                       gbdt=R(pc['GBDT'][cells[i]]), min_soh=R(yrec[d['cell'] == cells[i]].min()))
                                  for i in top5],
                      paired_vs_gbdt=pstats(pc[m], pc['GBDT'], cells),
                      percell_gap_quantiles=[R(q) for q in np.percentile(dc, [10, 25, 50, 75, 90])],
                      spearman_gap_vs_minsoh=R(ss.spearmanr(dc, [yrec[d['cell'] == c].min() for c in cells]).correlation))
        # SOH>=72 restricted gap per cell
        if (yrec < 72).any():
            subN = cellmacro(P[m], d, yrec >= 72); subG = cellmacro(P['GBDT'], d, yrec >= 72)
            gap[m]['paired_vs_gbdt_soh_ge72'] = pstats(subN, subG, cells)
    r['nn_vs_gbdt_gap'] = gap
    # extrapolation: how much of the low-SOH test region lies below the training-fold label range
    ext = []
    for f, (tr, va, te) in enumerate(folds):
        ytr = yrec[np.isin(d['cell'], np.concatenate([tr, va]))]
        mte = np.isin(kc, te)
        ext.append(dict(fold=f, train_min=R(ytr.min()), train_q01=R(np.percentile(ytr, 1)), train_q05=R(np.percentile(ytr, 5)),
                        test_min=R(ym[mte].min()), frac_test_pts_below_train_min=R(np.mean(ym[mte] < ytr.min())),
                        frac_test_pts_below_train_q05=R(np.mean(ym[mte] < np.percentile(ytr, 5)))))
    r['label_range_extrapolation'] = ext
    # ---------------- 5. per-cell R0-A2 correlates
    minsoh = np.array([yrec[d['cell'] == c].min() for c in cells])
    nrpt = np.array([np.unique(d['rpt'][d['cell'] == c]).size for c in cells])
    a2 = np.array([pc['A2'][c] for c in cells]); gb = np.array([pc['GBDT'][c] for c in cells])
    cor = {}
    for lab, v in [('min_soh', minsoh), ('A2_rmse', a2), ('GBDT_rmse', gb), ('n_rpt', nrpt),
                   ('A2_minus_GBDT', a2 - gb)]:
        if np.std(v) == 0:
            continue
        sp = ss.spearmanr(dR, v); pe = ss.pearsonr(dR, v)
        cor[lab] = dict(spearman=R(sp.correlation), spearman_p=R(sp.pvalue), pearson=R(pe.statistic), pearson_p=R(pe.pvalue))
    # group: eta^2 from one-way ANOVA
    gl = np.array([cg[c] for c in cells])
    grand = dR.mean()
    ssb = sum(np.sum(gl == g) * (dR[gl == g].mean() - grand) ** 2 for g in np.unique(gl))
    sst = np.sum((dR - grand) ** 2)
    fa = ss.f_oneway(*[dR[gl == g] for g in np.unique(gl)])
    cor['group'] = dict(eta2=R(ssb / sst), anova_F=R(fa.statistic), anova_p=R(fa.pvalue), kruskal_p=r['R0_vs_A2_by_group']['kruskal_p'])
    r['percell_R0minusA2_correlates'] = cor
    # tertiles of A2 error
    q = np.percentile(a2, [33.3, 66.7])
    ter = []
    for lo, hi, lab in [(-np.inf, q[0], 'low'), (q[0], q[1], 'mid'), (q[1], np.inf, 'high')]:
        s = (a2 >= lo) & (a2 < hi)
        ter.append(dict(A2_tertile=lab, n=int(s.sum()), A2=R(a2[s].mean()), diff=R(dR[s].mean()), improved=int(np.sum(dR[s] < 0))))
    r['R0minusA2_by_A2_error_tertile'] = ter
    # save per-cell table
    np.savez(f'{OUT}/percell_{ds}.npz', cells=np.array(cells), group=gl, min_soh=minsoh,
             **{f'rmse_{m}': np.array([pc[m][c] for c in cells]) for m in models})
    np.savez(f'{OUT}/cellrpt_{ds}.npz', key=uk, cell=kc, y=ym, group=kgroup, **{f'pred_{m}': pm[m] for m in models})

with open(f'{OUT}/results.json', 'w') as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
