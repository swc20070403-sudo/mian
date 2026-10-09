"""Aggregate redundancy diagnostics into results.json (cell-macro RMSE + paired per-cell stats)."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from jobs import WORK, OUT, DS, load, oof
sys.path.insert(0, WORK)
from core import cell_rmse
from stats import paired


def r3(x):
    if isinstance(x, dict):
        return {k: r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r3(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), 3) if abs(x) >= 1e-3 or x == 0 else float(f'{x:.3g}')
    if isinstance(x, np.integer):
        return int(x)
    return x


def percell(pred, d):
    r = cell_rmse(pred, d['y'], d['cell'], d['rpt'])
    cells = sorted(r)
    return np.array([r[c] for c in cells])


def collect(ds, part, keys):
    d = load(ds)
    P = {k: np.full(d['y'].size, np.nan) for k in keys}
    for f in range(5):
        p = os.path.join(OUT, f'{ds}_{part}_f{f}.npz')
        if not os.path.exists(p):
            return None
        z = np.load(p)
        for k in keys:
            P[k][z['idx']] = z[k]
    return P


def pstat(a, b):
    s = paired(a, b)
    return dict(diff=s['diff'], ci=[s['lo'], s['hi']], p=s['p'], improved=f"{s['improved']}/{s['n']}", dz=s['dz'])


def saved(ds, d, mode):
    ex = DS[ds][1]
    if mode == 'GBDT':
        pred = np.full(d['y'].size, np.nan)
        for f in range(5):
            z = np.load(os.path.join(WORK, 'results', ex, f'GBDT_f{f}.npz')); pred[z['idx']] = z['pred']
        return percell(pred, d)
    per = []
    for s in range(3):
        pred = np.full(d['y'].size, np.nan)
        for f in range(5):
            z = np.load(os.path.join(WORK, 'results', ex, f'{mode}_f{f}_s{s}.npz')); pred[z['idx']] = z['pred']
        per.append(percell(pred, d))
    return np.mean(per, 0)


res = {}
for ds in ['LFP', 'NMC']:
    d = load(ds)
    R = {}
    ref = {m: saved(ds, d, m) for m in (['R0', 'A2', 'GBDT'] + (['A1'] if ds == 'NMC' else []))}
    ref['A2_seedavg_pred'] = percell(oof(ds, d, 'A2'), d)
    ref['R0_seedavg_pred'] = percell(oof(ds, d, 'R0'), d)
    R['reference_saved'] = {k: v.mean() for k, v in ref.items()}
    # (a)
    ka = ['ridge_w_cond', 'ridge_w_cond_tau', 'ridge_vabs_cond', 'ridge_h_tau', 'gbdt_w_cond', 'gbdt_w_cond_tau',
          'gbdt_vabs_cond']
    Pa = collect(ds, 'a', ka)
    if Pa is not None:
        pa = {k: percell(Pa[k], d) for k in ka}
        R['a_waveform_only'] = {
            'rmse': {k: v.mean() for k, v in pa.items()},
            'gbdt_w_cond_vs_featGBDT_saved': pstat(pa['gbdt_w_cond'], ref['GBDT']),
            'gbdt_w_cond_tau_vs_featGBDT_saved': pstat(pa['gbdt_w_cond_tau'], ref['GBDT']),
            'ridge_w_cond_vs_ridge_h_tau': pstat(pa['ridge_w_cond'], pa['ridge_h_tau']),
            'gbdt_vabs_cond_vs_gbdt_w_cond': pstat(pa['gbdt_vabs_cond'], pa['gbdt_w_cond']),
        }
        hp = [json.load(open(os.path.join(OUT, f'{ds}_a_f{f}.json'))) for f in range(5)]
        R['a_waveform_only']['selected'] = {k: [h[k].get('alpha', h[k].get('hp')) for h in hp] for k in hp[0]}
    # (b)
    kb = ['gbdt_h_tau', 'gbdt_h_tau_w']
    Pb = collect(ds, 'b', kb)
    if Pb is not None:
        pb = {k: percell(Pb[k], d) for k in kb}
        hp = [json.load(open(os.path.join(OUT, f'{ds}_b_f{f}.json'))) for f in range(5)]
        R['b_added_value_gbdt'] = {
            'rmse': {k: v.mean() for k, v in pb.items()},
            'h_tau_w_minus_h_tau': pstat(pb['gbdt_h_tau_w'], pb['gbdt_h_tau']),
            'h_tau_vs_saved_full_grid_GBDT': pstat(pb['gbdt_h_tau'], ref['GBDT']),
            'selected_hp': {k: [h[k]['hp'] for h in hp] for k in kb},
        }
    # (c)
    cj = [os.path.join(OUT, f'{ds}_c_f{f}.json') for f in range(5)]
    if all(os.path.exists(p) for p in cj):
        C = [json.load(open(p)) for p in cj]
        out = {'ev_first10_mean_over_folds': np.mean([c['ev'] for c in C], 0).tolist(),
               'ev_cum10_per_fold': [c['ev_cum10'] for c in C],
               'ev_cum10_mean': float(np.mean([c['ev_cum10'] for c in C])),
               'ev_pc1_mean': float(np.mean([c['ev'][0] for c in C]))}
        sst = np.sum([c['sst'] for c in C], 0); sstw = np.sum([c['sst_w'] for c in C])
        for nm in ['cond', 'ridge', 'gbdt']:
            sse = np.sum([c[f'{nm}_sse'] for c in C], 0)
            out[f'{nm}_r2_per_pc_pooled'] = (1 - sse / sst).tolist()
            out[f'{nm}_r2_top10_varweighted_pooled'] = float(1 - sse.sum() / sst.sum())
            out[f'{nm}_r2_top10_per_fold'] = [c[f'{nm}_r2_top10'] for c in C]
            out[f'{nm}_r2_w_total_pooled'] = float(np.sum([c['sst_w'] * c[f'{nm}_r2_w'] for c in C]) / sstw)
            if nm != 'cond':
                ssec = np.sum([c['cond_sse'] for c in C], 0)
                out[f'{nm}_r2_within_condition_per_pc'] = (1 - sse / ssec).tolist()
                out[f'{nm}_r2_within_condition_top10'] = float(1 - sse.sum() / ssec.sum())
        R['c_redundancy_pca'] = out
    cj = [os.path.join(OUT, f'{ds}_c2_f{f}.json') for f in range(5)]
    if all(os.path.exists(p) for p in cj):
        C = [json.load(open(p)) for p in cj]
        out = {'within_cond_var_share_of_total_w': float(np.mean([c['within_cond_var_share_of_total_w'] for c in C])),
               'ev_first10_mean_over_folds': np.mean([c['ev'] for c in C], 0).tolist(),
               'ev_cum10_mean': float(np.mean([c['ev_cum10'] for c in C]))}
        sst = np.sum([c['sst'] for c in C], 0); sstw = np.sum([c['sst_w'] for c in C])
        for nm in ['ridge', 'gbdt']:
            sse = np.sum([c[f'{nm}_sse'] for c in C], 0)
            out[f'{nm}_r2_per_pc_pooled'] = (1 - sse / sst).tolist()
            out[f'{nm}_r2_top10_varweighted_pooled'] = float(1 - sse.sum() / sst.sum())
            out[f'{nm}_r2_top10_per_fold'] = [c[f'{nm}_r2_top10'] for c in C]
            out[f'{nm}_r2_w_within_cond_total_pooled'] = float(np.sum([c['sst_w'] * c[f'{nm}_r2_w'] for c in C]) / sstw)
        R['c2_redundancy_pca_condition_centred'] = out
    # (d)
    kd = ['base_A2', 'A2_w_cond', 'A2_h_tau', 'A2_h_tau_w', 'A2_vabs_cond', 'A2_h_tau_cond', 'base_R0', 'R0_h_tau',
          'R0_h_tau_w']
    Pd = collect(ds, 'd', kd)
    if Pd is not None:
        pdd = {k: percell(Pd[k], d) for k in kd}
        dd = {'rmse': {k: v.mean() for k, v in pdd.items()}}
        for k in ['A2_w_cond', 'A2_h_tau', 'A2_h_tau_w', 'A2_vabs_cond', 'A2_h_tau_cond']:
            dd[f'{k}_minus_A2'] = pstat(pdd[k], pdd['base_A2'])
        dd['A2_h_tau_w_minus_A2_h_tau'] = pstat(pdd['A2_h_tau_w'], pdd['A2_h_tau'])
        dd['A2_w_cond_minus_A2_h_tau'] = pstat(pdd['A2_w_cond'], pdd['A2_h_tau'])
        for k in ['R0_h_tau', 'R0_h_tau_w']:
            dd[f'{k}_minus_R0'] = pstat(pdd[k], pdd['base_R0'])
        dd['R0_h_tau_w_minus_R0_h_tau'] = pstat(pdd['R0_h_tau_w'], pdd['R0_h_tau'])
        dd['A2_h_tau_vs_featGBDT_saved'] = pstat(pdd['A2_h_tau'], ref['GBDT'])
        # residual variance explained at cell-RPT level (RMSE^2 reduction)
        y = d['y']
        key = d['cell'].astype(np.int64) * 10000 + d['rpt'].astype(np.int64)
        uk, inv = np.unique(key, return_inverse=True)
        agg = lambda v: np.bincount(inv, v) / np.bincount(inv)
        r0 = agg(y - Pd['base_A2'])
        dd['A2_residual_sd_cellrpt'] = float(r0.std())
        for k in ['A2_w_cond', 'A2_h_tau', 'A2_h_tau_w', 'A2_vabs_cond', 'A2_h_tau_cond']:
            r1 = agg(y - Pd[k])
            dd[f'{k}_cellrpt_MSE_reduction'] = float(1 - (r1 ** 2).sum() / (r0 ** 2).sum())
        hp = [json.load(open(os.path.join(OUT, f'{ds}_d_f{f}.json'))) for f in range(5)]
        dd['selected_hp'] = {k: [h[k]['hp'] for h in hp] for k in hp[0] if isinstance(hp[0][k], dict)}
        R['d_residual_predictability'] = dd
        ke = ['A2_tau', 'A2_tau_cond', 'A2_h', 'A2_w_cond_tau']
        Pe = collect(ds, 'e', ke)
        if Pe is not None:
            pe = {k: percell(Pe[k], d) for k in ke}
            ee = {'rmse': {k: v.mean() for k, v in pe.items()}}
            for k in ke:
                ee[f'{k}_minus_A2'] = pstat(pe[k], pdd['base_A2'])
            ee['A2_w_cond_tau_minus_A2_h_tau'] = pstat(pe['A2_w_cond_tau'], pdd['A2_h_tau'])
            ee['A2_h_minus_A2_h_tau'] = pstat(pe['A2_h'], pdd['A2_h_tau'])
            R['e_residual_localisation_extra'] = ee
        kg = ['gbdt_w_cond_v0', 'gbdt_w_cond_tau_v0', 'A2_w_cond_v0']
        Pg = collect(ds, 'g', kg)
        if Pg is not None and Pa is not None:
            pg = {k: percell(Pg[k], d) for k in kg}
            R['g_absolute_level_extra'] = {
                'rmse': {k: v.mean() for k, v in pg.items()},
                'gbdt_w_cond_v0_minus_gbdt_w_cond': pstat(pg['gbdt_w_cond_v0'], pa['gbdt_w_cond']),
                'gbdt_w_cond_tau_v0_minus_gbdt_w_cond_tau': pstat(pg['gbdt_w_cond_tau_v0'], pa['gbdt_w_cond_tau']),
                'gbdt_w_cond_tau_v0_vs_featGBDT_saved': pstat(pg['gbdt_w_cond_tau_v0'], ref['GBDT']),
                'A2_w_cond_v0_minus_A2_w_cond': pstat(pg['A2_w_cond_v0'], pdd['A2_w_cond']),
            }
        kf = ['gbdt_h_tau_pc', 'A2_h_tau_pc', 'A2_pc_cond']
        Pf = collect(ds, 'f', kf)
        if Pf is not None and Pb is not None:
            pf = {k: percell(Pf[k], d) for k in kf}
            R['f_waveform_pc10_extra'] = {
                'rmse': {k: v.mean() for k, v in pf.items()},
                'gbdt_h_tau_pc_minus_gbdt_h_tau': pstat(pf['gbdt_h_tau_pc'], pb['gbdt_h_tau']),
                'A2_h_tau_pc_minus_A2_h_tau': pstat(pf['A2_h_tau_pc'], pdd['A2_h_tau']),
                'A2_pc_cond_minus_A2': pstat(pf['A2_pc_cond'], pdd['base_A2']),
            }
    res[ds] = R

with open(os.path.join(HERE, 'results.json'), 'w') as fh:
    json.dump(r3(res), fh, indent=1)
print(json.dumps(r3(res), indent=1))
