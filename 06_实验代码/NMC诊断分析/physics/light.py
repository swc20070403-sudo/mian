"""Task 1 (curves by SOH bin), task 2 Spearman per time point, task 3a selected features, task 3b correlations."""
import json, numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

CONDS = [(s, p) for s in (20, 50, 90) for p in ('dchg', 'chg')]
EDGES = [100.01, 95, 90, 85, 80, 75, 70, 60, 40]
res = {'task1': {}, 'task2_spearman': {}, 'task3a': {}, 'task3b': {}}
S = {'S1': slice(1, 31), 'S2': slice(31, 41), 'S3': slice(41, 101)}


def rank_resid(a, b):
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    rb = rb - rb.mean(); ra = ra - ra.mean()
    return ra - (ra @ rb) / (rb @ rb) * rb


for kind in ['dV', 'shape']:
    fig, axes = plt.subplots(2, 6, figsize=(22, 7.5), sharex=True)
    for r, ds in enumerate(['LFP', 'NMC']):
        d = load(ds)
        V = d['V'].astype(float); dV = V - V[:, :1]
        amp = np.abs(V[:, 40] - V[:, 0])
        X = dV if kind == 'dV' else dV / amp[:, None]
        for c, (soc, pt) in enumerate(CONDS):
            m = (d['soc'] == soc) & (d['ptype'] == pt)
            ax = axes[r, c]
            curves, labs, ns = [], [], []
            for b in range(len(EDGES) - 1):
                mb = m & (d['y'] <= EDGES[b]) & (d['y'] > EDGES[b + 1])
                if mb.sum() < 20: continue
                curves.append(X[mb].mean(0)); labs.append(f'{EDGES[b+1]:.0f}-{min(EDGES[b],100):.0f}'); ns.append(int(mb.sum()))
            cmap = plt.get_cmap('viridis')
            for k, (cv, lb) in enumerate(zip(curves, labs)):
                ax.plot(cv * (1000 if kind == 'dV' else 1), color=cmap(k / max(len(curves) - 1, 1)), lw=1.4, label=f'SOH {lb} (n={ns[k]})')
            ax.set_title(f'{ds} SOC{soc} {pt}', fontsize=10)
            ax.axvspan(30, 40, color='0.9', zorder=0)
            if c == 0: ax.set_ylabel('dV = V - V[0] (mV)' if kind == 'dV' else 'dV / |V[40]-V[0]|')
            if r == 1: ax.set_xlabel('time (s)')
            ax.legend(fontsize=6.5, loc='best')
            if kind == 'dV':
                M = np.array(curves); m0 = M[0]
                s = M @ m0 / (m0 @ m0)
                resid = M - s[:, None] * m0
                tot = ((M - m0) ** 2).sum()
                frac = 1 - (resid ** 2).sum() / tot if tot > 0 else np.nan
                # record-level physical descriptors (within condition)
                Vm = V[m]; ym = d['y'][m]
                a31 = np.abs(Vm[:, 31] - Vm[:, 30]); a40 = np.abs(Vm[:, 40] - Vm[:, 30])
                ohm_frac = a31 / a40
                resid_pol = np.abs(Vm[:, 100] - Vm[:, 30]) / a40
                relax41 = np.abs(Vm[:, 41] - Vm[:, 40]) / a40
                res['task1'].setdefault(ds, {})[f'{soc}_{pt}'] = dict(
                    bins=labs, n=ns,
                    amp_V40mV0_mV_by_bin=[round(float(np.abs(cv[40]) * 1000), 3) for cv in M],
                    amp_ratio_oldest_vs_freshest=round(float(np.abs(M[-1, 40]) / np.abs(M[0, 40])), 3),
                    scale_explained_fraction=round(float(frac), 3),
                    rho_soh_a31=round(float(spearmanr(a31, ym)[0]), 3),
                    rho_soh_a40=round(float(spearmanr(a40, ym)[0]), 3),
                    rho_soh_ohmic_fraction=round(float(spearmanr(ohm_frac, ym)[0]), 3),
                    rho_soh_relax_jump_fraction=round(float(spearmanr(relax41, ym)[0]), 3),
                    rho_soh_residual_polarisation_fraction=round(float(spearmanr(resid_pol, ym)[0]), 3))
    fig.suptitle(('Mean dV curves' if kind == 'dV' else 'Mean shape-normalised curves dV/|V[40]-V[0]|') +
                 ' by true-SOH bin (shaded: 1C step, 30-40 s); top LFP, bottom NMC', fontsize=12)
    fig.tight_layout()
    fig.savefig(f'{OUT}/curves_{kind}_by_soh.png', dpi=110); plt.close(fig)

# task 2 Spearman per time point within condition
for ds in ['LFP', 'NMC']:
    d = load(ds); V = d['V'].astype(float); dV = V - V[:, :1]
    shp = dV / np.abs(V[:, 40] - V[:, 0])[:, None]
    out = {}
    for soc, pt in CONDS:
        m = (d['soc'] == soc) & (d['ptype'] == pt)
        o = {}
        for nm, X in [('dV', dV), ('shape', shp)]:
            rho = np.array([spearmanr(X[m, t], d['y'][m])[0] if t > 0 else np.nan for t in range(101)])
            o[nm] = {sg: dict(mean_abs=round(float(np.nanmean(np.abs(rho[sl]))), 3),
                             max_abs=round(float(np.nanmax(np.abs(rho[sl]))), 3),
                             argmax_t=int(sl.start + np.nanargmax(np.abs(rho[sl]))),
                             sign_at_max=int(np.sign(rho[sl.start + np.nanargmax(np.abs(rho[sl]))])))
                     for sg, sl in S.items()}
            o[nm + '_rho_t'] = [None if not np.isfinite(x) else round(float(x), 3) for x in rho]
        out[f'{soc}_{pt}'] = o
    for nm in ['dV', 'shape']:
        out[f'summary_{nm}'] = {sg: dict(mean_abs_over_conds=round(float(np.mean([out[f'{s}_{p}'][nm][sg]['mean_abs'] for s, p in CONDS])), 3),
                                         max_abs_over_conds=round(float(np.max([out[f'{s}_{p}'][nm][sg]['max_abs'] for s, p in CONDS])), 3))
                                for sg in S}
    res['task2_spearman'][ds] = out

# task 3a selected features per fold
for ds, exp in [('LFP', 'cv'), ('NMC', 'nmc')]:
    o = {}
    for f in range(5):
        sel = [str(s) for s in np.load(f'{W}/results/{exp}/R0_f{f}_s0.npz')['sel']]
        o[f'fold{f}'] = dict(n_abs=sum(s in ABS for s in sel), abs=[(s, sel.index(s) + 1) for s in sel if s in ABS],
                             n_wav=sum(s in WAV for s in sel), wav=[s for s in sel if s in WAV],
                             families={fam: sum(family(s) == fam for s in sel) for fam in sorted(set(map(family, sel)))},
                             top10=sel[:10])
    res['task3a'][ds] = o

# task 3b correlations within condition
for ds in ['LFP', 'NMC']:
    d = load(ds); names = list(d['names'])
    dev = d['F'][:, names.index('ctx_soc_dev')]; coul = d['F'][:, names.index('ctx_soc_coul')]
    V0 = d['V'][:, 0].astype(float)
    out = {}
    for soc, pt in CONDS:
        m = (d['soc'] == soc) & (d['ptype'] == pt)
        y = d['y'][m]
        ok = np.isfinite(dev[m])
        within = []
        for c in np.unique(d['cell'][m]):
            mc = m & (d['cell'] == c)
            if mc.sum() >= 5 and np.ptp(d['y'][mc]) > 0:
                within.append(spearmanr(V0[mc], d['y'][mc])[0])
        pr = spearmanr(rank_resid(V0[m][ok], dev[m][ok]), rank_resid(y[ok], dev[m][ok]))[0]
        out[f'{soc}_{pt}'] = dict(
            n=int(m.sum()),
            rho_V0_SOH=round(float(spearmanr(V0[m], y)[0]), 3),
            rho_V0_SOH_within_cell_median=round(float(np.nanmedian(within)), 3),
            rho_socdev_SOH=round(float(spearmanr(dev[m][ok], y[ok])[0]), 3),
            rho_V0_socdev=round(float(spearmanr(V0[m][ok], dev[m][ok])[0]), 3),
            partial_rho_V0_SOH_given_socdev=round(float(pr), 3),
            V0_range_mV=round(float(np.ptp(V0[m]) * 1000), 1), V0_sd_mV=round(float(V0[m].std() * 1000), 1),
            socdev_mean_pct=round(float(np.nanmean(dev[m]) * 100), 2), socdev_sd_pct=round(float(np.nanstd(dev[m]) * 100), 2),
            corr_frac_missing_socdev=round(float((~ok).mean()), 3))
    out['mean_abs_rho_V0_SOH'] = round(float(np.mean([abs(out[f'{s}_{p}']['rho_V0_SOH']) for s, p in CONDS])), 3)
    out['mean_abs_rho_socdev_SOH'] = round(float(np.mean([abs(out[f'{s}_{p}']['rho_socdev_SOH']) for s, p in CONDS])), 3)
    res['task3b'][ds] = out

json.dump(res, open(f'{OUT}/light.json', 'w'), indent=1)
