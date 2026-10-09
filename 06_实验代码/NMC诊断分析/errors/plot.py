import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/nmcdiag/errors'
r = json.load(open(f'{OUT}/results.json'))
r3 = json.load(open(f'{OUT}/results3.json'))
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
COL = {'R0': '#2a78d6', 'A2': '#eb6834', 'GBDT': '#1baf7a'}
LAB = {'R0': 'R0 (A2 + waveform)', 'A2': 'A2 (features + tau)', 'GBDT': 'GBDT'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})

bins = [b['bin'] for b in r['NMC']['by_soh_bin']['R0']['bins']]
fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, constrained_layout=True)
w = 0.26
x = np.arange(len(bins))
for j, ds in enumerate(['LFP', 'NMC']):
    for i, met in enumerate(['rmse', 'bias']):
        ax = axes[i, j]
        for k, m in enumerate(['R0', 'A2', 'GBDT']):
            vals = [b[met] for b in r[ds]['by_soh_bin'][m]['bins']]
            v = np.array([np.nan if q is None else q for q in vals])
            ax.bar(x + (k - 1) * w, v, w * 0.92, color=COL[m], label=LAB[m], zorder=3)
        ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        if met == 'bias':
            ax.axhline(0, color=INK2, lw=0.8, zorder=4)
        ns = [b['n_pts'] for b in r[ds]['by_soh_bin']['R0']['bins']]
        if met == 'rmse':
            top = ax.get_ylim()[1] * 1.12
            for xi, n in zip(x, ns):
                ax.text(xi, top * 0.995, 'no data' if n == 0 else f'n={n}', ha='center', va='top', fontsize=7.5, color=INK2)
            ax.set_ylim(0, top)
        h = r[ds]['headline']
        ax.set_title(f"{ds}  ({'RMSE' if met == 'rmse' else 'mean bias, pred - true'})   "
                     f"cell-macro: R0 {h['R0']:.3f} / A2 {h['A2']:.3f} / GBDT {h['GBDT']:.3f}" if met == 'rmse' else
                     f"{ds}  (mean bias, pred - true)", fontsize=9, loc='left')
        ax.set_ylabel('RMSE (pp)' if met == 'rmse' else 'bias (pp)')
        if i == 1:
            ax.set_xticks(x, bins)
            ax.set_xlabel('true SOH bin (%), cell-RPT level, 6 conditions x 3 seeds averaged')
axes[0, 0].legend(frameon=False, loc='center left', fontsize=8)
fig.savefig(f'{OUT}/fig_error_by_soh_bin.png', dpi=160)

# figure 2: R0-A2 by region, seed-averaged and per seed
regs = ['first_rpt', 'soh_ge90_excl_first', 'soh_80_90', 'soh_72_80', 'soh_lt72', 'all']
rl = ['first RPT\n(SOH=100)', '>=90\n(excl. first)', '80-90', '72-80', '<72', 'all']
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True, constrained_layout=True)
for j, ds in enumerate(['LFP', 'NMC']):
    ax = axes[j]
    xs = np.arange(len(regs))
    sa = [r3[ds]['seedavg'].get(g, {}).get('diff', np.nan) for g in regs]
    lo = [r3[ds]['seedavg'].get(g, {}).get('lo', np.nan) for g in regs]
    hi = [r3[ds]['seedavg'].get(g, {}).get('hi', np.nan) for g in regs]
    sa, lo, hi = np.array(sa, float), np.array(lo, float), np.array(hi, float)
    ax.bar(xs, sa, 0.55, color=COL['R0'], zorder=3, label='seed-averaged (95% CI)')
    ax.errorbar(xs, sa, yerr=[sa - lo, hi - sa], fmt='none', ecolor=INK, lw=1, capsize=3, zorder=4)
    for s, mk in zip(range(3), ['o', 's', '^']):
        v = [r3[ds][f'seed{s}'].get(g, {}).get('diff', np.nan) for g in regs]
        ax.scatter(xs + 0.33, v, s=22, marker=mk, facecolor=SURF, edgecolor=INK2, zorder=5, label=f'seed {s}')
    ax.axhline(0, color=INK2, lw=0.8)
    for xi, v in zip(xs, sa):
        if not np.isfinite(v):
            ax.text(xi, 0.05, 'no data', ha='center', fontsize=7.5, color=INK2)
    ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks(xs, rl, fontsize=8)
    ax.set_title(f'{ds}: R0 - A2 per-cell RMSE by SOH region (negative = waveform helps)', fontsize=9, loc='left')
axes[0].set_ylabel('R0 - A2 (pp)')
axes[0].set_ylim(-1.1, 0.5)
axes[1].legend(frameon=False, fontsize=7.5, loc='lower left')
fig.savefig(f'{OUT}/fig_R0minusA2_by_region.png', dpi=160)
print('ok')
