import json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
L = json.load(open(f'{OUT}/light.json')); A = json.load(open(f'{OUT}/analyze.json'))
CONDS = [f'{s}_{p}' for s in (20, 50, 90) for p in ('dchg', 'chg')]
fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
col = {'LFP': '#2a6fdb', 'NMC': '#d1495b'}
for ds in ['LFP', 'NMC']:
    for nm, ls in [('dV', '-'), ('shape', '--')]:
        R = np.array([[np.nan if v is None else abs(v) for v in L['task2_spearman'][ds][c][nm + '_rho_t']] for c in CONDS])
        ax[0].plot(np.nanmean(R, 0), ls, color=col[ds], label=f'{ds} {nm}')
        ax[0].fill_between(np.arange(101), np.nanmin(R, 0), np.nanmax(R, 0), color=col[ds], alpha=0.08)
ax[0].axvspan(30, 40, color='0.9', zorder=0); ax[0].set_xlabel('time (s)'); ax[0].set_ylabel('|Spearman rho| with SOH (within condition)')
ax[0].set_title('a  per-time-point |rho|: mean over 6 conditions (band: min-max)'); ax[0].legend(fontsize=8)
sets = ['tau', 'V0', 'amp', 'amp_V0', 'shape', 'dV', 'dV_V0', 'absV', 'abs', 'K50noabs', 'K50noabs_V0', 'K50', 'K50_dV']
x = np.arange(len(sets)); w = 0.4
for k, ds in enumerate(['LFP', 'NMC']):
    v = [A['gbdt'][ds].get(s, np.nan) for s in sets]
    b = ax[1].bar(x + (k - 0.5) * w, v, w, color=col[ds], label=ds)
    for xi, vi in zip(x, v):
        ax[1].text(xi + (k - 0.5) * w, min(vi, 3.0) + 0.03, f'{vi:.2f}', ha='center', fontsize=6.5, rotation=90)
ax[1].set_ylim(0, 3.4); ax[1].set_xticks(x); ax[1].set_xticklabels(sets, rotation=45, ha='right', fontsize=8)
ax[1].set_ylabel('cell-macro RMSE (pp)'); ax[1].set_title('b  HistGBDT by input set (+tau; +cond one-hot for raw sets)'); ax[1].legend()
tags = ['A2', 'R0', 'R0raw', 'A2noabs', 'R0noabs', 'R0rawnoabs']
r = A['nn']['NMC']['rmse']
for i, t in enumerate(tags):
    ss = [r[f'{t}_s{s}'] for s in range(3)]
    ax[2].bar(i, r[f'{t}_3seed'], color='#888' if t.startswith('A2') else '#d1495b', alpha=0.8)
    ax[2].scatter([i] * 3, ss, color='k', s=12, zorder=3)
    ax[2].text(i, r[f'{t}_3seed'] + 0.03, f'{r[t + "_3seed"]:.3f}', ha='center', fontsize=8)
ax[2].set_xticks(range(len(tags))); ax[2].set_xticklabels(tags, fontsize=8); ax[2].set_ylim(1.5, 2.35)
ax[2].set_ylabel('cell-macro RMSE (pp), 3-seed mean; dots = seeds')
ax[2].set_title('c  NMC neural variants (noabs: 28 level features removed from backbone)')
fig.tight_layout(); fig.savefig(f'{OUT}/summary_physics.png', dpi=110)
