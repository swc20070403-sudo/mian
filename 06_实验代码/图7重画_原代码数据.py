import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from figstyle import plt, panel, save, BLUE, GRAY, RED
L = '/home/user/mian/07_本地运行/结果回传/'
rd = lambda f: list(csv.DictReader(open(L + f, encoding='utf-8-sig')))
STY = {'R0': (BLUE, 'o', '-'), 'A2': (GRAY, 's', '--')}
fig = plt.figure(figsize=(6.7, 5.1)); gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.30)
# (a) LOGO
ax = fig.add_subplot(gs[0, 0]); g = rd('plot_3.1_logo_group_rmse.csv')
grp = np.array([int(r['group']) for r in g])
for m, off, tot, cv in [('A2', -0.14, 1.611, 1.500), ('R0', 0.14, 1.520, 1.377)]:
    c, mk, _ = STY[m]
    ax.plot(grp + off, [float(r[f'{m}_group_mean_RMSE']) for r in g], mk, color=c, ms=4.4, mec='white', mew=0.5,
            label=f'{m}  {tot:.3f} (CV {cv:.3f})')
ax.set_xticks(grp); ax.set_xlabel('Held-out cycling group'); ax.set_ylabel('Group RMSE (pp)'); ax.set_ylim(0, 4.4)
ax.legend(loc='upper right', fontsize=6.6, handletextpad=0.2, borderaxespad=0.2, title='Cell-macro RMSE, leave-one-group-out', title_fontsize=6.6)
panel(ax, 'a', x=-0.15)
# (b) robustness
ax = fig.add_subplot(gs[0, 1]); x = np.arange(5)
vals = {'R0': [1.377, 1.411, 1.421, 1.562, 1.700], 'A2': [1.500, 1.527, 1.538, 1.630, 1.770]}
for m, off in [('A2', -0.11), ('R0', 0.11)]:
    c, mk, _ = STY[m]; ax.plot(x + off, vals[m], mk, color=c, ms=4.6, mec='white', mew=0.5, label=m)
ax.axvline(2.5, color=GRAY, lw=0.6, ls=':')
ax.text(1.0, 1.0, 'Cycle-count bias\n(models fixed)', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.6)
ax.text(3.5, 1.0, 'Matched noise\n(models retrained)', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.6)
ax.set_xticks(x); ax.set_xticklabels(['Clean', 'τ +10%', 'τ −10%', 'Noise\n1 mV', 'Noise\n2 mV'], fontsize=7)
ax.set_ylabel('RMSE (pp)'); ax.legend(loc='upper left', fontsize=6.8, ncol=2, columnspacing=0.8, handletextpad=0.2)
panel(ax, 'b', x=-0.15)
# (c) pulses
ax = fig.add_subplot(gs[1, 0]); p = rd('plot_3.5_pulse_count.csv'); k = np.array([int(r['k']) for r in p])
for m in ['A2', 'R0']:
    c, mk, ls = STY[m]
    mean = [float(r[f'{m}_mean_over_combinations']) for r in p]; lo = [float(r[f'{m}_min']) for r in p]; hi = [float(r[f'{m}_max']) for r in p]
    ax.fill_between(k, lo, hi, color=c, alpha=0.14, lw=0)
    ax.plot(k, mean, marker=mk, color=c, ls=ls, ms=4.2, mec='white', mew=0.5, label=m)
ax.set_xticks(k); ax.set_xlabel('Pulses averaged per RPT'); ax.set_ylabel('RMSE (pp)')
ax.legend(loc='upper right', fontsize=6.8, ncol=2, columnspacing=0.8, handletextpad=0.2)
panel(ax, 'c', x=-0.15)
# (d) IG
ax = fig.add_subplot(gs[1, 1]); ig = rd('plot_3.6_integrated_gradients.csv')
t = np.array([int(r['sample_index']) for r in ig]); chg = np.array([float(r['mean_abs_IG_charge_pp']) for r in ig]); dch = np.array([float(r['mean_abs_IG_discharge_pp']) for r in ig])
sm = lambda v: np.convolve(np.pad(v, (2, 1), mode='edge'), np.ones(4) / 4, mode='valid')
ax.plot(t, chg, color=BLUE, lw=0.5, alpha=0.25); ax.plot(t, dch, color=RED, lw=0.5, alpha=0.25)
for (a, b), lab, col in [((0, 30), 'C/5', '#eef3f9'), ((30, 40), '1C', '#dce8f6'), ((40, 100), 'Rest', '#ffffff')]:
    ax.axvspan(a, b, color=col, lw=0, zorder=0)
    ax.text((a + b) / 2, 1.01, lab, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.8)
ax.plot(t, sm(chg), color=BLUE, lw=1.1, label='Charge'); ax.plot(t, sm(dch), color=RED, lw=1.1, ls='--', label='Discharge')
ax.set_xlim(0, 100); ax.set_ylim(0, max(chg.max(), dch.max()) * 1.25)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean |IG| per sample (pp)')
ax.legend(loc='upper right', fontsize=6.8, ncol=2, columnspacing=0.8); ax.grid(axis='x', visible=False)
panel(ax, 'd', x=-0.15)
save(fig, 'figs/fig7_general_original.png'); print('ok')
