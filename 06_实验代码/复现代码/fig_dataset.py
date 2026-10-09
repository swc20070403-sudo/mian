import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from figstyle import plt, panel, save, BLUE, GREEN, RED, GRAY, DARK
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

z = np.load('data_UConn-ILCC-LFP.npz')
V, y, cell, grp, soc, pt, cum = z['V'], z['y'], z['cell'], z['group'], z['soc'], z['ptype'], z['cum']
DOD = {1: 60, 2: 60, 3: 60, 4: 80, 5: 60, 6: 60, 7: 80, 8: 60, 9: 100, 10: 80, 11: 80}
t = np.arange(101)

fig = plt.figure(figsize=(6.7, 4.9))
gs = GridSpec(3, 2, figure=fig, height_ratios=[0.55, 1, 1], width_ratios=[1, 1], hspace=0.55, wspace=0.28)

# (a) current protocol
ax = fig.add_subplot(gs[0, 0])
I = np.r_[0, np.full(30, 0.2), np.full(10, 1.0), np.zeros(60)]
ax.step(t, I, where='pre', color=DARK, lw=1.1, label='Charge')
ax.step(t, -I, where='pre', color=DARK, lw=1.1, ls='--', label='Discharge')
for b in (30, 40):
    ax.axvline(b, color=GRAY, lw=0.6, ls=':')
ax.text(15, 1.55, 'C/5, 30 s', ha='center', fontsize=7)
ax.text(35, 1.55, '1C, 10 s', ha='center', fontsize=7, bbox=dict(fc='white', ec='none', pad=0.3))
ax.text(70, 1.55, 'Rest, 60 s', ha='center', fontsize=7)
ax.set_ylim(-1.35, 2.1); ax.set_xlim(0, 100)
ax.set_yticks([-1, 0, 1]); ax.set_ylabel('Current (C)')
ax.set_xticklabels([])
ax.legend(loc='center right', ncol=1, handlelength=1.8, fontsize=6.6, borderaxespad=0.2, bbox_to_anchor=(1.0, 0.66))
ax.grid(False)
panel(ax, 'a', x=-0.16)

# (b)(c) mean delta-V by SOH bin at 50% nominal SOC
bins = [(98, 101), (94, 98), (90, 94), (86, 90), (82, 86), (70, 82)]
labels = ['≥98', '94–98', '90–94', '86–90', '82–86', '<82']
cmap = mpl.colormaps['Blues']
cols = [cmap(v) for v in np.linspace(0.35, 1.0, len(bins))]
for k, (ptype, letter, title) in enumerate([('chg', 'b', 'Charge pulse, 50% nominal SOC'),
                                             ('dchg', 'c', 'Discharge pulse, 50% nominal SOC')]):
    ax = fig.add_subplot(gs[1 + k, 0])
    m0 = (pt == ptype) & (soc == 50)
    dV = (V - V[:, :1]) * 1000
    for (lo, hi), c, lab in zip(bins, cols, labels):
        m = m0 & (y >= lo) & (y < hi)
        ax.plot(t, dV[m].mean(0), color=c, lw=1.1, label=f'{lab} (n={m.sum()})')
    for b in (30, 40):
        ax.axvline(b, color=GRAY, lw=0.6, ls=':')
    ax.set_xlim(0, 100)
    ax.set_ylabel('ΔV (mV)')
    ax.set_title(title, loc='left', fontsize=8, pad=3)
    if k == 1:
        ax.set_xlabel('Time (s, 1 Hz resampled)')
    else:
        ax.set_xticklabels([])
    # inset: zoom on 1C segment end and early rest
    ins = ax.inset_axes([0.58, 0.40 if ptype == 'chg' else 0.10, 0.38, 0.48])
    for (lo, hi), c in zip(bins, cols):
        m = m0 & (y >= lo) & (y < hi)
        ins.plot(t, dV[m].mean(0), color=c, lw=1.0)
    ins.set_xlim(33, 46)
    seg = dV[m0][:, 33:47]
    ins.set_ylim(np.percentile(seg, 1), np.percentile(seg, 99))
    ins.tick_params(labelsize=6, length=1.5)
    if ptype == 'dchg':
        ins.xaxis.tick_top()
    ins.grid(False)
    ax.indicate_inset_zoom(ins, edgecolor=GRAY, lw=0.6)
    panel(ax, letter, x=-0.16)
    if k == 0:
        sm = mpl.cm.ScalarMappable(cmap=mpl.colors.ListedColormap(cols[::-1]),
                                   norm=mpl.colors.BoundaryNorm(np.arange(len(bins) + 1), len(bins)))

# (d) SOH trajectories coloured by depth of discharge
ax = fig.add_subplot(gs[:, 1])
dcol = {60: BLUE, 80: GREEN, 100: RED}
dls = {60: '-', 80: '--', 100: '-.'}
for c in np.unique(cell):
    m = (cell == c) & (pt == 'chg') & (soc == 20)
    o = np.argsort(cum[m])
    g = grp[m][0]
    ax.plot(cum[m][o] / 1000, y[m][o], color=dcol[DOD[g]], ls=dls[DOD[g]], lw=0.8, alpha=0.85)
for dd in (60, 80, 100):
    ax.plot([], [], color=dcol[dd], ls=dls[dd], lw=1.2, label=f'{dd}% DOD' + (lambda n: f' ({n} group' + ('s)' if n > 1 else ')'))(sum(1 for g in range(1, 12) if DOD[g] == dd)))
ax.set_xlabel('Cumulative cycles (×10$^3$)')
ax.set_ylabel('Measured SOH (%)')
ax.set_ylim(71, 101)
ax.set_xlim(left=0)
ax.legend(loc='upper right', handlelength=2.4, bbox_to_anchor=(1.0, 0.89))
ax.text(0.97, 0.97, '64 cells · 2,823 RPTs\n16,938 pulse records', transform=ax.transAxes, ha='right', va='top',
        fontsize=7.2, color=DARK)
panel(ax, 'd', x=-0.13, y=1.012)

# shared SOH-bin legend for (b),(c)
handles = [mpl.lines.Line2D([], [], color=c, lw=1.6) for c in cols]
fig.legend(handles, labels, title='SOH bin (%)', loc='upper center', bbox_to_anchor=(0.30, 0.035), ncol=6,
           fontsize=6.8, title_fontsize=7, handlelength=1.5, columnspacing=0.9)
save(fig, 'figs/fig1_dataset.png')
print('saved')
