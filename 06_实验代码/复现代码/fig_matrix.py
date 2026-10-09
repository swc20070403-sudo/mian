import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from figstyle import plt, panel, save, BLUE, GREEN, RED, GRAY, DARK

# values reported in Table 4 of the manuscript (cell-macro RMSE, SOH pp)
B = ['MLP', 'ResNet', 'FT-Transformer']
W = ['CNN', 'MLP', 'TCN', 'PW-Trans.', 'PatchTST']
M = np.array([[1.5088, 1.4084, 1.4365, 1.4109, 1.3774],
              [1.4370, 1.3974, 1.4334, 1.4411, 1.4452],
              [1.5551, 1.5733, 1.5973, 1.5549, 1.5982]])
A2 = 1.4999

fig = plt.figure(figsize=(6.7, 4.75))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 0.42], width_ratios=[1.08, 1], hspace=0.62, wspace=0.42)

# (a) heatmap, diverging around A2
ax = fig.add_subplot(gs[0, 0])
cmap = mpl.colors.LinearSegmentedColormap.from_list('div', ['#0067d6', '#7fbfff', '#f4f4f2', '#ff9a9a', '#d9141c'])
norm = mpl.colors.TwoSlopeNorm(vmin=1.36, vcenter=A2, vmax=1.62)
im = ax.imshow(M, cmap=cmap, norm=norm, aspect='auto')
for i in range(3):
    for j in range(5):
        v = M[i, j]
        dark = abs(norm(v) - 0.5) > 0.33
        ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=7.6,
                color='white' if dark else DARK, fontweight='bold' if (i, j) == (0, 4) else 'normal')
ax.add_patch(mpl.patches.Rectangle((3.5, -0.5), 1, 1, fill=False, ec=DARK, lw=1.6))
ax.set_xticks(range(5)); ax.set_xticklabels(['CNN', 'MLP', 'TCN', 'PW-\nTrans.', 'Patch-\nTST'], fontsize=7.2)
ax.set_yticks(range(3)); ax.set_yticklabels(['MLP', 'ResNet', 'FT-Trans.'])
ax.set_xlabel('Waveform complement encoder'); ax.set_ylabel('Handcrafted backbone encoder')
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-0.5, 5), minor=True); ax.set_yticks(np.arange(-0.5, 3), minor=True)
ax.grid(which='minor', color='white', lw=1.5); ax.tick_params(which='minor', length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.set_ticks([1.38, 1.44, A2, 1.56, 1.62]); cb.set_ticklabels(['1.38', '1.44', 'A2', '1.56', '1.62'])
cb.ax.tick_params(labelsize=6.8, length=2); cb.outline.set_linewidth(0.5)
cb.set_label('RMSE (pp)', fontsize=7.5)
panel(ax, 'a', x=-0.25, y=1.06)

# (b) interaction plot
ax = fig.add_subplot(gs[0, 1])
sty = [(BLUE, 'o', '-'), (GREEN, 's', '--'), (RED, '^', '-.')]
x = np.arange(5)
for i, (c, mk, ls) in enumerate(sty):
    ax.plot(x, M[i], color=c, marker=mk, ls=ls, ms=4.8, mec='white', mew=0.6, lw=1.3, label=['MLP', 'ResNet', 'FT-Trans.'][i])
ax.axhline(A2, color=GRAY, ls=(0, (4, 2)), lw=1.0)
ax.text(4.35, A2 + 0.004, 'A2', color=GRAY, fontsize=7.2, va='bottom', ha='right')
ax.annotate('R0', (4, M[0, 4]), xytext=(4, 1.345), ha='center', fontsize=7.5, fontweight='bold',
            arrowprops=dict(arrowstyle='-', color=DARK, lw=0.6))
ax.set_xticks(x); ax.set_xticklabels(['CNN', 'MLP', 'TCN', 'PW-\nTrans.', 'Patch-\nTST'], fontsize=7.2)
ax.set_xlim(-0.35, 4.35); ax.set_ylim(1.33, 1.69)
ax.set_xlabel('Waveform complement encoder'); ax.set_ylabel('RMSE (pp)')
ax.legend(loc='upper center', ncol=3, fontsize=6.8, columnspacing=0.8, handlelength=2.2,
          bbox_to_anchor=(0.5, 1.01), borderaxespad=0.2)
panel(ax, 'b', x=-0.2, y=1.06)

# (c) variance decomposition
ax = fig.add_subplot(gs[1, :])
rows = [('All 15 pairings', [84.4, 3.8, 11.8]), ('Without FT-Transformer', [0.1, 52.3, 47.6])]
cols = [BLUE, GREEN, RED]
names = ['Backbone', 'Complement', 'Pairing']
for r, (lab, vals) in enumerate(rows):
    left = 0
    for v, c, nm in zip(vals, cols, names):
        if v > 0.05:
            ax.barh(r, v, left=left, color=c, height=0.62, edgecolor='white', linewidth=1.2,
                    label=nm if r == 0 else None)
            if v >= 8:
                ax.text(left + v / 2, r, f'{v:.1f}%', ha='center', va='center', color='white', fontsize=7.4,
                        fontweight='bold')
            elif v >= 2:
                ax.text(left + v / 2, r - 0.36, f'{v:.1f}%', ha='center', va='bottom', color=DARK, fontsize=6.8)
        left += v
ax.text(0.3, 1 - 0.36, 'Backbone 0.1%', ha='left', va='bottom', fontsize=6.8, color=DARK)
ax.set_yticks([0, 1]); ax.set_yticklabels([r[0] for r in rows])
ax.set_ylim(1.45, -0.85)
ax.set_xlim(0, 100); ax.set_xlabel('Share of between-pairing sum of squares (%)')
ax.grid(axis='y', visible=False)
ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.93), handlelength=1.2, columnspacing=1.6)
panel(ax, 'c', x=-0.165, y=1.0)
save(fig, 'figs/fig3_matrix.png')
print('ok')
