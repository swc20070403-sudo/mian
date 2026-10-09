import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from figstyle import plt, panel, save, BLUE, GRAY, DARK
K = [10, 20, 30, 50, 75, 100, 144]
d = [0.1388, 0.1219, 0.0906, 0.0, 0.0365, 0.0409, -0.0019]
lo = [0.0891, 0.0770, 0.0565, 0, 0.0177, 0.0208, -0.0187]
hi = [0.1880, 0.1690, 0.1249, 0, 0.0570, 0.0619, 0.0151]
par = [16481, 17121, 17761, 19041, 20641, 22241, 25057]
fig, ax = plt.subplots(figsize=(4.4, 2.6))
x = np.arange(len(K))
ax.axhline(0, color=GRAY, lw=0.8, ls='--')
for i in range(len(K)):
    if K[i] == 50:
        ax.plot(x[i], 0, marker='D', color=DARK, ms=5.5)
        continue
    ns = lo[i] <= 0 <= hi[i]
    ax.errorbar(x[i], d[i], yerr=[[d[i] - lo[i]], [hi[i] - d[i]]], fmt='o', ms=5.5, color=BLUE,
                mfc='white' if ns else BLUE, mec=BLUE, capsize=2.5, lw=1.0)
for i in range(len(K)):
    ax.text(x[i], max(hi[i], 0) + 0.012, f'{par[i]/1000:.1f}k', ha='center', va='bottom', fontsize=6.6, color='#52514e')
ax.set_xticks(x); ax.set_xticklabels([f'{k}\n(R0)' if k == 50 else str(k) for k in K])
ax.set_xlabel('Number of handcrafted features $K$')
ax.set_ylabel('ΔRMSE vs. $K$ = 50 (pp)')
ax.set_ylim(-0.04, 0.22)
save(fig, 'figs/figS2_K.png')
