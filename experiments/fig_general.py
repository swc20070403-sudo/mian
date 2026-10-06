import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from figstyle import plt, panel, save, BLUE, GREEN, RED, GRAY, DARK
from evalres import summary
from core import load_data

LFP = 'UConn-ILCC-LFP'
dl = load_data(LFP)
cell_group = {c: g for c, g in zip(dl['cell'], dl['group'])}
STY = {'R0': (BLUE, 'o', '-'), 'A2': (GRAY, 's', '--'), 'GBDT': (GREEN, 'D', '-.')}
ORDER = ['A2', 'R0', 'GBDT']

fig = plt.figure(figsize=(6.7, 5.1))
gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.30)

# (a) leave-one-group-out
ax = fig.add_subplot(gs[0, 0])
logo = {m: summary('logo', LFP, m, nf=11) for m in ORDER}
cv = {m: summary('cv', LFP, m) for m in ORDER}
groups = np.arange(1, 12)
off = {'A2': -0.22, 'R0': 0.0, 'GBDT': 0.22}
for m in ORDER:
    cells, pc, _ = logo[m]
    gm = [np.mean([v for c, v in zip(cells, pc) if cell_group[c] == g]) for g in groups]
    c, mk, ls = STY[m]
    ax.plot(groups + off[m], gm, mk, color=c, ms=4.4, mec='white', mew=0.5,
            label=f'{m}  {pc.mean():.3f} (CV {cv[m][1].mean():.3f})')
ax.set_xticks(groups); ax.set_xlabel('Held-out cycling group'); ax.set_ylabel('Group RMSE (pp)')
ax.set_ylim(0, 5.7)
ax.legend(loc='upper right', fontsize=6.6, handletextpad=0.2, borderaxespad=0.2,
          title='Cell-macro RMSE, leave-one-group-out', title_fontsize=6.6)
panel(ax, 'a', x=-0.15)

# (b) robustness: cycle-count bias and matched sensor noise
ax = fig.add_subplot(gs[0, 1])
rob = json.load(open('results/robustness.json'))
scen = [('Clean', None), ('τ +10%', 'tau+10'), ('τ −10%', 'tau-10'), ('Noise\n1 mV', 'noisy1'), ('Noise\n2 mV', 'noisy2')]
x = np.arange(len(scen))
for m in ORDER:
    vals = []
    for lab, key in scen:
        if key is None:
            vals.append(cv[m][1].mean())
        elif key.startswith('tau'):
            vals.append(np.mean(list(rob[key][m].values())))
        else:
            s = summary(key, f'UConn-ILCC-LFP-n{key[-1]}', m)
            vals.append(s[1].mean() if s else np.nan)
    c, mk, ls = STY[m]
    ax.plot(x + off[m] * 0.8, vals, mk, color=c, ms=4.6, mec='white', mew=0.5, label=m)
ax.axvline(2.5, color=GRAY, lw=0.6, ls=':')
ax.text(1.0, 1.0, 'Cycle-count bias\n(models fixed)', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.6)
ax.text(3.5, 1.0, 'Matched noise\n(models retrained)', transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.6)
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in scen], fontsize=7)
ax.set_ylabel('RMSE (pp)')
ax.legend(loc='lower right', fontsize=6.8, ncol=3, columnspacing=0.8, handletextpad=0.2)
panel(ax, 'b', x=-0.15)

# (c) number of pulses
ax = fig.add_subplot(gs[1, 0])
pe = json.load(open('results/pulses_ensemble.json'))['pulses']
ks = np.arange(1, 7)
for m in ORDER:
    mean = [pe[m][str(k)][0] for k in ks]; lo = [pe[m][str(k)][1] for k in ks]; hi = [pe[m][str(k)][2] for k in ks]
    c, mk, ls = STY[m]
    ax.fill_between(ks, lo, hi, color=c, alpha=0.14, lw=0)
    ax.plot(ks, mean, marker=mk, color=c, ls=ls, ms=4.2, mec='white', mew=0.5, label=m)
ax.set_xticks(ks); ax.set_xlabel('Pulses averaged per RPT'); ax.set_ylabel('RMSE (pp)')
ax.legend(loc='upper right', fontsize=6.8, ncol=3, columnspacing=0.8, handletextpad=0.2)
panel(ax, 'c', x=-0.15)

# (d) integrated-gradient attribution profile
ax = fig.add_subplot(gs[1, 1])
att = np.load('results/attribution.npz')
cond = list(att['cond']); ci = att['cond_ig']
t = np.arange(101)
chg = ci[[i for i, c in enumerate(cond) if c.startswith('chg')]].mean(0)
dch = ci[[i for i, c in enumerate(cond) if c.startswith('dchg')]].mean(0)
sm = lambda v: np.convolve(np.pad(v, (2, 1), mode='edge'), np.ones(4) / 4, mode='valid')
ax.plot(t, chg, color=BLUE, lw=0.5, alpha=0.25)
ax.plot(t, dch, color=RED, lw=0.5, alpha=0.25)
chg, dch = sm(chg), sm(dch)
for (a, b), lab, col in [((0, 30), 'C/5', '#eef3f9'), ((30, 40), '1C', '#dce8f6'), ((40, 100), 'Rest', '#ffffff')]:
    ax.axvspan(a, b, color=col, lw=0, zorder=0)
    ax.text((a + b) / 2, 1.01, lab, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.8)
ax.plot(t, chg, color=BLUE, lw=1.1, label='Charge')
ax.plot(t, dch, color=RED, lw=1.1, ls='--', label='Discharge')
ax.set_xlim(0, 100); ax.set_ylim(0, max(chg.max(), dch.max()) * 1.35)
ax.set_xlabel('Time (s)'); ax.set_ylabel('Mean |IG| per sample (pp)')
ax.legend(loc='upper right', fontsize=6.8, ncol=2, columnspacing=0.8)
ax.grid(axis='x', visible=False)
panel(ax, 'd', x=-0.15)
save(fig, 'figs/fig8_general.png')
print('ok')
