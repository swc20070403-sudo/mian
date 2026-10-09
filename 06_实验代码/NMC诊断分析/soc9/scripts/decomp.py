import sys, os, json
W = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/work'
D = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/nmcdiag/soc9'
sys.path.insert(0, W)
import numpy as np
from stats import paired
d3 = dict(np.load(W + '/data_UConn-ILCC-NMC.npz')); d9 = dict(np.load(D + '/data_UConn-ILCC-NMC.npz'))
def preds(rdir, mo, n, seeds):
    out = []
    for s in seeds:
        p = np.full(n, np.nan)
        for f in range(5):
            r = np.load(f'{rdir}/{mo}_f{f}.npz' if s is None else f'{rdir}/{mo}_f{f}_s{s}.npz'); p[r['idx']] = r['pred']
        out.append(p)
    return out
def decomp(p, d):
    """per cell: record RMSE, group-mean (bias) RMSE, within-group RMS deviation; MSE_rec = MSE_bias + MSE_within"""
    e = p - d['y']
    key = d['cell'].astype(np.int64) * 10000 + d['rpt']
    uk, inv = np.unique(key, return_inverse=True)
    gm = np.bincount(inv, e) / np.bincount(inv)
    w = e - gm[inv]
    cells = np.unique(d['cell'])
    rec = np.array([np.sqrt(np.mean(e[d['cell'] == c] ** 2)) for c in cells])
    kc = uk // 10000
    bias = np.array([np.sqrt(np.mean(gm[kc == c] ** 2)) for c in cells])
    within = np.array([np.sqrt(np.mean(w[d['cell'] == c] ** 2)) for c in cells])
    return rec, bias, within
rnd = lambda r: {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)) for k, v in r.items()}
out = {}
for tag, d, rdir, seeds in [('soc9', d9, D + '/runs', [0, 1]), ('soc3_s01', d3, W + '/results/nmc', [0, 1]), ('soc3_s012', d3, W + '/results/nmc', [0, 1, 2])]:
    R = {}
    for mo in ['R0', 'A2', 'GBDT']:
        ss = [None] if mo == 'GBDT' else seeds
        rd = (D + '/runs') if (mo == 'GBDT' and tag == 'soc9') else rdir
        parts = [decomp(p, d) for p in preds(rd, mo, len(d['y']), ss)]
        R[mo] = [np.mean([q[i] for q in parts], 0) for i in range(3)]
    o = {}
    for i, nm in enumerate(['record', 'group_bias', 'within_group']):
        o[nm] = {mo: round(float(R[mo][i].mean()), 4) for mo in R}
        o[nm]['R0_minus_A2'] = rnd(paired(R['R0'][i], R['A2'][i]))
    out[tag] = o
    print(tag)
    for nm, v in o.items():
        r = v['R0_minus_A2']
        print(f"  {nm:12s} R0 {v['R0']:.3f} A2 {v['A2']:.3f} GBDT {v['GBDT']:.3f} | R0-A2 {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}] p={r['p']:.3f} impr {r['improved']}/44")
json.dump(out, open(D + '/decomp.json', 'w'), indent=1)
