"""Per-seed robustness of the R0-A2 pattern by SOH region (saved OOF predictions only)."""
import os, sys, json
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

OUT = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/nmcdiag/errors'
src = open(f'{OUT}/analyze.py').read().split('res = {}')[0]
ns = {}
exec(compile(src, 'analyze_helpers', 'exec'), ns)
load, collect, cellmacro, pstats, R, DS = ns['load'], ns['collect'], ns['cellmacro'], ns['pstats'], ns['R'], ns['DS']

res = {}
for ds, (name, exp, models) in DS.items():
    d = load(name)
    P = {m: collect(d, exp, m) for m in ['R0', 'A2']}
    cells = sorted(np.unique(d['cell']).tolist())
    y = d['y']
    key = d['cell'].astype(np.int64) * 10000 + d['rpt'].astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv); ym = np.bincount(inv, y) / cnt
    tau_k = np.bincount(inv, d['cum']) / cnt
    out = res[ds] = {}
    regions = {'first_rpt': d['cum'] == 0, 'soh_ge90_excl_first': (y >= 90) & (d['cum'] > 0),
               'soh_80_90': (y >= 80) & (y < 90), 'soh_72_80': (y >= 72) & (y < 80), 'soh_lt72': y < 72,
               'all_excl_first': d['cum'] > 0, 'all': np.ones(y.size, bool)}
    for s in range(3):
        row = {}
        for lab, msk in regions.items():
            if not msk.any():
                continue
            a = cellmacro([P['R0'][s]], d, msk); b = cellmacro([P['A2'][s]], d, msk)
            st = pstats(a, b, cells)
            row[lab] = dict(diff=st['diff'], p=st['p'], improved=st['improved'], n=st['n'])
        # first-RPT bias per seed
        f = tau_k == 0
        row['first_rpt_bias'] = {m: R(np.mean((np.bincount(inv, P[m][s]) / cnt)[f] - ym[f])) for m in ['R0', 'A2']}
        out[f'seed{s}'] = row
    # pooled 3-seed (seed-averaged per-cell RMSE) by region
    pooled = {}
    for lab, msk in regions.items():
        if not msk.any():
            continue
        a = cellmacro(P['R0'], d, msk); b = cellmacro(P['A2'], d, msk)
        pooled[lab] = pstats(a, b, cells)
    out['seedavg'] = pooled

json.dump(res, open(f'{OUT}/results3.json', 'w'), indent=1)
for ds, x in res.items():
    print('=====', ds)
    for k, v in x.items():
        print(k)
        for kk, vv in v.items():
            print('   ', kk, vv)
