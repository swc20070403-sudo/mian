import sys, os, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import load_data, cell_rmse, cell_mae

def collect(exp, dname, mode, seeds=(0,1,2), nf=5):
    d = load_data(dname)
    per_seed = {}
    for s in seeds:
        pred = np.full(d['y'].size, np.nan)
        ok = True
        for f in range(nf):
            p = f'results/{exp}/{mode}_f{f}' + ('' if mode in ('GBDT','SVR') else f'_s{s}') + '.npz'
            if not os.path.exists(p): ok = False; break
            z = np.load(p); pred[z['idx']] = z['pred']
        if ok: per_seed[s] = pred
        if mode in ('GBDT','SVR'): break
    return d, per_seed

def summary(exp, dname, mode, nf=5):
    d, ps = collect(exp, dname, mode, nf=nf)
    if not ps: return None
    m = np.isfinite(list(ps.values())[0])
    r = {}
    for s, p in ps.items():
        cr = cell_rmse(p[m], d['y'][m], d['cell'][m], d['rpt'][m])
        for c, v in cr.items(): r.setdefault(c, []).append(v)
    cells = sorted(r)
    per_cell = np.array([np.mean(r[c]) for c in cells])
    return cells, per_cell, len(ps)

if __name__ == '__main__':
    exp, dname = sys.argv[1], sys.argv[2]; nf = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    base = None
    for mode in sys.argv[4:] if len(sys.argv) > 4 else ['R0','A2','A1','GBDT','SVR']:
        s = summary(exp, dname, mode, nf)
        if s is None: continue
        cells, pc, ns = s
        print(f'{mode}: cells={len(cells)} seeds={ns} RMSE={pc.mean():.4f}')
