"""Collect per-cell RMSE of every re-implementation experiment into a CSV and a JSON summary."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from evalres import summary
from core import load_data, cv_folds, cell_rmse
from stats import paired

rows, summ = [], {}


def add(exp, dname, mode, nf=5, label=None):
    s = summary(exp, dname, mode, nf=nf)
    if s is None:
        return None
    cells, pc, ns = s
    lab = label or mode
    for c, v in zip(cells, pc):
        rows.append((exp, dname, lab, int(c), float(v)))
    summ.setdefault(exp, {})[lab] = round(float(pc.mean()), 4)
    return cells, pc


LFP, NMC = 'UConn-ILCC-LFP', 'UConn-ILCC-NMC'
for m in ['R0', 'A2', 'GBDT', 'SVR']:
    add('cv', LFP, m)
for m in ['R0', 'A2', 'GBDT']:
    add('logo', LFP, m, nf=11)
    add('noisy1', LFP + '-n1', m)
    add('noisy2', LFP + '-n2', m)
for m in ['R0', 'A2', 'A1', 'GBDT']:
    add('nmc', NMC, m)

# conventional baselines (BASE files)
d = load_data(LFP)
if len(glob.glob('results/cv/BASE_f*.npz')) == 5:
    keys = [k for k in np.load('results/cv/BASE_f0.npz').files if k != 'idx']
    groups = {}
    for k in keys:
        base = k.rsplit('_s', 1)[0] if k.startswith('RF_') else k
        groups.setdefault(base, []).append(k)
    folds = cv_folds(d['cell'])
    fo = {c: i for i, (a, b, t) in enumerate(folds) for c in t}
    for base, ks in groups.items():
        per = {}
        for k in ks:
            pred = np.full(d['y'].size, np.nan)
            for f in range(5):
                z = np.load(f'results/cv/BASE_f{f}.npz'); pred[z['idx']] = z[k]
            for c, v in cell_rmse(pred, d['y'], d['cell'], d['rpt']).items():
                per.setdefault(c, []).append(v)
        cells = sorted(per); pc = np.array([np.mean(per[c]) for c in cells])
        for c, v in zip(cells, pc):
            rows.append(('cv', LFP, base, int(c), float(v)))
        summ['cv'][base] = round(float(pc.mean()), 4)

for name in ['robustness', 'pulses_ensemble', 'cost']:
    p = f'results/{name}.json'
    if os.path.exists(p):
        js = json.load(open(p))
        if name == 'robustness':
            summ['robustness'] = {k: {m: round(float(np.mean(list(v.values()))), 4) for m, v in d2.items()}
                                  for k, d2 in js.items()}
        elif name == 'pulses_ensemble':
            summ['pulses'] = js['pulses']
        else:
            summ['cost'] = js
os.makedirs('summary', exist_ok=True)
with open('summary/per_cell_rmse.csv', 'w') as fh:
    fh.write('experiment,dataset,model,cell,rmse_pp\n')
    for r in rows:
        fh.write(','.join(map(str, r)) + '\n')
json.dump(summ, open('summary/results_summary.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k not in ('pulses', 'cost')}, indent=1))
