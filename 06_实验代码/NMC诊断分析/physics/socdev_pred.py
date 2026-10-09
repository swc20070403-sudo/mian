"""Side test: how well do dV / V0 / non-level features predict ctx_soc_dev (true minus nominal SOC)? Fixed-hp GBDT, cell-disjoint folds."""
import json, numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from common import *
out = {}
for ds in ['LFP', 'NMC']:
    d = load(ds); names = list(d['names'])
    dev = d['F'][:, names.index('ctx_soc_dev')] * 100
    V = d['V'].astype(float); dV = V - V[:, :1]; oh = onehot(d)
    noabs = [j for j, n in enumerate(names[:143]) if n not in ABS]
    sets = {'V0+cond': np.c_[V[:, :1], oh], 'dV+cond': np.c_[dV[:, 1:], oh], 'nonlevel_feats+cond': np.c_[d['F'][:, noabs], oh],
            'cond_only': oh}
    o = {}
    for nm, X in sets.items():
        pred = np.full(dev.size, np.nan)
        for f in range(5):
            itr, iva, ite = split_idx(d, f)
            m = HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=31, max_iter=300, random_state=0).fit(X[itr], dev[itr])
            pred[ite] = m.predict(X[ite])
        r2 = 1 - np.mean((pred - dev) ** 2) / np.var(dev)
        c = cond_id(d)
        within = np.mean([spearmanr(pred[c == k], dev[c == k])[0] for k in range(6)])
        o[nm] = dict(rmse_pct=round(float(np.sqrt(np.mean((pred - dev) ** 2))), 3), r2=round(float(r2), 3),
                     mean_within_cond_spearman=round(float(within), 3))
    o['socdev_sd_pct'] = round(float(dev.std()), 3)
    out[ds] = o
    print(ds, o)
json.dump(out, open(f'{OUT}/socdev_pred.json', 'w'), indent=1)
