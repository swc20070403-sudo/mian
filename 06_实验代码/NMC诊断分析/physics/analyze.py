import json, glob, os, numpy as np
from common import *

SETS = ['K50_dV', 'K50', 'K50noabs', 'K50noabswav', 'K50noabs_V0', 'abs', 'tau', 'V0', 'amp', 'amp_V0', 'shape', 'dV', 'dV_V0', 'absV']
SAVED = {'LFP': 'cv', 'NMC': 'nmc'}


def oof_gbdt(ds, s, d):
    pred = np.full(d['y'].size, np.nan)
    for f in range(5):
        p = f'{OUT}/gbdt/{ds}_{s}_f{f}.npz'
        if not os.path.exists(p): return None
        z = np.load(p); pred[z['idx']] = z['pred']
    return pred


def oof_saved(ds, mode, seed, d):
    pred = np.full(d['y'].size, np.nan)
    for f in range(5):
        p = f'{W}/results/{SAVED[ds]}/{mode}_f{f}' + ('' if mode == 'GBDT' else f'_s{seed}') + '.npz'
        z = np.load(p); pred[z['idx']] = z['pred']
    return pred


def oof_nn(ds, tag, d, seed=0):
    pred = np.full(d['y'].size, np.nan)
    for f in range(5):
        p = f'{OUT}/nn/{ds}_{tag}_f{f}_s{seed}.npz'
        if not os.path.exists(p): return None
        z = np.load(p); pred[z['idx']] = z['pred']
    return pred


res = {'gbdt': {}, 'gbdt_paired': {}, 'nn': {}, 'perm': {}}
for ds in ['LFP', 'NMC']:
    d = load(ds)
    pcs = {}
    for s in SETS:
        p = oof_gbdt(ds, s, d)
        if p is None: continue
        cells, pc = per_cell(p, d); pcs[s] = pc
    cells, pcs['saved_GBDT'] = per_cell(oof_saved(ds, 'GBDT', 0, d), d)
    for m in ['R0', 'A2']:
        pcs[f'saved_{m}_3seed'] = np.mean([per_cell(oof_saved(ds, m, s, d), d)[1] for s in range(3)], 0)
    res['gbdt'][ds] = {k: round(float(v.mean()), 3) for k, v in pcs.items()}
    comps = [('K50noabs', 'K50'), ('K50noabswav', 'K50'), ('K50noabs_V0', 'K50noabs'), ('abs', 'tau'), ('abs', 'K50'),
             ('amp', 'tau'), ('amp_V0', 'amp'), ('shape', 'tau'), ('dV', 'tau'), ('dV', 'amp'), ('shape', 'amp'), ('shape', 'dV'),
             ('dV_V0', 'dV'), ('absV', 'dV'), ('K50', 'dV'), ('K50', 'saved_GBDT'), ('K50_dV', 'K50'), ('V0', 'tau'), ('dV', 'abs'), ('shape', 'K50')]
    res['gbdt_paired'][ds] = {f'{a} - {b}': pstat(pcs[a], pcs[b]) for a, b in comps if a in pcs and b in pcs}
    # permutation importance
    for s in ['K50', 'K50noabs']:
        agg = {}
        for f in range(5):
            p = f'{OUT}/gbdt/{ds}_{s}_f{f}.npz'
            if not os.path.exists(p): break
            imp = json.loads(str(np.load(p)['perm']))
            for g, (m, sd, n) in imp.items():
                agg.setdefault(g, []).append((m, n))
        else:
            res['perm'].setdefault(ds, {})[s] = {g: dict(mean_dRMSE=round(float(np.mean([a for a, _ in v])), 3),
                                                          per_fold=[round(a, 3) for a, _ in v],
                                                          n_feat_per_fold=[n for _, n in v], folds_present=len(v))
                                                  for g, v in sorted(agg.items(), key=lambda kv: -np.mean([a for a, _ in kv[1]]))}
    # NN
    if ds == 'NMC':
        nnp = {}
        for m in ['R0', 'A2']:
            for s_ in range(3):
                nnp[f'{m}_s{s_}'] = per_cell(oof_saved(ds, m, s_, d), d)[1]
        for tag in ['R0raw', 'A2noabs', 'R0noabs', 'R0rawnoabs']:
            for s_ in range(3):
                p = oof_nn(ds, tag, d, s_)
                if p is not None: nnp[f'{tag}_s{s_}'] = per_cell(p, d)[1]
        for tag in ['R0', 'A2', 'R0raw', 'A2noabs', 'R0noabs', 'R0rawnoabs']:
            if all(f'{tag}_s{s_}' in nnp for s_ in range(3)):
                nnp[f'{tag}_3seed'] = np.mean([nnp[f'{tag}_s{s_}'] for s_ in range(3)], 0)
        res['nn'][ds] = {'rmse': {k: round(float(v.mean()), 3) for k, v in nnp.items()}}
        comps = []
        for suf in ['3seed', 's0', 's1', 's2']:
            comps += [(f'R0_{suf}', f'A2_{suf}'), (f'R0raw_{suf}', f'R0_{suf}'), (f'R0raw_{suf}', f'A2_{suf}'),
                      (f'A2noabs_{suf}', f'A2_{suf}'), (f'R0noabs_{suf}', f'A2noabs_{suf}'), (f'R0noabs_{suf}', f'R0_{suf}'),
                      (f'R0rawnoabs_{suf}', f'A2noabs_{suf}'), (f'R0rawnoabs_{suf}', f'R0noabs_{suf}'), (f'R0rawnoabs_{suf}', f'R0raw_{suf}')]
        comps += [('R0_s1', 'R0_s0'), ('R0_s2', 'R0_s0'), ('A2_s1', 'A2_s0'), ('A2_s2', 'A2_s0')]
        res['nn'][ds]['paired'] = {f'{a} - {b}': pstat(nnp[a], nnp[b]) for a, b in comps if a in nnp and b in nnp}
        # seed noise: sd over seeds of macro RMSE
        res['nn'][ds]['seed_sd_macro'] = {t: round(float(np.std([nnp[f'{t}_s{s_}'].mean() for s_ in range(3)], ddof=1)), 3)
                                          for t in ['R0', 'A2', 'R0raw', 'A2noabs', 'R0noabs', 'R0rawnoabs'] if f'{t}_s2' in nnp}
json.dump(res, open(f'{OUT}/analyze.json', 'w'), indent=1)
for k in ['gbdt', 'gbdt_paired', 'nn']:
    for ds, v in res[k].items():
        print('==', k, ds)
        if isinstance(v, dict) and 'paired' in v:
            print(v['rmse']); print('seed sd', v['seed_sd_macro'])
            for a, b in v['paired'].items(): print('  ', a, b)
        elif k == 'gbdt':
            print(v)
        else:
            for a, b in v.items(): print('  ', a, b)
for ds, v in res['perm'].items():
    for s, g in v.items():
        print('== perm', ds, s)
        for a, b in g.items(): print('  ', f'{a:12s}', b['mean_dRMSE'], b['per_fold'], b['n_feat_per_fold'])
