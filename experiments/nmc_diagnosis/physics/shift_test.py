"""Empirically flag absolute-level features: recompute features on V+c and compare."""
import sys, json, numpy as np
sys.path.insert(0, '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/work')
from features import waveform_features
W = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/work'
OUT = '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/nmcdiag/physics/out'
res = {}
for ds in ['LFP', 'NMC']:
    z = np.load(f'{W}/data_UConn-ILCC-{ds}.npz', allow_pickle=False)
    rng = np.random.default_rng(0)
    idx = rng.choice(z['y'].size, 300, replace=False)
    V = z['V'][idx].astype(float)
    F0, names = waveform_features(V)
    F1, _ = waveform_features(V + 0.1)
    sd = np.nanstd(z['F'][:, :143], 0)
    eff = np.nanmedian(np.abs(F1 - F0), 0) / np.maximum(sd, 1e-15)
    res[ds] = {n: float(e) for n, e in zip(names, eff)}
json.dump(res, open(f'{OUT}/shift_test.json', 'w'), indent=1)
names = list(res['LFP'])
for n in names:
    a, b = res['LFP'][n], res['NMC'][n]
    if max(a, b) > 1e-6:
        print(f'{n:22s} LFP {a:9.3f}  NMC {b:9.3f}')
