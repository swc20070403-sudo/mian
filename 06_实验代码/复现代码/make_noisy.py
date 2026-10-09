import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from features import waveform_features
z = dict(np.load('data_UConn-ILCC-LFP.npz'))
names = list(z['names'])
for sig in [1.0, 2.0]:
    rng = np.random.default_rng(int(sig * 100))
    V = z['V'] + rng.normal(0, sig / 1000, z['V'].shape)
    Fw, _ = waveform_features(V)
    F = z['F'].copy(); F[:, :143] = Fw
    F[:, names.index('ctx_dcir')] = np.abs(V[:, 40] - V[:, 30]) / (1.2 - 0.24)
    out = dict(z); out['V'] = V; out['F'] = F
    np.savez_compressed(f'data_UConn-ILCC-LFP-n{int(sig)}.npz', **out)
    print('saved', sig)
