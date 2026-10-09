"""Clean records, build cumulative cycle count, compute 150 candidate features; save npz."""
import sys, time
import numpy as np
from safeload import load
from features import waveform_features

D = '/home/user/reil-uconn/fine-tuning-for-rapid-soh-estimation/processed_data/'


def prep(name, socs_keep=None, max_step=0.1, max_range=0.5):
    d = load(D + f'{name}/data_slowpulse_1.pkl')
    n0 = d['cell_id'].size
    keep = np.ones(n0, bool)
    if socs_keep is not None:
        keep &= np.isin(d['soc'], socs_keep)
    d = {k: v[keep] for k, v in d.items()}
    V = d['voltage']; y = d['soh'].astype(float)
    cyc = d['num_cycles'].astype(float)
    ok = np.isfinite(cyc)
    ok &= np.all((V >= 2.5) & (V <= 4.5), 1)
    rngV = V.max(1) - V.min(1)
    ok &= (rngV >= 0.005) & (rngV <= max_range)
    ok &= np.abs(np.diff(V, axis=1)).max(1) <= max_step
    ok &= (y >= 0) & (y <= 110)
    n_conds = len(np.unique(d['soc'])) * 2
    print(name, 'records before', keep.sum(), 'cell-rpt', len(set(zip(d['cell_id'], d['rpt']))))
    # group completeness and label consistency
    key = d['cell_id'] * 1000 + d['rpt']
    for k in np.unique(key):
        m = key == k
        if (ok[m].sum() != n_conds) or (np.ptp(y[m]) > 1e-5):
            ok[m] = False
    d = {k: v[ok] for k, v in d.items()}
    V = d['voltage']; y = d['soh'].astype(float)
    print(name, 'records after', ok.sum(), 'removed', (~ok).sum(),
          'cell-rpt', len(set(zip(d['cell_id'], d['rpt']))), 'cells', len(np.unique(d['cell_id'])),
          'SOH', y.min().round(2), y.max().round(2))
    # cumulative cycles: cumulative sum of retained inter-RPT intervals
    cum = np.zeros(y.size)
    for c in np.unique(d['cell_id']):
        m = d['cell_id'] == c
        rpts = np.unique(d['rpt'][m])
        tot = 0.0
        for r in rpts:
            mr = m & (d['rpt'] == r)
            tot += d['num_cycles'][mr][0]
            cum[mr] = tot
    t = time.time()
    Fw, names = waveform_features(V)
    print('features', Fw.shape, f'{time.time() - t:.1f}s')
    soc = d['soc'].astype(float); socc = d['soc - coulomb'].astype(float)
    direc = np.where(d['pulse_type'] == 'chg', 1.0, -1.0)
    dcir = np.array([d[f'dcir_{p}_{s}'][i] for i, (p, s) in enumerate(zip(d['pulse_type'], d['soc']))])
    ctx = np.stack([soc / 100, socc / 100, (socc - soc) / 100, direc, dcir, cum, d['rpt'].astype(float)], 1)
    ctx_names = ['ctx_soc_nom', 'ctx_soc_coul', 'ctx_soc_dev', 'ctx_dir', 'ctx_dcir', 'ctx_cycles', 'ctx_rpt']
    F = np.concatenate([Fw, ctx], 1)
    np.savez_compressed(f'data_{name}.npz', F=F, names=np.array(names + ctx_names), V=V, y=y,
                        cell=d['cell_id'], group=d['group_id'], rpt=d['rpt'], soc=d['soc'],
                        ptype=d['pulse_type'], cum=cum)


if __name__ == '__main__':
    prep('UConn-ILCC-LFP')
    prep('UConn-ILCC-NMC', socs_keep=[20, 50, 90], max_step=0.3, max_range=0.8)
