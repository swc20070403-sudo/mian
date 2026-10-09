import sys, json
sys.path.insert(0, '/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/work')
import numpy as np
from safeload import load
d = load('/home/user/reil-uconn/fine-tuning-for-rapid-soh-estimation/processed_data/UConn-ILCC-NMC/data_slowpulse_1.pkl')
V = d['voltage']; y = d['soh'].astype(float); cyc = d['num_cycles'].astype(float)
max_step, max_range = 0.3, 0.8
f = {}
f['finite_cyc'] = np.isfinite(cyc)
f['V_in_2.5_4.5'] = np.all((V >= 2.5) & (V <= 4.5), 1)
rngV = V.max(1) - V.min(1)
f['range_ok'] = (rngV >= 0.005) & (rngV <= max_range)
f['step_ok'] = np.abs(np.diff(V, axis=1)).max(1) <= max_step
f['soh_ok'] = (y >= 0) & (y <= 110)
ok = np.ones(len(y), bool)
for k, v in f.items(): ok &= v
out = {}
socs = np.unique(d['soc'])
print('record-level pass rate by soc x ptype:')
for s in socs:
    for p in ['chg', 'dchg']:
        m = (d['soc'] == s) & (d['pulse_type'] == p)
        fails = {k: int((~v[m]).sum()) for k, v in f.items() if (~v[m]).sum() > 0}
        print(s, p, m.sum(), 'pass', ok[m].sum(), fails)
        out[f'{s}_{p}'] = dict(n=int(m.sum()), passed=int(ok[m].sum()), fails=fails)
key = d['cell_id'] * 1000 + d['rpt']
uk = np.unique(key)
# per cell-rpt: which conditions pass and label consistency
def complete(socs_req):
    keep_k = []
    for k in uk:
        m = key == k
        mm = m & np.isin(d['soc'], socs_req)
        nreq = len(socs_req) * 2
        if mm.sum() == nreq and ok[mm].sum() == nreq and np.ptp(y[m]) <= 1e-5:
            keep_k.append(k)
    return np.array(keep_k)
for name, sr in [('3soc', [20, 50, 90]), ('9soc', list(socs)), ('8soc_no10', [20,30,40,50,60,70,80,90]), ('8soc_no90', [10,20,30,40,50,60,70,80]), ('7soc_20_80', [20,30,40,50,60,70,80])]:
    kk = complete(sr)
    cells = np.unique(kk // 1000)
    print(name, 'complete cell-rpt', len(kk), 'of', len(uk), 'records', len(kk) * len(sr) * 2, 'cells', len(cells))
    out[name] = dict(cell_rpt=int(len(kk)), records=int(len(kk) * len(sr) * 2), cells=int(len(cells)))
# label consistency check alone
lab_bad = sum(np.ptp(y[key == k]) > 1e-5 for k in uk)
print('cell-rpt with inconsistent labels', lab_bad)
# per-cell-rpt count of failing SOC conditions
nfail = np.array([(~ok[key == k]).sum() for k in uk])
print('distribution of failing records per cell-rpt (9 SOC):', np.unique(nfail, return_counts=True))
json.dump(out, open('/tmp/claude-0/-home-user-mian/1d84ac3c-62fa-58c0-8bb6-3fefff5673a1/scratchpad/nmcdiag/soc9/clean_inspect.json', 'w'), indent=1)
