"""Add zero-mean Gaussian noise to every pulse voltage of the LFP pickle (experiments 3.3 / 3.4).

Noise is added to the raw 101-point voltage BEFORE feature extraction; the publisher DCIR fields are
recomputed from the noisy trace with the publisher's own formula |V[40]-V[30]| / (1.2 - 0.24 A), which
reproduces the stored fields exactly on clean data (checked below), so every voltage-derived input sees
the same noise.  One fixed noise realisation per sigma (seed below) -> R0 and A2 see identical data.

    python scripts/make_noisy_pkl.py            (system Python 3.10; numpy only)
"""
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
PROJECT = WORK.parents[1]
sys.path.insert(0, str(PROJECT))
from dataloader.uconn_data import load_uconn_pulse_dict  # noqa: E402

SOURCE = PROJECT / "data" / "UConn-ILCC-LFP" / "data_slowpulse_1.pkl"
VARIANTS = {"noise_1p0mV": 1.0e-3, "noise_2p0mV": 2.0e-3, "noise_0p5mV": 0.5e-3}
SEEDS = {"noise_1p0mV": 20261009, "noise_2p0mV": 20261010, "noise_0p5mV": 20261011}
DELTA_I = 1.2 - 0.24


def dcir(voltage):
    return np.abs(voltage[:, 40] - voltage[:, 30]) / DELTA_I


data = load_uconn_pulse_dict(SOURCE)
voltage = np.asarray(data["voltage"], dtype=np.float64)
soc = np.asarray(data["soc"]); pulse = np.asarray(data["pulse_type"]).astype(str)
for s in (20, 50, 90):
    for p in ("chg", "dchg"):
        m = (soc == s) & (pulse == p)
        stored = np.asarray(data[f"dcir_{p}_{s}"], dtype=np.float64)[m]
        ok = np.isfinite(stored)
        err = np.max(np.abs(stored[ok] - dcir(voltage[m][ok])))
        assert err < 1e-9, (s, p, err)
print("publisher DCIR formula reproduced on clean data (max err < 1e-9)")

for name, sigma in VARIANTS.items():
    out = WORK / "data" / f"LFP_slowpulse_{name}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEEDS[name])
    noisy_v = voltage + rng.normal(0.0, sigma, size=voltage.shape)
    noisy = dict(data)
    noisy["voltage"] = noisy_v
    for s in (20, 50, 90):
        for p in ("chg", "dchg"):
            key = f"dcir_{p}_{s}"
            arr = np.asarray(data[key], dtype=np.float64).copy()
            m = (soc == s) & (pulse == p) & np.isfinite(arr)
            arr[m] = dcir(noisy_v[m])
            noisy[key] = arr
    with out.open("wb") as handle:
        pickle.dump(noisy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    noise = noisy_v - voltage
    print(f"{name}: sigma={sigma*1e3:.1f} mV seed={SEEDS[name]} realised sd={noise.std()*1e3:.4f} mV "
          f"-> {out.name} sha256={hashlib.sha256(out.read_bytes()).hexdigest()[:16]}")
