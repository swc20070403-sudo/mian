"""3.7: time the ORIGINAL feature code of scripts/extract_lfp_features.py on one CPU thread.

The descriptor block of extract_lfp_features.main() (from ``features: dict`` up to ``feature_matrix =``) is cut
out of the frozen source at run time and executed verbatim (no copy, no edit), with the module's own helper
functions.  It computes all 150 candidates; 143 are voltage-derived, the 7 context columns are table look-ups
(negligible).  Two timings: (a) one pulse per call (batch 1), 200 calls, median; (b) all 17 130 pulses in one
call, divided by the number of pulses.

    python scripts/time_feature_extraction.py      (system Python 3.10: numpy, pandas, PyWavelets)
"""
import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import contextlib
import io
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
PROJECT = WORK.parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import extract_lfp_features as ex  # noqa: E402

src = (PROJECT / "scripts" / "extract_lfp_features.py").read_text(encoding="utf-8")
start = src.index("    features: dict[str, np.ndarray] = {}")
stop = src.index("    feature_matrix = np.column_stack")
block = compile(textwrap.dedent(src[start:stop]), "extract_lfp_features.main[descriptor block]", "exec")

data = ex.load_source(PROJECT / "data" / "UConn-ILCC-LFP" / "data_slowpulse_1.pkl")
N = len(np.asarray(data["cell_id"]))


def run(rows: np.ndarray) -> int:
    sub = {k: (np.asarray(v)[rows] if np.asarray(v).shape[:1] == (N,) else v) for k, v in data.items()}
    voltage = np.asarray(sub["voltage"], dtype=np.float64)
    env = dict(vars(ex))
    env.update(data=sub, voltage=voltage, n_rows=voltage.shape[0], soc=np.asarray(sub["soc"], dtype=np.int64),
               pulse_type=np.asarray(sub["pulse_type"]).astype(str), rpt=np.asarray(sub["rpt"], dtype=np.int64),
               soc_coulomb=np.asarray(sub["soc - coulomb"], dtype=np.float64),
               cumulative_cycles=np.zeros(voltage.shape[0]))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(block, env)
    return len(env["features"])


assert run(np.arange(5)) == 150
rng = np.random.default_rng(0)
single = []
for i in rng.choice(N, 200, replace=False):
    t0 = time.perf_counter_ns()
    run(np.array([i]))
    single.append(time.perf_counter_ns() - t0)
t0 = time.perf_counter()
run(np.arange(N))
batch = time.perf_counter() - t0
out = {"features_computed": 150, "voltage_derived": 143, "threads": 1,
       "single_pulse_ms_median": float(np.median(single) / 1e6),
       "single_pulse_ms_p05": float(np.percentile(single, 5) / 1e6),
       "single_pulse_ms_p95": float(np.percentile(single, 95) / 1e6), "single_pulse_calls": len(single),
       "batch_all_pulses_s": batch, "batch_pulses": int(N), "batch_ms_per_pulse": batch / N * 1e3,
       "python": sys.version.split()[0], "numpy": np.__version__}
(WORK / "results" / "posthoc").mkdir(parents=True, exist_ok=True)
(WORK / "results" / "posthoc" / "compute_cost_features.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(out)
