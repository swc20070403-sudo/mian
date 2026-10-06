# Re-implementation and supplementary experiments

This folder contains the CPU re-implementation of the manuscript's protocol. It was used for the generalisation, robustness and deployment experiments (Section 4.6 and Supplementary Note S6) and for the regenerated figures (Fig. 1, Fig. 3, Fig. 8 and Supplementary Fig. S2).

## Data

The processed pulse files come from the publisher's public repository
<https://github.com/REIL-UConn/fine-tuning-for-rapid-soh-estimation> (`processed_data/UConn-ILCC-LFP/data_slowpulse_1.pkl` and `processed_data/UConn-ILCC-NMC/data_slowpulse_1.pkl`). `safeload.py` reads them with an allow-listed unpickler that accepts only NumPy arrays.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Cleaning, cumulative cycle count, 150 candidate features | `prep.py` | `data_UConn-ILCC-LFP.npz`, `data_UConn-ILCC-NMC.npz` |
| Matched-noise copies (σ = 1, 2 mV) | `make_noisy.py` | `data_UConn-ILCC-LFP-n1.npz`, `-n2.npz` |
| Five-fold reproduction (R0, A2, GBDT) | `experiments.py cv` | `results/cv/` |
| Leave-one-cycling-group-out | `experiments.py logo` | `results/logo/` |
| NMC/Gr dataset | `experiments.py nmc` | `results/nmc/` |
| Matched noise | `experiments.py noisy1`, `noisy2` | `results/noisy1/`, `results/noisy2/` |
| Conventional regressors (ridge, τ-only and without-τ controls), RBF-SVR | `experiments.py baselight`, `svr` | `results/cv/BASEL_*`, `SVR_*` |
| Cycle-count bias, test-only noise, pulse count, attribution, cost | `extra.py robust pulses attr cost` | `results/*.json`, `results/attribution.npz` |
| Per-cell summary | `summarize.py` | `summary/per_cell_rmse.csv`, `summary/results_summary.json` |

Model, loss, training and metrics live in `core.py`; the feature definitions follow Supplementary Note S1 (`features.py`); paired bootstrap and sign-flip tests are in `stats.py`.

Set `OMP_NUM_THREADS=1` when running several workers in parallel. Neural runs use PyTorch on CPU with one thread each.

## Key results (cell-macro RMSE, SOH percentage points)

| Scenario | R0 | A2 | GBDT |
|---|---:|---:|---:|
| Five-fold cell-disjoint (re-implementation) | 1.427 | 1.474 | 1.028 |
| Leave-one-group-out | 1.560 | 1.594 | 1.401 |
| Cycle count +10% / −10% | 1.449 / 1.477 | 1.497 / 1.518 | 1.117 / 1.176 |
| Matched noise 1 mV / 2 mV | 1.620 / 1.715 | 1.678 / 1.760 | 1.372 / 1.424 |
| NMC/Gr five-fold | 2.030 | 2.026 | 0.935 |

`summary/results_summary.json` holds the full set.
