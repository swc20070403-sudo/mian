"""Training stages for the Sec. 4.6 rerun (and the NMC 3x5 encoder matrix) with the ORIGINAL GPU code.

Every stage calls the frozen training loops unchanged, exactly like experiments/run_followup.py and
outputs/nmc_original_code_20261007/scripts/run_nmc_original.py do:
  R0 = run_encoder_benchmark_formal._train_one, setting patchtst__current_mlp   (--deterministic, as the matrix)
  A2 = run_supplemental_ablation_formal._train_one, arm A2_handcrafted_age_mono  (default args, as the ablation)
Label-derived SOC columns excluded (experiments/run_clean_rerun.install_feature_exclusion), K = 50,
lambda = 0.1, init seeds 0/1/2, Top-50 / imputation / standardisation re-fitted on the training cells of
every split.  Only the split (logo) or the input data (noise) or the data set + setting (nmc_matrix) differ.

Stages (training venv  "D:\\swc PINN\\.venv-fusion-experiments\\Scripts\\python.exe"):
  logo        3.1  11 leave-one-cycling-group-out splits x {R0, A2} x 3 seeds        = 66 runs
  noise       3.3  sigma = 1 mV and 2 mV, train + test noisy, 5 folds x {R0, A2} x 3   = 60 runs
  nmc_matrix  4    NMC 3x5 matrix minus R0 (reused from nmc_original_code_20261007)   = 210 runs
                   (--part i/n splits the 14 settings over n processes)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
PROJECT = WORK.parents[1]
os.environ["PULSEFUSION_RESULTS_ROOT"] = str(WORK / "results")  # must precede importing run_clean_rerun
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from experiments import run_clean_rerun as clean  # noqa: E402

LFP_PULSE = PROJECT / "data/UConn-ILCC-LFP/data_slowpulse_1.pkl"
LFP_FEATURES = PROJECT / "data/LFP_150_candidate_features.csv"
LFP_CLEAN_IDS = PROJECT / "data/data_cleaning_lfp/cleaned_row_ids.npy"
NMC = PROJECT / "outputs/nmc_original_code_20261007"
NMC_PULSE = NMC / "data/NMC_slowpulse_soc20_50_90.pkl"
NMC_FEATURES = NMC / "features_nmc/LFP_150_candidate_features.csv"
NMC_CLEAN_IDS = NMC / "cleaning/cleaned_row_ids.npy"
R0, A2 = "patchtst__current_mlp", "A2_handcrafted_age_mono"
SEEDS = (0, 1, 2)
NOISE = {"noise_1p0mV": 1.0, "noise_2p0mV": 2.0}


def load_table(pulse: Path, features: Path, clean_ids: Path):
    clean.install_feature_exclusion()
    clean.install_clean_a1_context()
    import dataloader.fusion_data as fd
    from dataloader.data_cleaning import load_cleaned_row_ids, restrict_to_cleaned_rows

    return restrict_to_cleaned_rows(fd.load_fusion_table(pulse, features, include_age_context=False),
                                    load_cleaned_row_ids(clean_ids))


def logo_folds(table) -> list[dict]:
    """Hold out one cycling group; validation = one randomly drawn cell from each remaining group."""
    group_of = {}
    for c, g in zip(table.cell_id.tolist(), table.group_id.tolist()):
        if group_of.setdefault(c, g) != g:
            raise RuntimeError(f"cell {c} in two groups")
    groups = sorted(set(group_of.values()))
    folds = []
    for g in groups:
        test = sorted(c for c, gg in group_of.items() if gg == g)
        rng = np.random.default_rng(42 + g)
        valid = sorted(int(rng.choice(sorted(c for c, gg in group_of.items() if gg == other)))
                       for other in groups if other != g)
        train = sorted(c for c in group_of if c not in test and c not in valid)
        folds.append({"fold": int(g), "held_out_group": int(g), "train_cells": train, "valid_cells": valid,
                      "test_cells": test})
    from run_encoder_benchmark_formal import _assert_disjoint_folds

    _assert_disjoint_folds(folds)
    return folds


def write_spec(out: Path, payload: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def done(out: Path) -> bool:
    return (out / "metrics.json").is_file() and (out / "test_predictions.npz").is_file()


def train_settings(table, folds, root: Path, settings, meta: dict, voltage_mode="delta") -> None:
    import run_encoder_benchmark_formal as bench
    import run_supplemental_ablation_formal as abl
    from dataloader.fusion_data import fit_fusion_preprocessor

    bench_args = bench.parse_args(["--device", "cuda", "--deterministic"])
    bench_args.output_root = root / "matrix"
    abl_args = abl.parse_args(["--device", "cuda"])
    for f in folds:
        k = f["fold"]
        train_idx = np.where(np.isin(table.cell_id, f["train_cells"]))[0]
        pre = fit_fusion_preprocessor(table, train_idx, 50, voltage_mode=voltage_mode)
        bench.save_fold_artifacts(root / "matrix", f, pre)
        for name in settings:
            for seed in SEEDS:
                is_matrix = "__" in name
                out = (root / "matrix" if is_matrix else root / "ablation") / name / f"fold_{k}" / f"init_{seed}"
                if done(out):
                    continue
                write_spec(out, {**meta, "setting": name, "fold": k, "init_seed": seed, "split": f,
                                 "excluded_features": list(clean.EXCLUDED_FEATURES)})
                started = time.perf_counter()
                if is_matrix:
                    result = bench._train_one(bench_args, table, f, name, k, seed, out, time.perf_counter() + 3600, pre)
                    rmse = result["test_cell_macro_RMSE"]
                else:
                    abl._train_one(abl_args, table, f, abl.ARM_BY_NAME[name], k, seed, out,
                                   time.perf_counter() + 3600, pre)
                    rmse = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["test_cell_macro_RMSE"]
                print(f"[{meta['experiment']}] {name} fold={k} seed={seed} cell-RMSE={rmse:.4f} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)


def stage_logo() -> None:
    table = load_table(LFP_PULSE, LFP_FEATURES, LFP_CLEAN_IDS)
    folds = logo_folds(table)
    root = WORK / "results" / "logo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "logo_folds.json").write_text(json.dumps(folds, indent=1), encoding="utf-8")
    for f in folds:
        print(f"group {f['fold']}: train/valid/test cells {len(f['train_cells'])}/{len(f['valid_cells'])}/"
              f"{len(f['test_cells'])}", flush=True)
    train_settings(table, folds, root, (R0, A2), {"experiment": "logo_3.1"})


def stage_noise(only: str | None) -> None:
    from run_encoder_benchmark_formal import make_cell_folds

    for name, sigma in NOISE.items():
        if only and only != name:
            continue
        pulse = WORK / "data" / f"LFP_slowpulse_{name}.pkl"
        feats = WORK / f"features_{name}" / "LFP_150_candidate_features.csv"
        table = load_table(pulse, feats, LFP_CLEAN_IDS)
        folds = make_cell_folds(table.cell_id, 5, 42)
        train_settings(table, folds, WORK / "results" / name, (R0, A2),
                       {"experiment": "matched_noise_3.3", "sigma_mV": sigma, "pulse_data": str(pulse),
                        "feature_csv": str(feats), "cleaned_rows": "original clean row ids (not re-cleaned)"})


def stage_nmc_matrix(part: str) -> None:
    from run_encoder_benchmark_formal import SETTINGS, make_cell_folds

    i, n = (int(x) for x in part.split("/"))
    settings = [s for s in SETTINGS if s != R0]
    # interleave the slow FT-Transformer / Transformer settings over the parts
    settings = settings[i::n]
    table = load_table(NMC_PULSE, NMC_FEATURES, NMC_CLEAN_IDS)
    folds = make_cell_folds(table.cell_id, 5, 42)
    root = WORK / "results" / "nmc_matrix"
    print(f"part {part}: {settings}", flush=True)
    import run_encoder_benchmark_formal as bench
    from dataloader.fusion_data import fit_fusion_preprocessor

    for f in folds:  # fold artifacts must equal those of the reused NMC R0 runs before anything is trained
        pre = fit_fusion_preprocessor(table, np.where(np.isin(table.cell_id, f["train_cells"]))[0], 50,
                                      voltage_mode="delta")
        bench.save_fold_artifacts(root / "matrix", f, pre)
        for fn in ("preprocessor.json", "split.json", "selected_features.csv"):
            a = (root / "matrix" / f"fold_{f['fold']}" / fn).read_bytes()
            b = (NMC / "results/encoder_benchmark" / f"fold_{f['fold']}" / fn).read_bytes()
            if a != b:
                raise RuntimeError(f"NMC fold {f['fold']} {fn} differs from the reused R0 runs")
    print("NMC fold artifacts identical to nmc_original_code_20261007 (R0 reuse valid)", flush=True)
    train_settings(table, folds, root, settings, {"experiment": "nmc_matrix_4", "pulse_data": str(NMC_PULSE),
                                                  "feature_csv": str(NMC_FEATURES)})


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("logo", "noise", "nmc_matrix"))
    p.add_argument("--only")
    p.add_argument("--part", default="0/1")
    a = p.parse_args(argv)
    if a.stage == "logo":
        stage_logo()
    elif a.stage == "noise":
        stage_noise(a.only)
    else:
        stage_nmc_matrix(a.part)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
