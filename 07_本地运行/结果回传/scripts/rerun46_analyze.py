"""Statistics + deliverables for the Sec. 4.6 rerun and the NMC encoder matrix (trains nothing).

Metric everywhere = the paper's cell-macro RMSE (pp): per cell-RPT average the 6 pulse conditions, RMSE over
the cell's RPTs, per cell average the 3 init seeds, then average cells (run_encoder_benchmark_formal.cell_macro_metrics
+ paired_differences).  Paired statistics = frozen paired_bootstrap (5,000 cell resamples stratified by outer fold /
held-out group) and paired_randomization_p (5,000 two-sided sign flips), improved cells, d_z = mean/sd(ddof=1).

Writes deliverables/results_summary.json, per_cell_rmse.csv, 3.1/3.5/3.6/4 plot CSVs.   Training venv (numpy + scipy).
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
PROJECT = WORK.parents[1]
sys.path.insert(0, str(PROJECT))
from run_encoder_benchmark_formal import (HANDCRAFTED_ENCODERS, SETTINGS, WAVEFORM_ENCODERS, _stable_seed,  # noqa: E402
                                          paired_bootstrap, paired_randomization_p)

ORIG = PROJECT / "results/clean_rerun_20260927"
NMC = PROJECT / "outputs/nmc_original_code_20261007/results"
RES = WORK / "results"
DEL = WORK / "deliverables"
R0, A2 = "patchtst__current_mlp", "A2_handcrafted_age_mono"
N = 5000
PER_CELL = []  # rows for per_cell_rmse.csv
LABEL = {"cnn": "CNN", "mlp": "MLP", "tcn": "TCN", "transformer": "PW-Transformer", "patchtst": "PatchTST",
         "current_mlp": "MLP", "rtdl_resnet": "ResNet", "rtdl_ft_transformer": "FT-Transformer"}


# ----------------------------------------------------------------------------- loading
def runs(root: Path, setting: str, expected: int | None = None) -> dict[tuple[int, int], dict[int, float]]:
    """{(fold, seed): {cell: RMSE}} from metrics.json of every run."""
    out = {}
    for p in sorted((root / setting).glob("fold_*/init_*/metrics.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        out[(int(m["fold"]), int(m["init_seed"]))] = {int(r["cell_id"]): float(r["RMSE"]) for r in m["cell_metrics"]}
    if expected is not None and len(out) != expected:
        raise RuntimeError(f"{root / setting}: {len(out)} runs, expected {expected}")
    return out


def from_json(path: Path, model: str) -> dict[tuple[int, int], dict[int, float]]:
    return {(r["fold"], r["seed"]): {int(c): v for c, v in r["cells"].items()}
            for r in json.loads(path.read_text(encoding="utf-8")) if r["model"] == model}


def record(experiment: str, dataset: str, model: str, per_run: dict) -> None:
    for (fold, seed), cells in sorted(per_run.items()):
        for c, v in sorted(cells.items()):
            PER_CELL.append({"experiment": experiment, "dataset": dataset, "model": model, "cell": c, "seed": seed,
                             "RMSE": v, "fold": fold})


def seed_mean(per_run: dict, seeds=(0, 1, 2)) -> dict[tuple[int, int], float]:
    acc = defaultdict(list)
    for (fold, seed), cells in per_run.items():
        if seed in seeds:
            for c, v in cells.items():
                acc[(fold, c)].append(v)
    if any(len(v) != len(seeds) for v in acc.values()):
        raise RuntimeError("missing seeds")
    return {k: float(np.mean(v)) for k, v in acc.items()}


def absolute(values: dict, name: str) -> dict:
    rows = [{"fold": f, "difference": v} for (f, _), v in sorted(values.items())]
    ci = paired_bootstrap(rows, N, _stable_seed(name + "_absolute"))["ci95"]
    return {"RMSE": float(np.mean([r["difference"] for r in rows])), "ci95": ci, "n_cells": len(rows)}


def paired(cand_runs: dict, base_runs: dict, name: str) -> dict:
    """cand - base, paired by cell (seed-averaged); also per-seed mean differences."""
    a, b = seed_mean(cand_runs), seed_mean(base_runs)
    if set(a) != set(b):
        # different splits (e.g. LOGO vs 5-fold): pair by cell only, stratify by the candidate's fold
        a_c = {c: (f, v) for (f, c), v in a.items()}
        b_c = {c: v for (_, c), v in b.items()}
        if set(a_c) != set(b_c):
            raise RuntimeError(f"{name}: unpaired cells")
        rows = [{"fold": f, "cell_id": c, "difference": v - b_c[c]} for c, (f, v) in sorted(a_c.items())]
        per_seed = []
        for s in (0, 1, 2):
            sa, sb = seed_mean(cand_runs, (s,)), seed_mean(base_runs, (s,))
            sb_c = {c: v for (_, c), v in sb.items()}
            per_seed.append(float(np.mean([v - sb_c[c] for (_, c), v in sa.items()])))
    else:
        rows = [{"fold": f, "cell_id": c, "difference": a[(f, c)] - b[(f, c)]} for f, c in sorted(a)]
        per_seed = []
        for s in (0, 1, 2):
            sa, sb = seed_mean(cand_runs, (s,)), seed_mean(base_runs, (s,))
            per_seed.append(float(np.mean([sa[k] - sb[k] for k in sa])))
    d = np.array([r["difference"] for r in rows])
    boot = paired_bootstrap(rows, N, _stable_seed(name + "_bootstrap"))
    return {"contrast": name, "difference": boot["mean_difference"], "ci95": boot["ci95"],
            "p_signflip": paired_randomization_p(rows, N, _stable_seed(name + "_randomization")),
            "improved_cells": int((d < 0).sum()), "n_cells": len(rows),
            "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None,
            "per_seed_difference": per_seed, "seeds": 3, "folds": len({r["fold"] for r in rows}),
            "resamples": N}


def model_block(per_run: dict, name: str) -> dict:
    out = absolute(seed_mean(per_run), name)
    out["per_seed_RMSE"] = [float(np.mean(list(seed_mean(per_run, (s,)).values()))) for s in (0, 1, 2)]
    out["seeds"] = 3
    out["folds"] = len({f for f, _ in per_run})
    return out


def mae_of(root, setting):
    acc = defaultdict(list)
    for p in sorted((root / setting).glob("fold_*/init_*/metrics.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        for r in m["cell_metrics"]:
            acc[(m["fold"], r["cell_id"])].append(r["MAE"])
    return float(np.mean([np.mean(v) for v in acc.values()]))


# ----------------------------------------------------------------------------- experiments
def original_lfp():
    r0 = runs(ORIG / "encoder_benchmark", R0, 15)
    a2 = runs(ORIG / "supplemental_ablation" / "runs", A2, 15)
    record("main_5fold", "LFP", "R0", r0)
    record("main_5fold", "LFP", "A2", a2)
    return r0, a2


def exp_nmc():
    r0 = runs(NMC / "encoder_benchmark", R0, 15)
    a2 = runs(NMC / "supplemental_ablation", A2, 15)
    record("2.1_nmc_5fold", "NMC", "R0", r0)
    record("2.1_nmc_5fold", "NMC", "A2", a2)
    return {"R0": {**model_block(r0, "nmc_R0"), "MAE": mae_of(NMC / "encoder_benchmark", R0)},
            "A2": {**model_block(a2, "nmc_A2"), "MAE": mae_of(NMC / "supplemental_ablation", A2)},
            "R0_minus_A2": paired(r0, a2, "R0 - A2"),
            "note": "Same numbers as nmc_original_code_20261007/results/analysis (identical bootstrap seeds)."}


def exp_logo(r0_5, a2_5):
    root = RES / "logo"
    folds = json.loads((root / "logo_folds.json").read_text(encoding="utf-8"))
    r0 = runs(root / "matrix", R0, 33)
    a2 = runs(root / "ablation", A2, 33)
    record("3.1_logo", "LFP", "R0", r0)
    record("3.1_logo", "LFP", "A2", a2)
    s0, s2 = seed_mean(r0), seed_mean(a2)
    groups = []
    for f in folds:
        g = f["fold"]
        cells = f["test_cells"]
        r = [s0[(g, c)] for c in cells]
        a = [s2[(g, c)] for c in cells]
        groups.append({"group": g, "n_cells": len(cells), "R0_group_mean_RMSE": float(np.mean(r)),
                       "A2_group_mean_RMSE": float(np.mean(a)), "R0_minus_A2": float(np.mean(r) - np.mean(a)),
                       "R0_improved_cells": int(sum(x < y for x, y in zip(r, a))),
                       "n_train_cells": len(f["train_cells"]), "n_valid_cells": len(f["valid_cells"])})
    with (DEL / "plot_3.1_logo_group_rmse.csv").open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(groups[0]))
        w.writeheader()
        w.writerows(groups)
    return {"R0": model_block(r0, "logo_R0"), "A2": model_block(a2, "logo_A2"),
            "R0_minus_A2": paired(r0, a2, "LOGO R0 - A2"),
            "R0_LOGO_minus_R0_5fold": paired(r0, r0_5, "R0 LOGO - R0 5-fold"),
            "A2_LOGO_minus_A2_5fold": paired(a2, a2_5, "A2 LOGO - A2 5-fold"),
            "per_group": groups,
            "split": "test = one cycling group (6 cells; group 11: 4); validation = 1 random cell from each other "
                     "group (10 cells, rng seed 42+group); train = rest (48 or 50 cells); Top-50 refitted per split; "
                     "bootstrap stratified by held-out group"}


def exp_cycle(r0_5, a2_5):
    out = {}
    for factor in ("1.1", "0.9"):
        path = RES / "posthoc" / f"cycle_x{factor}.json"
        r0, a2 = from_json(path, "R0"), from_json(path, "A2")
        record(f"3.2_cycles_x{factor}", "LFP", "R0", r0)
        record(f"3.2_cycles_x{factor}", "LFP", "A2", a2)
        out[f"x{factor}"] = {"R0": model_block(r0, f"cyc{factor}_R0"), "A2": model_block(a2, f"cyc{factor}_A2"),
                             "R0_minus_A2": paired(r0, a2, f"cycles x{factor} R0 - A2"),
                             "R0_minus_R0_clean": paired(r0, r0_5, f"cycles x{factor} R0 - R0 clean"),
                             "A2_minus_A2_clean": paired(a2, a2_5, f"cycles x{factor} A2 - A2 clean")}
    out["note"] = ("tau (raw cumulative cycle count) of TEST rows multiplied before the train-fold standardisation; "
                   "the original 30 trained models, not retrained.")
    return out


def exp_noise(r0_5, a2_5):
    out = {}
    for name in ("noise_1p0mV", "noise_2p0mV"):
        root = RES / name
        r0, a2 = runs(root / "matrix", R0, 15), runs(root / "ablation", A2, 15)
        record(f"3.3_matched_{name}", "LFP", "R0", r0)
        record(f"3.3_matched_{name}", "LFP", "A2", a2)
        out[name] = {"R0": model_block(r0, f"{name}_R0"), "A2": model_block(a2, f"{name}_A2"),
                     "R0_minus_A2": paired(r0, a2, f"{name} R0 - A2"),
                     "R0_minus_R0_clean": paired(r0, r0_5, f"{name} R0 - R0 clean"),
                     "A2_minus_A2_clean": paired(a2, a2_5, f"{name} A2 - A2 clean")}
    return out


def exp_tnoise(r0_5, a2_5):
    path = RES / "posthoc" / "test_noise_0p5mV.json"
    r0, a2 = from_json(path, "R0"), from_json(path, "A2")
    record("3.4_test_only_noise_0p5mV", "LFP", "R0", r0)
    record("3.4_test_only_noise_0p5mV", "LFP", "A2", a2)
    return {"R0": model_block(r0, "tn_R0"), "A2": model_block(a2, "tn_A2"),
            "R0_minus_A2": paired(r0, a2, "test-noise R0 - A2"),
            "R0_minus_R0_clean": paired(r0, r0_5, "test-noise R0 - R0 clean"),
            "A2_minus_A2_clean": paired(a2, a2_5, "test-noise A2 - A2 clean")}


def exp_pulses():
    rows = json.loads((RES / "posthoc" / "pulse_subsets.json").read_text(encoding="utf-8"))
    by = defaultdict(dict)  # (model, combo) -> {(fold, seed): cells}
    for r in rows:
        by[(r["model"], tuple(r["combo"]))][(r["fold"], r["seed"])] = {int(c): v for c, v in r["cells"].items()}
    curve = []
    summary = {}
    for k in range(1, 7):
        combos = sorted({c for (_, c) in by if len(c) == k})
        entry = {"k": k, "n_combinations": len(combos)}
        for model in ("R0", "A2"):
            vals = {c: float(np.mean(list(seed_mean(by[(model, c)]).values()))) for c in combos}
            best = min(vals, key=vals.get)
            worst = max(vals, key=vals.get)
            entry.update({f"{model}_mean_over_combinations": float(np.mean(list(vals.values()))),
                          f"{model}_min": vals[best], f"{model}_min_combo": "+".join(best),
                          f"{model}_max": vals[worst], f"{model}_max_combo": "+".join(worst)})
        diffs = [float(np.mean(list(seed_mean(by[("R0", c)]).values())) - np.mean(list(seed_mean(by[("A2", c)]).values())))
                 for c in combos]
        entry["R0_minus_A2_mean_over_combinations"] = float(np.mean(diffs))
        entry["combinations_where_R0_better"] = int(sum(d < 0 for d in diffs))
        curve.append(entry)
    with (DEL / "plot_3.5_pulse_count.csv").open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(curve[0]))
        w.writeheader()
        w.writerows(curve)
    with (DEL / "plot_3.5_pulse_subsets_all_combinations.csv").open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.writer(h)
        w.writerow(["k", "combination", "R0_RMSE", "A2_RMSE"])
        for (model, c) in sorted(by):
            if model == "R0":
                w.writerow([len(c), "+".join(c), float(np.mean(list(seed_mean(by[("R0", c)]).values()))),
                            float(np.mean(list(seed_mean(by[("A2", c)]).values())))])
    chg = ("20chg", "50chg", "90chg")
    dchg = ("20dchg", "50dchg", "90dchg")
    full = tuple(sorted({c for (_, c) in by if len(c) == 6}))[0]
    for model in ("R0", "A2"):
        record("3.5_three_charge_pulses", "LFP", model, by[(model, chg)])
        record("3.5_three_discharge_pulses", "LFP", model, by[(model, dchg)])
    summary = {
        "curve": curve,
        "three_charge_pulses": {"R0": model_block(by[("R0", chg)], "chg3_R0"), "A2": model_block(by[("A2", chg)], "chg3_A2"),
                                "R0_minus_A2": paired(by[("R0", chg)], by[("A2", chg)], "3 charge pulses R0 - A2"),
                                "R0_minus_R0_all6": paired(by[("R0", chg)], by[("R0", full)], "3 charge R0 - R0 all six")},
        "three_discharge_pulses": {"R0": model_block(by[("R0", dchg)], "dchg3_R0"), "A2": model_block(by[("A2", dchg)], "dchg3_A2"),
                                   "R0_minus_A2": paired(by[("R0", dchg)], by[("A2", dchg)], "3 discharge pulses R0 - A2")},
        "note": "models trained on all six pulses are unchanged; only the per-RPT averaging uses a subset of the pulses "
                "(saved test predictions of the 30 original runs). Every subset has >= 1 pulse in every cell-RPT group.",
    }
    return summary


def exp_ig():
    d = json.loads((RES / "posthoc" / "integrated_gradients.json").read_text(encoding="utf-8"))
    chg, dchg = np.array(d["chg"]), np.array(d["dchg"])
    seg = lambda i: "pre-pulse (0)" if i == 0 else "C/5 (1-30)" if i <= 30 else "1C (31-40)" if i <= 40 else "rest (41-100)"
    with (DEL / "plot_3.6_integrated_gradients.csv").open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.writer(h)
        w.writerow(["sample_index", "segment", "mean_abs_IG_charge_pp", "mean_abs_IG_discharge_pp"])
        for i in range(101):
            w.writerow([i, seg(i), chg[i], dchg[i]])
    per_run = {p: np.array(d["per_run"][p]) for p in ("chg", "dchg")}
    out = {"steps": d["steps"], "baseline": "all-zero standardised waveform",
           "unit": "SOH pp (attribution x target_std x 100)", "runs": len(per_run["chg"]),
           "completeness_median_relative_error": d["completeness_median_relative_error"]}
    for p, curve in (("charge", chg), ("discharge", dchg)):
        key = "chg" if p == "charge" else "dchg"
        out[p] = {}
        for name, sl in (("C/5 (1-30)", slice(1, 31)), ("1C (31-40)", slice(31, 41)), ("rest (41-100)", slice(41, 101))):
            run_vals = per_run[key][:, sl].mean(axis=1)
            out[p][name] = {"mean_abs_IG_per_point": float(curve[sl].mean()),
                            "range_over_15_runs": [float(run_vals.min()), float(run_vals.max())],
                            "share_of_total_abs_IG": float(curve[sl].sum() / curve[1:].sum())}
    return out


def exp_cost():
    m = json.loads((RES / "posthoc" / "compute_cost_models.json").read_text(encoding="utf-8"))
    f = json.loads((RES / "posthoc" / "compute_cost_features.json").read_text(encoding="utf-8"))
    return {"models": m, "feature_extraction_143": f,
            "note": "FLOPs = torch.utils.flop_counter (matmul/conv/attention, 2 FLOPs per multiply-add; element-wise "
                    "ops not counted); latency = CPU, torch.set_num_threads(1), batch 1, 50 warm-up + 1000 timed "
                    "forward passes, median; fold-0 seed-0 models.  Feature time = the frozen extractor's descriptor "
                    "block (all 150 columns, 143 voltage-derived) on one thread, system Python."}


# ----------------------------------------------------------------------------- NMC / LFP encoder matrix
def decompose(table: np.ndarray) -> dict:
    g = table.mean()
    ss_tot = float(((table - g) ** 2).sum())
    ss_p = float(table.shape[1] * ((table.mean(axis=1) - g) ** 2).sum())
    ss_w = float(table.shape[0] * ((table.mean(axis=0) - g) ** 2).sum())
    return {"primary": 100 * ss_p / ss_tot, "waveform": 100 * ss_w / ss_tot,
            "pairing": 100 * (ss_tot - ss_p - ss_w) / ss_tot, "SS_total": ss_tot}


def matrix(dataset: str, roots: dict[str, Path]) -> dict:
    per_setting = {s: runs(roots[s], s, 15) for s in SETTINGS}
    for s, r in per_setting.items():
        record("4_encoder_matrix", dataset, s, r)
    means = {s: seed_mean(r) for s, r in per_setting.items()}
    keys = sorted(means[R0])
    if any(sorted(m) != keys for m in means.values()):
        raise RuntimeError("cells differ between settings")
    H, W = HANDCRAFTED_ENCODERS, WAVEFORM_ENCODERS

    M = np.array([[[means[f"{w}__{h}"][k] for k in keys] for w in W] for h in H])  # (3, 5, cells)

    def table_of(sel):
        return M[:, :, sel].mean(axis=2)

    tab = table_of(np.arange(len(keys)))
    out = {"rows_primary": [LABEL[h] for h in H], "cols_waveform": [LABEL[w] for w in W],
           "cell_macro_RMSE": tab.tolist()}
    # bootstrap of the decomposition: cells resampled within outer folds (as for LFP Fig. 4d)
    rng = np.random.default_rng(_stable_seed(dataset + "_matrix_decomposition"))
    by_fold = defaultdict(list)
    for i, k in enumerate(keys):
        by_fold[k[0]].append(i)
    boots = {"all": [], "noFT": []}
    for _ in range(N):
        sel = np.concatenate([rng.choice(by_fold[f], len(by_fold[f]), replace=True) for f in sorted(by_fold)])
        t = table_of(sel)
        boots["all"].append(decompose(t))
        boots["noFT"].append(decompose(t[:2]))
    for name, t in (("all_15", tab), ("without_FT_Transformer", tab[:2])):
        d = decompose(t)
        b = boots["all" if name == "all_15" else "noFT"]
        out[f"SS_shares_{name}"] = {k: {"percent": d[k], "ci95": [float(np.percentile([x[k] for x in b], 2.5)),
                                                                  float(np.percentile([x[k] for x in b], 97.5))]}
                                    for k in ("primary", "waveform", "pairing")}
        out[f"SS_shares_{name}"]["P(pairing > waveform)"] = float(np.mean([x["pairing"] > x["waveform"] for x in b]))
    # two-way ANOVA with the 3 seeds as replicates (seed-level cell-macro RMSE)
    from scipy import stats

    seed_tab = np.array([[[np.mean(list(seed_mean(per_setting[f"{w}__{h}"], (s,)).values())) for s in (0, 1, 2)]
                          for w in W] for h in H])  # (3,5,3)
    a, b, n = seed_tab.shape
    gm = seed_tab.mean()
    cm = seed_tab.mean(axis=2)
    ss_a = b * n * ((cm.mean(axis=1) - gm) ** 2).sum()
    ss_b = a * n * ((cm.mean(axis=0) - gm) ** 2).sum()
    ss_ab = n * ((cm - cm.mean(axis=1, keepdims=True) - cm.mean(axis=0, keepdims=True) + gm) ** 2).sum()
    ss_e = ((seed_tab - cm[..., None]) ** 2).sum()
    df_ab, df_e = (a - 1) * (b - 1), a * b * (n - 1)
    F = (ss_ab / df_ab) / (ss_e / df_e)
    out["interaction_test_seeds_as_replicates"] = {"F": float(F), "df": [df_ab, df_e],
                                                   "p": float(stats.f.sf(F, df_ab, df_e))}
    ranks = {}
    for i, h in enumerate(H):
        order = np.argsort(tab[i])
        ranks[LABEL[h]] = {LABEL[W[j]]: int(np.where(order == j)[0][0] + 1) for j in range(len(W))}
    out["waveform_rank_within_primary"] = ranks
    flat = sorted(((tab[i, j], f"{LABEL[H[i]]} + {LABEL[W[j]]}") for i in range(3) for j in range(5)))
    out["ranking_all_15"] = [{"rank": r + 1, "combination": c, "RMSE": float(v)} for r, (v, c) in enumerate(flat)]
    out["best"] = out["ranking_all_15"][0]
    out["MLP+PatchTST_rank"] = next(x for x in out["ranking_all_15"] if x["combination"] == "MLP + PatchTST")
    inv = {f"{LABEL[h]} + {LABEL[w]}": f"{w}__{h}" for h in H for w in W}
    best_setting = inv[out["best"]["combination"]]
    if best_setting != R0:
        out["best_minus_MLP+PatchTST"] = paired(per_setting[best_setting], per_setting[R0],
                                                f"{dataset} {out['best']['combination']} - MLP + PatchTST")
    out["seeds"], out["folds"], out["n_cells"] = 3, 5, len(keys)
    return out


def main() -> int:
    DEL.mkdir(parents=True, exist_ok=True)
    r0_5, a2_5 = original_lfp()
    summary = {
        "metric": "cell-macro RMSE (pp): 6 pulse conditions averaged per cell-RPT -> per-cell RMSE -> mean over 3 "
                  "seeds per cell -> mean over cells",
        "statistics": "fold-stratified cell bootstrap (5,000) 95% CI; two-sided sign-flip test (5,000); "
                      "improved cells = cells with negative difference; d_z = mean/sd of per-cell differences",
        "LFP_reference_5fold": {"R0": model_block(r0_5, "R0"), "A2": model_block(a2_5, "A2"),
                                "R0_minus_A2": paired(r0_5, a2_5, "R0 - A2")},
        "2.1_NMC": exp_nmc(),
        "3.1_LOGO": exp_logo(r0_5, a2_5),
        "3.2_cycle_count_error": exp_cycle(r0_5, a2_5),
        "3.3_matched_noise": exp_noise(r0_5, a2_5),
        "3.4_test_only_noise_0p5mV": exp_tnoise(r0_5, a2_5),
        "3.5_pulse_count": exp_pulses(),
        "3.6_integrated_gradients": exp_ig(),
        "3.7_compute_cost": exp_cost(),
    }
    nmc_roots = {s: (PROJECT / "outputs/nmc_original_code_20261007/results/encoder_benchmark" if s == R0
                     else RES / "nmc_matrix" / "matrix") for s in SETTINGS}
    summary["4_encoder_matrix_NMC"] = matrix("NMC", nmc_roots)
    summary["4_encoder_matrix_LFP_recomputed"] = matrix("LFP", {s: ORIG / "encoder_benchmark" for s in SETTINGS})
    (DEL / "results_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    with (DEL / "per_cell_rmse.csv").open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["experiment", "dataset", "model", "cell", "seed", "RMSE", "fold"])
        w.writeheader()
        w.writerows(PER_CELL)
    with (DEL / "plot_4_nmc_matrix.csv").open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.writer(h)
        m = summary["4_encoder_matrix_NMC"]
        w.writerow(["primary"] + m["cols_waveform"])
        for name, row in zip(m["rows_primary"], m["cell_macro_RMSE"]):
            w.writerow([name] + row)
    print(f"wrote results_summary.json and {len(PER_CELL)} per-cell rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
