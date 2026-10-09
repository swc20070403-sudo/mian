"""Sec. 4.6 analyses that need NO retraining: the paper's own clean R0 / A2 runs are reloaded from
results/clean_rerun_20260927 (model.pt + the train-fold preprocessor, refitted and checked against the saved
preprocessor.json), their saved test predictions are reproduced first, then:

  cycle  3.2  test-time cumulative cycle count tau x 1.1 and x 0.9 (models not retrained)
  tnoise 3.4  test voltages + N(0, 0.5 mV) (features re-extracted from the noisy trace), clean-trained models
  pulses 3.5  per RPT, average the predictions of only k of the 6 pulses (all C(6,k) subsets)
  ig     3.6  integrated gradients of R0 w.r.t. the 101-point standardised waveform input
  cost   3.7  parameters, FLOPs per pulse, single-thread CPU latency (batch 1, 1000 reps, median)

Writes raw per-cell values to results/posthoc/*.json; statistics are done in rerun46_analyze.py.
Training venv.   python scripts/rerun46_posthoc.py {verify,cycle,tnoise,pulses,ig,cost}
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parents[1]
PROJECT = WORK.parents[1]
os.environ["PULSEFUSION_RESULTS_ROOT"] = str(WORK / "results")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rerun46_train as tr  # noqa: E402
from experiments import run_clean_rerun as clean  # noqa: E402

ORIG = PROJECT / "results" / "clean_rerun_20260927"
R0_ROOT = ORIG / "encoder_benchmark"
A2_ROOT = ORIG / "supplemental_ablation"
OUT = WORK / "results" / "posthoc"
CONDITIONS = [(20, "chg"), (50, "chg"), (90, "chg"), (20, "dchg"), (50, "dchg"), (90, "dchg")]


def run_dir(model: str, fold: int, seed: int) -> Path:
    if model == "R0":
        return R0_ROOT / tr.R0 / f"fold_{fold}" / f"init_{seed}"
    return A2_ROOT / "runs" / tr.A2 / f"fold_{fold}" / f"init_{seed}"


def fold_prep_path(model: str, fold: int) -> Path:
    return (R0_ROOT / f"fold_{fold}" if model == "R0" else A2_ROOT / "folds" / f"fold_{fold}") / "preprocessor.json"


_CACHE = {}


def setup():
    if "table" not in _CACHE:
        from dataloader.fusion_data import fit_fusion_preprocessor
        from run_encoder_benchmark_formal import _preprocessor_payload, make_cell_folds

        table = tr.load_table(tr.LFP_PULSE, tr.LFP_FEATURES, tr.LFP_CLEAN_IDS)
        folds = make_cell_folds(table.cell_id, 5, 42)
        pres = {}
        for f in folds:
            pre = fit_fusion_preprocessor(table, np.where(np.isin(table.cell_id, f["train_cells"]))[0], 50,
                                          voltage_mode="delta")
            payload = json.loads(json.dumps(_preprocessor_payload(pre)))
            for model in ("R0", "A2"):
                saved = json.loads(fold_prep_path(model, f["fold"]).read_text(encoding="utf-8"))
                if saved != payload:
                    raise RuntimeError(f"refitted preprocessor differs from saved ({model}, fold {f['fold']})")
            pres[f["fold"]] = pre
        _CACHE.update(table=table, folds=folds, pres=pres)
    return _CACHE["table"], _CACHE["folds"], _CACHE["pres"]


def load_model(model: str, fold: int, seed: int, device):
    import torch

    ckpt = torch.load(run_dir(model, fold, seed) / "model.pt", map_location="cpu", weights_only=False)
    if model == "R0":
        from Model.encoder_benchmark import EncoderBenchmarkModel

        net = EncoderBenchmarkModel("patchtst", "current_mlp", handcrafted_dim=50, branch_embedding_dim=32, dropout=0.2)
    else:
        import Model.supplemental_ablation as sm

        cfg = ckpt["model_config"]
        net = sm.SupplementalSOHModel(cfg["mode"], use_time=cfg["use_time"], patch_len=cfg["patch_len"],
                                      patch_stride=cfg["patch_stride"], handcrafted_dim=50, context_dim=2,
                                      branch_embedding_dim=32, dropout=0.2)
    net.load_state_dict(ckpt["state_dict"])
    return net.to(device).eval()


def arrays_for(table, idx, pre, tau_factor: float = 1.0, voltage_table=None):
    """make_fusion_arrays; optionally scale the RAW cumulative cycle count, or take voltage/features from
    another (noisy) table with identical row order."""
    from dataloader.fusion_data import _cumulative_cycle_time, make_fusion_arrays

    src = table if voltage_table is None else voltage_table
    arrays = list(make_fusion_arrays(src, idx, pre, include_time=True))
    if tau_factor != 1.0:
        tau = _cumulative_cycle_time(table)[idx] * tau_factor
        arrays[3] = ((tau - pre.time_mean) / pre.time_std).astype(np.float32).reshape(-1, 1)
    return arrays


def predict(net, arrays, pre, device) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for s in range(0, len(arrays[0]), 4096):
            w, h, t = (torch.from_numpy(np.asarray(a[s:s + 4096])).float().to(device) for a in (arrays[0], arrays[1], arrays[3]))
            out.append(net(w, h, t).cpu().numpy().reshape(-1))
    return (np.concatenate(out) * pre.target_std + pre.target_mean) * 100.0


def cell_rmse(cell, rpt, true, pred) -> dict[int, float]:
    from run_encoder_benchmark_formal import cell_macro_metrics

    return {r["cell_id"]: r["RMSE"] for r in cell_macro_metrics(cell, rpt, true, pred)["cells"]}


def each_run(fn):
    """Call fn(model, fold, seed, net, idx, pre) for the 30 original runs; collect per-cell RMSE dicts."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, folds, pres = setup()
    results = {}
    for model in ("R0", "A2"):
        for f in folds:
            idx = np.where(np.isin(table.cell_id, f["test_cells"]))[0]
            for seed in (0, 1, 2):
                net = load_model(model, f["fold"], seed, device)
                results[(model, f["fold"], seed)] = fn(model, f["fold"], seed, net, idx, pres[f["fold"]], device)
    return results


def dump(name: str, results: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"model": m, "fold": f, "seed": s, "cells": {str(c): v for c, v in cells.items()}}
            for (m, f, s), cells in sorted(results.items())]
    (OUT / f"{name}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------- stages
def stage_verify() -> None:
    table, _, _ = setup()

    def fn(model, fold, seed, net, idx, pre, device):
        pred = predict(net, arrays_for(table, idx, pre), pre, device)
        with np.load(run_dir(model, fold, seed) / "test_predictions.npz") as d:
            if not np.array_equal(d["row_id"], table.row_id[idx]):
                raise RuntimeError("row order differs")
            diff = float(np.max(np.abs(d["pred_soh_percent"] - pred)))
        saved = {r["cell_id"]: r["RMSE"] for r in json.loads((run_dir(model, fold, seed) / "metrics.json")
                                                             .read_text(encoding="utf-8"))["cell_metrics"]}
        mine = cell_rmse(table.cell_id[idx], table.rpt[idx], table.soh_percent[idx], pred)
        cdiff = max(abs(saved[c] - mine[c]) for c in saved)
        print(f"{model} fold={fold} seed={seed}: max |pred - saved| = {diff:.2e} pp; max cell-RMSE diff {cdiff:.2e}")
        if diff > 1e-3:
            raise RuntimeError("cannot reproduce saved predictions")
        return mine

    dump("reproduced_original", each_run(fn))


def stage_cycle() -> None:
    table, _, _ = setup()
    for factor in (1.1, 0.9):
        def fn(model, fold, seed, net, idx, pre, device, factor=factor):
            pred = predict(net, arrays_for(table, idx, pre, tau_factor=factor), pre, device)
            return cell_rmse(table.cell_id[idx], table.rpt[idx], table.soh_percent[idx], pred)

        dump(f"cycle_x{factor:.1f}", each_run(fn))
        print(f"cycle x{factor} done", flush=True)


def stage_tnoise() -> None:
    table, _, _ = setup()
    noisy = tr.load_table(tr.WORK / "data" / "LFP_slowpulse_noise_0p5mV.pkl",
                          tr.WORK / "features_noise_0p5mV" / "LFP_150_candidate_features.csv", tr.LFP_CLEAN_IDS)
    for key in ("row_id", "cell_id", "rpt", "soc", "pulse_type", "soh_percent"):
        if not np.array_equal(getattr(table, key), getattr(noisy, key)):
            raise RuntimeError(f"noisy table differs in {key}")
    if noisy.feature_names != table.feature_names:
        raise RuntimeError("feature columns differ")

    def fn(model, fold, seed, net, idx, pre, device):
        pred = predict(net, arrays_for(table, idx, pre, voltage_table=noisy), pre, device)
        return cell_rmse(table.cell_id[idx], table.rpt[idx], table.soh_percent[idx], pred)

    dump("test_noise_0p5mV", each_run(fn))


def stage_pulses() -> None:
    """Uses the saved test predictions (identical to the reloaded models, see verify)."""
    table, folds, _ = setup()
    out = {}
    skipped = {}
    combos = [c for k in range(1, 7) for c in itertools.combinations(range(6), k)]
    for model in ("R0", "A2"):
        for f in folds:
            for seed in (0, 1, 2):
                with np.load(run_dir(model, f["fold"], seed) / "test_predictions.npz") as d:
                    cell, rpt, soc = d["cell_id"].astype(int), d["rpt"].astype(int), d["soc"].astype(int)
                    pulse, true, pred = d["pulse_type"].astype(str), d["true_soh_percent"], d["pred_soh_percent"]
                cond = np.full(len(cell), -1)
                for j, (s, p) in enumerate(CONDITIONS):
                    cond[(soc == s) & (pulse == p)] = j
                for combo in combos:
                    m = np.isin(cond, combo)
                    per_cell = {}
                    for c in np.unique(cell):
                        all_rpts = np.unique(rpt[cell == c])
                        mc = m & (cell == c)
                        rpts = np.unique(rpt[mc])
                        skipped[(model, combo)] = skipped.get((model, combo), 0) + len(all_rpts) - len(rpts)
                        err = [pred[mc & (rpt == r)].mean() - true[mc & (rpt == r)].mean() for r in rpts]
                        per_cell[int(c)] = float(np.sqrt(np.mean(np.square(err))))
                    out[(model, f["fold"], seed, combo)] = per_cell
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"model": m, "fold": f, "seed": s, "combo": [f"{CONDITIONS[j][0]}{CONDITIONS[j][1]}" for j in combo],
             "cells": {str(c): v for c, v in cells.items()}} for (m, f, s, combo), cells in out.items()]
    (OUT / "pulse_subsets.json").write_text(json.dumps(rows), encoding="utf-8")
    sk = [{"model": m, "combo": [f"{CONDITIONS[j][0]}{CONDITIONS[j][1]}" for j in combo], "cell_rpt_groups_without_any_pulse_of_subset_summed_over_seeds": n}
          for (m, combo), n in skipped.items() if n]
    (OUT / "pulse_subsets_skipped_rpts.json").write_text(json.dumps(sk, indent=1), encoding="utf-8")
    print(f"pulse subsets done: {len(combos)} combos; subsets with missing RPTs: {len(sk)}")


def stage_ig(steps: int = 32) -> None:
    import torch

    table, _, _ = setup()
    curves = {"chg": [], "dchg": []}
    completeness = []

    def fn(model, fold, seed, net, idx, pre, device):
        if model != "R0":
            return {}
        arrays = arrays_for(table, idx, pre)
        wave = torch.from_numpy(arrays[0]).float().to(device)
        hand = torch.from_numpy(arrays[1]).float().to(device)
        tau = torch.from_numpy(arrays[3]).float().to(device)
        total = torch.zeros_like(wave)
        alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps  # midpoint Riemann sum
        for a in alphas:
            x = (a * wave).requires_grad_(True)  # baseline = all-zero standardised waveform
            y = net(x, hand, tau).sum()
            total += torch.autograd.grad(y, x)[0]
        ig = (wave * total / steps).squeeze(1).detach().cpu().numpy() * pre.target_std * 100.0  # pp of SOH
        with torch.no_grad():
            gap = (net(wave, hand, tau) - net(torch.zeros_like(wave), hand, tau)).cpu().numpy().reshape(-1) * pre.target_std * 100.0
        completeness.append(float(np.median(np.abs(ig.sum(axis=1) - gap)) / max(float(np.median(np.abs(gap))), 1e-12)))
        pulse = table.pulse_type[idx].astype(str)
        for p in ("chg", "dchg"):
            curves[p].append(np.abs(ig[pulse == p]).mean(axis=0))
        return {}

    each_run(fn)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {p: np.mean(v, axis=0).tolist() for p, v in curves.items()}
    payload["per_run"] = {p: [c.tolist() for c in v] for p, v in curves.items()}
    payload["steps"] = steps
    payload["completeness_median_relative_error"] = float(np.median(completeness))
    (OUT / "integrated_gradients.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"IG done over {len(curves['chg'])} R0 runs; completeness median rel. error {np.median(completeness):.3e}")


def stage_cost(reps: int = 1000) -> None:
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    torch.set_num_threads(1)
    table, folds, pres = setup()
    f = folds[0]
    pre = pres[0]
    idx = np.where(np.isin(table.cell_id, f["test_cells"]))[0]
    arrays = arrays_for(table, idx, pre)
    x = [torch.from_numpy(arrays[i][:1]).float() for i in (0, 1, 3)]
    rows = []
    for model in ("R0", "A2"):
        net = load_model(model, 0, 0, torch.device("cpu"))
        params = sum(p.numel() for p in net.parameters())
        # FLOPs with autograd enabled so nn.TransformerEncoderLayer takes its ordinary (countable) path;
        # FlopCounterMode counts matmul/conv/attention FLOPs (2 per MAC), not element-wise ops.
        with FlopCounterMode(display=False) as counter:
            net(*x)
        flops = int(counter.get_total_flops())
        with torch.inference_mode():
            for _ in range(50):
                net(*x)
            times = []
            for _ in range(reps):
                t0 = time.perf_counter_ns()
                net(*x)
                times.append(time.perf_counter_ns() - t0)
        rows.append({"model": model, "parameter_count": int(params), "flops_per_pulse": flops,
                     "latency_ms_median": float(np.median(times) / 1e6),
                     "latency_ms_p05": float(np.percentile(times, 5) / 1e6),
                     "latency_ms_p95": float(np.percentile(times, 95) / 1e6), "reps": reps,
                     "threads": torch.get_num_threads(), "torch": torch.__version__})
        print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "compute_cost_models.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("verify", "cycle", "tnoise", "pulses", "ig", "cost"))
    stage = p.parse_args(argv).stage
    {"verify": stage_verify, "cycle": stage_cycle, "tnoise": stage_tnoise, "pulses": stage_pulses,
     "ig": stage_ig, "cost": stage_cost}[stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
