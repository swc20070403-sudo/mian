"""Render deliverables/results_summary.md from deliverables/results_summary.json (+ the run log section)."""
import json
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
S = json.loads((WORK / "deliverables" / "results_summary.json").read_text(encoding="utf-8"))
L = []


def ci(x):
    return f"[{x[0]:.3f}, {x[1]:.3f}]"


def pf(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def model_row(name, m, extra=""):
    seeds = ", ".join(f"{v:.3f}" for v in m["per_seed_RMSE"])
    return f"| {name} | {m['RMSE']:.3f} | {ci(m['ci95'])} | {seeds} | {m['n_cells']} | {m['folds']} |{extra}"


def contrast_row(name, c):
    seeds = ", ".join(f"{v:+.3f}" for v in c["per_seed_difference"])
    dz = "—" if c["dz"] is None else f"{c['dz']:+.2f}"
    return (f"| {name} | {c['difference']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | {pf(c['p_signflip'])} | "
            f"{c['improved_cells']}/{c['n_cells']} | {dz} | {seeds} |")


MH = ["| 模型 | 电芯宏平均 RMSE (pp) | 95% CI | 各种子 RMSE (0/1/2) | 电芯数 | 折/组数 |", "|---|---:|---:|---:|---:|---:|"]
CH = ["| 对比 | 差值 (pp) | 95% CI | p（符号翻转） | 改善电芯 | d_z | 各种子差值 (0/1/2) |",
      "|---|---:|---:|---:|---:|---:|---:|"]


def block(title, d, models=("R0", "A2"), contrasts=()):
    L.extend([f"**{title}**", ""] + MH)
    for m in models:
        L.append(model_row(m, d[m]))
    L.extend([""] + CH)
    for label, key in contrasts:
        L.append(contrast_row(label, d[key]))
    L.append("")


ref = S["LFP_reference_5fold"]
L += ["# 4.6 节原代码重跑结果（2026-10-09）", "",
      "全部结果由论文原 GPU 代码（冻结的 `_train_one` 训练循环、冻结的特征提取脚本）得到，没有使用 CPU 复现代码。",
      "", f"- 指标：{S['metric']}。", f"- 统计：{S['statistics']}。每项均为 3 个种子（0/1/2）；折数见表。",
      "- 表中“各种子”一栏给出每个种子单独计算的结果，便于看种子间波动；CI 与 p 按论文口径只对电芯重抽样，不含种子方差。", "",
      "参照（论文主结果，LFP 5 折）：", ""] + MH + [model_row("R0", ref["R0"]), model_row("A2", ref["A2"]), ""] + CH + [
      contrast_row("R0 − A2", ref["R0_minus_A2"]), ""]
hl = WORK / "deliverables" / "report_highlights.md"
if hl.is_file():
    L += [hl.read_text(encoding="utf-8"), ""]

# 2.1
n = S["2.1_NMC"]
L += ["## 二、1 NMC 补全（表 S6 “待补”格）", "",
      f"原代码 NMC 结果（44 个电芯，5 折，3 种子）：R0 RMSE **{n['R0']['RMSE']:.3f}** {ci(n['R0']['ci95'])}，"
      f"R0 MAE **{n['R0']['MAE']:.3f}** pp；A2 RMSE {n['A2']['RMSE']:.3f}，A2 MAE {n['A2']['MAE']:.3f} pp。", ""]
block("NMC", n, contrasts=[("R0 − A2", "R0_minus_A2")])

# 3.1
g = S["3.1_LOGO"]
L += ["## 三、1 留一工况组（LOGO）", "", f"划分：{g['split']}。", ""]
block("LOGO（11 组 × 3 种子 × 2 模型 = 66 次训练）", g,
      contrasts=[("LOGO：R0 − A2", "R0_minus_A2"), ("R0：LOGO − 5 折", "R0_LOGO_minus_R0_5fold"),
                 ("A2：LOGO − 5 折", "A2_LOGO_minus_A2_5fold")])
L += ["每组组均 RMSE（该组测试电芯的电芯 RMSE 平均，种子已平均）：", "",
      "| 留出组 | 电芯数 | R0 | A2 | R0 − A2 | R0 更优电芯 |", "|---:|---:|---:|---:|---:|---:|"]
for r in g["per_group"]:
    L.append(f"| {r['group']} | {r['n_cells']} | {r['R0_group_mean_RMSE']:.3f} | {r['A2_group_mean_RMSE']:.3f} | "
             f"{r['R0_minus_A2']:+.3f} | {r['R0_improved_cells']}/{r['n_cells']} |")
L.append("")

# 3.2
c = S["3.2_cycle_count_error"]
L += ["## 三、2 累计循环数 ±10% 误差（模型不重训）", "", c["note"], ""]
for k, lab in (("x1.1", "τ × 1.1"), ("x0.9", "τ × 0.9")):
    block(lab, c[k], contrasts=[("R0 − A2", "R0_minus_A2"), ("R0 − R0(无误差)", "R0_minus_R0_clean"),
                                ("A2 − A2(无误差)", "A2_minus_A2_clean")])

# 3.3
z = S["3.3_matched_noise"]
L += ["## 三、3 匹配噪声（训练与测试都加噪，重训）", ""]
for k, lab in (("noise_1p0mV", "σ = 1 mV"), ("noise_2p0mV", "σ = 2 mV")):
    block(lab + "（5 折 × 3 种子 × 2 模型 = 30 次训练）", z[k],
          contrasts=[("R0 − A2", "R0_minus_A2"), ("R0 − R0(无噪声)", "R0_minus_R0_clean"),
                     ("A2 − A2(无噪声)", "A2_minus_A2_clean")])

# 3.4
t = S["3.4_test_only_noise_0p5mV"]
L += ["## 三、4 只在测试加噪声（σ = 0.5 mV，干净数据训练的原模型）", ""]
block("测试噪声 0.5 mV", t, contrasts=[("R0 − A2", "R0_minus_A2"), ("R0 − R0(无噪声)", "R0_minus_R0_clean"),
                                      ("A2 − A2(无噪声)", "A2_minus_A2_clean")])

# 3.5
p = S["3.5_pulse_count"]
L += ["## 三、5 脉冲数", "", p["note"], "",
      "| 每 RPT 脉冲数 k | 组合数 | R0 平均 | R0 范围（最小～最大） | A2 平均 | A2 范围 | R0 − A2 平均 | R0 更优组合 |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|"]
for e in p["curve"]:
    L.append(f"| {e['k']} | {e['n_combinations']} | {e['R0_mean_over_combinations']:.3f} | {e['R0_min']:.3f}～{e['R0_max']:.3f} | "
             f"{e['A2_mean_over_combinations']:.3f} | {e['A2_min']:.3f}～{e['A2_max']:.3f} | "
             f"{e['R0_minus_A2_mean_over_combinations']:+.3f} | {e['combinations_where_R0_better']}/{e['n_combinations']} |")
L += ["", "最好/最差组合：" + "；".join(f"k={e['k']}: R0 最好 {e['R0_min_combo']}，最差 {e['R0_max_combo']}" for e in p["curve"][:5]), ""]
block("只用 3 个充电脉冲（20/50/90 % 充电）", p["three_charge_pulses"],
      contrasts=[("R0 − A2", "R0_minus_A2"), ("R0(3 充电) − R0(6 脉冲)", "R0_minus_R0_all6")])
block("只用 3 个放电脉冲（参考）", p["three_discharge_pulses"], contrasts=[("R0 − A2", "R0_minus_A2")])

# 3.6
ig = S["3.6_integrated_gradients"]
L += ["## 三、6 积分梯度（R0 的 101 点波形输入）", "",
      f"{ig['steps']} 步（中点黎曼和），基线 = {ig['baseline']}，单位 = {ig['unit']}；15 个 R0 模型（5 折 × 3 种子）的全部测试记录；"
      f"手工特征与 τ 固定为实测值。完备性检验（ΣIG 与 f(x) − f(基线) 的相对误差中位数）= {ig['completeness_median_relative_error']:.3f}。"
      "采样点 0 是 ΔV 的首点（恒为 0，IG 恒为 0）。", "",
      "| 脉冲 | 段 | 每点平均 abs(IG) (pp) | 15 个模型的范围 | 占总 abs(IG) |", "|---|---|---:|---:|---:|"]
for pulse, lab in (("charge", "充电"), ("discharge", "放电")):
    for seg, v in ig[pulse].items():
        L.append(f"| {lab} | {seg} | {v['mean_abs_IG_per_point']:.4f} | {v['range_over_15_runs'][0]:.4f}～{v['range_over_15_runs'][1]:.4f} | "
                 f"{100 * v['share_of_total_abs_IG']:.1f}% |")
L.append("")

# 3.7
cc = S["3.7_compute_cost"]
L += ["## 三、7 计算开销", "", cc["note"], "", "| 模型 | 参数量 | FLOPs/脉冲 | CPU 单线程时延中位数 (ms) | 5%～95% (ms) |",
      "|---|---:|---:|---:|---:|"]
for m in cc["models"]:
    L.append(f"| {m['model']} | {m['parameter_count']:,} | {m['flops_per_pulse']:,} | {m['latency_ms_median']:.3f} | "
             f"{m['latency_ms_p05']:.3f}～{m['latency_ms_p95']:.3f} |")
f = cc["feature_extraction_143"]
L += ["", f"143 项波形特征提取（原脚本，单线程）：单个脉冲逐次计算中位数 **{f['single_pulse_ms_median']:.2f} ms**"
      f"（5%～95%：{f['single_pulse_ms_p05']:.2f}～{f['single_pulse_ms_p95']:.2f} ms，{f['single_pulse_calls']} 次）；"
      f"一次处理全部 {f['batch_pulses']:,} 个脉冲 {f['batch_all_pulses_s']:.1f} s，摊到每个脉冲 {f['batch_ms_per_pulse']:.2f} ms。", ""]

# 4
for key, title in (("4_encoder_matrix_NMC", "## 四、NMC 上的 3 × 5 编码器矩阵"),
                   ("4_encoder_matrix_LFP_recomputed", "### 对照：LFP 矩阵（同一脚本重算，与论文一致）")):
    m = S[key]
    L += [title, "", f"电芯宏平均 RMSE (pp)，{m['n_cells']} 个电芯，{m['folds']} 折 × {m['seeds']} 种子。", "",
          "| 主分支 \\ 波形分支 | " + " | ".join(m["cols_waveform"]) + " |", "|---|" + "---:|" * 5]
    for name, row in zip(m["rows_primary"], m["cell_macro_RMSE"]):
        L.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    L += ["", "平方和分解（组合间差异的百分比；95% CI = 按折分层电芯 bootstrap 5,000 次）：", "",
          "| 组合集合 | 主分支 | 波形分支 | 组合依赖 | P(组合依赖 > 波形分支) |", "|---|---:|---:|---:|---:|"]
    for sk, lab in (("SS_shares_all_15", "全部 15 种"), ("SS_shares_without_FT_Transformer", "去掉 FT-Transformer（10 种）")):
        s = m[sk]
        L.append(f"| {lab} | " + " | ".join(f"{s[x]['percent']:.1f}% [{s[x]['ci95'][0]:.1f}, {s[x]['ci95'][1]:.1f}]"
                                           for x in ("primary", "waveform", "pairing")) + f" | {s['P(pairing > waveform)']:.3f} |")
    it = m["interaction_test_seeds_as_replicates"]
    L += ["", f"以种子为重复的双因素方差分析，交互项 F({it['df'][0]}, {it['df'][1]}) = {it['F']:.2f}，p = {pf(it['p'])}。", "",
          "每种主分支下 5 种波形编码器的排名（1 = 最好）：", "",
          "| 主分支 | " + " | ".join(m["cols_waveform"]) + " |", "|---|" + "---:|" * 5]
    for name, ranks in m["waveform_rank_within_primary"].items():
        L.append(f"| {name} | " + " | ".join(str(ranks[w]) for w in m["cols_waveform"]) + " |")
    L += ["", f"最优组合：{m['best']['combination']}（{m['best']['RMSE']:.3f}）；MLP + PatchTST（R0）排第 "
          f"{m['MLP+PatchTST_rank']['rank']}/15（{m['MLP+PatchTST_rank']['RMSE']:.3f}）。", "",
          "全部 15 种排序：" + "；".join(f"{x['rank']}. {x['combination']} {x['RMSE']:.3f}" for x in m["ranking_all_15"]), ""]
    if "best_minus_MLP+PatchTST" in m:
        L += CH + [contrast_row(f"{m['best']['combination']} − MLP + PatchTST", m["best_minus_MLP+PatchTST"]), ""]

extra = WORK / "deliverables" / "report_text_sections.md"
if extra.is_file():
    L += [extra.read_text(encoding="utf-8")]
(WORK / "deliverables" / "results_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote results_summary.md", len(L), "lines")
