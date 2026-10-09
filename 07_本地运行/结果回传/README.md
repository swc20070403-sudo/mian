# 本地原代码补跑结果（2026-10-09）

按 `07_本地运行/本地Claude任务说明.md` 用论文原 GPU 代码补跑的结果。**先看 `results_summary.md`**：开头是主要结果一览，后面依次是各项实验的表、补充图 S3 的说明、第二节第 2 项（特征计算方式）的回答，以及运行记录与全部偏差说明。

| 文件 | 内容 |
|---|---|
| `results_summary.md` / `results_summary.json` | 全部结果（同一组数值，json 可机读） |
| `per_cell_rmse.csv` | 实验、数据集、模型、电芯、种子、RMSE（另加 fold 列） |
| `plot_3.1_logo_group_rmse.csv` | 3.1 每组组均 RMSE |
| `plot_3.5_pulse_count.csv`、`plot_3.5_pulse_subsets_all_combinations.csv` | 3.5 脉冲数曲线与全部 63 种组合 |
| `plot_3.6_integrated_gradients.csv` | 3.6 充电/放电的逐点平均 abs(IG) 曲线 |
| `plot_4_nmc_matrix.csv` | 四、NMC 3 × 5 矩阵 |
| `figS3_bias_vs_soh_R0_only.png`、`figS3_bias_vs_soh_R0_A2.png`（另有 pdf/svg/tiff） | 补充图 S3 重画（600 dpi，无 GBDT） |
| `feature_offset_sensitivity.csv`、`top50_level_classification_by_fold.csv` | 第二节第 2 项的依据 |
| `scripts/`、`logs/` | 本次运行的驱动脚本和日志（本地工程 `outputs/rerun_46_20261009/`） |

本次共训练 336 次（LOGO 66、匹配噪声 60、NMC 矩阵 210），另外复用了论文的 30 个原模型做 3.2、3.4–3.7。
