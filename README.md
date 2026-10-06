# mian — LFP short-pulse SOH paper (English, journal format)

英文稿与补充材料，单栏、按 Elsevier 能源类顶刊（Applied Energy / J. Power Sources / eTransportation）格式排版。

## 文件 / Files

| 文件 | 内容 |
|---|---|
| `manuscript/LFP_SOH_PulseFusion_Manuscript.docx` | 英文正文（可编辑 Word，原生公式、三线表、编号参考文献） |
| `manuscript/LFP_SOH_PulseFusion_Supplementary.docx` | 补充材料（附录）：特征公式、PatchTST 公式、超参数网格、检验族、复现说明、算法 S1、表 S1–S7、图 S1–S5 |
| `manuscript/*_preview.pdf` | 两个文档的 PDF 预览（LibreOffice 渲染，字体以 Liberation Serif 代替 Times New Roman） |
| `manuscript/figures/` | 全部图片（600 dpi PNG） |
| `manuscript/format_review.md` | 参考的 14 篇相关顶刊论文及采用的格式规范 |
| `manuscript/source/` | 生成 Word 的源文件（Markdown + `build.py`），改文字后可一键重建 |
| `experiments/` | 补充实验的复现代码与结果汇总（见 `experiments/README.md`） |
| `manuscript/zh/LFP_SOH_PulseFusion_Manuscript_ZH.docx`（及 .pdf） | **中文对照版正文**：修订后英文稿的逐句中译，章节、公式、表格和编号与英文版一一对应（图片仍为英文） |
| `manuscript/zh/LFP_SOH_PulseFusion_Supplementary_ZH.docx`（及 .pdf） | 中文对照版补充材料 |
| `manuscript/zh/论文修改说明.docx`（及 .pdf、.md） | **修改说明**：原中文稿 → 修订稿的逐节对照、图表编号对照、新增实验结果、删减内容、需要您确认的事项 |
| `manuscript/zh/source/` | 中文版 Markdown 源文件与术语对照表 |

重建 Word：`cd manuscript/source && python build.py manuscript.md out.docx && python build.py supplementary.md out_supp.docx --supp`（需要 pandoc 与 python-docx；图片路径为 `../figures`）。中文版：`cd manuscript/zh/source && python build.py manuscript_zh.md out.docx --lang zh`（补充材料加 `--supp`）。

## 主要改动 / Change log

1. **翻译与压缩**：中文稿约 2.1 万汉字（不含参考文献） → 英文正文约 7,100 词（不含摘要、图注、表格与参考文献；连同摘要、图注、表格与术语表合计约 9,600 词，低于 1 万）。删去重复的限定语，边际效应压缩为一句，负面结果集中在 4.7 节最后一段（两句）。
2. **结构**：Highlights（5 条，均 ≤85 字符）、摘要（248 词）、关键词、Nomenclature、1–5 章、CRediT、利益声明、数据可用性、致谢、参考文献（50 篇，Elsevier 编号格式）。
3. **移入补充材料**：附录 A 特征公式、PatchTST 标准公式、算法 1、超参数网格、检验族列表、老化工况表、输入信息表、消融配置表、分层结果表、特征数量表，以及原图 9–12。
4. **图**：新绘图 1（数据集与脉冲响应，按 SOH 着色，取代原图 1、图 2 的 3D 图）、图 3（编码器矩阵，按表 7 数值重绘，删去边际效应子图）、图 8（新实验）、图 S2（K 敏感性）；图 2 使用您提供的合并版框架图；原图 5–8 保留为图 4–7。
5. **补充实验**（基于公开数据的独立 CPU 复现，详见补充说明 S6 与表 S7）：
   - 复现检验：GBDT 1.028（原 1.022）、A2 1.474（原 1.500）、R0 1.427（原 1.377）；波形增量方向与显著性一致（−0.046 pp，p < 0.001），最好/最差电芯同为 12 号与 3 号。
   - 未见循环工况（留一组）：波形增量仍显著（−0.035 pp，p = 0.008）；R0 误差仅升 9%，GBDT 升 36%。
   - 循环计数 ±10% 误差：R0 误差仅升 1.5–3.5%，GBDT 升 8.6–14.4%。
   - 匹配噪声（1、2 mV，训练与测试同噪声）：波形增量保持（−0.058、−0.045 pp）。
   - 脉冲数量：仅用三个充电脉冲即达 1.432 pp（六脉冲 1.427 pp），测试时间减半。
   - 归因（积分梯度）：贡献均匀分布于整个瞬态，支持“区段间联合形状信息”的论点。
   - 计算开销：19,041 参数（74 kB）、0.42 MFLOPs/脉冲、单核 0.52 ms。

## 需要您补充或确认 / To do

完整清单（8 项）见 `manuscript/zh/论文修改说明.docx` 第六部分，要点如下：

- 作者、单位、通信作者邮箱、基金、CRediT、代码与数据仓库链接（文中以方括号标出）。
- 图 2：上传版本为 2000 px 宽，投稿前请替换为原始高分辨率图。
- 补充表 S5 中“Reported”一列：中文稿未给出的数值（Ridge、随机森林、GPR，以及不含 τ 的 Ridge/随机森林）标为 n.r.，请用您原始结果文件中的数值替换。“Re-implemented”一列为复现值（Ridge、RBF-SVR、仅 τ 的对照、不含 τ 的 Ridge 与 GBDT；有原值可比的均与原值接近：仅 τ ≈2.9–3.1，不含 τ 的 GBDT 1.698 vs 原 1.663，SVR 1.007 vs 原 1.026）；全输入随机森林、GPR 和不含 τ 的随机森林因内存与耗时未复现（标为 —）。
- NMC/石墨数据上未观察到波形增量（R0 2.030 vs A2 2.026 pp，GBDT 0.935 pp）。正文只在局限段用一句话交代，详细见补充说明 S6；是否保留请您决定。
- 只在测试时加噪声（训练用干净数据）时，所有基于特征的模型误差都会大幅上升，已在补充说明 S6 中说明，正文局限段一句话带过。
- 新增实验来自独立 CPU 复现：复现的波形增量（−0.046 pp，42/64）小于原文（−0.123 pp，58/64），与原代码的差异已在补充说明 S6 披露。可用原代码在原划分下重跑，或保留现在的写法。
- 请确认新题名、关键词，以及“学习器依赖”和“单调约束为正则项”这两处表述。
