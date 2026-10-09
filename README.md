# 磷酸铁锂电池短时脉冲 SOH 估计论文

**题目**：Fusing short-pulse voltage waveforms with handcrafted features for rapid state-of-health estimation of lithium iron phosphate batteries: a controlled comparison of encoder pairings

**当前版本**：
- 英文稿按投稿模板排版：A4、双倍行距、行号；摘要分为 Background / Method / Significant findings。
- 中文版与英文稿内容一致。
- 全文只与神经网络比较，树模型等传统回归器的对照已删除。
- 引言采用以老师版本为底的融合稿；全文统一用“主分支 / 补充分支”（primary / supplementary branch）。

## 目录

| 文件夹 | 内容 |
|---|---|
| **01_投稿稿件_英文/** | **当前投稿版**：`正文_Manuscript.docx`、`补充材料_Supplementary.docx`（各附 PDF 预览） |
| **02_中文版/** | `中文版正文`、`中文版补充材料`（Word + PDF）；`引言定稿_中英对照.docx`；`引言融合稿_老师版本+修订稿（含分析）.docx` |
| **03_修改说明/** | `论文修改说明`：与原中文稿逐节对照，第七、八部分为最近两轮更新；`参考期刊格式调研.md` |
| **04_图片/** | 正文图 1–7 和补充图 S1–S5（600 dpi 原图，投稿时单独上传用这些） |
| **05_源文件/** | 生成 Word 的 Markdown 源文件和脚本。`英文/`、`中文/` 各有一个 `build.py`；`压缩Word图片.py` 用来解决 Word 中图片显示不出来的问题 |
| **06_实验代码/** | `复现代码/`：CPU 复现程序；`复现结果汇总/`：复现结果；`NMC诊断分析/`：复现代码在 NMC 上没有波形增量的原因分析（含中文说明） |
| **07_本地运行/** | `本地Claude任务说明.md`：交给本地 Claude 用原代码补跑的实验清单；`结果回传/`：本地跑完的结果放这里 |
| **08_旧版存档/** | 早期的 Elsevier 格式英文稿（仍含 GBDT 对照和旧引言，仅供参考），以及已删除的旧图 |

## 重新生成 Word

需要 pandoc 和 python-docx。

```bash
# 英文（投稿模板格式）
cd 05_源文件/英文
python build.py 正文_模板格式.md 正文.docx --layout T
python build.py 补充材料.md 补充材料.docx --supp --layout T

# 中文
cd ../中文
python build.py 中文版正文.md 中文版正文.docx --lang zh
python build.py 中文版补充材料.md 中文版补充材料.docx --supp --lang zh

# 可选：压缩 Word 内的图片，避免较老版本的 Word 显示不出来
python ../压缩Word图片.py 中文版正文.docx
```

图片从 `04_图片/` 读取。

## 待办

1. **本地补跑实验**（见 `07_本地运行/本地Claude任务说明.md`）：
   - 4.6 节全部实验改用原代码重跑；
   - NMC 上的 15 种编码器组合矩阵；
   - NMC 结果 R0 − A2 的 95% 置信区间；
   - 确认特征是在绝对电压还是首点作差电压上计算；
   - 重画补充图 S3，去掉 GBDT 曲线。
2. **作者信息**：作者、单位、通讯地址、电话、邮箱、基金信息、CRediT（文中方括号处）。
3. **图 2 高分辨率原图**：现为 2000 px 宽，投稿前替换。
