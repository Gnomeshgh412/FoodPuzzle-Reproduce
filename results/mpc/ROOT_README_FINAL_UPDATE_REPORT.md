# 根目录 README 最终更新报告

## 1. 本阶段目标

本阶段目标是编写面向导师、课程汇报和后续复现实验对接的根目录 `README.md`，用最终确定的方法、路径和结果说明本仓库的 FoodPuzzle MFP / MPC 近似复现工作。

本阶段只修改文档，不调用 LLM/API，不运行 prediction、evaluation、Agent、reconstruction 或 split，不修改任何 formal result 文件，不修改 metrics。

## 2. README.md 更新内容

根目录 `README.md` 已重写为中文为主的正式说明文档，包含：

- 项目简介；
- MFP / MPC 任务说明；
- 与 FoodPuzzle 原文和官方公开资源的关系；
- 官方已公开内容与未完整公开内容；
- 仓库目录结构；
- MPC 数据重建与 split 说明；
- zero-shot、BM25 ICL、Scientific Agent 方法实现；
- MFP / MPC evaluation 差异；
- 当前正式 MPC 结果；
- 关键复现策略与最终决策；
- 简版重新运行流程；
- 复现限制；
- 重要注意事项。

README 中明确说明本仓库是 paper/code/data-aligned reconstruction，不是 official exact reproduction，也不与论文 Table 2 做严格数值比较。

## 3. 是否包含仓库结构

已包含。README 使用目录树说明：

- `code/`：核心复现代码；
- `data/`：原始数据、processed 数据和 official evidence；
- `results/`：formal results、split 和分析报告；
- `artifacts/`：archived historical results，不作为当前 formal results。

并逐项说明了 `code/`、`data/`、`results/`、`artifacts/` 的用途。

## 4. 是否包含数据说明

已包含。README 明确说明：

- MPC 官方公开 task 缺少 `partial_molecules` 和 `n`；
- 本仓库基于 FlavorDB full molecule set 和 `missing_molecules` 重建 MPC 输入；
- `partial_molecules = full_molecules - missing_molecules`；
- `n = len(missing_molecules)`；
- reconstructed MPC task 总样本数为 710；
- MPC split 使用 seed=42、80/10/10，得到 train/dev/test = 568/71/71；
- 这是 FlavorDB-derived reconstruction，不是 official exact input。

## 5. 是否包含与原文差异

已包含。README 明确列出：

- official train/dev/test split 未公开；
- split code 未公开；
- full Scientific Agent code 未公开；
- zero-shot / ICL / Agent prompt 未完整公开；
- Scientist / Reviewer prompt 未公开；
- parser / normalization 细节未公开；
- 当前使用 DeepSeek，与原文 GPT-3.5 / Gemini / LLaMA3 设置不同；
- 当前结果不能与论文 Table 2 严格数值比较。

## 6. 是否包含关键策略

已包含。README 记录了最终采用的关键复现策略：

- MPC input 使用 FlavorDB-derived reconstruction；
- split 由 `code/split_data.py` 统一负责；
- Agent 使用 official task2 food-centered evidence；
- 每个 food 使用 10 条 evidence snippets；
- BM25 demonstrations top-k = 3；
- Reviewer 默认不直接接收 raw evidence；
- prompt 鼓励 exactly n when possible / best n candidates / never exceed n；
- protocol-constrained output normalization 只使用 model output、`partial_molecules` 和 `n`，不使用 gold `missing_molecules`；
- MPC evaluation 使用 functional group set precision / recall / F1。

## 7. 是否包含结果汇总

已包含。README 中记录当前正式 MPC 三方法结果：

| Method | Average F1 | Average Precision | Average Recall |
| --- | ---: | ---: | ---: |
| Zero-shot | 0.23661028972959447 | 0.33909958802034773 | 0.18677851848688762 |
| BM25 ICL | 0.28099623215215647 | 0.3550211491455769 | 0.23892580313662173 |
| Scientific Agent | 0.2659615161035022 | 0.34558690304088524 | 0.2223102853798969 |

同时记录了 `zero_f1_count`、`failed_functional_group_prediction_count` 和 `predicted_count_not_equal_n` 诊断指标。

## 8. 是否避免记录试错过程

已避免。README 只记录最终确定的方法、路径、设置、结果和复现边界，不记录调试过程、失败过程、smoke test 过程、retry 过程或 original Agent 与 v2 的试错细节。

README 没有把 archived original Agent 写成 formal result，也没有把 historical v2 目录写成 formal path。

## 9. 是否调用 LLM/API

否。本阶段没有调用 LLM/API。

## 10. 是否运行 prediction/evaluation/Agent/reconstruction/split

否。本阶段没有运行 prediction、evaluation、Agent、reconstruction 或 split。

## 11. 是否修改 formal results

否。本阶段没有修改任何 formal result 文件，没有修改 predictions，也没有修改 metrics。

## 12. 验证结果

已运行：

```bash
python3 -m py_compile code/*.py
python3 code/validate_data.py
```

验证结果：

- `py_compile` 通过；
- `validate_data.py` 通过，输出 `DATA_AUDIT_STATUS: PASS`；
- MFP formal result 行数仍为 71 / 71 / 71；
- MPC zero-shot / BM25 ICL / Scientific Agent formal evaluation details 均为 71 行；
- MPC formal summary 指标保持不变：
  - zero-shot average_f1 = 0.23661028972959447；
  - BM25 ICL average_f1 = 0.28099623215215647；
  - Scientific Agent average_f1 = 0.2659615161035022。

## 13. 是否 ready for final submission / presentation

是。根目录 README 已整理为面向汇报和后续复现实验对接的正式说明文档，当前仓库可以进入 final submission / presentation 准备阶段。
