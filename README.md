# FoodPuzzle 近似复现仓库说明

## 1. 项目简介

本仓库基于 FoodPuzzle 论文与官方公开资源，对 MFP 与 MPC 两个任务进行近似复现。

当前复现使用论文描述、官方公开 task data、FlavorDB、official collected evidence 和公开 evaluation 思路进行 paper/code/data-aligned reconstruction。由于官方未完整公开 train/dev/test split、完整 Scientific Agent code、prompt 和 DSPy signature，本仓库不是 official exact reproduction。

现有冻结结果使用 DeepSeek `deepseek-v4-flash`，不是原文 GPT-3.5 / Gemini / LLaMA3 设置；当前数值只用于本仓库 reconstructed split 内部比较，不与论文 Table 2 做严格数值比较。

新的多模型复现实验通过 AIHubMix 接入以下模型：

- `coding-glm-4.7-free`
- `gpt-4.1-free`
- `xiaomi-mimo-v2.5-free`

各模型结果统一写入 `results/<method>/<task>/<model>/`。现有 DeepSeek 冻结结果位于对应任务目录的 `deepseek-v4-flash/` 子目录。

## 2. 任务说明

### 2.1 MFP

MFP 是 Molecule-to-Food Prediction。

任务形式：

- 输入：molecules；
- 输出：food / category。

MFP results：

- `results/Only-Deepseek/{zero-shot,icl,agent}/mfp/`
- `results/Multi-Models/{zero-shot,icl,agent}/mfp/`

### 2.2 MPC

MPC 是 Molecule Profile Completion。

任务形式：

- 输入：
  - `target_food`
  - `partial_molecules`
  - `n`
- 输出：
  - `missing_molecules` / `predicted_molecules`

## 3. 与原文和官方仓库的关系

### 3.1 官方公开内容

本仓库使用的官方公开或论文明确描述内容包括：

- FlavorDB 数据；
- FoodPuzzle task data；
- `collected_evidences_task1.pkl`；
- `collected_evidences_task2.pkl`；
- 官方公开 evaluation 思路；
- 论文中的 zero-shot、BM25 ICL、Scientific Agent 方法描述。

### 3.2 官方未完整公开内容

- official train/dev/test split；
- split 代码；
- full Scientific Agent code；
- zero-shot / ICL / Agent prompts；
- Scientist / Reviewer prompt；
- parser / normalization 细节；
- 完整 DSPy signature。

### 3.3 本仓库的处理原则

- 尽量遵循论文方法流程；
- prediction 阶段不使用 test gold label；
- Agent 不使用 FlavorDB full molecule list 作为 evidence；
- Agent 不使用 zero-shot / ICL predictions 作为输入；
- 不手工修 predictions；
- 不改变 evaluation metric；
- 所有与官方不完全一致的部分均标注为 reconstruction。

## 4. 仓库结构

```text
FoodPuzzle-Reproduce/
├── code/                         # 核心复现代码
├── data/                         # 原始数据、processed 数据、official evidence
├── results/                      # formal results、split、README 和报告
└── README.md                     # 根目录说明文档
```

### `code/`

- `reconstruct_mpc_data.py`：MPC 数据重建；
- `split_data.py`：data split；
- `validate_data.py`：数据与结果完整性检查；
- `Only-Deepseek/`：DeepSeek 的 zero-shot、BM25 ICL、Scientific Agent、Multi-Agent、Optimized-Agent 和评测实现；
- `Multi-Models/`：AIHubMix 多模型 zero-shot、BM25 ICL、Scientific Agent 和统一评测实现；
- `code/README.md`：代码与运行入口说明。

### `data/`

- `data/raw/`：FlavorDB；
- `data/processed/`：processed / reconstructed task data；
- `data/collected_evidences`：official evidence。

### `results/`

- `results/splits`：reconstructed train/dev/test split；
- `results/Only-Deepseek`：仅 DeepSeek 的三种 baseline 结果与独立评测缓存；
- `results/Multi-Models`：多模型的三种 baseline 结果与 GPT-4.1-free 共享评测缓存；
- `results/Only-Deepseek/multi-agent`：DeepSeek Multi-Agent 冻结结果；
- `results/Only-Deepseek/optimized-agent`：DeepSeek Optimized-Agent 冻结结果；
- `results/Only-Deepseek/优化实验`：已执行的分轮优化实验与审查产物。

## 5. 当前正式结果

下表为修改 MPC 官方兼容解析和共享 judge cache 之前的历史冻结结果。新的多模型实验完成后，应以 `results/<method>/<task>/<model>/` 中使用统一评测配置生成的结果为准。

### 5.1 MFP formal results

MFP 使用 food/category prediction accuracy 作为主要指标。

| Method | Accuracy | Correct | Total |
|---|---:|---:|---:|
| Zero-shot | 0.2112676056338028 | 15 | 71 |
| BM25 ICL | 0.23943661971830985 | 17 | 71 |
| Scientific Agent | 0.29577464788732394 | 21 | 71 |

MFP 当前结果排序：

```text
Scientific Agent > BM25 ICL > Zero-shot
```

### 5.2 MPC formal results

MPC 使用 functional group set precision / recall / F1 作为主要指标。

| Method | Average F1 | Average Precision | Average Recall |
|---|---:|---:|---:|
| Zero-shot | 0.23661028972959447 | 0.33909958802034773 | 0.18677851848688762 |
| BM25 ICL | 0.28099623215215647 | 0.3550211491455769 | 0.23892580313662173 |
| Scientific Agent | 0.2659615161035022 | 0.34558690304088524 | 0.2223102853798969 |

MPC 当前结果排序：

```text
BM25 ICL > Scientific Agent > Zero-shot
```

## 6. AIHubMix 多模型复现

四个生成模型分别使用对应环境变量；下列三个 AIHubMix 模型使用独立密钥，DeepSeek 使用 `DEEPSEEK_API_KEY`：

```text
AIHUBMIX_CODING_GLM_4_7_FREE_API_KEY
AIHUBMIX_GPT_4_1_FREE_API_KEY
AIHUBMIX_XIAOMI_MIMO_V2_5_FREE_API_KEY
```

正式 provider 名称如下：

```text
aihubmix-coding-glm-4.7-free
aihubmix-gpt-4.1-free
aihubmix-xiaomi-mimo-v2.5-free
```

MFP 和 MPC 的正式评测统一使用 `gpt-4.1-free`。MPC 中该模型负责预测分子的官能团映射，所有生成模型与方法共享：

```text
results/Multi-Models/shared_cache/gpt-4.1-free_functional_group_cache.json
```

cache metadata 会绑定 provider、model、endpoint、prompt 版本和 53 类词表哈希，避免其他 judge 意外复用同一路径。

## 7. 关键复现策略与最终决策

1. 任务边界
   - MFP 和 MPC 均保留 zero-shot、BM25 ICL、Scientific Agent 三条方法支线；
   - MFP 是 food/category prediction；
   - MPC 是 molecule profile completion；
   - 两个任务使用不同 evaluation metric，不做跨任务数值排序。

2. Data and split
   - `split_data.py` 统一负责 MFP / MPC split；
   - 使用 fixed seed = 42、ratio = 80/10/10；
   - MFP 总样本数为 709，实际落盘 train/dev/test = 567/71/71；
   - MPC reconstructed 总样本数为 710，实际落盘 train/dev/test = 568/71/71。

3. MFP prediction strategy
   - zero-shot 直接基于 molecule input 预测 food/category；
   - BM25 ICL 使用 train split 构建 demonstrations；
   - Scientific Agent 保留 MFP 的 entropy-based Starting Point Identification；
   - MFP prediction/parser 只做 non-gold output parsing / format cleanup；
   - MFP prediction 阶段不使用 test gold food/category。

4. MFP evaluation
   - MFP evaluation 直接比较 predicted food/category 与 gold label；
   - gold label 只在 evaluation 阶段使用。

5. MPC input reconstruction
   - 使用 FlavorDB-derived reconstruction；
   - `partial_molecules = full_molecules - missing_molecules`；
   - `n = len(missing_molecules)`。

6. MPC Agent evidence
   - 使用 official task2 food-centered evidence；
   - 每个 food 使用 10 条 evidence snippets；
   - 不使用 FlavorDB full molecule list 作为 evidence。

7. MPC Agent prompt
   - 鼓励 exactly `n` when possible；
   - never exceed `n`；
   - 不预测 `partial_molecules` 中已有分子。

8. MPC output normalization
   - 只使用 model output、`partial_molecules`、`n`；
   - 不使用 gold `missing_molecules`；
   - 与 MFP 的 non-gold parsing / format cleanup 原则一致。

9. MPC evaluation
   - MPC 使用 functional group set F1；
   - predicted molecules 走 LLM functional group extraction；
   - cache 不改变 evaluation 公式。

10. 统一禁止事项
   - prediction 阶段不使用当前 test sample 的 gold label；
   - Agent 不使用 zero-shot / ICL predictions 作为输入；
   - 不手工修 predictions；
   - 不改变 evaluation metric。
