# FoodPuzzle Only-DeepSeek 复现

本项目使用 DeepSeek 复现 FoodPuzzle 的两个任务：

- **MFP (Molecular Food Prediction)**：根据风味分子集合预测食物来源。
- **MPC (Molecular Profile Completion)**：根据食物、已知分子和缺失数量，预测缺失分子。

仓库中的数据和 evidence 来自 FoodPuzzle 官方仓库；由于官方没有公开完整的推理代码、prompt、Agent 实现和官方 split，Zero-shot、BM25 ICL 和 Scientific Agent 都是根据论文描述做的近似复现。正式实验统一使用 DeepSeek 作为生成模型和评测裁判模型。

## 目录结构

```text
FoodPuzzle-Reproduce/
├── code/
│   ├── Only-Deepseek/          # Only-DeepSeek 推理与评测代码
│   ├── reconstruct_mpc_data.py # 重建 MPC partial molecules
│   └── split_data.py           # 生成自建 train/dev/test split
├── data/
│   ├── raw/flavordb.db        # 官方 FlavorDB 数据
│   ├── processed/              # 官方 task 数据与 MPC 重建数据
│   └── collected_evidences/    # 官方离线 evidence
├── results/
│   ├── splits/                 # MFP/MPC 自建数据划分
│   └── Only-Deepseek/
│       ├── zero-shot/         # Zero-shot 预测与评测
│       ├── icl/               # BM25 ICL 预测与评测
│       ├── agent/             # Scientific Agent 预测与评测
│       └── shared_cache/      # DeepSeek 评测共享缓存
└── scripts/
    ├── run_only_deepseek_mfp.sh
    └── run_only_deepseek_mpc.sh

```

## `code/Only-Deepseek` 文件说明

### `zero_shot.py`

根据论文任务定义近似复现的 Zero-shot baseline。

- MFP 仅向模型提供分子集合，不读取测试食物答案。
- MPC 仅提供食物、`partial_molecules` 和 `n`，不读取测试 `missing_molecules`。
- 该文件不是官方代码，prompt 是按照论文任务定义重建的。

### `bm25_icl.py`

根据论文 BM25 In-context Learning 方法近似复现。

- 只从标注训练集检索 demonstrations。
- 训练 passage 包含 `Food + Molecules`。
- 每个测试样本检索 BM25 top-3 demonstrations。
- 该文件不是官方代码；官方只在论文中公开了方法描述，没有公开 BM25 实现。

### `scientific_agent.py`

根据论文 Scientific Agent 架构近似复现的正式 Agent 主线。

- MFP 从训练集频率计算分子信息熵，选择最多 10 个低熵 starting points。
- MPC 直接根据食物获取官方离线 evidence。
- Scientist 接收任务输入、evidence 和三个 BM25 demonstrations，生成三个 hypotheses。
- Reviewer 只从三个 hypotheses 中选择最终答案。
- 该文件不是官方代码。官方未公开 Agent 源码和完整 prompt，因此属于论文级近似复现。

### `evaluation.py`

对官方公开 `evaluation.py` 的 DeepSeek 适配版，不是官方原文件。

- MFP gold category 按官方行为从 FlavorDB 的 `entity_alias_readable` 和第一层 category 获取。
- MFP 使用 DeepSeek 将自由文本预测映射到官方 macro categories，代替论文中的 GPT-3.5 裁判。
- MPC gold functional groups 保留官方公开代码的解析控制流。
- MPC 使用官方固定的 53 个官能团候选，由 DeepSeek 提取预测分子的官能团，最后计算 sample-average F1。
- 评测结构与官方公开代码对齐，但裁判模型已替换为 DeepSeek，因此应称为 **DeepSeek-adapted evaluation**。

## 数据来源说明

| 文件 | 来源 |
|---|---|
| `data/raw/flavordb.db` | FoodPuzzle 官方公开资产，与官方文件逐字节一致 |
| `data/processed/MFP_tasks.jsonl` | FoodPuzzle 官方公开任务数据 |
| `data/processed/MPC_tasks.jsonl` | FoodPuzzle 官方公开任务数据，只公开 missing molecules |
| `data/collected_evidences/*.pkl` | FoodPuzzle 官方公开 evidence |
| `data/processed/MPC_reconstructed_tasks.jsonl` | 本项目根据 FlavorDB 重建，不是官方原始 partial input |
| `results/splits/` | 本项目的固定随机 80/10/10 split，不是官方 split |

MPC 重建使用：

```text
partial_molecules = FlavorDB full_molecules - official missing_molecules
n = len(official missing_molecules)
```

## 当前结果

| 任务 | Zero-shot | BM25 ICL | Scientific Agent |
|---|---:|---:|---:|
| MFP Accuracy | 0.2113 | **0.3521** | 0.2958 |
| MPC Functional-group F1 | 0.5535 | **0.5778** | 0.5311 |

上述结果使用自建 split、重建 MPC input 和 DeepSeek 裁判，不能与论文表格中的 GPT-3.5、Gemini 或 LLaMA 结果直接比较。

## 官方资料

- Paper: <https://arxiv.org/abs/2409.12832>
- Official repository: <https://github.com/tenghaohuang/FoodPuzzle>
