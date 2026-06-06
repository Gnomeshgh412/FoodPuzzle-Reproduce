# code/ README

## 1. 目录用途

`code/` 保存 FoodPuzzle 复现用的核心脚本，当前主要支持 MFP 和 MPC 两条支线：

- MFP: Molecule Flavor Prediction；
- MPC: Molecule Profile Completion。

## 2. 文件说明

### `reconstruct_mpc_data.py`

功能：

- 从公开 `data/processed/MPC_tasks.jsonl` 和 `data/raw/flavordb.db` 重建 MPC 可用输入；
- 根据 FlavorDB 中 food 的 full molecule set 与 MPC gold `missing_molecules` 生成 `partial_molecules` 和 `n`；
- 只负责 MPC reconstruction，不负责 train/dev/test split；
- 不调用 LLM/API。

支持任务：

- MPC only。

输入：

- `data/processed/MPC_tasks.jsonl`
- `data/raw/flavordb.db`

输出：

- `data/processed/MPC_reconstructed_tasks.jsonl`

### `split_data.py`

功能：

- 统一负责 MFP 和 MPC 的 train/dev/test split；
- 使用 seed 和 ratio 做 row-level reconstructed split；
- 不调用 LLM/API。

支持任务：

- MFP；
- MPC。

MFP 输入：

- `data/processed/MFP_tasks.jsonl`

MFP 输出：

- `results/splits/mfp/train.jsonl`
- `results/splits/mfp/dev.jsonl`
- `results/splits/mfp/test.jsonl`
- `results/splits/mfp/train_ids.txt`
- `results/splits/mfp/dev_ids.txt`
- `results/splits/mfp/test_ids.txt`
- `results/splits/mfp/split_metadata.json`

MPC 输入：

- `data/processed/MPC_reconstructed_tasks.jsonl`

MPC 输出：

- `results/splits/mpc/train.jsonl`
- `results/splits/mpc/dev.jsonl`
- `results/splits/mpc/test.jsonl`

### `validate_data.py`

功能：

- 检查 raw / processed data、MPC reconstructed data、MPC split 和 FlavorDB schema；
- 用于确认文件存在、JSONL 可解析、SQLite 表结构可读取；
- 不调用 LLM/API；
- 不修改 formal results。

支持任务：

- shared。

输入：

- `data/raw/flavordb.db`
- `data/processed/MFP_tasks.jsonl`
- `data/processed/MPC_tasks.jsonl`
- `data/processed/MPC_reconstructed_tasks.jsonl`
- `results/splits/mpc/train.jsonl`
- `results/splits/mpc/dev.jsonl`
- `results/splits/mpc/test.jsonl`

输出：

- stdout audit；
- 不写 result 文件。

### `zero_shot.py`

功能：

- 运行 zero-shot prediction；
- 支持 `--task mfp` / `--task mpc`；
- 使用 `--use-llm` 时调用 LLM/API；
- 支持 `--resume`。

支持任务：

- MFP；
- MPC。

MFP 输入：

- `results/splits/mfp/test.jsonl`

MFP 输出：

- `results/zero-shot/mfp/predictions.jsonl`

MPC 输入：

- `results/splits/mpc/test.jsonl`

MPC prompt 使用 test sample 的 input-visible fields：

- `target_food`
- `partial_molecules`
- `n`

MPC 输出：

- `results/zero-shot/mpc/predictions.jsonl`

### `bm25_icl.py`

功能：

- 运行 BM25 in-context learning prediction；
- 输出 retrieval metadata；
- 支持 `--task mfp` / `--task mpc`；
- 使用 `--use-llm` 时调用 LLM/API；
- 支持 `--resume`。

支持任务：

- MFP；
- MPC。

MFP 输入：

- train: `results/splits/mfp/train.jsonl`
- test: `results/splits/mfp/test.jsonl`

MFP 输出：

- `results/icl/mfp/predictions.jsonl`
- `results/icl/mfp/retrieval_metadata.jsonl`

MPC 输入：

- train: `results/splits/mpc/train.jsonl`
- test: `results/splits/mpc/test.jsonl`

MPC 输出：

- `results/icl/mpc/predictions.jsonl`
- `results/icl/mpc/retrieval_metadata.jsonl`

### `scientific_agent.py`

功能：

- 运行 Scientific Agent prediction；
- 支持 BM25 demonstrations、local evidence、Scientist / Reviewer 流程；
- 支持 `--task mfp` / `--task mpc`；
- 使用 `--use-llm` 时调用 LLM/API；
- 支持 `--resume`。

支持任务：

- MFP；
- MPC。

MFP 输入：

- train: `results/splits/mfp/train.jsonl`
- test: `results/splits/mfp/test.jsonl`
- evidence: official task1 / MFP evidence file，按当前 MFP formal run 配置指定；

MFP 输出：

- `results/agent/mfp/predictions.jsonl`
- `results/agent/mfp/retrieval_metadata.jsonl`
- `results/agent/mfp/evidence_metadata.jsonl`
- `results/agent/mfp/hypotheses_metadata.jsonl`

MPC 输入：

- train: `results/splits/mpc/train.jsonl`
- test: `results/splits/mpc/test.jsonl`
- evidence: `data/collected_evidences/collected_evidences_task2.pkl`

MPC formal Agent 使用：

- official task2 food-centered evidence: `data/collected_evidences/collected_evidences_task2.pkl`；
- BM25 demonstrations top-k = 3；
- evidence snippets = 10；
- `reviewer_evidence_mode = none`；
- exact-n prompt；
- protocol-constrained output normalization。

MPC 输出：

- `results/agent/mpc/predictions.jsonl`
- `results/agent/mpc/retrieval_metadata.jsonl`
- `results/agent/mpc/evidence_metadata.jsonl`
- `results/agent/mpc/hypothesis_metadata.jsonl`

MPC normalization 只使用：

- model output；
- input `partial_molecules`；
- input `n`。

它不使用 gold `missing_molecules`，不使用 zero-shot / ICL predictions，不使用 FlavorDB full molecule list 作为 evidence。

### `evaluation.py`

功能：

- 运行 MFP / MPC evaluation；
- 不重新生成 predictions；
- 使用 `--use-llm` 时可能调用 LLM/API。

支持任务：

- MFP；
- MPC。

MFP 输入：

- gold: MFP test split；
- pred: MFP prediction file；
- db: `data/raw/flavordb.db`。

MFP 输出：

- `results/zero-shot/mfp/evaluation_details.jsonl`
- `results/icl/mfp/evaluation_details.jsonl`
- `results/agent/mfp/evaluation_details.jsonl`
- summary 输出按具体 MFP formal run 配置保存。

MPC 输入：

- gold: `results/splits/mpc/test.jsonl`；
- pred: 对应方法的 `predictions.jsonl`；
- db: `data/raw/flavordb.db`。

MPC 输出：

- `evaluation_details.jsonl`
- `evaluation_summary.json`
- `predicted_functional_group_cache.json`

MPC formal evaluation 使用 `official_llm`：

- gold `missing_molecules` 通过 FlavorDB 映射到 `functional_groups`；
- predicted `predicted_molecules` 通过 LLM functional group extraction 映射到 official fixed 53-item functional group vocabulary；
- 使用 functional group set precision / recall / F1；
- `predicted_count != n` 只作为诊断字段，不作为 error；
- empty prediction 合法处理为 F1=0；
- functional group cache 用于 resume 和减少重复 LLM 调用，不改变 evaluation 逻辑。

MPC evaluation 任务是 missing molecule list completion。原文评价思路是 functional group set precision / recall / F1：gold `missing_molecules` 需要用 FlavorDB 映射到 functional groups，predicted `predicted_molecules` 需要通过 LLM functional group extraction 映射到固定 vocabulary。因此 MPC evaluation 命令需要 `--db`、`--use-llm`、`--mpc-eval-mode official_llm`、`--functional-group-cache`、`--save-details` 和 `--save-summary-json`。

## 3. 重要边界

1. MPC 是 FlavorDB-derived reconstruction；
2. official split 未公开，当前 split 是 seed=42 的 reconstructed split；
3. official Agent code / prompts / DSPy signature 未完整公开；
4. 当前 Agent 是 paper-aligned reconstruction；
5. 当前 formal results 使用 DeepSeek `deepseek-v4-flash`，不是原文 GPT-3.5 / Gemini / LLaMA3；
6. MPC evaluation 使用 DeepSeek 做 predicted molecule -> functional group extraction；
7. 不与论文 Table 2 严格数值比较。

## 4. 禁止事项

1. prediction 阶段不得使用当前 test sample 的 gold `missing_molecules`；
2. Agent 不得使用 zero-shot / ICL predictions 作为输入；
3. Agent 不得使用 FlavorDB full molecule list 作为 evidence；
4. 不得手工修正 low-score samples；
5. 不得改变 evaluation metric；
6. API key 不得写入代码、README 或结果文件；
7. evaluation 不得重新生成 predictions。
