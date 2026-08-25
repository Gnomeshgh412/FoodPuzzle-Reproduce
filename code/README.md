# code/ README

## 目录用途

`code/` 保存 FoodPuzzle MFP 和 MPC 的数据处理、baseline、Agent 与评测代码。当前代码按 provider 和实验主线分开，不在源码中保存 API key。

## 共享数据工具

- `reconstruct_mpc_data.py`：从公开 MPC task 和 FlavorDB 重建 `partial_molecules` 与 `n`。
- `split_data.py`：使用固定 seed 生成 MFP / MPC reconstructed train/dev/test split。
- `validate_data.py`：检查 JSONL、SQLite schema、数据与 split 的基本完整性。

## DeepSeek 复现

`Only-Deepseek/` 包含：

- `zero_shot.py`：zero-shot baseline；
- `bm25_icl.py`：BM25 in-context learning baseline；
- `scientific_agent.py`：Scientist / Reviewer Agent baseline；
- `multi_agent.py`：异构 Multi-Agent 实验；
- `optimized_agent.py`：当前优化 Agent 研究源码；
- `evaluation.py`：MFP 类别映射与 MPC 官能团评测。

对应 runner：

```bash
bash scripts/run_only_deepseek_mpc.sh
bash scripts/run_multi_agent.sh
bash scripts/run_optimized_agent.sh all
```

`run_optimized_agent.sh` 会检查版本和元数据兼容性。运行前应先阅读根目录的交接与审计记录，避免用当前研究源码覆盖历史冻结结果。

## AIHubMix 多模型复现

`Multi-Models/` 包含 zero-shot、BM25 ICL、Scientific Agent 和统一评测代码。正式 provider 名称为：

- `aihubmix-coding-glm-4.7-free`
- `aihubmix-gpt-4.1-free`
- `aihubmix-xiaomi-mimo-v2.5-free`

运行入口：

```bash
bash scripts/run_multi_models.sh
```

MPC 多模型评测使用独立的 GPT-4.1-free 官能团 cache；它与 DeepSeek cache 不互相复用。

## 安全与复现边界

1. prediction 阶段不得使用当前 test sample 的 gold label。
2. Agent 不得把 zero-shot / ICL prediction 当作输入证据。
3. API key 只通过本地环境变量或 `.env.local` 提供，不得写入代码、结果或文档。
4. reconstructed split 不是官方未公开 split，结果不与论文 Table 2 做严格数值对比。
5. `results/Only-Deepseek/optimized-agent/` 的冻结产物和当前 `optimized_agent.py` 可能属于不同版本，以 `run_metadata.json` 和 `audit_records/` 为准。
