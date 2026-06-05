# MPC BM25 ICL Result

## 目录用途

本目录保存 MPC BM25 ICL baseline 在 reconstructed test split 上的正式 prediction、retrieval metadata、functional group evaluation summary、per-sample evaluation details 和 predicted molecule functional group extraction cache。

## 当前状态

```text
task: MPC
method: BM25 ICL
retrieval corpus: results/splits/mpc/train.jsonl
eval split: results/splits/mpc/test.jsonl
top_k: 3
BM25 implementation: pure Python
generation provider: DeepSeek
evaluation provider: DeepSeek
evaluation: official-code-aligned functional group set F1
```

## 正式结果

```text
samples_evaluated: 71
average_f1: 0.28099623215215647
average_precision: 0.3550211491455769
average_recall: 0.23892580313662173
zero_f1_count: 1
predicted_count_not_equal_n: 24
unique_predicted_molecule_count: 779
llm_functional_group_prediction_count: 779
failed_functional_group_prediction_count: 29
```

## Prediction 状态

```text
predictions rows: 71
unique ids: 71
error rows: 0
empty prediction rows: 0
predicted_molecules_non_list: 0
count_equal_n: 47
count_not_equal_n: 24
```

## 文件说明

- `predictions.jsonl`: BM25 ICL 生成的 `predicted_molecules`。
- `retrieval_metadata.jsonl`: 每个 test query 检索到的 train demonstrations 记录。
- `evaluation_summary.json`: aggregate functional group evaluation 输出。
- `evaluation_details.jsonl`: per-sample functional group precision / recall / F1 detail。
- `predicted_functional_group_cache.json`: predicted molecule 到 functional groups 的 LLM extraction cache。
