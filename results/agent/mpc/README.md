# MPC Scientific Agent Result

## 目录用途

本目录保存 MPC Scientific Agent 在 reconstructed test split 上的正式 prediction、metadata、functional group evaluation summary、per-sample evaluation details 和 predicted molecule functional group extraction cache。

## 当前状态

```text
task: MPC
method: Scientific Agent
evidence route: official_task2_food_centered_evidence
official evidence: data/collected_evidences/collected_evidences_task2.pkl
BM25 demonstrations: results/splits/mpc/train.jsonl
agent structure: two-stage Scientist / Reviewer
reviewer_evidence_mode: none
evidence snippets per food: 10
eval split: results/splits/mpc/test.jsonl
provider: DeepSeek
model: deepseek-v4-flash
evaluation: official-code-aligned functional group set F1
```

## 正式结果

```text
samples_evaluated: 71
average_f1: 0.2659615161035022
average_precision: 0.34558690304088524
average_recall: 0.2223102853798969
zero_f1_count: 3
predicted_count_not_equal_n: 30
unique_predicted_molecule_count: 1258
llm_functional_group_prediction_count: 1258
failed_functional_group_prediction_count: 37
scientist_parse_failures: 10
reviewer_parse_failures: 0
```

## Prediction 状态

```text
predictions rows: 71
unique ids: 71
error rows: 0
empty prediction rows: 0
predicted_molecules_non_list: 0
predicted_count <= n: 71
predicted_count == n: 41
predicted_count < n: 30
predicted_count > n: 0
truncated_to_n: 13
removed_partial_molecule_count_total: 138
```

## 文件说明

- `predictions.jsonl`: Agent 最终 `predicted_molecules`。
- `retrieval_metadata.jsonl`: Agent 使用的 BM25 train demonstrations。
- `evidence_metadata.jsonl`: official task2 food-centered evidence lookup 和 usage metadata。
- `hypothesis_metadata.jsonl`: Scientist hypotheses、Reviewer output 和 normalization metadata。
- `evaluation_summary.json`: aggregate functional group evaluation 输出。
- `evaluation_details.jsonl`: per-sample functional group precision / recall / F1 detail。
- `predicted_functional_group_cache.json`: predicted molecule 到 functional groups 的 LLM extraction cache。
