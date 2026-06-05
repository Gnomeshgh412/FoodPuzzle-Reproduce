# MPC Zero-shot Result

## 目录用途

本目录保存 MPC zero-shot baseline 在 reconstructed test split 上的正式 prediction、functional group evaluation summary、per-sample evaluation details 和 predicted molecule functional group extraction cache。

## 当前状态

```text
task: MPC
method: zero-shot
mode: missing molecule prediction
generation provider: DeepSeek
evaluation provider: DeepSeek
evaluation: official-code-aligned functional group set F1
eval split: results/splits/mpc/test.jsonl
```

## 正式结果

```text
samples_evaluated: 71
average_f1: 0.23661028972959447
average_precision: 0.33909958802034773
average_recall: 0.18677851848688762
zero_f1_count: 6
predicted_count_not_equal_n: 5
unique_predicted_molecule_count: 1463
llm_functional_group_prediction_count: 1463
failed_functional_group_prediction_count: 51
```

## Prediction 状态

```text
predictions rows: 71
unique ids: 71
error rows: 0
empty prediction rows: 0
predicted_molecules_non_list: 0
count_equal_n: 66
count_not_equal_n: 5
```

## 文件说明

- `predictions.jsonl`: zero-shot 生成的 `predicted_molecules`。
- `evaluation_summary.json`: aggregate functional group evaluation 输出。
- `evaluation_details.jsonl`: per-sample functional group precision / recall / F1 detail。
- `predicted_functional_group_cache.json`: predicted molecule 到 functional groups 的 LLM extraction cache。
