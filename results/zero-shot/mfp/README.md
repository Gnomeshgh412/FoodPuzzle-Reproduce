# MFP Zero-shot Result

## 目录用途

本目录保存 MFP zero-shot free-text baseline 在 reconstructed test split 上的正式 prediction、evaluation summary、per-sample evaluation details 和 run metadata。

## 当前状态

```text
task: MFP
method: zero-shot
mode: free-text prediction
generation provider: DeepSeek
evaluation: LLM category mapping accuracy
eval split: results/splits/mfp/test.jsonl
```

## 正式结果

```text
correct: 15 / 71
accuracy: 0.2112676056338028
parse_failures: 0
llm_mapping_failures: 0
```

## 文件说明

- `predictions.jsonl`: zero-shot 生成的 free-text `predicted_food`。
- `evaluation_summary.txt`: aggregate evaluation 输出。
- `evaluation_details.jsonl`: per-sample LLM category mapping detail。
- `run_metadata.json`: 方法、split、provider、结果指标和路径元数据。