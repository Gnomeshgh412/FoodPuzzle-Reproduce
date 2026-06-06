# MFP BM25 ICL Result

## 目录用途

本目录保存 MFP BM25 ICL baseline 在 reconstructed test split 上的正式 prediction、retrieval metadata、evaluation summary、per-sample evaluation details 和 run metadata。

## 当前状态

```text
task: MFP
method: BM25 ICL
retrieval corpus: results/splits/mfp/train.jsonl
eval split: results/splits/mfp/test.jsonl
top_k: 3
BM25 implementation: pure Python
generation provider: DeepSeek
model: deepseek-v4-flash
evaluation: LLM category mapping accuracy
```

## 正式结果

```text
correct: 17 / 71
accuracy = 0.23943661971830985
parse_failures: 0
llm_mapping_failures: 0
```

## 文件说明

- `predictions.jsonl`: BM25 ICL 生成的 free-text `predicted_food`。
- `retrieval_metadata.jsonl`: 每个 test query 检索到的 train demonstrations 记录。
- `evaluation_summary.txt`: aggregate evaluation 输出。
- `evaluation_details.jsonl`: per-sample LLM category mapping detail。
- `run_metadata.json`: 方法、split、retrieval、provider、结果指标和路径元数据。
