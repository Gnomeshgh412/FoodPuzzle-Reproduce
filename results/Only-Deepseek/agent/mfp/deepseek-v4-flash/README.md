# MFP Scientific Agent Result

## 目录用途

本目录保存 MFP Scientific Agent 在 reconstructed test split 上的正式 prediction、metadata、evaluation summary、per-sample evaluation details 和诊断输出。

## 当前状态

```text
task: MFP
method: Scientific Agent
evidence route: answer_masked_official_evidence
official evidence: data/collected_evidences/collected_evidences_task1.pkl
BM25 demonstrations: results/Only-Deepseek/icl/mfp/deepseek-v4-flash/retrieval_metadata.jsonl
agent structure: two-stage Scientist / Reviewer
eval split: results/splits/mfp/test.jsonl
provider: DeepSeek
model: deepseek-v4-flash
```

## 正式结果

```text
correct: 21 / 71
accuracy = 0.29577464788732394
parse_failures: 0
llm_mapping_failures: 0
scientist_parse_failures: 0
reviewer_parse_failures: 0
```

## 文件说明

- `predictions.jsonl`: Agent 最终 free-text `predicted_food`。
- `evidence_metadata.jsonl`: selected molecules、answer-masked official evidence 和 masking metadata。
- `retrieval_metadata.jsonl`: Agent 使用的 BM25 train demonstrations。
- `hypotheses_metadata.jsonl`: Scientist hypotheses 和 Reviewer output。
- `evaluation_summary.txt`: aggregate MFP evaluation 输出。
- `evaluation_details.jsonl`: per-sample LLM category mapping detail。
- `run_metadata.json`: 方法、split、evidence、provider、baseline 和结果指标元数据。
