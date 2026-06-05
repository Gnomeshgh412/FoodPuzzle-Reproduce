# MFP Reconstructed Split

## 目录用途

本目录保存基于公开 `data/processed/MFP_tasks.jsonl` 重建的 MFP train/dev/test split，用于 MFP zero-shot、BM25 ICL 和 Scientific Agent 的公平比较。

## 当前状态

```text
task: MFP
source: data/processed/MFP_tasks.jsonl
seed: 42
ratio: 80/10/10
train: 567
dev: 71
test: 71
official split: unavailable
stratified: false
near-duplicate decontamination: false
```

## 文件说明

- `train.jsonl`: MFP BM25 ICL 和 Scientific Agent 的 retrieval corpus。
- `dev.jsonl`: reconstructed dev split，当前未用于正式 baseline。
- `test.jsonl`: MFP zero-shot / ICL / Agent 使用的 evaluation split。