# MPC Reconstructed Split

## 目录用途

本目录保存基于公开 MPC 数据和 FlavorDB 重建的 MPC train/dev/test split，用于 MPC zero-shot、BM25 ICL 和 Scientific Agent 的公平比较。

## 当前状态

```text
task: MPC
source: data/processed/MPC_reconstructed_tasks.jsonl
seed: 42
ratio: 80/10/10
train: 568
dev: 71
test: 71
official split: unavailable
stratified: false
near-duplicate decontamination: false
```

## 文件说明

- `train.jsonl`: MPC BM25 ICL 和 Scientific Agent 的 retrieval corpus。
- `dev.jsonl`: reconstructed dev split，当前未用于正式 baseline。
- `test.jsonl`: MPC zero-shot / ICL / Agent 使用的 evaluation split。
