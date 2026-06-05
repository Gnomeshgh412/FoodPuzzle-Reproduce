# processed 数据说明

本目录保存 FoodPuzzle 复现实验使用的处理后 JSONL task 文件。

## 文件说明

- `MFP_tasks.jsonl`: 官方公开 MFP task 文件，字段包括 `id`、`task`、`actual_food`、`molecules`。
- `MPC_tasks.jsonl`: 官方公开 MPC task 文件，字段包括 `id`、`task`、`food`、`missing_molecules`。
- `MPC_reconstructed_tasks.jsonl`: FlavorDB-derived MPC reconstructed task 文件，不是 official exact reproduction。

## MPC_reconstructed_tasks.jsonl

公开 `MPC_tasks.jsonl` 缺少论文 MPC 输入需要的 official `partial_molecules` 和 `n`。本文件使用本地 `data/raw/flavordb.db` 派生重建：

```text
partial_molecules = FlavorDB 中 target_food 的 full molecule set - missing_molecules
n = len(missing_molecules)
```

每条样本只保留必要任务字段：

```text
id
task
target_food
partial_molecules
n
missing_molecules
```

审计信息不写入正式数据文件，只记录在 `results/splits/mpc/MPC_RECONSTRUCTION_AUDIT.md` 和 `results/splits/mpc/split_report.json`。

后续模型推理时不能把 `missing_molecules`、FlavorDB full molecule set 或完整 FlavorDB food-to-molecule list 暴露给模型。
