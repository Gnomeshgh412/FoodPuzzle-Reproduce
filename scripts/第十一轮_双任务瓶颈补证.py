#!/usr/bin/env python3
"""第十一轮双任务瓶颈补证。

本脚本只做阶段诊断，不改变 Scientist、Reviewer 或评测器：
1. 为当前 MFP dev 构造与既有 BM25 基线同口径的训练示例检索元数据；
2. 把 MFP Scientist 的三个具体食物候选拆成三份评测输入，并统计具体食物命中；
3. 在 MPC 训练集上做 exact-profile clustered 5-fold OOF，仅用各折
   训练 profile 中出现的分子，检查 H1 在 N/1.25N/1.5N/2N/3N 的召回。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[1]
默认输出根目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第十一轮/双任务瓶颈补证"


def 加载模块(名称: str, 路径: Path) -> Any:
    spec = importlib.util.spec_from_file_location(名称, 路径)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{路径}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 读取_jsonl(路径: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in 路径.read_text(encoding="utf-8").splitlines() if line.strip()]


def 写入_jsonl(路径: Path, rows: list[dict[str, Any]]) -> None:
    路径.parent.mkdir(parents=True, exist_ok=True)
    if 路径.exists():
        raise RuntimeError(f"为避免覆盖旧产物，已停止：{路径}")
    路径.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def 归一化(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def 准备_mfp(输出根目录: Path) -> None:
    bm25 = 加载模块("mfp_bm25", 项目根目录 / "code/Only-Deepseek/bm25_icl.py")
    train = 读取_jsonl(项目根目录 / "results/splits/mfp/train.jsonl")
    dev = 读取_jsonl(项目根目录 / "results/splits/mfp/dev.jsonl")
    index = bm25.BM25Index(train, "mfp")
    rows = []
    for query in dev:
        retrieved = index.retrieve(query, 3)
        rows.append(
            {
                "id": query["id"],
                "retrieved": [
                    {
                        "id": item["id"],
                        "rank": item["rank"],
                        "score": item["score"],
                        "actual_food": item["actual_food"],
                        "target_food": item["target_food"],
                    }
                    for item in retrieved
                ],
            }
        )
    output = 输出根目录 / "MFP_完整科学家阶段分解/开发集BM25检索元数据.jsonl"
    写入_jsonl(output, rows)
    if {str(row["id"]) for row in rows} != {str(row["id"]) for row in dev}:
        raise RuntimeError("MFP dev 检索 ID 覆盖不完整")
    print(json.dumps({"状态": "通过", "样本数": len(rows), "输出": str(output)}, ensure_ascii=False, indent=2))


def 分析_mfp(输出根目录: Path) -> None:
    directory = 输出根目录 / "MFP_完整科学家阶段分解_联网重试"
    dev = 读取_jsonl(项目根目录 / "results/splits/mfp/dev.jsonl")
    by_id = {str(row["id"]): row for row in dev}
    metadata = 读取_jsonl(directory / "假设与审查元数据.jsonl")
    rank_rows: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    reviewer_rows: list[dict[str, Any]] = []
    exact_top1 = exact_top3 = reviewer_exact = 0
    valid_three = reviewer_category_like = 0
    category_names = {
        "additive", "animal product", "animalproduct", "beverage alcoholic", "beverage",
        "cereal", "dairy", "dish", "essential oil", "fish seafood", "flower", "fruit",
        "fungus", "herb", "legume", "meat", "nut", "plant", "seed", "spice", "vegetable",
    }
    for record in metadata:
        row_id = str(record["id"])
        gold = 归一化(by_id[row_id].get("actual_food"))
        hypotheses = record.get("hypotheses") or []
        if len(hypotheses) == 3:
            valid_three += 1
        foods = [str(item.get("predicted_food") or "").strip() for item in hypotheses[:3]]
        foods += [""] * (3 - len(foods))
        for rank, food in enumerate(foods, 1):
            rank_rows[rank].append({"id": row_id, "predicted_food": food})
        if 归一化(foods[0]) == gold:
            exact_top1 += 1
        if gold in {归一化(food) for food in foods if 归一化(food)}:
            exact_top3 += 1
        reviewer = str((record.get("reviewer_output") or {}).get("predicted_food") or "").strip()
        reviewer_rows.append({"id": row_id, "predicted_food": reviewer})
        if 归一化(reviewer) == gold:
            reviewer_exact += 1
        if 归一化(reviewer).replace(" ", "") in {x.replace(" ", "") for x in category_names}:
            reviewer_category_like += 1
    for rank in (1, 2, 3):
        写入_jsonl(directory / f"科学家第{rank}候选_官方类别评测输入.jsonl", rank_rows[rank])
    写入_jsonl(directory / "审查器最终预测_官方类别评测输入.jsonl", reviewer_rows)
    summary = {
        "样本数": len(dev),
        "Scientist有效三候选样本数": valid_three,
        "Scientist_Top1具体食物命中数": exact_top1,
        "Scientist_Top3具体食物召回数": exact_top3,
        "Reviewer具体食物命中数": reviewer_exact,
        "Reviewer似乎直接输出宏类别的样本数": reviewer_category_like,
        "说明": "具体食物仅作诊断；官方宏类别 accuracy 需分别评测三个 Scientist 候选和 Reviewer。",
    }
    path = directory / "具体食物阶段诊断.json"
    if path.exists():
        raise RuntimeError(f"为避免覆盖旧产物，已停止：{path}")
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def 汇总_mfp类别(输出根目录: Path) -> None:
    directory = 输出根目录 / "MFP_完整科学家阶段分解_联网重试"
    names = ["科学家第1候选", "科学家第2候选", "科学家第3候选", "审查器最终预测"]
    tables = {
        name: {str(row["id"]): row for row in 读取_jsonl(directory / f"{name}_官方类别逐样本.jsonl")}
        for name in names
    }
    ids = sorted(tables[names[0]], key=lambda value: (len(value), value))
    details: list[dict[str, Any]] = []
    for row_id in ids:
        candidate_rows = [tables[name][row_id] for name in names[:3]]
        reviewer = tables[names[3]][row_id]
        candidate_correct = [bool(row.get("correct")) for row in candidate_rows]
        categories = [row.get("predicted_category") for row in candidate_rows]
        details.append(
            {
                "id": row_id,
                "gold类别": reviewer.get("gold_category"),
                "Scientist候选类别": categories,
                "Scientist各候选是否正确": candidate_correct,
                "Scientist_Top3类别oracle正确": any(candidate_correct),
                "Scientist候选类别数": len({value for value in categories if value}),
                "Reviewer预测类别": reviewer.get("predicted_category"),
                "Reviewer是否正确": bool(reviewer.get("correct")),
                "Reviewer在oracle可正确时选错": any(candidate_correct) and not bool(reviewer.get("correct")),
                "Reviewer在候选均错时补救": not any(candidate_correct) and bool(reviewer.get("correct")),
            }
        )
    total = len(details)
    oracle = sum(bool(row["Scientist_Top3类别oracle正确"]) for row in details)
    reviewer_correct = sum(bool(row["Reviewer是否正确"]) for row in details)
    lost = sum(bool(row["Reviewer在oracle可正确时选错"]) for row in details)
    rescued = sum(bool(row["Reviewer在候选均错时补救"]) for row in details)
    summary = {
        "样本数": total,
        "Scientist第1候选宏类别accuracy": sum(bool(row["Scientist各候选是否正确"][0]) for row in details) / total,
        "Scientist第2候选宏类别accuracy": sum(bool(row["Scientist各候选是否正确"][1]) for row in details) / total,
        "Scientist第3候选宏类别accuracy": sum(bool(row["Scientist各候选是否正确"][2]) for row in details) / total,
        "Scientist_Top3宏类别oracle_accuracy": oracle / total,
        "Reviewer宏类别accuracy": reviewer_correct / total,
        "Reviewer相对Top3_oracle损失样本数": lost,
        "Reviewer在Top3均错时补救样本数": rescued,
        "Top3_oracle与Reviewer净差样本数": oracle - reviewer_correct,
        "平均Scientist候选类别数": sum(int(row["Scientist候选类别数"]) for row in details) / total,
        "瓶颈判据": "若 Top3 类别 oracle 显著高于 Reviewer，说明 Reviewer 选择损失存在；但 Scientist oracle 自身的绝对上限仍决定生成瓶颈是否同时存在。",
    }
    写入_jsonl(directory / "官方类别阶段分解逐样本.jsonl", details)
    path = directory / "官方类别阶段分解汇总.json"
    if path.exists():
        raise RuntimeError(f"为避免覆盖旧产物，已停止：{path}")
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def 谱签名(row: dict[str, Any]) -> str:
    values = [
        归一化(value)
        for value in list(row.get("partial_molecules") or []) + list(row.get("missing_molecules") or [])
        if 归一化(value)
    ]
    return "|".join(sorted(set(values)))


def 完整谱聚类分折(rows: list[dict[str, Any]], folds: int) -> dict[int, int]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[谱签名(row)].append(index)
    sizes = [0] * folds
    mapping: dict[int, int] = {}
    for _, indices in sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0])):
        fold = min(range(folds), key=lambda value: (sizes[value], value))
        sizes[fold] += len(indices)
        for index in indices:
            mapping[index] = fold
    return mapping


def 分析_mpc(输出根目录: Path) -> None:
    agent = 加载模块("mpc_agent", 项目根目录 / "code/Only-Deepseek/optimized_agent.py")
    rows = 读取_jsonl(项目根目录 / "results/splits/mpc/train.jsonl")
    folds = 5
    mapping = 完整谱聚类分折(rows, folds)
    multipliers = (1.0, 1.25, 1.5, 2.0, 3.0)
    details: list[dict[str, Any]] = []
    for fold in range(folds):
        fit = [row for index, row in enumerate(rows) if mapping[index] != fold]
        held = [row for index, row in enumerate(rows) if mapping[index] == fold]
        model = agent.MPCStructureModel(
            fit, embeddings=None, ablation="full",
            db_path=项目根目录 / "data/raw/flavordb.db",
            calibrate_residuals=False,
        )
        for row in held:
            gold = {归一化(value) for value in row.get("missing_molecules") or [] if 归一化(value)}
            n = len(gold)
            if n <= 0:
                continue
            query = {
                "id": f"{row.get('id')}:第十一轮H1深度诊断",
                "target_food": row.get("target_food"),
                "partial_molecules": list(row.get("partial_molecules") or []),
                "n": n,
            }
            ranked_items = model._boundary_training_items(query, exclude_train_index=None, limit=len(model.training_universe))
            ranking = [归一化(item.get("molecule")) for item in ranked_items if 归一化(item.get("molecule"))]
            record: dict[str, Any] = {
                "样本编号": str(row.get("id")), "折": fold + 1, "N": n,
                "训练任务内候选数": len(ranking), "gold分子数": len(gold),
                "gold在该折训练候选域内数": len(gold & set(ranking)),
            }
            for multiplier in multipliers:
                k = min(len(ranking), max(n, int(math.ceil(n * multiplier))))
                hits = len(gold & set(ranking[:k]))
                label = str(multiplier).replace(".", "_")
                record[f"K_{label}"] = k
                record[f"命中_{label}"] = hits
            details.append(record)
        print(f"[MPC H1深度诊断] 完成第 {fold + 1}/{folds} 折", flush=True)
    total_gold = sum(int(row["gold分子数"]) for row in details)
    summary: dict[str, Any] = {
        "协议": "exact-profile clustered 5-fold OOF",
        "候选边界": "仅各折训练 processed MPC task profiles 中出现的分子；无 FlavorDB 额外候选",
        "样本数": len(details), "gold分子位置数": total_gold,
        "训练任务内候选域召回": sum(int(row["gold在该折训练候选域内数"]) for row in details) / total_gold,
        "各深度": {},
    }
    for multiplier in multipliers:
        label = str(multiplier).replace(".", "_")
        hits = sum(int(row[f"命中_{label}"]) for row in details)
        summary["各深度"][f"{multiplier}N"] = {
            "命中数": hits, "micro_recall": hits / total_gold,
            "相对N新增命中": hits - sum(int(row["命中_1_0"]) for row in details),
        }
    directory = 输出根目录 / "MPC_H1排名深度诊断"
    写入_jsonl(directory / "训练OOF逐样本.jsonl", details)
    summary_path = directory / "训练OOF指标汇总.json"
    if summary_path.exists():
        raise RuntimeError(f"为避免覆盖旧产物，已停止：{summary_path}")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("动作", choices=["准备MFP", "分析MFP", "汇总MFP类别", "分析MPC"])
    parser.add_argument("--输出根目录", type=Path, default=默认输出根目录)
    args = parser.parse_args()
    if args.动作 == "准备MFP":
        准备_mfp(args.输出根目录)
    elif args.动作 == "分析MFP":
        分析_mfp(args.输出根目录)
    elif args.动作 == "汇总MFP类别":
        汇总_mfp类别(args.输出根目录)
    else:
        分析_mpc(args.输出根目录)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
