#!/usr/bin/env python3
"""第三轮 MPC：固定 Scientist Bank，只给 Reviewer 增加 UniMol 集合证据。"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[1]
第一轮脚本 = 项目根目录 / "scripts/第一轮_MPC_查询条件集合审查器验证.py"
第二轮脚本 = 项目根目录 / "scripts/第二轮_MPC_面向H1的两阶段集合审查器验证.py"
默认输出目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第三轮/MPC_UniMol集合证据审查器"


def 加载脚本(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 向量统计(names: set[str], reference: Any | None, embeddings: Any) -> list[float]:
    vectors = [embeddings.vector(name) for name in names]
    vectors = [vector for vector in vectors if vector is not None]
    coverage = len(vectors) / max(1, len(names))
    if not vectors or reference is None:
        return [0.0, 0.0, 0.0, coverage]
    similarities = sorted(float(vector @ reference) for vector in vectors)
    return [
        sum(similarities) / len(similarities),
        similarities[0],
        similarities[-1],
        coverage,
    ]


def 集合证据(
    proposal: set[str], h1: set[str], partial: set[str], embeddings: Any,
) -> tuple[list[float], dict[str, Any]]:
    partial_profile = embeddings.profile(sorted(partial))
    h1_profile = embeddings.profile(sorted(h1))
    proposal_profile = embeddings.profile(sorted(proposal))
    added = proposal - h1
    removed = h1 - proposal
    proposal_alignment = embeddings.cosine(proposal_profile, partial_profile)
    h1_alignment = embeddings.cosine(h1_profile, partial_profile)
    proposal_to_h1 = embeddings.cosine(proposal_profile, h1_profile)
    added_stats = 向量统计(added, partial_profile, embeddings)
    removed_stats = 向量统计(removed, partial_profile, embeddings)
    proposal_stats = 向量统计(proposal, partial_profile, embeddings)
    vector = [
        proposal_alignment,
        proposal_alignment - h1_alignment,
        proposal_to_h1,
        1.0 - proposal_to_h1,
        *added_stats,
        *removed_stats,
        added_stats[0] - removed_stats[0],
        *proposal_stats,
        len(added) / max(1, len(h1)),
        len(removed) / max(1, len(h1)),
        math.log1p(len(partial)) / math.log1p(500),
    ]
    ledger = {
        "候选对部分谱余弦": proposal_alignment,
        "相对H1部分谱余弦变化": proposal_alignment - h1_alignment,
        "候选对H1余弦": proposal_to_h1,
        "新增分子数": len(added),
        "删除分子数": len(removed),
        "新增分子UniMol覆盖率": added_stats[3],
        "删除分子UniMol覆盖率": removed_stats[3],
        "候选分子UniMol覆盖率": proposal_stats[3],
        "新增分子对部分谱平均相似度": added_stats[0],
        "删除分子对部分谱平均相似度": removed_stats[0],
    }
    return vector, ledger


def 增强记录(record: dict[str, Any], source: dict[str, Any], embeddings: Any, base: Any) -> dict[str, Any]:
    result = copy.deepcopy(record)
    h1 = {base.归一化(x) for x in result.get("H1") or [] if base.归一化(x)}
    partial = {base.归一化(x) for x in source.get("partial_molecules") or [] if base.归一化(x)}
    for candidate in result["候选"]:
        proposal = {base.归一化(x) for x in candidate.get("提案") or [] if base.归一化(x)}
        evidence, ledger = 集合证据(proposal, h1, partial, embeddings)
        candidate["基础特征"] = list(candidate["特征"])
        candidate["UniMol集合特征"] = evidence
        candidate["UniMol集合账本"] = ledger
    result["部分分子数"] = len(partial)
    return result


def 特征视图(records: list[dict[str, Any]], use_unimol: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        cloned = dict(record)
        cloned["候选"] = []
        for candidate in record["候选"]:
            item = dict(candidate)
            item["特征"] = list(candidate["基础特征"])
            if use_unimol:
                item["特征"].extend(candidate["UniMol集合特征"])
            cloned["候选"].append(item)
        result.append(cloned)
    return result


def 生成OOF查询(
    rows: list[dict[str, Any]], folds: int, agent: Any, db: Path,
    embeddings: Any, base: Any, label: str,
) -> list[dict[str, Any]]:
    mapping = base.分折(rows, folds)
    records: list[dict[str, Any]] = []
    for fold in range(folds):
        fit = [row for index, row in enumerate(rows) if mapping[index] != fold]
        held = [row for index, row in enumerate(rows) if mapping[index] == fold]
        model = agent.MPCStructureModel(fit, None, "full", db, calibrate_residuals=False)
        print(f"[{label}] 第 {fold + 1}/{folds} 折：训练 {len(fit)}，验证 {len(held)}", flush=True)
        for source in held:
            record = base.构造查询记录(model, source, fold)
            if record is not None:
                records.append(增强记录(record, source, embeddings, base))
    return records


def 内层训练预测(records: list[dict[str, Any]], inner_folds: int, reviewer: Any, use_unimol: bool) -> tuple[Any, float]:
    predictions: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for fold in range(inner_folds):
        fit = 特征视图([x for x in records if x["折"] != fold], use_unimol)
        held = 特征视图([x for x in records if x["折"] == fold], use_unimol)
        model = reviewer.拟合(fit)
        predictions.extend((record, reviewer.预测(model, record)) for record in held)
    threshold = reviewer.选择门槛(predictions)
    final_model = reviewer.拟合(特征视图(records, use_unimol))
    return final_model, threshold


def 选中候选(record: dict[str, Any], prediction: dict[str, Any], threshold: float, reviewer: Any) -> tuple[int, dict[str, Any]]:
    index = 0 if math.isinf(threshold) else reviewer.选择索引(prediction, threshold)
    return index, record["候选"][index]


def bootstrap下界(values: list[float], seed: str) -> float:
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(5000))
    return means[max(0, int(0.025 * len(means)) - 1)]


def 汇总(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    gains = [float(x[f"{prefix}官能团F1增益"]) for x in rows]
    molecule_gains = [float(x[f"{prefix}具体分子F1增益"]) for x in rows]
    fold_gains = {
        str(fold): sum(float(x[f"{prefix}官能团F1增益"]) for x in rows if x["外层折"] == fold)
        / max(1, sum(1 for x in rows if x["外层折"] == fold))
        for fold in range(1, 6)
    }
    return {
        "平均官能团F1增益": sum(gains) / len(gains),
        "官能团F1增益bootstrap_95%下界": bootstrap下界(gains, f"第三轮MPC-{prefix}"),
        "平均具体分子F1增益": sum(molecule_gains) / len(molecule_gains),
        "胜负平": {
            "胜": sum(x > 1e-12 for x in gains),
            "负": sum(x < -1e-12 for x in gains),
            "平": sum(abs(x) <= 1e-12 for x in gains),
        },
        "各外层折官能团增益": fold_gains,
    }


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("为保证 v16 Bank 候选排序可复现，必须使用 PYTHONHASHSEED=0 启动实验")
    parser = argparse.ArgumentParser()
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument("--外层折数", type=int, default=5)
    parser.add_argument("--内层折数", type=int, default=4)
    args = parser.parse_args()
    args.输出目录.mkdir(parents=True, exist_ok=True)
    base = 加载脚本("第三轮MPC基础框架", 第一轮脚本)
    reviewer = 加载脚本("第三轮MPC两阶段审查器", 第二轮脚本)
    agent = base.加载模块(项目根目录 / "code/Only-Deepseek/optimized_agent.py")
    db = 项目根目录 / "data/raw/flavordb.db"
    rows = base.读取_jsonl(项目根目录 / "results/splits/mpc/train.jsonl")
    embeddings = agent.EmbeddingStore(项目根目录 / "data/structure/unimol/unimol_embeddings.npz")
    protocol = {
        "实验名称": "MPC UniMol集合证据审查器",
        "唯一变化": "在冻结两阶段Reviewer的基础特征后追加候选分子集合相对部分已知分子谱的UniMol结构证据",
        "固定部分": "H1、v16 Bank20、exact-N、候选生成、两阶段Reviewer结构、门槛选择规则",
        "严格嵌套OOF": True,
        "外层折数": args.外层折数,
        "内层折数": args.内层折数,
        "正式测试集是否读取": False,
        "API调用次数": 0,
        "Gold是否进入特征": False,
        "Python哈希种子": 0,
    }
    (args.输出目录 / "冻结实验方案.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outer_map = base.分折(rows, args.外层折数)
    all_results: list[dict[str, Any]] = []
    for outer in range(args.外层折数):
        outer_fit = [row for index, row in enumerate(rows) if outer_map[index] != outer]
        outer_held = [row for index, row in enumerate(rows) if outer_map[index] == outer]
        inner_records = 生成OOF查询(
            outer_fit, args.内层折数, agent, db, embeddings, base,
            f"第三轮外层{outer + 1}的内层",
        )
        baseline_model, baseline_threshold = 内层训练预测(inner_records, args.内层折数, reviewer, False)
        unimol_model, unimol_threshold = 内层训练预测(inner_records, args.内层折数, reviewer, True)
        outer_model = agent.MPCStructureModel(outer_fit, None, "full", db, calibrate_residuals=False)
        fold_results: list[dict[str, Any]] = []
        for source in outer_held:
            record = base.构造查询记录(outer_model, source, outer)
            if record is None:
                continue
            record = 增强记录(record, source, embeddings, base)
            baseline_record = 特征视图([record], False)[0]
            unimol_record = 特征视图([record], True)[0]
            baseline_prediction = reviewer.预测(baseline_model, baseline_record)
            unimol_prediction = reviewer.预测(unimol_model, unimol_record)
            baseline_index, baseline_choice = 选中候选(baseline_record, baseline_prediction, baseline_threshold, reviewer)
            unimol_index, unimol_choice = 选中候选(unimol_record, unimol_prediction, unimol_threshold, reviewer)
            fold_results.append({
                "样本编号": record["样本编号"],
                "外层折": outer + 1,
                "目标食物": record["目标食物"],
                "候选数量": len(record["候选"]),
                "无UniMol门槛": None if math.isinf(baseline_threshold) else baseline_threshold,
                "无UniMol选择": baseline_choice["候选类型"],
                "无UniMol官能团F1增益": baseline_choice["真实官能团增益"],
                "无UniMol具体分子F1增益": baseline_choice["真实分子F1增益"],
                "UniMol门槛": None if math.isinf(unimol_threshold) else unimol_threshold,
                "UniMol选择": unimol_choice["候选类型"],
                "UniMol官能团F1增益": unimol_choice["真实官能团增益"],
                "UniMol具体分子F1增益": unimol_choice["真实分子F1增益"],
                "UniMol选择账本": record["候选"][unimol_index]["UniMol集合账本"],
                "无UniMol选择索引": baseline_index,
                "UniMol选择索引": unimol_index,
            })
        all_results.extend(fold_results)
        baseline_gain = sum(float(x["无UniMol官能团F1增益"]) for x in fold_results) / max(1, len(fold_results))
        unimol_gain = sum(float(x["UniMol官能团F1增益"]) for x in fold_results) / max(1, len(fold_results))
        print(f"[MPC外层 {outer + 1}/{args.外层折数}] 无UniMol={baseline_gain:.6f}，UniMol={unimol_gain:.6f}", flush=True)
    baseline_summary = 汇总(all_results, "无UniMol")
    unimol_summary = 汇总(all_results, "UniMol")
    paired = [
        float(x["UniMol官能团F1增益"]) - float(x["无UniMol官能团F1增益"])
        for x in all_results
    ]
    paired_molecule = [
        float(x["UniMol具体分子F1增益"]) - float(x["无UniMol具体分子F1增益"])
        for x in all_results
    ]
    comparison = {
        "平均官能团F1增益差": sum(paired) / len(paired),
        "配对bootstrap_95%下界": bootstrap下界(paired, "第三轮MPC配对"),
        "平均具体分子F1增益差": sum(paired_molecule) / len(paired_molecule),
        "相对无UniMol改进恶化不变": {
            "改进": sum(x > 1e-12 for x in paired),
            "恶化": sum(x < -1e-12 for x in paired),
            "不变": sum(abs(x) <= 1e-12 for x in paired),
        },
    }
    wins = unimol_summary["胜负平"]
    fold_values = list(unimol_summary["各外层折官能团增益"].values())
    admitted = bool(
        unimol_summary["平均官能团F1增益"] > 0.0075
        and comparison["平均官能团F1增益差"] > 0
        and comparison["配对bootstrap_95%下界"] > 0
        and sum(x >= 0 for x in fold_values) >= 4
        and wins["胜"] >= 2 * max(1, wins["负"])
        and comparison["平均具体分子F1增益差"] >= 0
    )
    summary = {
        "协议": protocol,
        "冻结无UniMol两阶段Reviewer": baseline_summary,
        "UniMol集合证据Reviewer": unimol_summary,
        "配对比较": comparison,
        "是否通过准入": admitted,
    }
    (args.输出目录 / "逐样本结果.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in all_results),
        encoding="utf-8",
    )
    (args.输出目录 / "指标汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    conclusion = {
        "结论": "通过：可进入下一阶段" if admitted else "未通过：不接入正式代理",
        "原因": "必须同时获得稳定官能团增益、相对冻结Reviewer的正配对下界、折级稳定性以及具体分子不下降。",
    }
    (args.输出目录 / "结论说明.json").write_text(json.dumps(conclusion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
