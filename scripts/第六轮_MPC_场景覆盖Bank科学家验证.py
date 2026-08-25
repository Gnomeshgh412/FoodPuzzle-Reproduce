#!/usr/bin/env python3
"""第六轮 MPC：只替换 v16 两步终态到 Bank20 的场景覆盖选择器。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[1]
基础脚本 = 项目根目录 / "scripts/第一轮_MPC_查询条件集合审查器验证.py"
默认输出目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第六轮/MPC_场景覆盖Bank科学家"


def 加载基础模块() -> Any:
    spec = importlib.util.spec_from_file_location("第六轮MPC基础", 基础脚本)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载基础脚本：{基础脚本}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 平衡分配(clusters: list[list[int]], folds: int) -> dict[int, int]:
    sizes = [0] * folds
    mapping: dict[int, int] = {}
    for indices in sorted(clusters, key=lambda values: (-len(values), values[0])):
        fold = min(range(folds), key=lambda value: (sizes[value], value))
        sizes[fold] += len(indices)
        for index in indices:
            mapping[index] = fold
    return mapping


def 近重复分折(rows: list[dict[str, Any]], folds: int, normalize: Any, threshold: float = 0.9) -> dict[int, int]:
    profiles = [
        {
            normalize(value)
            for value in (row.get("partial_molecules") or []) + (row.get("missing_molecules") or [])
            if normalize(value)
        }
        for row in rows
    ]
    parent = list(range(len(rows)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            union_size = len(profiles[left] | profiles[right])
            similarity = len(profiles[left] & profiles[right]) / union_size if union_size else 1.0
            if similarity >= threshold:
                union(left, right)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        grouped[find(index)].append(index)
    return 平衡分配(list(grouped.values()), folds)


def 构造高效完整候选(model: Any, query: dict[str, Any], normalize: Any) -> list[dict[str, Any]]:
    """与主程序 H1/v16 所需字段等价，但跳过无关的 3D/感知字段。"""
    partial = {normalize(value) for value in query.get("partial_molecules") or [] if normalize(value)}
    retrieved = model._build_retrieved_support(query, exclude_train_index=None, top_k=10)
    idf_retrieved, idf_counts = model._build_idf_retrieved_support(query, exclude_train_index=None, top_k=15)
    context = model._build_query_context(query)
    training = set(model.training_universe)
    primary_by_key: dict[str, list[float]] = {}
    for candidate in model.training_universe:
        if candidate in partial:
            continue
        values = model._query_features(
            query,
            candidate,
            retrieved,
            excluded_profile=None,
            expected_attributes=None,
            query_context=context,
        )
        primary_by_key[candidate] = [values[index] for index in model.PRIMARY_FEATURE_INDICES]
    candidates = [candidate for candidate in model.universe if candidate not in partial]
    primary = [primary_by_key.get(candidate, [0.0, 0.0, 0.0, 0.0]) for candidate in candidates]
    if model.ranker is not None and candidates:
        occurrence_scores = [float(value) for value in model.ranker.decision_function(primary)]
    else:
        occurrence_scores = [0.20 * x[0] + 0.30 * x[1] + 0.20 * x[2] + 0.30 * x[3] for x in primary]
    group_demand = model._predict_group_demand(query)
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError("第六轮高效候选构造需要 numpy") from exc
    if not hasattr(model, "_第六轮官能团矩阵"):
        group_index = {group: index for index, group in enumerate(model.functional_group_vocabulary)}
        matrix = np.zeros((len(model.universe), len(group_index)), dtype=np.float32)
        for row_index, candidate in enumerate(model.universe):
            for group in model.functional_group_sets.get(candidate, set()):
                column = group_index.get(group)
                if column is not None:
                    matrix[row_index, column] = 1.0
        model._第六轮官能团矩阵 = matrix
        model._第六轮候选行号 = {candidate: index for index, candidate in enumerate(model.universe)}
    coefficients = np.asarray([2.0 * value - 1.0 for value in group_demand], dtype=np.float32)
    all_group_scores = (
        model._第六轮官能团矩阵 @ coefficients / max(1, len(group_demand))
        if group_demand else np.zeros(len(model.universe), dtype=np.float32)
    )
    items: list[dict[str, Any]] = []
    for candidate, values, occurrence_score in zip(candidates, primary, occurrence_scores):
        group_score = float(all_group_scores[model._第六轮候选行号[candidate]])
        items.append({
            "molecule": model.display_names[candidate],
            "occurrence_score": occurrence_score,
            "frequency_prior": values[0],
            "cooccurrence": values[1],
            "cooccurrence_max": values[2],
            "retrieved_profile_support": retrieved.get(candidate, 0.0),
            "idf_retrieved_support": idf_retrieved.get(candidate, 0.0),
            "idf_retrieved_profile_count": idf_counts.get(candidate, 0),
            "functional_group_demand_score": group_score,
            "来自训练谱": candidate in training,
        })
    items.sort(key=lambda item: (-float(item["occurrence_score"]), normalize(item["molecule"])))
    for rank, item in enumerate(items, 1):
        item["occurrence_rank"] = rank
    return items


def 构造未压缩终态(
    model: Any, items: list[dict[str, Any]], n: int,
    posterior: dict[str, dict[int, float]], normalize: Any,
    stable_unique: Any, addition_pool_size: int = 100,
    first_step_beam_size: int = 8,
) -> dict[str, Any]:
    ordered = sorted(items, key=lambda item: (int(item.get("occurrence_rank") or 10**9), normalize(item.get("molecule"))))
    h1 = stable_unique(normalize(item.get("molecule")) for item in ordered[:n] if normalize(item.get("molecule")))
    baseline_groups = model._functional_group_set(set(h1))
    baseline_expected = model._expected_group_f1(baseline_groups, posterior)
    empty = {"h1": h1, "h1_groups": sorted(baseline_groups), "actions": [], "one_step": [], "two_step": [], "additions": []}
    if n <= 0 or len(h1) != n or len(ordered) <= n:
        return empty
    h1_set = set(h1)
    outside = [item for item in ordered if normalize(item.get("molecule")) not in h1_set]

    def ranked_keys(field: str) -> list[str]:
        ranked = sorted(outside, key=lambda item: (-float(item.get(field) or 0.0), int(item.get("occurrence_rank") or 10**9), normalize(item.get("molecule"))))
        return stable_unique(normalize(item.get("molecule")) for item in ranked)

    occurrence = stable_unique(normalize(item.get("molecule")) for item in outside)
    source_lists = [occurrence, ranked_keys("idf_retrieved_support"), ranked_keys("retrieved_profile_support"), ranked_keys("functional_group_demand_score")]
    additions = occurrence[: min(70, addition_pool_size)]
    for source in source_lists[1:]:
        for key in source[:10]:
            if key not in additions:
                additions.append(key)
            if len(additions) >= addition_pool_size:
                break
        if len(additions) >= addition_pool_size:
            break
    for key in occurrence:
        if key not in additions:
            additions.append(key)
        if len(additions) >= addition_pool_size:
            break
    additions = additions[:addition_pool_size]
    molecule_groups = {key: set(model.functional_group_sets.get(key, set())) for key in set(h1) | set(additions)}

    def group_counts(proposal: list[str]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for key in proposal:
            counts.update(molecule_groups.get(key, set()))
        return counts

    def exchanged_signature(counts: Counter[str], remove_key: str, add_key: str) -> tuple[str, ...]:
        updated = counts.copy()
        updated.subtract(molecule_groups.get(remove_key, set()))
        updated.update(molecule_groups.get(add_key, set()))
        return tuple(sorted(group for group, count in updated.items() if count > 0))

    def make_action(proposal: list[str], path: list[dict[str, str]], signature: tuple[str, ...]) -> dict[str, Any]:
        groups = set(signature)
        expected = model._expected_group_f1(groups, posterior)
        return {
            "depth": len(path),
            "path": path,
            "remove_keys": [step["remove_key"] for step in path],
            "add_keys": [step["add_key"] for step in path],
            "remove_key": path[-1]["remove_key"],
            "add_key": path[-1]["add_key"],
            "predicted_expected_f1": expected,
            "predicted_expected_f1_gain": expected - baseline_expected,
            "removed_groups": sorted(baseline_groups - groups),
            "added_groups": sorted(groups - baseline_groups),
            "proposal_groups": sorted(groups),
            "proposal": proposal,
        }

    one_by_signature: dict[tuple[str, ...], dict[str, Any]] = {}
    baseline_counts = group_counts(h1)
    for remove_key in h1:
        for add_key in additions:
            signature = exchanged_signature(baseline_counts, remove_key, add_key)
            previous = one_by_signature.get(signature)
            path_key = (remove_key, add_key)
            previous_key = (previous["remove_keys"][0], previous["add_keys"][0]) if previous is not None else None
            if previous is None or path_key < previous_key:
                proposal = list(h1)
                proposal[proposal.index(remove_key)] = add_key
                one_by_signature[signature] = make_action(proposal, [{"remove_key": remove_key, "add_key": add_key}], signature)
    one_step = list(one_by_signature.values())
    first_beam = model._v16_select_quality_diverse(one_step, first_step_beam_size, quality_quota=min(4, first_step_beam_size), diversity_weight=0.25)
    two_by_signature: dict[tuple[str, ...], dict[str, Any]] = {}
    for first in first_beam:
        current = list(first["proposal"])
        counts = group_counts(current)
        first_remove = first["remove_keys"][0]
        first_add = first["add_keys"][0]
        for remove_key in current:
            if remove_key == first_add:
                continue
            for add_key in additions:
                if add_key in current or add_key == first_remove:
                    continue
                signature = exchanged_signature(counts, remove_key, add_key)
                path = list(first["path"]) + [{"remove_key": remove_key, "add_key": add_key}]
                path_key = (tuple(step["remove_key"] for step in path), tuple(step["add_key"] for step in path))
                previous = two_by_signature.get(signature)
                previous_key = (tuple(previous["remove_keys"]), tuple(previous["add_keys"])) if previous is not None else None
                if previous is None or path_key < previous_key:
                    proposal = list(current)
                    proposal[proposal.index(remove_key)] = add_key
                    two_by_signature[signature] = make_action(proposal, path, signature)
    combined = dict(one_by_signature)
    for signature, action in two_by_signature.items():
        previous = combined.get(signature)
        if previous is None or float(action["predicted_expected_f1_gain"]) > float(previous["predicted_expected_f1_gain"]):
            combined[signature] = action
    return {
        "h1": h1,
        "h1_groups": sorted(baseline_groups),
        "actions": list(combined.values()),
        "one_step": one_step,
        "two_step": list(two_by_signature.values()),
        "additions": additions,
    }


def 构造训练场景(model: Any, query: dict[str, Any], normalize: Any, top_k: int = 15) -> list[tuple[set[str], float]]:
    merged: defaultdict[tuple[str, ...], float] = defaultdict(float)
    neighbours = model._retrieved_profiles(query, top_k=top_k, exclude_train_index=None)
    for score, index in neighbours:
        missing = {normalize(value) for value in model.rows[index].get("missing_molecules") or [] if normalize(value)}
        groups = model._functional_group_set(missing)
        if groups:
            merged[tuple(sorted(groups))] += max(0.0, float(score))
    if not merged:
        return []
    total = sum(merged.values())
    if total <= 1e-12:
        weight = 1.0 / len(merged)
        return [(set(signature), weight) for signature in sorted(merged)]
    return [(set(signature), value / total) for signature, value in sorted(merged.items())]


def 场景覆盖选择(
    actions: list[dict[str, Any]], h1_groups: set[str],
    scenarios: list[tuple[set[str], float]], limit: int,
    f1: Any,
) -> list[dict[str, Any]]:
    if limit <= 0 or not actions:
        return []
    base_scores = [f1(h1_groups, groups) for groups, _ in scenarios]
    utility: dict[int, list[float]] = {}
    for action in actions:
        groups = set(action.get("proposal_groups") or [])
        utility[id(action)] = [max(0.0, f1(groups, scenario) - baseline) for (scenario, _), baseline in zip(scenarios, base_scores)]
    selected: list[dict[str, Any]] = []
    covered = [0.0] * len(scenarios)
    remaining = list(actions)
    while remaining and len(selected) < limit:
        best: tuple[tuple[float, float, int, tuple[str, ...], tuple[str, ...]], dict[str, Any]] | None = None
        for action in remaining:
            values = utility[id(action)]
            marginal = sum(weight * max(0.0, value - old) for value, old, (_, weight) in zip(values, covered, scenarios))
            key = (
                marginal,
                float(action.get("predicted_expected_f1_gain") or 0.0),
                -int(action.get("depth") or 1),
                tuple(action.get("remove_keys") or []),
                tuple(action.get("add_keys") or []),
            )
            if best is None or key > best[0]:
                best = (key, action)
        if best is None:
            break
        action = best[1]
        selected.append(action)
        covered = [max(old, value) for old, value in zip(covered, utility[id(action)])]
        remaining.remove(action)
    return selected


def oracle_gain(actions: list[dict[str, Any]], h1_groups: set[str], gold_groups: set[str], f1: Any) -> float:
    baseline = f1(h1_groups, gold_groups)
    return max([0.0] + [f1(set(action.get("proposal_groups") or []), gold_groups) - baseline for action in actions])


def bootstrap下界(values: list[float], seed: str) -> float:
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(5000))
    return means[max(0, int(0.025 * len(means)) - 1)]


def 运行协议(
    rows: list[dict[str, Any]], mapping: dict[int, int], folds: int,
    protocol_name: str, agent: Any, db: Path, base: Any,
    per_fold_limit: int = 0,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for fold in range(folds):
        fit = [row for index, row in enumerate(rows) if mapping[index] != fold]
        held = [row for index, row in enumerate(rows) if mapping[index] == fold]
        if per_fold_limit > 0:
            held = held[:per_fold_limit]
        model = agent.MPCStructureModel(fit, None, "full", db, calibrate_residuals=False)
        for source in held:
            gold = {base.归一化(value) for value in source.get("missing_molecules") or [] if base.归一化(value)}
            n = len(gold)
            query = {"id": f"{source.get('id')}:第六轮", "target_food": source.get("target_food"), "partial_molecules": list(source.get("partial_molecules") or []), "n": n}
            items = 构造高效完整候选(model, query, base.归一化)
            posterior = model._predict_group_cardinality_posterior(query)
            raw = 构造未压缩终态(model, items, n, posterior, base.归一化, agent.stable_unique)
            actions = raw["actions"]
            h1_groups = set(raw["h1_groups"])
            old_bank = model._v16_select_quality_diverse(actions, 20, quality_quota=10, diversity_weight=0.20)
            old_slate = model._v16_select_quality_diverse(old_bank, 5, quality_quota=2, diversity_weight=0.35)
            scenarios = 构造训练场景(model, query, base.归一化, top_k=15)
            new_bank = 场景覆盖选择(actions, h1_groups, scenarios, 20, base.集合F1)
            new_slate = model._v16_select_quality_diverse(new_bank, 5, quality_quota=2, diversity_weight=0.35)
            gold_groups = model._functional_group_set(gold)
            raw_gain = oracle_gain(actions, h1_groups, gold_groups, base.集合F1)
            old_bank_gain = oracle_gain(old_bank, h1_groups, gold_groups, base.集合F1)
            new_bank_gain = oracle_gain(new_bank, h1_groups, gold_groups, base.集合F1)
            old_slate_gain = oracle_gain(old_slate, h1_groups, gold_groups, base.集合F1)
            new_slate_gain = oracle_gain(new_slate, h1_groups, gold_groups, base.集合F1)
            all_proposals = [action.get("proposal") or [] for action in new_bank]
            details.append({
                "协议": protocol_name,
                "折": fold + 1,
                "样本编号": str(source.get("id")),
                "目标食物": source.get("target_food"),
                "N": n,
                "训练场景数": len(scenarios),
                "一步唯一终态数": len(raw["one_step"]),
                "两步唯一终态数": len(raw["two_step"]),
                "未压缩唯一终态数": len(actions),
                "旧Bank动作数": len(old_bank),
                "新Bank动作数": len(new_bank),
                "未压缩Oracle增益": raw_gain,
                "旧Bank20_Oracle增益": old_bank_gain,
                "场景覆盖Bank20_Oracle增益": new_bank_gain,
                "旧Slate5_Oracle增益": old_slate_gain,
                "新Slate5_Oracle增益": new_slate_gain,
                "新旧Bank增益差": new_bank_gain - old_bank_gain,
                "新Bank相对未压缩regret": raw_gain - new_bank_gain,
                "新Bank全部exact_N": all(len(proposal) == n and len(set(proposal)) == n for proposal in all_proposals),
            })
        print(f"[MPC第六轮 {protocol_name} {fold + 1}/{folds}] 已完成 {len(held)} 条", flush=True)
    return details


def 汇总协议(details: list[dict[str, Any]], protocol: str, folds: int) -> dict[str, Any]:
    selected = [row for row in details if row["协议"] == protocol]
    differences = [float(row["新旧Bank增益差"]) for row in selected]
    raw = sum(float(row["未压缩Oracle增益"]) for row in selected) / len(selected)
    old_bank = sum(float(row["旧Bank20_Oracle增益"]) for row in selected) / len(selected)
    new_bank = sum(float(row["场景覆盖Bank20_Oracle增益"]) for row in selected) / len(selected)
    old_slate = sum(float(row["旧Slate5_Oracle增益"]) for row in selected) / len(selected)
    new_slate = sum(float(row["新Slate5_Oracle增益"]) for row in selected) / len(selected)
    fold_differences = {
        str(fold): sum(float(row["新旧Bank增益差"]) for row in selected if row["折"] == fold)
        / max(1, sum(row["折"] == fold for row in selected))
        for fold in range(1, folds + 1)
    }
    return {
        "样本数": len(selected),
        "平均训练场景数": sum(int(row["训练场景数"]) for row in selected) / len(selected),
        "平均未压缩唯一终态数": sum(int(row["未压缩唯一终态数"]) for row in selected) / len(selected),
        "未压缩Oracle增益": raw,
        "旧Bank20_Oracle增益": old_bank,
        "场景覆盖Bank20_Oracle增益": new_bank,
        "旧Slate5_Oracle增益": old_slate,
        "新Slate5_Oracle增益": new_slate,
        "新Bank相对旧Bank平均增益": sum(differences) / len(differences),
        "配对bootstrap_95%下界": bootstrap下界(differences, f"第六轮-{protocol}"),
        "新Bank捕获未压缩Oracle比例": new_bank / raw if raw > 0 else 0.0,
        "新Bank正收益查询数": sum(float(row["场景覆盖Bank20_Oracle增益"]) > 1e-12 for row in selected),
        "未压缩正收益查询数": sum(float(row["未压缩Oracle增益"]) > 1e-12 for row in selected),
        "新Slate捕获新Bank比例": new_slate / new_bank if new_bank > 0 else 0.0,
        "新旧胜负平": {"胜": sum(x > 1e-12 for x in differences), "负": sum(x < -1e-12 for x in differences), "平": sum(abs(x) <= 1e-12 for x in differences)},
        "各折新旧Bank增益差": fold_differences,
        "exact_N通过": all(bool(row["新Bank全部exact_N"]) for row in selected),
    }


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("必须使用 PYTHONHASHSEED=0 启动实验")
    parser = argparse.ArgumentParser()
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument("--折数", type=int, default=5)
    parser.add_argument("--跳过近重复压力测试", action="store_true")
    parser.add_argument("--每折最多样本", type=int, default=0, help="仅供冒烟检查；正式实验必须为0")
    args = parser.parse_args()
    args.输出目录.mkdir(parents=True, exist_ok=True)
    base = 加载基础模块()
    agent = base.加载模块(项目根目录 / "code/Only-Deepseek/optimized_agent.py")
    rows = base.读取_jsonl(项目根目录 / "results/splits/mpc/train.jsonl")
    db = 项目根目录 / "data/raw/flavordb.db"
    frozen = {
        "实验名称": "MPC场景覆盖Bank科学家验证",
        "唯一变化": "将v16未压缩两步终态到Bank20的expected-F1加几何多样性选择，替换为训练近邻完整缺失官能团场景上的正效用最大覆盖",
        "冻结部分": "H1、全H1 remove、add100、beam8、两步终态、Bank预算20、Slate5选择器、Reviewer、执行器、无UniMol",
        "场景数": 15,
        "场景来源": "仅外层训练折；按现有food+partial profile检索；场景为训练查询完整missing官能团集合",
        "场景权重": "非负检索分数归一化；相同场景合并；总权重为零时均匀",
        "场景效用": "max(0, F1(proposal,scenario)-F1(H1,scenario))",
        "Bank选择": "固定20步最大加权边际场景覆盖；并列依次按predicted expected-F1、浅深度、路径字典序",
        "主协议": "exact full-profile clustered 5-fold OOF",
        "压力协议": "full-profile Jaccard>=0.9连通分量 grouped 5-fold OOF",
        "正式测试集是否读取": False,
        "正式cache是否读取": False,
        "API调用次数": 0,
        "Python哈希种子": 0,
    }
    (args.输出目录 / "冻结实验方案.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    details = 运行协议(rows, base.分折(rows, args.折数), args.折数, "完整profile聚类OOF", agent, db, base, args.每折最多样本)
    protocols = ["完整profile聚类OOF"]
    if not args.跳过近重复压力测试:
        details.extend(运行协议(rows, 近重复分折(rows, args.折数, base.归一化), args.折数, "近重复profile压力OOF", agent, db, base, args.每折最多样本))
        protocols.append("近重复profile压力OOF")
    summary = {"冻结方案": frozen, "各协议": {name: 汇总协议(details, name, args.折数) for name in protocols}}
    main_metrics = summary["各协议"]["完整profile聚类OOF"]
    stress_metrics = summary["各协议"].get("近重复profile压力OOF")
    summary["预注册准入判定"] = {
        "主协议配对下界大于0": main_metrics["配对bootstrap_95%下界"] > 0,
        "主协议捕获未压缩Oracle至少90%": main_metrics["新Bank捕获未压缩Oracle比例"] >= 0.90,
        "主协议覆盖至少90%正查询": main_metrics["新Bank正收益查询数"] >= math.ceil(0.90 * main_metrics["未压缩正收益查询数"]),
        "主协议五折无大幅退化": min(main_metrics["各折新旧Bank增益差"].values()) > -0.01,
        "压力协议不为负": stress_metrics is None or stress_metrics["新Bank相对旧Bank平均增益"] >= 0,
        "exact_N": main_metrics["exact_N通过"] and (stress_metrics is None or stress_metrics["exact_N通过"]),
    }
    summary["是否冒烟检查"] = args.每折最多样本 > 0
    summary["是否通过全部准入"] = bool(not summary["是否冒烟检查"] and all(summary["预注册准入判定"].values()))
    (args.输出目录 / "逐样本结果.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in details), encoding="utf-8")
    (args.输出目录 / "指标汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
