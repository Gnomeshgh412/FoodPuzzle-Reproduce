#!/usr/bin/env python3
"""第五轮 MPC：只比较 Scientist/H1 主干，不训练或调用 Reviewer。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[1]
基础脚本 = 项目根目录 / "scripts/第一轮_MPC_查询条件集合审查器验证.py"
默认输出目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第五轮/MPC_强H1科学家复原"


def 加载基础模块() -> Any:
    spec = importlib.util.spec_from_file_location("第五轮MPC基础", 基础脚本)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载基础脚本：{基础脚本}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 随机行分折(rows: list[dict[str, Any]], folds: int) -> dict[int, int]:
    indices = list(range(len(rows)))
    random.Random(20260804).shuffle(indices)
    return {index: position % folds for position, index in enumerate(indices)}


def 构造训练谱候选(model: Any, query: dict[str, Any], normalize: Any) -> list[dict[str, Any]]:
    """只打分具有训练 occurrence 的候选，复原旧版 H1 的候选域。"""
    partial = {normalize(value) for value in query.get("partial_molecules") or [] if normalize(value)}
    retrieved = model._build_retrieved_support(query, exclude_train_index=None, top_k=10)
    context = model._build_query_context(query)
    candidates: list[tuple[str, list[float]]] = []
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
        candidates.append((candidate, values))
    if model.ranker is not None and candidates:
        current_scores = [
            float(value)
            for value in model.ranker.decision_function(
                [[values[index] for index in model.PRIMARY_FEATURE_INDICES] for _, values in candidates]
            )
        ]
    else:
        current_scores = [
            0.20 * values[0] + 0.30 * values[1] + 0.20 * values[2] + 0.30 * values[3]
            for _, values in candidates
        ]
    items = [
        {
            "molecule": model.display_names[candidate],
            "occurrence_score": score,
            "frequency_prior": values[0],
            "cooccurrence": values[1],
            "cooccurrence_max": values[2],
            "retrieved_profile_support": values[3],
        }
        for (candidate, values), score in zip(candidates, current_scores)
    ]
    items.sort(key=lambda item: (-float(item["occurrence_score"]), normalize(item["molecule"])))
    for rank, item in enumerate(items, 1):
        item["occurrence_rank"] = rank
    return items


def 简单H1(items: list[dict[str, Any]], n: int, training_universe: set[str], normalize: Any) -> tuple[list[str], list[dict[str, Any]]]:
    eligible = [item for item in items if normalize(item.get("molecule")) in training_universe]
    ranked = sorted(
        eligible,
        key=lambda item: (
            -(
                0.20 * float(item.get("frequency_prior") or 0.0)
                + 0.30 * float(item.get("cooccurrence") or 0.0)
                + 0.20 * float(item.get("cooccurrence_max") or 0.0)
            ),
            normalize(item.get("molecule")),
        ),
    )
    if len(ranked) < n:
        seen = {normalize(item.get("molecule")) for item in ranked}
        ranked.extend(item for item in items if normalize(item.get("molecule")) not in seen)
    for rank, item in enumerate(ranked, 1):
        item["复原H1名次"] = rank
    keys = [normalize(item.get("molecule")) for item in ranked[:n] if normalize(item.get("molecule"))]
    return keys, ranked


def 一次检索残差(ranked: list[dict[str, Any]], n: int, normalize: Any) -> list[str]:
    if n <= 0 or len(ranked) <= n:
        return [normalize(item.get("molecule")) for item in ranked[:n]]
    protected = ranked[: max(0, n - 1)]
    boundary = ranked[max(0, n - 1) : min(len(ranked), n + 5)]
    supported = [item for item in boundary if float(item.get("retrieved_profile_support") or 0.0) > 0.0]
    residual = min(
        supported,
        key=lambda item: (
            -float(item.get("retrieved_profile_support") or 0.0),
            int(item.get("复原H1名次") or 10**9),
            normalize(item.get("molecule")),
        ),
        default=None,
    )
    selected = list(protected) + ([residual] if residual is not None else [])
    seen = {normalize(item.get("molecule")) for item in selected}
    for item in ranked:
        key = normalize(item.get("molecule"))
        if key and key not in seen:
            selected.append(item)
            seen.add(key)
        if len(selected) >= n:
            break
    return [normalize(item.get("molecule")) for item in selected[:n]]


def 评估集合(model: Any, prediction: list[str], gold: set[str], base: Any) -> tuple[float, float]:
    molecule_f1 = base.集合F1(set(prediction), gold)
    predicted_groups = model._functional_group_set(set(prediction))
    gold_groups = model._functional_group_set(gold)
    return base.集合F1(predicted_groups, gold_groups), molecule_f1


def 运行协议(
    rows: list[dict[str, Any]], mapping: dict[int, int], folds: int,
    protocol_name: str, agent: Any, db: Path, base: Any,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for fold in range(folds):
        fit = [row for index, row in enumerate(rows) if mapping[index] != fold]
        held = [row for index, row in enumerate(rows) if mapping[index] == fold]
        model = agent.MPCStructureModel(fit, None, "full", db, calibrate_residuals=False)
        training_universe = set(model.training_universe)
        for source in held:
            gold = {base.归一化(value) for value in source.get("missing_molecules") or [] if base.归一化(value)}
            n = len(gold)
            if n <= 0:
                continue
            query = {
                "id": f"{source.get('id')}:第五轮H1",
                "target_food": source.get("target_food"),
                "partial_molecules": list(source.get("partial_molecules") or []),
                "n": n,
            }
            items = 构造训练谱候选(model, query, base.归一化)
            if len(items) < n:
                raise RuntimeError(f"样本 {source.get('id')} 的训练谱候选少于 N：{len(items)} < {n}")
            current = [base.归一化(item.get("molecule")) for item in sorted(items, key=lambda item: int(item.get("occurrence_rank") or 10**9))[:n]]
            simple, ranked = 简单H1(items, n, training_universe, base.归一化)
            residual = 一次检索残差(ranked, n, base.归一化)
            current_fg, current_molecule = 评估集合(model, current, gold, base)
            simple_fg, simple_molecule = 评估集合(model, simple, gold, base)
            residual_fg, residual_molecule = 评估集合(model, residual, gold, base)
            details.append({
                "协议": protocol_name,
                "折": fold + 1,
                "样本编号": str(source.get("id")),
                "目标食物": source.get("target_food"),
                "N": n,
                "当前H1官能团F1": current_fg,
                "当前H1具体分子F1": current_molecule,
                "复原H1官能团F1": simple_fg,
                "复原H1具体分子F1": simple_molecule,
                "复原H1官能团增益": simple_fg - current_fg,
                "复原H1具体分子增益": simple_molecule - current_molecule,
                "复原H1加一次检索残差官能团F1": residual_fg,
                "复原H1加一次检索残差具体分子F1": residual_molecule,
                "复原H1加一次检索残差官能团增益": residual_fg - current_fg,
                "复原H1加一次检索残差具体分子增益": residual_molecule - current_molecule,
                "检索残差是否改变": residual != simple,
                "当前H1是否exact_N": len(current) == n and len(set(current)) == n,
                "复原H1是否exact_N": len(simple) == n and len(set(simple)) == n,
                "检索残差是否exact_N": len(residual) == n and len(set(residual)) == n,
            })
        print(f"[MPC第五轮 {protocol_name} {fold + 1}/{folds}] 已完成 {len(held)} 条", flush=True)
    return details


def bootstrap下界(values: list[float], seed: str) -> float:
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(5000)
    )
    return means[max(0, int(0.025 * len(means)) - 1)]


def 汇总方法(details: list[dict[str, Any]], prefix: str, folds: int, protocol: str) -> dict[str, Any]:
    selected = [row for row in details if row["协议"] == protocol]
    fg = [float(row[f"{prefix}官能团增益"]) for row in selected]
    molecule = [float(row[f"{prefix}具体分子增益"]) for row in selected]
    fold_gains = {
        str(fold): sum(float(row[f"{prefix}官能团增益"]) for row in selected if row["折"] == fold)
        / max(1, sum(row["折"] == fold for row in selected))
        for fold in range(1, folds + 1)
    }
    absolute_fg_field = f"{prefix}官能团F1"
    absolute_molecule_field = f"{prefix}具体分子F1"
    result = {
        "样本数": len(selected),
        "平均官能团F1": sum(float(row[absolute_fg_field]) for row in selected) / len(selected),
        "平均具体分子F1": sum(float(row[absolute_molecule_field]) for row in selected) / len(selected),
        "相对当前H1平均官能团增益": sum(fg) / len(fg),
        "相对当前H1官能团增益bootstrap_95%下界": bootstrap下界(fg, f"第五轮-{protocol}-{prefix}"),
        "相对当前H1平均具体分子增益": sum(molecule) / len(molecule),
        "胜负平": {"胜": sum(x > 1e-12 for x in fg), "负": sum(x < -1e-12 for x in fg), "平": sum(abs(x) <= 1e-12 for x in fg)},
        "各折官能团增益": fold_gains,
    }
    result["官方指标准入"] = bool(
        result["相对当前H1平均官能团增益"] > 0
        and result["相对当前H1官能团增益bootstrap_95%下界"] > 0
        and sum(value >= 0 for value in fold_gains.values()) >= 4
        and result["胜负平"]["胜"] > result["胜负平"]["负"]
    )
    result["双目标准入"] = bool(result["官方指标准入"] and result["相对当前H1平均具体分子增益"] >= 0)
    return result


def 当前H1汇总(details: list[dict[str, Any]], protocol: str) -> dict[str, Any]:
    selected = [row for row in details if row["协议"] == protocol]
    return {
        "样本数": len(selected),
        "平均官能团F1": sum(float(row["当前H1官能团F1"]) for row in selected) / len(selected),
        "平均具体分子F1": sum(float(row["当前H1具体分子F1"]) for row in selected) / len(selected),
        "exact_N样本数": sum(int(row["当前H1是否exact_N"]) for row in selected),
    }


def 检索残差自身贡献(details: list[dict[str, Any]], protocol: str, folds: int) -> dict[str, Any]:
    selected = [row for row in details if row["协议"] == protocol]
    fg = [
        float(row["复原H1加一次检索残差官能团F1"]) - float(row["复原H1官能团F1"])
        for row in selected
    ]
    molecule = [
        float(row["复原H1加一次检索残差具体分子F1"]) - float(row["复原H1具体分子F1"])
        for row in selected
    ]
    fold_gains = {
        str(fold): sum(
            float(row["复原H1加一次检索残差官能团F1"]) - float(row["复原H1官能团F1"])
            for row in selected if row["折"] == fold
        ) / max(1, sum(row["折"] == fold for row in selected))
        for fold in range(1, folds + 1)
    }
    return {
        "平均官能团F1贡献": sum(fg) / len(fg),
        "官能团贡献bootstrap_95%下界": bootstrap下界(fg, f"第五轮-{protocol}-检索残差自身"),
        "平均具体分子F1贡献": sum(molecule) / len(molecule),
        "胜负平": {"胜": sum(x > 1e-12 for x in fg), "负": sum(x < -1e-12 for x in fg), "平": sum(abs(x) <= 1e-12 for x in fg)},
        "各折官能团贡献": fold_gains,
    }


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("必须使用 PYTHONHASHSEED=0 启动实验")
    parser = argparse.ArgumentParser()
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument("--折数", type=int, default=5)
    args = parser.parse_args()
    args.输出目录.mkdir(parents=True, exist_ok=True)
    base = 加载基础模块()
    agent = base.加载模块(项目根目录 / "code/Only-Deepseek/optimized_agent.py")
    rows = base.读取_jsonl(项目根目录 / "results/splits/mpc/train.jsonl")
    db = 项目根目录 / "data/raw/flavordb.db"
    protocol = {
        "实验名称": "MPC强H1科学家复原验证",
        "唯一研究对象": "Scientist/H1候选排序",
        "比较方法": ["当前masked-query pairwise排序器在历史候选域的H1", "训练profile occurrence/co-occurrence复原H1", "复原H1加一次检索边界替换"],
        "候选域": "仅训练profile中出现且不在partial中的分子；这是旧版occurrence H1的非零信号候选域，避免遍历无训练统计的FlavorDB全目录",
        "冻结部分": "数据、候选名称归一、官能团解析、exact-N、无UniMol、无Reviewer、无Action Bank执行",
        "随机行OOF": "用于与旧版本及当前随机切分结果连续比较",
        "完整profile聚类OOF": "用于排除完全重复profile跨折记忆",
        "正式测试集是否读取": False,
        "API调用次数": 0,
        "Python哈希种子": 0,
    }
    (args.输出目录 / "冻结实验方案.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    row_details = 运行协议(rows, 随机行分折(rows, args.折数), args.折数, "随机行OOF", agent, db, base)
    cluster_details = 运行协议(rows, base.分折(rows, args.折数), args.折数, "完整profile聚类OOF", agent, db, base)
    details = row_details + cluster_details
    summary: dict[str, Any] = {"实验方案": protocol, "各协议": {}}
    for protocol_name in ("随机行OOF", "完整profile聚类OOF"):
        summary["各协议"][protocol_name] = {
            "当前H1": 当前H1汇总(details, protocol_name),
            "复原H1": 汇总方法(details, "复原H1", args.折数, protocol_name),
            "复原H1加一次检索残差": 汇总方法(details, "复原H1加一次检索残差", args.折数, protocol_name),
            "一次检索残差相对复原H1的自身贡献": 检索残差自身贡献(details, protocol_name, args.折数),
        }
    cluster = summary["各协议"]["完整profile聚类OOF"]
    summary["最终判定"] = {
        "复原H1是否通过双协议官方指标准入": bool(
            summary["各协议"]["随机行OOF"]["复原H1"]["官方指标准入"]
            and cluster["复原H1"]["官方指标准入"]
        ),
        "检索残差是否通过双协议官方指标准入": bool(
            summary["各协议"]["随机行OOF"]["复原H1加一次检索残差"]["官方指标准入"]
            and cluster["复原H1加一次检索残差"]["官方指标准入"]
        ),
        "解释规则": "双协议均通过才可进入冻结候选；否则不进入正式测试，并重新定位Scientist内部瓶颈。",
        "瓶颈更新": "简单occurrence手工加权不是可恢复的强主干；当前pairwise H1在训练侧明显更强，下一步不继续复刻旧公式。",
    }
    (args.输出目录 / "逐样本结果.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in details), encoding="utf-8"
    )
    (args.输出目录 / "指标汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
