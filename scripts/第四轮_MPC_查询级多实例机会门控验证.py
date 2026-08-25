#!/usr/bin/env python3
"""第四轮 MPC：冻结 Scientist 与动作排序，只替换查询级机会门控。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[1]
第一轮脚本 = 项目根目录 / "scripts/第一轮_MPC_查询条件集合审查器验证.py"
第二轮脚本 = 项目根目录 / "scripts/第二轮_MPC_面向H1的两阶段集合审查器验证.py"
默认输出目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第四轮/MPC_查询级多实例机会门控"


def 加载脚本(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 差分(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def 多实例特征(record: dict[str, Any]) -> list[float]:
    keep = record["候选"][0]["特征"]
    actions = [差分(action["特征"], keep) for action in record["候选"][1:]]
    if not actions:
        return [0.0] * (len(keep) * 3 + 5)
    pooled: list[float] = []
    for index in range(len(keep)):
        values = [row[index] for row in actions]
        pooled.extend([max(values), sum(values) / len(values), statistics.pstdev(values)])
    predicted = [float(action["特征"][0]) for action in record["候选"][1:]]
    ordered = sorted(predicted, reverse=True)
    pooled.extend([
        ordered[0] if ordered else 0.0,
        ordered[1] if len(ordered) > 1 else 0.0,
        ordered[2] if len(ordered) > 2 else 0.0,
        sum(x > 0 for x in predicted) / max(1, len(predicted)),
        len(actions) / 20.0,
    ])
    return pooled


def 有帕累托机会(record: dict[str, Any]) -> bool:
    return any(
        float(action["真实官能团增益"]) > 1e-12
        and float(action["真实分子F1增益"]) >= -1e-12
        for action in record["候选"][1:]
    )


def 拟合查询门控(records: list[dict[str, Any]]) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = [多实例特征(record) for record in records]
    y = [int(有帕累托机会(record)) for record in records]
    weights = []
    for record in records:
        best = max(
            (
                float(action["真实官能团增益"])
                for action in record["候选"][1:]
                if float(action["真实分子F1增益"]) >= -1e-12
            ),
            default=0.0,
        )
        weights.append(0.02 + max(0.0, best))
    if len(set(y)) < 2:
        raise RuntimeError("查询级机会标签缺少正例或负例")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000, random_state=0),
    )
    model.fit(x, y, logisticregression__sample_weight=weights)
    return model


def 条件动作索引(record: dict[str, Any], action_model: Any, reviewer: Any) -> int:
    keep = record["候选"][0]
    admitted: list[int] = []
    rank_scores: dict[int, float] = {}
    for index, action in enumerate(record["候选"][1:], 1):
        delta = 差分(action["特征"], keep["特征"])
        probability = float(action_model.gate.predict_proba([delta])[0, 1])
        if probability >= 0.5:
            admitted.append(index)
            rank_scores[index] = (
                reviewer.线性分数(action_model.ranker, action["特征"])
                if action_model.ranker is not None
                else probability
            )
    return max(admitted, key=lambda index: (rank_scores[index], -index), default=0)


def 查询预测(record: dict[str, Any], query_gate: Any, action_model: Any, reviewer: Any) -> dict[str, Any]:
    probability = float(query_gate.predict_proba([多实例特征(record)])[0, 1])
    action_index = 条件动作索引(record, action_model, reviewer)
    return {"查询机会概率": probability, "条件动作索引": action_index}


def 选择查询门槛(predictions: list[tuple[dict[str, Any], dict[str, Any]]]) -> float:
    best: tuple[float, float, int, float] | None = None
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        fg_gains: list[float] = []
        molecule_gains: list[float] = []
        for record, prediction in predictions:
            index = prediction["条件动作索引"] if prediction["查询机会概率"] >= threshold else 0
            choice = record["候选"][index]
            fg_gains.append(float(choice["真实官能团增益"]))
            molecule_gains.append(float(choice["真实分子F1增益"]))
        wins = sum(x > 1e-12 for x in fg_gains)
        losses = sum(x < -1e-12 for x in fg_gains)
        mean_fg = sum(fg_gains) / len(fg_gains)
        mean_molecule = sum(molecule_gains) / len(molecule_gains)
        if wins >= 10 and mean_fg > 0 and mean_molecule >= 0 and wins >= 2 * max(1, losses):
            candidate = (mean_fg, mean_molecule, -losses, threshold)
            if best is None or candidate > best:
                best = candidate
    return best[3] if best is not None else math.inf


def 内层训练(
    records: list[dict[str, Any]], folds: int, reviewer: Any,
) -> tuple[Any, float, Any, float]:
    baseline_predictions: list[tuple[dict[str, Any], dict[str, Any]]] = []
    query_predictions: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for fold in range(folds):
        fit = [x for x in records if x["折"] != fold]
        held = [x for x in records if x["折"] == fold]
        action_model = reviewer.拟合(fit)
        query_gate = 拟合查询门控(fit)
        baseline_predictions.extend((x, reviewer.预测(action_model, x)) for x in held)
        query_predictions.extend((x, 查询预测(x, query_gate, action_model, reviewer)) for x in held)
    baseline_threshold = reviewer.选择门槛(baseline_predictions)
    query_threshold = 选择查询门槛(query_predictions)
    final_action_model = reviewer.拟合(records)
    final_query_gate = 拟合查询门控(records)
    return final_action_model, baseline_threshold, final_query_gate, query_threshold


def bootstrap下界(values: list[float], seed: str) -> float:
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(5000))
    return means[max(0, int(0.025 * len(means)) - 1)]


def 汇总(rows: list[dict[str, Any]], prefix: str, folds: int) -> dict[str, Any]:
    fg = [float(x[f"{prefix}官能团F1增益"]) for x in rows]
    molecule = [float(x[f"{prefix}具体分子F1增益"]) for x in rows]
    return {
        "平均官能团F1增益": sum(fg) / len(fg),
        "官能团F1增益bootstrap_95%下界": bootstrap下界(fg, f"第四轮MPC-{prefix}"),
        "平均具体分子F1增益": sum(molecule) / len(molecule),
        "胜负平": {"胜": sum(x > 1e-12 for x in fg), "负": sum(x < -1e-12 for x in fg), "平": sum(abs(x) <= 1e-12 for x in fg)},
        "各外层折官能团增益": {
            str(fold): sum(float(x[f"{prefix}官能团F1增益"]) for x in rows if x["外层折"] == fold)
            / max(1, sum(1 for x in rows if x["外层折"] == fold))
            for fold in range(1, folds + 1)
        },
    }


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("必须使用 PYTHONHASHSEED=0 启动实验")
    parser = argparse.ArgumentParser()
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument("--外层折数", type=int, default=5)
    parser.add_argument("--内层折数", type=int, default=4)
    args = parser.parse_args()
    args.输出目录.mkdir(parents=True, exist_ok=True)
    base = 加载脚本("第四轮MPC基础", 第一轮脚本)
    reviewer = 加载脚本("第四轮MPC两阶段", 第二轮脚本)
    agent = base.加载模块(项目根目录 / "code/Only-Deepseek/optimized_agent.py")
    rows = base.读取_jsonl(项目根目录 / "results/splits/mpc/train.jsonl")
    db = 项目根目录 / "data/raw/flavordb.db"
    protocol = {
        "实验名称": "MPC查询级多实例机会门控",
        "唯一变化": "将动作级概率兼任查询拒绝门控改为对完整Bank做置换不变聚合的查询级机会门控",
        "查询级正标签": "Bank中存在官能团F1正增益且具体分子F1不下降的动作",
        "固定部分": "H1、v16 Bank20、动作特征、动作级模型、正动作排序器、exact-N",
        "严格嵌套OOF": True,
        "外层折数": args.外层折数,
        "内层折数": args.内层折数,
        "Python哈希种子": 0,
        "正式测试集是否读取": False,
        "API调用次数": 0,
    }
    (args.输出目录 / "冻结实验方案.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outer_map = base.分折(rows, args.外层折数)
    all_results: list[dict[str, Any]] = []
    for outer in range(args.外层折数):
        outer_fit = [row for index, row in enumerate(rows) if outer_map[index] != outer]
        outer_held = [row for index, row in enumerate(rows) if outer_map[index] == outer]
        inner_records = base.生成OOF查询(outer_fit, args.内层折数, agent, db, f"第四轮外层{outer + 1}的内层")
        action_model, baseline_threshold, query_gate, query_threshold = 内层训练(inner_records, args.内层折数, reviewer)
        outer_model = agent.MPCStructureModel(outer_fit, None, "full", db, calibrate_residuals=False)
        fold_results: list[dict[str, Any]] = []
        for source in outer_held:
            record = base.构造查询记录(outer_model, source, outer)
            if record is None:
                continue
            baseline_prediction = reviewer.预测(action_model, record)
            baseline_index = 0 if math.isinf(baseline_threshold) else reviewer.选择索引(baseline_prediction, baseline_threshold)
            query_prediction = 查询预测(record, query_gate, action_model, reviewer)
            query_index = query_prediction["条件动作索引"] if not math.isinf(query_threshold) and query_prediction["查询机会概率"] >= query_threshold else 0
            baseline_choice = record["候选"][baseline_index]
            query_choice = record["候选"][query_index]
            fold_results.append({
                "样本编号": record["样本编号"],
                "外层折": outer + 1,
                "目标食物": record["目标食物"],
                "候选数量": len(record["候选"]),
                "同步动作级门槛": None if math.isinf(baseline_threshold) else baseline_threshold,
                "同步动作级选择": baseline_choice["候选类型"],
                "同步动作级官能团F1增益": baseline_choice["真实官能团增益"],
                "同步动作级具体分子F1增益": baseline_choice["真实分子F1增益"],
                "查询级门槛": None if math.isinf(query_threshold) else query_threshold,
                "查询机会概率": query_prediction["查询机会概率"],
                "查询级选择": query_choice["候选类型"],
                "查询级官能团F1增益": query_choice["真实官能团增益"],
                "查询级具体分子F1增益": query_choice["真实分子F1增益"],
                "Bank是否有帕累托机会": 有帕累托机会(record),
            })
        all_results.extend(fold_results)
        baseline_gain = sum(float(x["同步动作级官能团F1增益"]) for x in fold_results) / max(1, len(fold_results))
        query_gain = sum(float(x["查询级官能团F1增益"]) for x in fold_results) / max(1, len(fold_results))
        print(f"[MPC第四轮外层 {outer + 1}/{args.外层折数}] 动作级={baseline_gain:.6f}，查询级={query_gain:.6f}", flush=True)
    baseline_summary = 汇总(all_results, "同步动作级", args.外层折数)
    query_summary = 汇总(all_results, "查询级", args.外层折数)
    paired = [float(x["查询级官能团F1增益"]) - float(x["同步动作级官能团F1增益"]) for x in all_results]
    paired_molecule = [float(x["查询级具体分子F1增益"]) - float(x["同步动作级具体分子F1增益"]) for x in all_results]
    comparison = {
        "平均官能团F1增益差": sum(paired) / len(paired),
        "配对bootstrap_95%下界": bootstrap下界(paired, "第四轮MPC配对"),
        "平均具体分子F1增益差": sum(paired_molecule) / len(paired_molecule),
        "改进恶化不变": {"改进": sum(x > 1e-12 for x in paired), "恶化": sum(x < -1e-12 for x in paired), "不变": sum(abs(x) <= 1e-12 for x in paired)},
        "帕累托机会查询数": sum(int(x["Bank是否有帕累托机会"]) for x in all_results),
        "机会查询中执行数": sum(int(x["Bank是否有帕累托机会"] and x["查询级选择"] != "保持H1") for x in all_results),
        "无机会查询中误执行数": sum(int(not x["Bank是否有帕累托机会"] and x["查询级选择"] != "保持H1") for x in all_results),
    }
    wins = query_summary["胜负平"]
    fold_values = list(query_summary["各外层折官能团增益"].values())
    admitted = bool(
        query_summary["平均官能团F1增益"] > 0.0075
        and comparison["平均官能团F1增益差"] > 0
        and comparison["配对bootstrap_95%下界"] > 0
        and sum(x >= 0 for x in fold_values) >= 4
        and wins["胜"] >= 2 * max(1, wins["负"])
        and query_summary["平均具体分子F1增益"] >= 0
        and comparison["平均具体分子F1增益差"] >= 0
    )
    summary = {
        "协议": protocol,
        "同步动作级Reviewer": baseline_summary,
        "查询级多实例机会门控": query_summary,
        "配对比较": comparison,
        "是否通过准入": admitted,
    }
    (args.输出目录 / "逐样本结果.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in all_results), encoding="utf-8")
    (args.输出目录 / "指标汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.输出目录 / "结论说明.json").write_text(json.dumps({"结论": "通过" if admitted else "未通过，不接入正式代理"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
