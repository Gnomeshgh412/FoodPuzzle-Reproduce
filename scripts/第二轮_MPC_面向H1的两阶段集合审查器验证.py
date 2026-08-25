#!/usr/bin/env python3
"""第二轮 MPC：固定候选与特征，只审计面向 H1 的两阶段 Reviewer。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[1]
基础脚本 = 项目根目录 / "scripts/第一轮_MPC_查询条件集合审查器验证.py"
默认输出目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第二轮/MPC_面向H1的两阶段集合审查器"


def 加载基础模块() -> Any:
    spec = importlib.util.spec_from_file_location("第二轮MPC基础框架", 基础脚本)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载基础脚本：{基础脚本}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class 两阶段模型:
    def __init__(self, gate: Any, ranker: Any | None):
        self.gate = gate
        self.ranker = ranker


def 差分(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def 拟合(records: list[dict[str, Any]]) -> 两阶段模型:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gate_x: list[list[float]] = []
    gate_y: list[int] = []
    gate_w: list[float] = []
    rank_x: list[list[float]] = []
    rank_y: list[int] = []
    rank_w: list[float] = []
    for record in records:
        keep = record["候选"][0]
        actions = record["候选"][1:]
        for action in actions:
            gain = float(action["真实官能团增益"])
            gate_x.append(差分(action["特征"], keep["特征"]))
            gate_y.append(int(gain > 1e-12))
            gate_w.append(0.01 + abs(gain))
        positives = [x for x in actions if float(x["真实官能团增益"]) > 1e-12]
        for i in range(len(positives)):
            for j in range(i + 1, len(positives)):
                delta = float(positives[i]["真实官能团增益"]) - float(positives[j]["真实官能团增益"])
                if abs(delta) <= 1e-12:
                    continue
                vector = 差分(positives[i]["特征"], positives[j]["特征"])
                label = int(delta > 0)
                rank_x.extend([vector, [-x for x in vector]])
                rank_y.extend([label, 1 - label])
                rank_w.extend([abs(delta), abs(delta)])
    if not gate_x or len(set(gate_y)) < 2:
        raise RuntimeError("action-vs-H1 训练数据不足")
    gate = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, random_state=0),
    )
    gate.fit(gate_x, gate_y, logisticregression__sample_weight=gate_w)
    ranker = None
    if rank_x and len(set(rank_y)) == 2:
        ranker = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=3000, random_state=0),
        )
        ranker.fit(rank_x, rank_y, logisticregression__sample_weight=rank_w)
    return 两阶段模型(gate, ranker)


def 线性分数(model: Any, features: list[float]) -> float:
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    transformed = scaler.transform([features])[0]
    return float(transformed @ classifier.coef_[0])


def 预测(model: 两阶段模型, record: dict[str, Any]) -> dict[str, Any]:
    keep = record["候选"][0]
    gate_probabilities = [0.5]
    rank_scores = [float("-inf")]
    for action in record["候选"][1:]:
        delta = 差分(action["特征"], keep["特征"])
        gate_probabilities.append(float(model.gate.predict_proba([delta])[0, 1]))
        rank_scores.append(
            线性分数(model.ranker, action["特征"])
            if model.ranker is not None
            else math.log(max(gate_probabilities[-1], 1e-12) / max(1.0 - gate_probabilities[-1], 1e-12))
        )
    return {"门控概率": gate_probabilities, "正动作排序分数": rank_scores}


def 选择索引(prediction: dict[str, Any], threshold: float) -> int:
    admitted = [
        index for index in range(1, len(prediction["门控概率"]))
        if prediction["门控概率"][index] >= threshold
    ]
    return max(admitted, key=lambda i: (prediction["正动作排序分数"][i], -i), default=0)


def 选择门槛(predictions: list[tuple[dict[str, Any], dict[str, Any]]]) -> float:
    best: tuple[float, int, float] | None = None
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        gains = [
            float(record["候选"][选择索引(pred, threshold)]["真实官能团增益"])
            for record, pred in predictions
        ]
        changed = sum(abs(x) > 1e-12 for x in gains)
        wins = sum(x > 1e-12 for x in gains)
        losses = sum(x < -1e-12 for x in gains)
        mean = sum(gains) / max(1, len(gains))
        if changed >= 10 and mean > 0 and wins >= 2 * max(1, losses):
            candidate = (mean, -losses, threshold)
            if best is None or candidate > best:
                best = candidate
    return best[2] if best is not None else math.inf


def bootstrap下界(values: list[float]) -> float:
    rng = random.Random(20260804)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(2000))
    return means[max(0, int(0.025 * len(means)) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument("--外层折数", type=int, default=5)
    parser.add_argument("--内层折数", type=int, default=4)
    args = parser.parse_args()
    args.输出目录.mkdir(parents=True, exist_ok=True)
    base = 加载基础模块()
    agent = base.加载模块(项目根目录 / "code/Only-Deepseek/optimized_agent.py")
    db = 项目根目录 / "data/raw/flavordb.db"
    rows = base.读取_jsonl(项目根目录 / "results/splits/mpc/train.jsonl")
    outer_map = base.分折(rows, args.外层折数)
    all_results: list[dict[str, Any]] = []
    gate_labels: list[int] = []
    gate_scores: list[float] = []
    gate_weights: list[float] = []
    for outer in range(args.外层折数):
        outer_fit = [row for idx, row in enumerate(rows) if outer_map[idx] != outer]
        outer_held = [row for idx, row in enumerate(rows) if outer_map[idx] == outer]
        inner_records = base.生成OOF查询(outer_fit, args.内层折数, agent, db, f"第二轮外层{outer + 1}的内层")
        inner_predictions: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for inner_fold in range(args.内层折数):
            fit_records = [x for x in inner_records if x["折"] != inner_fold]
            held_records = [x for x in inner_records if x["折"] == inner_fold]
            model = 拟合(fit_records)
            inner_predictions.extend((x, 预测(model, x)) for x in held_records)
        threshold = 选择门槛(inner_predictions)
        final_model = 拟合(inner_records)
        outer_model = agent.MPCStructureModel(outer_fit, None, "full", db, calibrate_residuals=False)
        fold_results: list[dict[str, Any]] = []
        for source in outer_held:
            record = base.构造查询记录(outer_model, source, outer)
            if record is None:
                continue
            pred = 预测(final_model, record)
            selected_index = 0 if math.isinf(threshold) else 选择索引(pred, threshold)
            selected = record["候选"][selected_index]
            for index, action in enumerate(record["候选"][1:], 1):
                gain = float(action["真实官能团增益"])
                gate_labels.append(int(gain > 1e-12))
                gate_scores.append(float(pred["门控概率"][index]))
                gate_weights.append(0.01 + abs(gain))
            fold_results.append({
                "样本编号": record["样本编号"],
                "外层折": outer + 1,
                "目标食物": record["目标食物"],
                "门槛": None if math.isinf(threshold) else threshold,
                "选择": selected["候选类型"],
                "选择动作门控概率": pred["门控概率"][selected_index] if selected_index else None,
                "官能团F1增益": selected["真实官能团增益"],
                "具体分子F1增益": selected["真实分子F1增益"],
                "候选数量": len(record["候选"]),
            })
        all_results.extend(fold_results)
        fold_gain = sum(x["官能团F1增益"] for x in fold_results) / max(1, len(fold_results))
        print(f"[第二轮外层 {outer + 1}/{args.外层折数}] 门槛={None if math.isinf(threshold) else threshold}，官能团增益={fold_gain:.6f}", flush=True)
    from sklearn.metrics import average_precision_score, roc_auc_score
    gains = [float(x["官能团F1增益"]) for x in all_results]
    molecule_gains = [float(x["具体分子F1增益"]) for x in all_results]
    fold_gains = {
        str(fold): sum(float(x["官能团F1增益"]) for x in all_results if x["外层折"] == fold)
        / max(1, sum(1 for x in all_results if x["外层折"] == fold))
        for fold in range(1, args.外层折数 + 1)
    }
    summary = {
        "实验名称": "MPC面向H1的两阶段集合审查器",
        "协议": {"严格嵌套OOF": True, "外层折数": args.外层折数, "内层折数": args.内层折数, "正式测试集是否读取": False, "API调用次数": 0, "UniMol是否使用": False, "固定部分": "H1、v16 Bank20、特征、exact-N解码"},
        "action_vs_H1_加权ROC_AUC": roc_auc_score(gate_labels, gate_scores, sample_weight=gate_weights),
        "action_vs_H1_加权平均精确率": average_precision_score(gate_labels, gate_scores, sample_weight=gate_weights),
        "平均官能团F1增益": sum(gains) / max(1, len(gains)),
        "官能团F1增益bootstrap_95%下界": bootstrap下界(gains),
        "平均具体分子F1增益": sum(molecule_gains) / max(1, len(molecule_gains)),
        "胜负平": {"胜": sum(x > 1e-12 for x in gains), "负": sum(x < -1e-12 for x in gains), "平": sum(abs(x) <= 1e-12 for x in gains)},
        "各外层折官能团增益": fold_gains,
    }
    summary["是否通过准入"] = bool(summary["平均官能团F1增益"] > 0.0075 and summary["官能团F1增益bootstrap_95%下界"] > 0 and sum(x >= 0 for x in fold_gains.values()) >= 4 and summary["胜负平"]["胜"] >= 2 * max(1, summary["胜负平"]["负"]) and summary["平均具体分子F1增益"] >= 0)
    (args.输出目录 / "逐样本结果.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in all_results), encoding="utf-8")
    (args.输出目录 / "指标汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
