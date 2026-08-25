#!/usr/bin/env python3
"""第一轮 MPC：严格嵌套 OOF 的查询条件化、可拒绝集合审查器。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[1]
默认输出目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第一轮/MPC_查询条件集合审查器"


def 加载模块(路径: Path) -> Any:
    spec = importlib.util.spec_from_file_location("优化代理_第一轮MPC", 路径)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{路径}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def 归一化(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def 谱签名(row: dict[str, Any]) -> str:
    values = [归一化(x) for x in (row.get("partial_molecules") or []) + (row.get("missing_molecules") or []) if 归一化(x)]
    return "|".join(sorted(set(values)))


def 分折(rows: list[dict[str, Any]], folds: int) -> dict[int, int]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        clusters[谱签名(row)].append(idx)
    sizes = [0] * folds
    mapping: dict[int, int] = {}
    for _, indices in sorted(clusters.items(), key=lambda x: (-len(x[1]), x[0])):
        fold = min(range(folds), key=lambda x: (sizes[x], x))
        sizes[fold] += len(indices)
        for idx in indices:
            mapping[idx] = fold
    return mapping


def 集合F1(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    return 2 * overlap / (len(left) + len(right))


def posterior边际(posterior: dict[str, dict[int, float]], group: str) -> float:
    return float(sum(posterior.get(group, {}).values()))


def 候选特征(
    action: dict[str, Any] | None,
    bank: dict[str, Any],
    items: list[dict[str, Any]],
    posterior: dict[str, dict[int, float]],
    query: dict[str, Any],
    model: Any,
    bank_rank: int,
) -> list[float]:
    item_by_key = {归一化(x.get("molecule")): x for x in items}
    h1_groups = set(bank.get("h1_groups") or [])
    partial_groups = model._functional_group_set({归一化(x) for x in query.get("partial_molecules") or [] if 归一化(x)})
    if action is None:
        proposal_groups = h1_groups
        added: set[str] = set()
        removed: set[str] = set()
        add_keys: list[str] = []
        remove_keys: list[str] = []
        predicted_gain = 0.0
        predicted_f1 = float(bank.get("h1_expected_f1") or 0.0)
        depth = 0
    else:
        proposal_groups = set(action.get("proposal_groups") or [])
        added = set(action.get("added_groups") or [])
        removed = set(action.get("removed_groups") or [])
        add_keys = [归一化(x) for x in action.get("add_keys") or [action.get("add_key")] if 归一化(x)]
        remove_keys = [归一化(x) for x in action.get("remove_keys") or [action.get("remove_key")] if 归一化(x)]
        predicted_gain = float(action.get("predicted_expected_f1_gain") or 0.0)
        predicted_f1 = float(action.get("predicted_expected_f1") or 0.0)
        depth = int(action.get("depth") or 1)
    add_marginals = [posterior边际(posterior, x) for x in added]
    remove_marginals = [posterior边际(posterior, x) for x in removed]

    def item_values(keys: list[str], field: str) -> list[float]:
        return [float(item_by_key.get(key, {}).get(field) or 0.0) for key in keys]

    add_occurrence = item_values(add_keys, "occurrence_score")
    remove_occurrence = item_values(remove_keys, "occurrence_score")
    add_retrieval = item_values(add_keys, "idf_retrieved_support")
    remove_retrieval = item_values(remove_keys, "idf_retrieved_support")
    n = int(query.get("n") or 0)
    return [
        predicted_gain,
        predicted_f1,
        depth / 2.0,
        len(added) / 10.0,
        len(removed) / 10.0,
        sum(add_marginals) / max(1, len(add_marginals)),
        max(add_marginals, default=0.0),
        sum(remove_marginals) / max(1, len(remove_marginals)),
        max(remove_marginals, default=0.0),
        (sum(add_marginals) - sum(remove_marginals)) / 10.0,
        sum(add_occurrence) / max(1, len(add_occurrence)),
        sum(remove_occurrence) / max(1, len(remove_occurrence)),
        sum(add_retrieval) / max(1, len(add_retrieval)),
        sum(remove_retrieval) / max(1, len(remove_retrieval)),
        len(proposal_groups) / 30.0,
        len(proposal_groups & partial_groups) / max(1, len(partial_groups)),
        len(proposal_groups - partial_groups) / 30.0,
        math.log1p(n) / math.log1p(250),
        math.log1p(len(query.get("partial_molecules") or [])) / math.log1p(500),
        0.0 if action is None else 1.0 / max(1, bank_rank),
    ]


def 构造查询记录(model: Any, source: dict[str, Any], fold: int) -> dict[str, Any] | None:
    gold = {归一化(x) for x in source.get("missing_molecules") or [] if 归一化(x)}
    n = len(gold)
    if n <= 0:
        return None
    query = {
        "id": f"{source.get('id')}:集合审查",
        "target_food": source.get("target_food"),
        "partial_molecules": list(source.get("partial_molecules") or []),
        "n": n,
    }
    items = model._boundary_training_items(query, exclude_train_index=None, limit=len(model.universe))
    posterior = model._predict_group_cardinality_posterior(query)
    bank = model._build_v16_scientist_bank(items, n, posterior)
    gold_groups = model._functional_group_set(gold)
    h1 = list(bank.get("h1") or [])
    baseline_group_f1 = 集合F1(set(bank.get("h1_groups") or []), gold_groups)
    baseline_molecule_f1 = 集合F1(set(h1), gold)
    candidates = [{
        "候选类型": "保持H1",
        "提案": h1,
        "特征": 候选特征(None, bank, items, posterior, query, model, 0),
        "真实官能团增益": 0.0,
        "真实分子F1增益": 0.0,
    }]
    for rank, action in enumerate(bank.get("actions") or [], 1):
        proposal = list(action.get("proposal") or [])
        candidates.append({
            "候选类型": f"动作{rank}",
            "提案": proposal,
            "特征": 候选特征(action, bank, items, posterior, query, model, rank),
            "真实官能团增益": 集合F1(set(action.get("proposal_groups") or []), gold_groups) - baseline_group_f1,
            "真实分子F1增益": 集合F1(set(proposal), gold) - baseline_molecule_f1,
        })
    return {
        "样本编号": str(source.get("id")),
        "折": fold,
        "目标食物": source.get("target_food"),
        "N": n,
        "H1": h1,
        "H1官能团F1": baseline_group_f1,
        "H1分子F1": baseline_molecule_f1,
        "候选": candidates,
    }


def 生成OOF查询(rows: list[dict[str, Any]], folds: int, agent: Any, db: Path, 标签: str) -> list[dict[str, Any]]:
    mapping = 分折(rows, folds)
    records: list[dict[str, Any]] = []
    for fold in range(folds):
        fit = [row for idx, row in enumerate(rows) if mapping[idx] != fold]
        held = [(idx, row) for idx, row in enumerate(rows) if mapping[idx] == fold]
        model = agent.MPCStructureModel(fit, None, "full", db, calibrate_residuals=False)
        print(f"[{标签}] 第 {fold + 1}/{folds} 折：训练 {len(fit)}，验证 {len(held)}", flush=True)
        for _, row in held:
            record = 构造查询记录(model, row, fold)
            if record is not None:
                records.append(record)
    return records


def pairwise数据(records: list[dict[str, Any]]) -> tuple[list[list[float]], list[int]]:
    X: list[list[float]] = []
    y: list[int] = []
    for record in records:
        candidates = record["候选"]
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                delta = float(candidates[i]["真实官能团增益"]) - float(candidates[j]["真实官能团增益"])
                if abs(delta) <= 1e-12:
                    continue
                diff = [a - b for a, b in zip(candidates[i]["特征"], candidates[j]["特征"])]
                label = int(delta > 0)
                X.extend([diff, [-x for x in diff]])
                y.extend([label, 1 - label])
    return X, y


def 拟合(records: list[dict[str, Any]]) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = pairwise数据(records)
    if not X or len(set(y)) < 2:
        raise RuntimeError("pairwise 训练数据不足")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, random_state=0))
    model.fit(X, y)
    return model


def 候选分数(model: Any, candidate: dict[str, Any]) -> float:
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    transformed = scaler.transform([candidate["特征"]])[0]
    return float(transformed @ classifier.coef_[0])


def 预测记录(model: Any, record: dict[str, Any]) -> dict[str, Any]:
    candidates = record["候选"]
    scores = [候选分数(model, x) for x in candidates]
    best = max(range(len(candidates)), key=lambda i: (scores[i], -i))
    keep_score = scores[0]
    probability = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, scores[best] - keep_score))))
    return {"最佳索引": best, "相对保持H1概率": probability, "分数": scores}


def 选择门槛(predictions: list[tuple[dict[str, Any], dict[str, Any]]]) -> float:
    best: tuple[float, int, float] | None = None
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        gains: list[float] = []
        for record, pred in predictions:
            idx = pred["最佳索引"] if pred["最佳索引"] != 0 and pred["相对保持H1概率"] >= threshold else 0
            gains.append(float(record["候选"][idx]["真实官能团增益"]))
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
    rng = random.Random(20260803)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(2000))
    return means[max(0, int(0.025 * len(means)) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument("--外层折数", type=int, default=5)
    parser.add_argument("--内层折数", type=int, default=4)
    args = parser.parse_args()
    args.输出目录.mkdir(parents=True, exist_ok=True)
    agent = 加载模块(项目根目录 / "code/Only-Deepseek/optimized_agent.py")
    db = 项目根目录 / "data/raw/flavordb.db"
    rows = 读取_jsonl(项目根目录 / "results/splits/mpc/train.jsonl")
    outer_map = 分折(rows, args.外层折数)
    all_results: list[dict[str, Any]] = []
    pair_labels: list[int] = []
    pair_scores: list[float] = []
    for outer in range(args.外层折数):
        outer_fit = [row for idx, row in enumerate(rows) if outer_map[idx] != outer]
        outer_held = [row for idx, row in enumerate(rows) if outer_map[idx] == outer]
        # 内层候选由各自训练折模型生成，标签只来自对应 held-out 查询。
        inner_records = 生成OOF查询(outer_fit, args.内层折数, agent, db, f"外层{outer + 1}的内层")
        inner_predictions: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for inner_fold in range(args.内层折数):
            fit_records = [x for x in inner_records if x["折"] != inner_fold]
            held_records = [x for x in inner_records if x["折"] == inner_fold]
            ranker = 拟合(fit_records)
            inner_predictions.extend((x, 预测记录(ranker, x)) for x in held_records)
        threshold = 选择门槛(inner_predictions)
        final_ranker = 拟合(inner_records)
        outer_model = agent.MPCStructureModel(outer_fit, None, "full", db, calibrate_residuals=False)
        fold_results: list[dict[str, Any]] = []
        for source in outer_held:
            record = 构造查询记录(outer_model, source, outer)
            if record is None:
                continue
            pred = 预测记录(final_ranker, record)
            chosen = pred["最佳索引"] if pred["最佳索引"] != 0 and pred["相对保持H1概率"] >= threshold else 0
            selected = record["候选"][chosen]
            for i in range(len(record["候选"])):
                for j in range(i + 1, len(record["候选"])):
                    delta = float(record["候选"][i]["真实官能团增益"]) - float(record["候选"][j]["真实官能团增益"])
                    if abs(delta) > 1e-12:
                        pair_labels.append(int(delta > 0))
                        pair_scores.append(pred["分数"][i] - pred["分数"][j])
            fold_results.append({
                "样本编号": record["样本编号"],
                "外层折": outer + 1,
                "目标食物": record["目标食物"],
                "门槛": None if math.isinf(threshold) else threshold,
                "选择": selected["候选类型"],
                "相对保持H1概率": pred["相对保持H1概率"],
                "官能团F1增益": selected["真实官能团增益"],
                "具体分子F1增益": selected["真实分子F1增益"],
                "候选数量": len(record["候选"]),
            })
        all_results.extend(fold_results)
        mean = sum(x["官能团F1增益"] for x in fold_results) / max(1, len(fold_results))
        print(f"[外层 {outer + 1}/{args.外层折数}] 门槛={None if math.isinf(threshold) else threshold}，官能团增益={mean:.6f}", flush=True)
    from sklearn.metrics import roc_auc_score
    gains = [float(x["官能团F1增益"]) for x in all_results]
    molecule_gains = [float(x["具体分子F1增益"]) for x in all_results]
    fold_gains = {
        str(fold): sum(float(x["官能团F1增益"]) for x in all_results if x["外层折"] == fold)
        / max(1, sum(1 for x in all_results if x["外层折"] == fold))
        for fold in range(1, args.外层折数 + 1)
    }
    summary = {
        "实验名称": "MPC 查询条件化可拒绝集合审查器",
        "协议": {"严格嵌套OOF": True, "外层折数": args.外层折数, "内层折数": args.内层折数, "正式测试集是否读取": False, "API调用次数": 0, "UniMol是否使用": False, "候选": "冻结v16 Bank20加保持H1"},
        "查询内pairwise_roc_auc": roc_auc_score(pair_labels, pair_scores) if len(set(pair_labels)) == 2 else None,
        "平均官能团F1增益": sum(gains) / max(1, len(gains)),
        "官能团F1增益bootstrap_95%下界": bootstrap下界(gains),
        "平均具体分子F1增益": sum(molecule_gains) / max(1, len(molecule_gains)),
        "胜负平": {"胜": sum(x > 1e-12 for x in gains), "负": sum(x < -1e-12 for x in gains), "平": sum(abs(x) <= 1e-12 for x in gains)},
        "各外层折官能团增益": fold_gains,
    }
    summary["是否通过准入"] = bool(summary["查询内pairwise_roc_auc"] is not None and summary["查询内pairwise_roc_auc"] > 0.5 and summary["平均官能团F1增益"] > 0.0075 and summary["官能团F1增益bootstrap_95%下界"] > 0 and sum(x >= 0 for x in fold_gains.values()) >= 4 and summary["胜负平"]["胜"] >= 2 * max(1, summary["胜负平"]["负"]))
    (args.输出目录 / "逐样本结果.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in all_results), encoding="utf-8")
    (args.输出目录 / "指标汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
