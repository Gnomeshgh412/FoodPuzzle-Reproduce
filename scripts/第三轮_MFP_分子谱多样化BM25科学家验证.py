#!/usr/bin/env python3
"""第三轮 MFP：只优化 Scientist 的分子谱多样化 BM25 候选集。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


项目根目录 = Path(__file__).resolve().parents[1]
基础脚本 = 项目根目录 / "scripts/第一轮_MFP_UniMol独占审查器验证.py"
默认输出目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第三轮/MFP_分子谱多样化BM25科学家"
第二轮结果目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第二轮/MFP_BM25候选与UniMol独占审查器"
候选权重 = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)


def 加载基础模块() -> Any:
    spec = importlib.util.spec_from_file_location("第三轮MFP基础框架", 基础脚本)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载基础脚本：{基础脚本}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 谱签名(row: dict[str, Any], base: Any) -> str:
    return "|".join(sorted({base.归一化(x) for x in row.get("molecules") or [] if base.归一化(x)}))


def 分折(rows: list[dict[str, Any]], folds: int, base: Any) -> dict[int, int]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[谱签名(row, base)].append(index)
    sizes = [0] * folds
    result: dict[int, int] = {}
    for signature, indices in sorted(clusters.items(), key=lambda x: (-len(x[1]), x[0])):
        fold = min(range(folds), key=lambda x: (sizes[x], x))
        sizes[fold] += len(indices)
        for index in indices:
            result[index] = fold
    return result


def 集合Jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


class 多样化BM25模型:
    """BM25 负责相关性，训练食物分子谱 Jaccard 只用于候选间去冗余。"""

    def __init__(self, rows: list[dict[str, Any]], base: Any, relevance_weight: float):
        self.base = base
        self.relevance_weight = relevance_weight
        self.bm25 = base.BM25候选模型(rows)
        self.rows = rows
        self.profile_by_food = {
            base.归一化(row.get("actual_food")): {
                base.归一化(x) for x in row.get("molecules") or [] if base.归一化(x)
            }
            for row in rows
        }

    def rank(self, row: dict[str, Any], top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ledger, _ = self.bm25.rank(row, len(self.rows))
        if not ledger:
            return [], {}
        maximum = max(float(x.get("score") or 0.0) for x in ledger)
        selected: list[dict[str, Any]] = []
        remaining = list(ledger)
        while remaining and len(selected) < top_k:
            best: tuple[tuple[float, float, int], dict[str, Any]] | None = None
            for item in remaining:
                profile = self.profile_by_food.get(self.base.归一化(item.get("food")), set())
                redundancy = max(
                    (
                        集合Jaccard(
                            profile,
                            self.profile_by_food.get(self.base.归一化(old.get("food")), set()),
                        )
                        for old in selected
                    ),
                    default=0.0,
                )
                relevance = float(item.get("score") or 0.0) / max(maximum, 1e-12)
                objective = self.relevance_weight * relevance - (1.0 - self.relevance_weight) * redundancy
                value = (objective, relevance, -int(item.get("rank") or 10**9))
                if best is None or value > best[0]:
                    best = (value, item)
            assert best is not None
            chosen = dict(best[1])
            chosen["原始BM25名次"] = chosen.get("rank")
            chosen["多样化选择名次"] = len(selected) + 1
            chosen["与已选最大分子谱Jaccard"] = round(
                max(
                    (
                        集合Jaccard(
                            self.profile_by_food.get(self.base.归一化(chosen.get("food")), set()),
                            self.profile_by_food.get(self.base.归一化(old.get("food")), set()),
                        )
                        for old in selected
                    ),
                    default=0.0,
                ),
                6,
            )
            chosen["rank"] = len(selected) + 1
            selected.append(chosen)
            remaining.remove(best[1])
        return selected, {
            "候选方法": "分子谱多样化BM25",
            "BM25相关性权重": self.relevance_weight,
            "使用UniMol": False,
            "使用宏类别": False,
        }


def 三候选(model: Any, row: dict[str, Any], base: Any) -> list[dict[str, Any]]:
    ledger, _ = model.rank(row, 3)
    if len(ledger) != 3:
        raise RuntimeError("无法构造三个具体食物候选")
    return [
        {
            "候选编号": f"C{index}",
            "具体食物": item.get("food"),
            "检索名次": item.get("原始BM25名次", item.get("rank")),
            "BM25分数": item.get("score"),
            "分子集合Jaccard": item.get("molecule_jaccard"),
            "与已选最大分子谱Jaccard": item.get("与已选最大分子谱Jaccard", 0.0),
        }
        for index, item in enumerate(ledger, 1)
    ]


def 候选指标(rows: list[dict[str, Any]], model: Any, categories: dict[str, str], base: Any) -> dict[str, Any]:
    category_hits = 0
    entity_hits = 0
    distinct_categories = 0
    details: list[dict[str, Any]] = []
    for row in rows:
        candidates = 三候选(model, row, base)
        gold_food = base.归一化(row.get("actual_food"))
        gold_category = categories.get(gold_food)
        candidate_foods = [base.归一化(x["具体食物"]) for x in candidates]
        candidate_categories = {categories.get(x) for x in candidate_foods if categories.get(x) is not None}
        category_hit = int(gold_category is not None and gold_category in candidate_categories)
        entity_hit = int(gold_food in candidate_foods)
        category_hits += category_hit
        entity_hits += entity_hit
        distinct_categories += len(candidate_categories)
        details.append({
            "样本编号": str(row.get("id")),
            "真实食物": row.get("actual_food"),
            "候选": candidates,
            "正确类别是否在候选中": bool(category_hit),
            "正确食物是否在候选中": bool(entity_hit),
            "候选不同宏类别数_仅评估": len(candidate_categories),
        })
    count = max(1, len(rows))
    return {
        "样本数": len(rows),
        "Top3宏类别覆盖数": category_hits,
        "Top3宏类别覆盖率": category_hits / count,
        "Top3具体食物覆盖数": entity_hits,
        "Top3具体食物覆盖率": entity_hits / count,
        "平均候选不同宏类别数_仅评估": distinct_categories / count,
        "逐样本": details,
    }


def bootstrap下界(values: list[float], seed: str) -> float:
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(5000))
    return means[max(0, int(0.025 * len(means)) - 1)]


def 训练侧严格OOF(train: list[dict[str, Any]], categories: dict[str, str], base: Any, folds: int) -> tuple[dict[str, Any], float]:
    mapping = 分折(train, folds, base)
    per_weight: dict[float, list[dict[str, Any]]] = {weight: [] for weight in (*候选权重, 1.0)}
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(folds):
        fit = [row for index, row in enumerate(train) if mapping[index] != fold]
        held = [row for index, row in enumerate(train) if mapping[index] == fold]
        fold_result: dict[str, Any] = {"折": fold + 1, "训练数": len(fit), "验证数": len(held), "各权重": {}}
        for weight in (*候选权重, 1.0):
            metrics = 候选指标(held, 多样化BM25模型(fit, base, weight), categories, base)
            per_weight[weight].extend(metrics.pop("逐样本"))
            fold_result["各权重"][str(weight)] = metrics
        fold_summaries.append(fold_result)
        print(f"[MFP训练侧OOF {fold + 1}/{folds}] 已完成 {len(held)} 条", flush=True)
    totals: dict[str, Any] = {}
    for weight, details in per_weight.items():
        totals[str(weight)] = {
            "Top3宏类别覆盖数": sum(int(x["正确类别是否在候选中"]) for x in details),
            "Top3宏类别覆盖率": sum(int(x["正确类别是否在候选中"]) for x in details) / len(details),
            "Top3具体食物覆盖数": sum(int(x["正确食物是否在候选中"]) for x in details),
            "平均候选不同宏类别数_仅评估": sum(x["候选不同宏类别数_仅评估"] for x in details) / len(details),
        }
    selected_weight = max(
        候选权重,
        key=lambda weight: (
            totals[str(weight)]["Top3宏类别覆盖数"],
            totals[str(weight)]["Top3具体食物覆盖数"],
            weight,
        ),
    )
    selected = per_weight[selected_weight]
    baseline = {x["样本编号"]: x for x in per_weight[1.0]}
    gains = [
        int(x["正确类别是否在候选中"]) - int(baseline[x["样本编号"]]["正确类别是否在候选中"])
        for x in selected
    ]
    fold_gains = []
    for fold in range(folds):
        ids = {str(row.get("id")) for index, row in enumerate(train) if mapping[index] == fold}
        current = [gain for row, gain in zip(selected, gains) if row["样本编号"] in ids]
        fold_gains.append(sum(current) / max(1, len(current)))
    paired = {
        "选择的BM25相关性权重": selected_weight,
        "相对普通BM25_Top3覆盖增益": sum(gains) / len(gains),
        "bootstrap_95%下界": bootstrap下界(gains, "第三轮MFP训练侧OOF"),
        "新增覆盖数": sum(x > 0 for x in gains),
        "丢失覆盖数": sum(x < 0 for x in gains),
        "不变数": sum(x == 0 for x in gains),
        "五折增益": fold_gains,
    }
    paired["是否通过训练侧准入"] = bool(
        paired["相对普通BM25_Top3覆盖增益"] > 0
        and paired["bootstrap_95%下界"] > 0
        and sum(x >= 0 for x in fold_gains) >= 4
        and paired["新增覆盖数"] > paired["丢失覆盖数"]
        and totals[str(selected_weight)]["Top3具体食物覆盖数"] >= totals["1.0"]["Top3具体食物覆盖数"]
    )
    return {"各折": fold_summaries, "总体": totals, "配对准入": paired}, selected_weight


def 开发集实验(
    dev: list[dict[str, Any]], train: list[dict[str, Any]], categories: dict[str, str],
    selected_weight: float, output_dir: Path, base: Any,
) -> dict[str, Any]:
    agent = base.加载模块("优化代理_第三轮MFP", 项目根目录 / "code/Only-Deepseek/optimized_agent.py")
    evaluation = base.加载模块("评测模块_第三轮MFP", 项目根目录 / "code/Only-Deepseek/evaluation.py")
    embeddings = agent.EmbeddingStore(项目根目录 / "data/structure/unimol/unimol_embeddings.npz")
    model = 多样化BM25模型(train, base, selected_weight)
    idf = agent.molecule_idf(train, "molecules")
    train_by_food = {base.归一化(x.get("actual_food")): x for x in train}
    with (项目根目录 / "data/collected_evidences/collected_evidences_task1.pkl").open("rb") as handle:
        raw_evidence = pickle.load(handle)
    evidence = {base.归一化(k): [str(v) for v in values] for k, values in raw_evidence.items() if isinstance(values, list)}
    evaluation.load_local_env_file()
    llm_config = evaluation.resolve_llm_config(SimpleNamespace(llm_provider="deepseek", llm_model="deepseek-v4-flash", llm_base_url=None))
    evaluation.require_api_key(llm_config)
    detail_path = output_dir / "开发集逐样本结果.jsonl"
    done = {str(x["样本编号"]) for x in base.读取_jsonl(detail_path)} if detail_path.is_file() else set()
    for index, row in enumerate(dev, 1):
        row_id = str(row.get("id"))
        if row_id in done:
            continue
        candidates = 三候选(model, row, base)
        snippets: list[str] = []
        for molecule in sorted(row.get("molecules") or [], key=lambda x: (-idf.get(base.归一化(x), 0.0), base.归一化(x)))[:8]:
            for evidence_text in evidence.get(base.归一化(molecule), [])[:2]:
                snippets.append(f"分子={molecule}｜证据={evidence_text}")
        evidence_text = "\n".join(snippets) or "没有可用的分子文本证据。"
        scientist = base.解析对象(base.稳健调用(evaluation, base.科学家消息(row, candidates, evidence_text), llm_config))
        structure = base.UniMol结构账本(row, candidates, train_by_food, embeddings, idf)
        reviewer = base.解析对象(base.稳健调用(evaluation, base.审查器消息(row, candidates, scientist, evidence_text, structure), llm_config))
        base.追加_jsonl(detail_path, {
            "样本编号": row_id,
            "真实食物": row.get("actual_food"),
            "固定候选": candidates,
            "固定第一名预测": candidates[0]["具体食物"],
            "科学家分析": scientist,
            "UniMol结构账本": structure,
            "第三轮UniMol审查预测": base.合法选择(reviewer, candidates),
            "第三轮UniMol审查输出": reviewer,
        })
        print(f"[MFP开发集 {index}/{len(dev)}] 已完成样本 {row_id}", flush=True)
    rows = base.读取_jsonl(detail_path)
    baseline_rows = {
        str(x["样本编号"]): x
        for x in base.读取_jsonl(第二轮结果目录 / "逐样本结果.jsonl")
    }
    comparable = []
    for row in rows:
        baseline = baseline_rows.get(str(row["样本编号"]))
        if baseline is None:
            raise RuntimeError(f"第二轮缺少样本 {row['样本编号']}")
        row = dict(row)
        row["第二轮UniMol审查预测"] = baseline["UniMol独占审查预测"]
        comparable.append(row)
    gains = [
        base.类别正确(row, categories, "第三轮UniMol审查预测")
        - base.类别正确(row, categories, "第二轮UniMol审查预测")
        for row in comparable
    ]
    result = {
        "第二轮普通BM25加UniMol审查器": base.指标(comparable, categories, "第二轮UniMol审查预测"),
        "第三轮多样化BM25加冻结UniMol审查器": base.指标(comparable, categories, "第三轮UniMol审查预测"),
        "相对第二轮配对结果": {
            "平均宏类别准确率增益": sum(gains) / len(gains),
            "bootstrap_95%下界": bootstrap下界(gains, "第三轮MFP开发集"),
            "改对数": sum(x > 0 for x in gains),
            "改错数": sum(x < 0 for x in gains),
            "不变数": sum(x == 0 for x in gains),
        },
    }
    paired = result["相对第二轮配对结果"]
    result["是否通过开发集准入"] = bool(
        paired["平均宏类别准确率增益"] > 0
        and paired["bootstrap_95%下界"] > 0
        and paired["改对数"] > paired["改错数"]
        and result["第三轮多样化BM25加冻结UniMol审查器"]["具体食物准确率"]
        >= result["第二轮普通BM25加UniMol审查器"]["具体食物准确率"]
    )
    return result


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("为保证候选集合排序可复现，必须使用 PYTHONHASHSEED=0 启动实验")
    parser = argparse.ArgumentParser()
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument("--折数", type=int, default=5)
    parser.add_argument("--跳过API", action="store_true")
    args = parser.parse_args()
    base = 加载基础模块()
    agent = base.加载模块("优化代理_第三轮MFP类别", 项目根目录 / "code/Only-Deepseek/optimized_agent.py")
    train = base.读取_jsonl(项目根目录 / "results/splits/mfp/train.jsonl")
    dev = base.读取_jsonl(项目根目录 / "results/splits/mfp/dev.jsonl")
    categories = agent.load_food_categories(项目根目录 / "data/raw/flavordb.db")
    args.输出目录.mkdir(parents=True, exist_ok=True)
    protocol = {
        "实验名称": "MFP分子谱多样化BM25科学家验证",
        "唯一变化": "Scientist的三个具体食物候选由普通BM25 Top3改为相关性与候选间分子谱去冗余联合选择",
        "固定部分": "BM25相关性、三个候选预算、Scientist提示、UniMol独占Reviewer及提示",
        "参数选择": "仅在训练集五折分组OOF选择BM25相关性权重，锁定后才查看开发集结果",
        "宏类别是否进入生成": False,
        "UniMol是否进入Scientist": False,
        "正式测试集是否读取": False,
        "Python哈希种子": 0,
        "API条件": "只有训练侧准入通过且未指定--跳过API时，才调用DeepSeek开发集",
        "候选权重": list(候选权重),
    }
    (args.输出目录 / "冻结实验方案.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit, selected_weight = 训练侧严格OOF(train, categories, base, args.折数)
    (args.输出目录 / "训练侧OOF候选审查.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {"训练侧OOF": audit["配对准入"], "锁定BM25相关性权重": selected_weight}
    if audit["配对准入"]["是否通过训练侧准入"] and not args.跳过API:
        summary["开发集"] = 开发集实验(dev, train, categories, selected_weight, args.输出目录, base)
    else:
        summary["开发集"] = {"是否执行": False, "原因": "训练侧未通过准入或显式跳过API"}
    (args.输出目录 / "指标汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
