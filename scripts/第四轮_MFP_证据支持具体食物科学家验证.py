#!/usr/bin/env python3
"""第四轮 MFP：只给 Scientist 增加文本证据支持的具体食物候选。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


项目根目录 = Path(__file__).resolve().parents[1]
第一轮脚本 = 项目根目录 / "scripts/第一轮_MFP_UniMol独占审查器验证.py"
第三轮脚本 = 项目根目录 / "scripts/第三轮_MFP_分子谱多样化BM25科学家验证.py"
默认输出目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第四轮/MFP_证据支持具体食物科学家"
第二轮结果目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第二轮/MFP_BM25候选与UniMol独占审查器"


def 加载脚本(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 文本归一化(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


class 证据支持BM25模型:
    """保留 BM25 前二名，只用多分子直接文本提及决定是否替换第三名。"""

    def __init__(self, rows: list[dict[str, Any]], evidence: dict[str, list[str]], base: Any):
        self.rows = rows
        self.base = base
        self.bm25 = base.BM25候选模型(rows)
        self.idf = self.bm25.idf
        self.food_by_key = {
            base.归一化(row.get("actual_food")): str(row.get("actual_food"))
            for row in rows if base.归一化(row.get("actual_food"))
        }
        phrases = {
            key: 文本归一化(food)
            for key, food in self.food_by_key.items()
            if len(文本归一化(food)) >= 4
        }
        self.mentions_by_molecule: dict[str, set[str]] = {}
        for molecule, snippets in evidence.items():
            text = f" {文本归一化(' '.join(snippets))} "
            mentions = {
                key for key, phrase in phrases.items()
                if phrase and f" {phrase} " in text
            }
            if mentions:
                self.mentions_by_molecule[molecule] = mentions

    def rank(self, row: dict[str, Any], top_k: int = 3) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        baseline, _ = self.bm25.rank(row, max(3, top_k))
        if len(baseline) < 3:
            return baseline, {}
        query = {self.base.归一化(x) for x in row.get("molecules") or [] if self.base.归一化(x)}
        support: dict[str, list[str]] = defaultdict(list)
        for molecule in sorted(query):
            for food_key in self.mentions_by_molecule.get(molecule, set()):
                support[food_key].append(molecule)
        protected = {self.base.归一化(x.get("food")) for x in baseline[:2]}
        eligible = [
            (food_key, molecules)
            for food_key, molecules in support.items()
            if len(set(molecules)) >= 2 and food_key not in protected
        ]
        selected: tuple[str, list[str]] | None = max(
            eligible,
            key=lambda item: (
                len(set(item[1])),
                sum(self.idf.get(molecule, 0.0) for molecule in set(item[1])),
                item[0],
            ),
            default=None,
        )
        result = [dict(x) for x in baseline[:3]]
        if selected is not None:
            food_key, molecules = selected
            result[2] = {
                "rank": 3,
                "food": self.food_by_key[food_key],
                "score": None,
                "molecule_jaccard": None,
                "source": "多分子直接文本证据",
                "直接支持分子数": len(set(molecules)),
                "直接支持分子": sorted(set(molecules)),
                "证据稀有度和": round(sum(self.idf.get(x, 0.0) for x in set(molecules)), 6),
            }
        for index, item in enumerate(result, 1):
            item["rank"] = index
        return result[:top_k], {
            "候选方法": "BM25前二名加多分子直接文本证据第三名",
            "最少直接支持分子数": 2,
            "使用宏类别": False,
            "使用UniMol": False,
            "是否触发证据替换": selected is not None,
        }


def 三候选(model: Any, row: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    ledger, diagnostics = model.rank(row, 3)
    if len(ledger) != 3:
        raise RuntimeError("无法构造三个具体食物候选")
    candidates = [
        {
            "候选编号": f"C{index}",
            "具体食物": item.get("food"),
            "检索名次": index,
            "候选来源": item.get("source"),
            "BM25分数": item.get("score"),
            "分子集合Jaccard": item.get("molecule_jaccard"),
            "直接支持分子数": item.get("直接支持分子数", 0),
            "直接支持分子": item.get("直接支持分子", []),
        }
        for index, item in enumerate(ledger, 1)
    ]
    return candidates, bool(diagnostics.get("是否触发证据替换"))


def bootstrap下界(values: list[float], seed: str) -> float:
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(5000))
    return means[max(0, int(0.025 * len(means)) - 1)]


def 训练侧OOF(
    rows: list[dict[str, Any]], evidence: dict[str, list[str]], categories: dict[str, str],
    folds: int, base: Any, third: Any,
) -> dict[str, Any]:
    mapping = third.分折(rows, folds, base)
    details: list[dict[str, Any]] = []
    fold_gains: list[float] = []
    for fold in range(folds):
        fit = [row for index, row in enumerate(rows) if mapping[index] != fold]
        held = [row for index, row in enumerate(rows) if mapping[index] == fold]
        baseline = base.BM25候选模型(fit)
        evidence_model = 证据支持BM25模型(fit, evidence, base)
        fold_values: list[int] = []
        for row in held:
            baseline_candidates, _ = 三候选(baseline, row)
            evidence_candidates, triggered = 三候选(evidence_model, row)
            gold_food = base.归一化(row.get("actual_food"))
            gold_category = categories.get(gold_food)
            baseline_foods = [base.归一化(x["具体食物"]) for x in baseline_candidates]
            evidence_foods = [base.归一化(x["具体食物"]) for x in evidence_candidates]
            baseline_hit = int(gold_category is not None and gold_category in {categories.get(x) for x in baseline_foods})
            evidence_hit = int(gold_category is not None and gold_category in {categories.get(x) for x in evidence_foods})
            gain = evidence_hit - baseline_hit
            fold_values.append(gain)
            details.append({
                "样本编号": str(row.get("id")),
                "折": fold + 1,
                "真实食物": row.get("actual_food"),
                "普通BM25候选": baseline_candidates,
                "证据支持候选": evidence_candidates,
                "是否触发证据替换": triggered,
                "普通BM25类别覆盖": bool(baseline_hit),
                "证据支持类别覆盖": bool(evidence_hit),
                "类别覆盖增益": gain,
                "普通BM25具体食物覆盖": gold_food in baseline_foods,
                "证据支持具体食物覆盖": gold_food in evidence_foods,
            })
        fold_gain = sum(fold_values) / max(1, len(fold_values))
        fold_gains.append(fold_gain)
        print(f"[MFP第四轮OOF {fold + 1}/{folds}] 增益={fold_gain:.6f}", flush=True)
    gains = [int(x["类别覆盖增益"]) for x in details]
    baseline_hits = sum(int(x["普通BM25类别覆盖"]) for x in details)
    evidence_hits = sum(int(x["证据支持类别覆盖"]) for x in details)
    baseline_entity = sum(int(x["普通BM25具体食物覆盖"]) for x in details)
    evidence_entity = sum(int(x["证据支持具体食物覆盖"]) for x in details)
    summary = {
        "样本数": len(details),
        "普通BM25_Top3类别覆盖": baseline_hits / len(details),
        "证据支持_Top3类别覆盖": evidence_hits / len(details),
        "平均类别覆盖增益": sum(gains) / len(gains),
        "bootstrap_95%下界": bootstrap下界(gains, "第四轮MFP证据候选"),
        "新增覆盖数": sum(x > 0 for x in gains),
        "丢失覆盖数": sum(x < 0 for x in gains),
        "不变数": sum(x == 0 for x in gains),
        "触发证据替换数": sum(int(x["是否触发证据替换"]) for x in details),
        "普通BM25_Top3具体食物覆盖数": baseline_entity,
        "证据支持_Top3具体食物覆盖数": evidence_entity,
        "五折增益": fold_gains,
    }
    summary["是否通过训练侧准入"] = bool(
        summary["平均类别覆盖增益"] > 0
        and summary["bootstrap_95%下界"] > 0
        and sum(x >= 0 for x in fold_gains) >= 4
        and summary["新增覆盖数"] > summary["丢失覆盖数"]
        and evidence_entity >= baseline_entity
    )
    return {"指标": summary, "逐样本": details}


def 开发集实验(
    dev: list[dict[str, Any]], train: list[dict[str, Any]], evidence: dict[str, list[str]],
    categories: dict[str, str], output_dir: Path, base: Any,
) -> dict[str, Any]:
    agent = base.加载模块("第四轮MFP代理", 项目根目录 / "code/Only-Deepseek/optimized_agent.py")
    evaluation = base.加载模块("第四轮MFP评测", 项目根目录 / "code/Only-Deepseek/evaluation.py")
    embeddings = agent.EmbeddingStore(项目根目录 / "data/structure/unimol/unimol_embeddings.npz")
    idf = agent.molecule_idf(train, "molecules")
    train_by_food = {base.归一化(x.get("actual_food")): x for x in train}
    model = 证据支持BM25模型(train, evidence, base)
    evaluation.load_local_env_file()
    llm_config = evaluation.resolve_llm_config(SimpleNamespace(llm_provider="deepseek", llm_model="deepseek-v4-flash", llm_base_url=None))
    evaluation.require_api_key(llm_config)
    detail_path = output_dir / "开发集逐样本结果.jsonl"
    done = {str(x["样本编号"]) for x in base.读取_jsonl(detail_path)} if detail_path.is_file() else set()
    for index, row in enumerate(dev, 1):
        row_id = str(row.get("id"))
        if row_id in done:
            continue
        candidates, triggered = 三候选(model, row)
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
            "是否触发证据替换": triggered,
            "固定候选": candidates,
            "科学家分析": scientist,
            "UniMol结构账本": structure,
            "第四轮预测": base.合法选择(reviewer, candidates),
            "第四轮审查输出": reviewer,
        })
        print(f"[MFP第四轮开发集 {index}/{len(dev)}] 已完成 {row_id}", flush=True)
    rows = base.读取_jsonl(detail_path)
    baseline = {str(x["样本编号"]): x for x in base.读取_jsonl(第二轮结果目录 / "逐样本结果.jsonl")}
    gains: list[int] = []
    for row in rows:
        row["第二轮预测"] = baseline[str(row["样本编号"])]["UniMol独占审查预测"]
        gains.append(base.类别正确(row, categories, "第四轮预测") - base.类别正确(row, categories, "第二轮预测"))
    result = {
        "第二轮": base.指标(rows, categories, "第二轮预测"),
        "第四轮": base.指标(rows, categories, "第四轮预测"),
        "配对": {
            "平均宏类别准确率增益": sum(gains) / len(gains),
            "bootstrap_95%下界": bootstrap下界(gains, "第四轮MFP开发集"),
            "改对数": sum(x > 0 for x in gains),
            "改错数": sum(x < 0 for x in gains),
            "不变数": sum(x == 0 for x in gains),
        },
    }
    result["是否通过开发集准入"] = bool(
        result["配对"]["平均宏类别准确率增益"] > 0
        and result["配对"]["bootstrap_95%下界"] > 0
        and result["配对"]["改对数"] > result["配对"]["改错数"]
        and result["第四轮"]["具体食物准确率"] >= result["第二轮"]["具体食物准确率"]
    )
    return result


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("必须使用 PYTHONHASHSEED=0 启动实验")
    parser = argparse.ArgumentParser()
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument("--折数", type=int, default=5)
    parser.add_argument("--跳过API", action="store_true")
    args = parser.parse_args()
    args.输出目录.mkdir(parents=True, exist_ok=True)
    base = 加载脚本("第四轮MFP基础", 第一轮脚本)
    third = 加载脚本("第四轮MFP分折", 第三轮脚本)
    agent = base.加载模块("第四轮MFP类别", 项目根目录 / "code/Only-Deepseek/optimized_agent.py")
    train = base.读取_jsonl(项目根目录 / "results/splits/mfp/train.jsonl")
    dev = base.读取_jsonl(项目根目录 / "results/splits/mfp/dev.jsonl")
    categories = agent.load_food_categories(项目根目录 / "data/raw/flavordb.db")
    with (项目根目录 / "data/collected_evidences/collected_evidences_task1.pkl").open("rb") as handle:
        raw = pickle.load(handle)
    evidence = {base.归一化(k): [str(x) for x in values] for k, values in raw.items() if isinstance(values, list)}
    protocol = {
        "实验名称": "MFP证据支持具体食物科学家",
        "唯一变化": "BM25第三候选可被至少两个查询分子的文本证据直接提及的具体食物替换",
        "固定部分": "BM25前二名、三个候选预算、Scientist提示、UniMol Reviewer",
        "宏类别是否进入生成": False,
        "Gold是否进入生成": False,
        "UniMol是否进入Scientist": False,
        "最少直接支持分子数": 2,
        "Python哈希种子": 0,
        "正式测试集是否读取": False,
        "API条件": "训练侧准入通过且未指定--跳过API时才调用MFP开发集DeepSeek",
    }
    (args.输出目录 / "冻结实验方案.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = 训练侧OOF(train, evidence, categories, args.折数, base, third)
    (args.输出目录 / "训练侧OOF证据候选审查.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {"训练侧OOF": audit["指标"]}
    if audit["指标"]["是否通过训练侧准入"] and not args.跳过API:
        summary["开发集"] = 开发集实验(dev, train, evidence, categories, args.输出目录, base)
    else:
        summary["开发集"] = {"是否执行": False, "原因": "训练侧未通过准入或显式跳过API"}
    (args.输出目录 / "指标汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
