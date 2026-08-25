#!/usr/bin/env python3
"""第六轮 MFP：用全食物名称词表审计官方文本证据能否产生开放具体食物候选。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[1]
第一轮脚本 = 项目根目录 / "scripts/第一轮_MFP_UniMol独占审查器验证.py"
第三轮脚本 = 项目根目录 / "scripts/第三轮_MFP_分子谱多样化BM25科学家验证.py"
默认输出目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第六轮/MFP_全食物名称证据桥科学家"


def 加载模块(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 文本归一化(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split())


def 加载食物名称词表(db_path: Path) -> dict[str, str]:
    """只读名称字段；不查询 entity_molecule_link 或任何分子谱。"""
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT entity_alias_readable, entity_alias, natural_source_name,
                   entity_alias_synonyms
            FROM food_entities
            """
        ).fetchall()
    phrase_to_foods: dict[str, set[str]] = defaultdict(set)
    for readable, alias, natural_name, synonyms in rows:
        display = str(readable or alias or natural_name or "").strip()
        if not display:
            continue
        names = [readable, alias, natural_name]
        if synonyms:
            names.extend(re.split(r"[,;|]", str(synonyms)))
        for name in names:
            phrase = 文本归一化(name)
            if len(phrase) >= 4:
                phrase_to_foods[phrase].add(display)
    # 同一短语对应多个实体时不作消歧，宁可放弃也不人为选择。
    return {
        phrase: next(iter(foods))
        for phrase, foods in phrase_to_foods.items()
        if len(foods) == 1
    }


def 构建证据提及索引(
    evidence: dict[str, list[str]], phrases: dict[str, str], agent: Any, base: Any,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    mentions: dict[str, set[str]] = {}
    direct_snippets = 0
    matched_snippets = 0
    for molecule, snippets in evidence.items():
        found: set[str] = set()
        for snippet in snippets:
            if agent.classify_evidence_snippet(str(snippet)) != "direct_occurrence":
                continue
            direct_snippets += 1
            text = f" {文本归一化(snippet)} "
            before = len(found)
            for phrase, food in phrases.items():
                if f" {phrase} " in text:
                    found.add(food)
            if len(found) > before:
                matched_snippets += 1
        if found:
            mentions[base.归一化(molecule)] = found
    return mentions, {
        "词表唯一名称短语数": len(phrases),
        "直接出现证据片段数": direct_snippets,
        "命中至少一个食物名称的直接证据片段数": matched_snippets,
        "有食物名称提及的分子数": len(mentions),
    }


class 全词表证据候选模型:
    def __init__(self, rows: list[dict[str, Any]], mentions: dict[str, set[str]], threshold: int, base: Any):
        self.base = base
        self.bm25 = base.BM25候选模型(rows)
        self.mentions = mentions
        self.threshold = threshold
        self.idf = self.bm25.idf

    def rank(self, row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        baseline, _ = self.bm25.rank(row, 3)
        if len(baseline) < 3:
            return baseline, {"是否替换": False}
        support: dict[str, set[str]] = defaultdict(set)
        for molecule in row.get("molecules") or []:
            key = self.base.归一化(molecule)
            for food in self.mentions.get(key, set()):
                support[food].add(key)
        protected = {self.base.归一化(x.get("food")) for x in baseline[:2]}
        eligible = [
            (food, molecules) for food, molecules in support.items()
            if len(molecules) >= self.threshold and self.base.归一化(food) not in protected
        ]
        selected = max(
            eligible,
            key=lambda item: (
                len(item[1]),
                sum(self.idf.get(molecule, 0.0) for molecule in item[1]),
                self.base.归一化(item[0]),
            ),
            default=None,
        )
        result = [dict(x) for x in baseline]
        if selected is not None:
            food, molecules = selected
            result[2] = {
                "rank": 3,
                "food": food,
                "source": "全食物名称词表中的多分子直接出现证据",
                "直接支持分子数": len(molecules),
                "直接支持分子": sorted(molecules),
            }
        return result, {"是否替换": selected is not None}


def 评估数据(
    fit: list[dict[str, Any]], held: list[dict[str, Any]], mentions: dict[str, set[str]],
    threshold: int, categories: dict[str, str], base: Any,
) -> list[dict[str, Any]]:
    baseline = base.BM25候选模型(fit)
    method = 全词表证据候选模型(fit, mentions, threshold, base)
    details: list[dict[str, Any]] = []
    for row in held:
        old, _ = baseline.rank(row, 3)
        new, diagnostics = method.rank(row)
        gold_food = base.归一化(row.get("actual_food"))
        gold_category = categories.get(gold_food)
        old_foods = [base.归一化(x.get("food")) for x in old]
        new_foods = [base.归一化(x.get("food")) for x in new]
        old_categories = {categories.get(x) for x in old_foods if categories.get(x)}
        new_categories = {categories.get(x) for x in new_foods if categories.get(x)}
        details.append({
            "样本编号": str(row.get("id")),
            "真实食物": row.get("actual_food"),
            "普通BM25候选": [x.get("food") for x in old],
            "证据桥候选": [x.get("food") for x in new],
            "是否替换": diagnostics["是否替换"],
            "普通具体食物命中": gold_food in old_foods,
            "证据桥具体食物命中": gold_food in new_foods,
            "普通宏类别覆盖": bool(gold_category and gold_category in old_categories),
            "证据桥宏类别覆盖": bool(gold_category and gold_category in new_categories),
        })
    return details


def 汇总(details: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(details)
    old_entity = sum(int(x["普通具体食物命中"]) for x in details)
    new_entity = sum(int(x["证据桥具体食物命中"]) for x in details)
    old_macro = sum(int(x["普通宏类别覆盖"]) for x in details)
    new_macro = sum(int(x["证据桥宏类别覆盖"]) for x in details)
    return {
        "样本数": n,
        "普通BM25_Top3具体食物召回": old_entity / n,
        "证据桥_Top3具体食物召回": new_entity / n,
        "具体食物召回增益": (new_entity - old_entity) / n,
        "普通BM25_Top3宏类别oracle": old_macro / n,
        "证据桥_Top3宏类别oracle": new_macro / n,
        "宏类别oracle增益": (new_macro - old_macro) / n,
        "触发替换数": sum(int(x["是否替换"]) for x in details),
        "具体食物新增命中数": sum(int(x["证据桥具体食物命中"] and not x["普通具体食物命中"]) for x in details),
        "宏类别新增覆盖数": sum(int(x["证据桥宏类别覆盖"] and not x["普通宏类别覆盖"]) for x in details),
        "宏类别丢失覆盖数": sum(int(x["普通宏类别覆盖"] and not x["证据桥宏类别覆盖"]) for x in details),
    }


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("必须使用 PYTHONHASHSEED=0 启动实验")
    parser = argparse.ArgumentParser()
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument("--折数", type=int, default=5)
    args = parser.parse_args()
    args.输出目录.mkdir(parents=True, exist_ok=True)

    base = 加载模块("第六轮MFP基础", 第一轮脚本)
    third = 加载模块("第六轮MFP分折", 第三轮脚本)
    agent = 加载模块("第六轮MFP代理", 项目根目录 / "code/Only-Deepseek/optimized_agent.py")
    train = base.读取_jsonl(项目根目录 / "results/splits/mfp/train.jsonl")
    dev = base.读取_jsonl(项目根目录 / "results/splits/mfp/dev.jsonl")
    db_path = 项目根目录 / "data/raw/flavordb.db"
    categories = agent.load_food_categories(db_path)
    phrases = 加载食物名称词表(db_path)
    with (项目根目录 / "data/collected_evidences/collected_evidences_task1.pkl").open("rb") as handle:
        raw_evidence = pickle.load(handle)
    evidence = {
        base.归一化(key): [str(x) for x in values]
        for key, values in raw_evidence.items() if isinstance(values, list)
    }
    mentions, index_audit = 构建证据提及索引(evidence, phrases, agent, base)

    protocol = {
        "实验名称": "MFP全食物名称证据桥Scientist验证",
        "唯一变化": "第三候选允许来自FlavorDB全食物名称词表，而非仅训练集食物名",
        "候选证据": "只有官方task1文本中被判为direct_occurrence的片段，并按独立输入分子数支持",
        "只读数据库字段": ["entity_alias_readable", "entity_alias", "natural_source_name", "entity_alias_synonyms"],
        "明确禁止": ["entity_molecule_link", "食物完整分子谱", "开发集Gold参与生成", "宏类别参与生成", "API", "正式测试集"],
        "冻结BM25": "前二名不变，证据候选只替换第三名",
        "预注册阈值": [1, 2],
        "训练OOF选择规则": "具体食物召回增益>0且宏类别oracle增益>=0；再按具体食物增益、宏类别增益、较高阈值排序",
        "开发集执行门槛": "至少一个阈值通过训练OOF选择规则",
        "索引审计": index_audit,
        "正式测试集是否读取": False,
        "API调用次数": 0,
    }
    (args.输出目录 / "冻结实验方案.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fold_map = third.分折(train, args.折数, base)
    oof_results: dict[str, Any] = {}
    for threshold in (1, 2):
        details: list[dict[str, Any]] = []
        fold_summaries: list[dict[str, Any]] = []
        for fold in range(args.折数):
            fit = [row for index, row in enumerate(train) if fold_map[index] != fold]
            held = [row for index, row in enumerate(train) if fold_map[index] == fold]
            fold_details = 评估数据(fit, held, mentions, threshold, categories, base)
            details.extend(fold_details)
            fold_summaries.append(汇总(fold_details))
        summary = 汇总(details)
        summary["五折"] = fold_summaries
        summary["通过训练OOF门槛"] = bool(summary["具体食物召回增益"] > 0 and summary["宏类别oracle增益"] >= 0)
        oof_results[str(threshold)] = {"指标": summary, "逐样本": details}
        print(f"[MFP第六轮 阈值={threshold}] {json.dumps(summary, ensure_ascii=False)}", flush=True)

    eligible = [
        threshold for threshold in (1, 2)
        if oof_results[str(threshold)]["指标"]["通过训练OOF门槛"]
    ]
    selected = max(
        eligible,
        key=lambda threshold: (
            oof_results[str(threshold)]["指标"]["具体食物召回增益"],
            oof_results[str(threshold)]["指标"]["宏类别oracle增益"],
            threshold,
        ),
        default=None,
    )
    output: dict[str, Any] = {"训练侧OOF": oof_results, "选择阈值": selected}
    if selected is not None:
        dev_details = 评估数据(train, dev, mentions, selected, categories, base)
        output["开发集"] = {"指标": 汇总(dev_details), "逐样本": dev_details}
    else:
        output["开发集"] = {"是否执行": False, "原因": "没有阈值同时提高具体食物召回且不降低宏类别oracle"}
    (args.输出目录 / "完整审计结果.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_output = {
        "训练侧阈值1": oof_results["1"]["指标"],
        "训练侧阈值2": oof_results["2"]["指标"],
        "选择阈值": selected,
        "开发集": output["开发集"].get("指标", output["开发集"]),
    }
    (args.输出目录 / "指标汇总.json").write_text(json.dumps(summary_output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary_output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
