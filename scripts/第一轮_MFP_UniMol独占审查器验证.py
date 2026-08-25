#!/usr/bin/env python3
"""第一轮 MFP 开发集验证：固定候选下审查器是否从 UniMol 获得独立增益。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pickle
import random
import time
from collections import Counter
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


项目根目录 = Path(__file__).resolve().parents[1]
默认输出目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第一轮/MFP_UniMol独占审查器"


def 加载模块(名称: str, 路径: Path) -> Any:
    spec = importlib.util.spec_from_file_location(名称, 路径)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块：{路径}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[名称] = module
    spec.loader.exec_module(module)
    return module


def 读取_jsonl(路径: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in 路径.read_text(encoding="utf-8").splitlines() if line.strip()]


def 追加_jsonl(路径: Path, 记录: dict[str, Any]) -> None:
    路径.parent.mkdir(parents=True, exist_ok=True)
    with 路径.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(记录, ensure_ascii=False, sort_keys=True) + "\n")


def 解析对象(文本: str) -> dict[str, Any]:
    try:
        value = json.loads(文本)
    except Exception:
        start, end = 文本.find("{"), 文本.rfind("}")
        if start < 0 or end <= start:
            return {}
        try:
            value = json.loads(文本[start : end + 1])
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def 稳健调用(evaluation: Any, messages: list[dict[str, str]], llm_config: dict[str, str]) -> str:
    """只重试明确的瞬时限流或服务端错误，不改变请求与模型。"""
    delays = (2, 4, 8, 15, 30, 30, 30)
    for attempt in range(len(delays) + 1):
        try:
            return evaluation.call_chat_completion(messages, llm_config)
        except evaluation.ChatCompletionHTTPError as exc:
            if exc.status not in {429, 500, 502, 503, 504} or attempt >= len(delays):
                raise
            delay = delays[attempt]
            print(f"DeepSeek瞬时错误HTTP {exc.status}，{delay}秒后重试第{attempt + 2}次。", flush=True)
            time.sleep(delay)
        except evaluation.EvaluationError as exc:
            if "API request failed" not in str(exc) or attempt >= len(delays):
                raise
            delay = delays[attempt]
            print(f"DeepSeek瞬时网络中断，{delay}秒后重试第{attempt + 2}次。", flush=True)
            time.sleep(delay)
    raise RuntimeError("DeepSeek瞬时错误重试耗尽")


def 归一化(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def 食物类别(食物: Any, 类别映射: dict[str, str]) -> str | None:
    return 类别映射.get(归一化(食物))


def 固定候选(model: Any, row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # 候选生成故意关闭 UniMol，仅按分子集合 Jaccard 排序。
    ledger, diagnostics = model.rank(row, top_k=30)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ledger:
        key = 归一化(item.get("food"))
        if not key or key in seen:
            continue
        candidates.append(
            {
                "候选编号": f"C{len(candidates) + 1}",
                "具体食物": item.get("food"),
                "检索名次": item.get("rank"),
                "分子集合Jaccard": item.get("molecule_jaccard"),
            }
        )
        seen.add(key)
        if len(candidates) == 3:
            break
    if len(candidates) != 3:
        raise RuntimeError("无法构造三个固定具体食物候选")
    return candidates, diagnostics


class BM25候选模型:
    """只依据训练侧具体食物分子谱检索，不使用 UniMol 或宏类别。"""

    def __init__(self, rows: list[dict[str, Any]], k1: float = 1.2, b: float = 0.75):
        self.rows = rows
        self.k1 = k1
        self.b = b
        self.profile_sets = [
            {归一化(x) for x in row.get("molecules") or [] if 归一化(x)}
            for row in rows
        ]
        df: Counter[str] = Counter()
        for profile in self.profile_sets:
            df.update(profile)
        count = max(1, len(rows))
        self.idf = {
            key: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for key, frequency in df.items()
        }
        self.average_length = sum(map(len, self.profile_sets)) / count

    def rank(self, row: dict[str, Any], top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        query = {归一化(x) for x in row.get("molecules") or [] if 归一化(x)}
        scored: list[tuple[float, int, float]] = []
        for index, profile in enumerate(self.profile_sets):
            norm = self.k1 * (
                1.0 - self.b + self.b * len(profile) / max(self.average_length, 1e-12)
            )
            score = sum(
                self.idf.get(term, 0.0) * (self.k1 + 1.0) / (1.0 + norm)
                for term in query & profile
            )
            union = query | profile
            jaccard = len(query & profile) / len(union) if union else 0.0
            scored.append((score, index, jaccard))
        scored.sort(key=lambda x: (-x[0], x[1]))
        ledger = [
            {
                "rank": rank,
                "food": self.rows[index].get("actual_food"),
                "score": round(score, 6),
                "molecule_jaccard": round(jaccard, 6),
                "source": "训练侧分子BM25",
            }
            for rank, (score, index, jaccard) in enumerate(scored[:top_k], 1)
        ]
        return ledger, {
            "候选方法": "训练侧分子BM25",
            "使用UniMol": False,
            "使用宏类别": False,
            "k1": self.k1,
            "b": self.b,
        }


def UniMol结构账本(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    train_by_food: dict[str, dict[str, Any]],
    embeddings: Any,
    idf: dict[str, float],
) -> list[dict[str, Any]]:
    query_names = [name for name in row.get("molecules") or [] if embeddings.vector(name) is not None]
    per_candidate: list[dict[str, Any]] = []
    pairwise_by_candidate: list[Any] = []
    for candidate in candidates:
        profile = train_by_food[归一化(candidate["具体食物"])].get("molecules") or []
        profile_names = [name for name in profile if embeddings.vector(name) is not None]
        q2f, f2q, nq, nf = embeddings.weighted_chamfer(query_names, profile_names, idf)
        if query_names and profile_names:
            qmat = embeddings.np.asarray([embeddings.vector(name) for name in query_names])
            fmat = embeddings.np.asarray([embeddings.vector(name) for name in profile_names])
            pairwise = qmat @ fmat.T
            best_index = pairwise.argmax(axis=1)
            matches = [
                {
                    "查询分子": str(query_names[i]),
                    "候选内最相近分子": str(profile_names[int(best_index[i])]),
                    "余弦相似度": round(float(pairwise[i, int(best_index[i])]), 6),
                    "训练侧稀有度": round(float(idf.get(归一化(query_names[i]), 0.0)), 6),
                }
                for i in range(len(query_names))
            ]
            matches.sort(key=lambda x: (-x["训练侧稀有度"], -x["余弦相似度"], x["查询分子"]))
            pairwise_by_candidate.append(pairwise.max(axis=1))
        else:
            matches = []
            pairwise_by_candidate.append(embeddings.np.zeros(len(query_names)))
        per_candidate.append(
            {
                "候选编号": candidate["候选编号"],
                "具体食物": candidate["具体食物"],
                "查询到候选覆盖": q2f,
                "候选到查询覆盖": f2q,
                "已映射查询分子数": nq,
                "已映射候选分子数": nf,
                "高稀有度诊断匹配": matches[:5],
            }
        )
    if query_names:
        matrix = embeddings.np.vstack(pairwise_by_candidate).T
        for cidx, item in enumerate(per_candidate):
            other = embeddings.np.max(embeddings.np.delete(matrix, cidx, axis=1), axis=1)
            advantages = [
                {
                    "查询分子": str(query_names[i]),
                    "相对另外候选的结构优势": round(float(matrix[i, cidx] - other[i]), 6),
                }
                for i in range(len(query_names))
            ]
            advantages.sort(key=lambda x: (-x["相对另外候选的结构优势"], x["查询分子"]))
            item["最强候选独占结构支持"] = advantages[:5]
    combined = [0.6 * x["查询到候选覆盖"] + 0.4 * x["候选到查询覆盖"] for x in per_candidate]
    order = sorted(range(len(combined)), key=lambda i: (-combined[i], i))
    for rank, idx in enumerate(order, 1):
        per_candidate[idx]["双向结构相对名次"] = rank
        per_candidate[idx]["双向结构相对分数"] = round(combined[idx], 6)
        per_candidate[idx]["相对第一名差值"] = round(combined[idx] - combined[order[0]], 6)
    return per_candidate


def 科学家消息(row: dict[str, Any], candidates: list[dict[str, Any]], evidence_text: str) -> list[dict[str, str]]:
    prompt = f"""FoodPuzzle MFP 开发实验。输入是一组分子，任务是从固定候选中选出最合理的具体食物名称。
输入分子：{json.dumps(row.get('molecules') or [], ensure_ascii=False)}
固定候选（候选由不含 UniMol 的分子重叠检索产生）：{json.dumps(candidates, ensure_ascii=False)}
分子证据：\n{evidence_text}

逐一审查三个候选。不得输出宏类别，不得改名、重排或创造候选。气味相似不等于存在性证据。
只返回 JSON：{{"候选分析":[{{"候选编号":"C1","具体食物":"...","支持":[],"冲突":[],"置信度":0.0}}]}}"""
    return [{"role": "system", "content": "你是严谨的风味科学家，只返回有效 JSON。"}, {"role": "user", "content": prompt}]


def 审查器消息(
    row: dict[str, Any], candidates: list[dict[str, Any]], scientist: dict[str, Any],
    evidence_text: str, structural_ledger: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    extra = (
        "本组不提供 UniMol；你只能核验科学家分析和文本证据。"
        if structural_ledger is None else
        "以下 UniMol 结构账本仅提供相对、双向和分子级核验信号，不是答案或类别标签：\n"
        + json.dumps(structural_ledger, ensure_ascii=False)
    )
    prompt = f"""FoodPuzzle MFP 独立审查。必须从三个固定候选中选择一个具体食物。
输入分子：{json.dumps(row.get('molecules') or [], ensure_ascii=False)}
固定候选：{json.dumps(candidates, ensure_ascii=False)}
科学家分析：{json.dumps(scientist, ensure_ascii=False)}
文本证据：\n{evidence_text}
{extra}

不得输出宏类别、创造第四个候选或把数值分数当作真值。只返回 JSON：
{{"选择的候选编号":"C1","具体食物":"...","支持":[],"冲突":[],"拒绝的主张":[]}}"""
    return [{"role": "system", "content": "你是保守且独立的科学审查器，只返回有效 JSON。"}, {"role": "user", "content": prompt}]


def 合法选择(output: dict[str, Any], candidates: list[dict[str, Any]]) -> str:
    by_id = {x["候选编号"]: str(x["具体食物"]) for x in candidates}
    by_name = {归一化(x["具体食物"]): str(x["具体食物"]) for x in candidates}
    cid = str(output.get("选择的候选编号") or "")
    name = 归一化(output.get("具体食物"))
    if cid in by_id and (not name or 归一化(by_id[cid]) == name):
        return by_id[cid]
    if name in by_name:
        return by_name[name]
    return str(candidates[0]["具体食物"])


def 指标(rows: list[dict[str, Any]], categories: dict[str, str], key: str) -> dict[str, Any]:
    correct_category = 0
    correct_entity = 0
    mapped = 0
    for row in rows:
        pred = row[key]
        gold = row["真实食物"]
        pc, gc = 食物类别(pred, categories), 食物类别(gold, categories)
        mapped += int(pc is not None and gc is not None)
        correct_category += int(pc is not None and gc is not None and pc == gc)
        correct_entity += int(归一化(pred) == 归一化(gold))
    n = max(1, len(rows))
    return {"样本数": len(rows), "宏类别准确率": correct_category / n, "具体食物准确率": correct_entity / n, "类别成功映射数": mapped}


def 类别正确(row: dict[str, Any], categories: dict[str, str], key: str) -> int:
    pred = 食物类别(row[key], categories)
    gold = 食物类别(row["真实食物"], categories)
    return int(pred is not None and gold is not None and pred == gold)


def 配对审查(rows: list[dict[str, Any]], categories: dict[str, str], baseline_key: str) -> dict[str, Any]:
    gains = [
        类别正确(row, categories, "UniMol独占审查预测") - 类别正确(row, categories, baseline_key)
        for row in rows
    ]
    rng = random.Random(f"MFP-UniMol-{baseline_key}-20260803")
    means = sorted(
        sum(gains[rng.randrange(len(gains))] for _ in gains) / len(gains)
        for _ in range(5000)
    )
    folds: list[list[float]] = [[] for _ in range(5)]
    for index, gain in enumerate(gains):
        folds[index % 5].append(gain)
    return {
        "平均宏类别准确率增益": sum(gains) / max(1, len(gains)),
        "bootstrap_95%下界": means[max(0, int(0.025 * len(means)) - 1)],
        "改对数": sum(x > 0 for x in gains),
        "改错数": sum(x < 0 for x in gains),
        "不变数": sum(x == 0 for x in gains),
        "五个固定分块增益": [sum(x) / max(1, len(x)) for x in folds],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--仅准备", action="store_true")
    parser.add_argument("--输出目录", type=Path, default=默认输出目录)
    parser.add_argument("--候选方法", choices=("jaccard", "bm25"), default="jaccard")
    args = parser.parse_args()
    agent = 加载模块("优化代理_第一轮MFP", 项目根目录 / "code/Only-Deepseek/optimized_agent.py")
    evaluation = 加载模块("评测模块_第一轮MFP", 项目根目录 / "code/Only-Deepseek/evaluation.py")
    train = 读取_jsonl(项目根目录 / "results/splits/mfp/train.jsonl")
    dev = 读取_jsonl(项目根目录 / "results/splits/mfp/dev.jsonl")
    categories = agent.load_food_categories(项目根目录 / "data/raw/flavordb.db")
    embeddings = agent.EmbeddingStore(项目根目录 / "data/structure/unimol/unimol_embeddings.npz")
    model = (
        BM25候选模型(train)
        if args.候选方法 == "bm25"
        else agent.MFPStructureModel(train, None, categories, "no_unimol")
    )
    idf = agent.molecule_idf(train, "molecules")
    train_by_food = {归一化(x.get("actual_food")): x for x in train}
    with (项目根目录 / "data/collected_evidences/collected_evidences_task1.pkl").open("rb") as handle:
        raw_evidence = pickle.load(handle)
    evidence = {归一化(k): [str(v) for v in values] for k, values in raw_evidence.items() if isinstance(values, list)}
    args.输出目录.mkdir(parents=True, exist_ok=True)
    config = {
        "实验名称": (
            "MFP BM25候选与UniMol独占审查器验证"
            if args.候选方法 == "bm25"
            else "MFP UniMol 独占审查器验证"
        ),
        "数据": "重建开发集，训练集仅作检索库",
        "开发样本数": len(dev),
        "候选生成": (
            "关闭UniMol和宏类别的训练侧分子BM25，固定三个具体食物"
            if args.候选方法 == "bm25"
            else "关闭 UniMol 的分子集合 Jaccard，固定三个具体食物"
        ),
        "对照": ["固定检索第一名", "无UniMol审查器", "UniMol独占结构审查器"],
        "宏类别是否进入生成": False,
        "正式测试集是否读取": False,
        "模型": "deepseek-v4-flash",
        "准入条件": "UniMol组同时优于固定第一名和无UniMol组；配对bootstrap下界大于0；至少4/5折非负；改错少于改对；具体食物准确率不下降",
    }
    (args.输出目录 / "冻结实验方案.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.仅准备:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0
    evaluation.load_local_env_file()
    llm_config = evaluation.resolve_llm_config(SimpleNamespace(llm_provider="deepseek", llm_model="deepseek-v4-flash", llm_base_url=None))
    evaluation.require_api_key(llm_config)
    detail_path = args.输出目录 / "逐样本结果.jsonl"
    done = {str(x["样本编号"]) for x in 读取_jsonl(detail_path)} if detail_path.is_file() else set()
    for index, row in enumerate(dev, 1):
        row_id = str(row.get("id"))
        if row_id in done:
            continue
        candidates, _ = 固定候选(model, row)
        snippets: list[str] = []
        for molecule in sorted(row.get("molecules") or [], key=lambda x: (-idf.get(归一化(x), 0.0), 归一化(x)))[:8]:
            for text in evidence.get(归一化(molecule), [])[:2]:
                snippets.append(f"分子={molecule}｜证据={text}")
        evidence_text = "\n".join(snippets) or "没有可用的分子文本证据。"
        scientist = 解析对象(稳健调用(evaluation, 科学家消息(row, candidates, evidence_text), llm_config))
        no_structure = 解析对象(稳健调用(evaluation, 审查器消息(row, candidates, scientist, evidence_text, None), llm_config))
        structure = UniMol结构账本(row, candidates, train_by_food, embeddings, idf)
        with_structure = 解析对象(稳健调用(evaluation, 审查器消息(row, candidates, scientist, evidence_text, structure), llm_config))
        record = {
            "样本编号": row_id,
            "真实食物": row.get("actual_food"),
            "固定候选": candidates,
            "固定第一名预测": candidates[0]["具体食物"],
            "科学家分析": scientist,
            "无UniMol审查输出": no_structure,
            "无UniMol审查预测": 合法选择(no_structure, candidates),
            "UniMol结构账本": structure,
            "UniMol独占审查输出": with_structure,
            "UniMol独占审查预测": 合法选择(with_structure, candidates),
        }
        追加_jsonl(detail_path, record)
        print(f"[{index}/{len(dev)}] 已完成样本 {row_id}", flush=True)
    rows = 读取_jsonl(detail_path)
    summary = {
        "固定第一名": 指标(rows, categories, "固定第一名预测"),
        "无UniMol审查器": 指标(rows, categories, "无UniMol审查预测"),
        "UniMol独占审查器": 指标(rows, categories, "UniMol独占审查预测"),
        "相对固定第一名的配对审查": 配对审查(rows, categories, "固定第一名预测"),
        "相对无UniMol审查器的配对审查": 配对审查(rows, categories, "无UniMol审查预测"),
    }
    checks = [summary["相对固定第一名的配对审查"], summary["相对无UniMol审查器的配对审查"]]
    summary["是否通过准入"] = bool(
        all(x["平均宏类别准确率增益"] > 0 for x in checks)
        and all(x["bootstrap_95%下界"] > 0 for x in checks)
        and all(sum(v >= 0 for v in x["五个固定分块增益"]) >= 4 for x in checks)
        and all(x["改对数"] > x["改错数"] for x in checks)
        and summary["UniMol独占审查器"]["具体食物准确率"] >= summary["无UniMol审查器"]["具体食物准确率"]
    )
    (args.输出目录 / "指标汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
