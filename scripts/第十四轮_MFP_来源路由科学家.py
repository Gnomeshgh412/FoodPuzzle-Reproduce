#!/usr/bin/env python3
"""MFP 第十四轮：冻结数据与 Reviewer，只改变 Scientist 的三候选来源路由。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from typing import Any


根目录 = Path(__file__).resolve().parents[1]
默认输出 = 根目录 / "results/Only-Deepseek/优化实验/第十四轮/MFP_来源路由科学家"
基线 = 根目录 / "results/Only-Deepseek/优化实验/第十一轮/双任务瓶颈补证/MFP_完整科学家阶段分解_联网重试"
随机种子 = 20260810


def 加载(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 读_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def 写_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"禁止覆盖：{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def 写_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"禁止覆盖：{path}")
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def 追加(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def 规范(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def 准备(output: Path) -> None:
    protocol = {
        "任务输出": "具体食物名称",
        "当前瓶颈": "原始Scientist把BM25类比、文献字符串和参数知识混在一次自由生成中，没有来源级假设构造规则。",
        "唯一改动": "候选1固定为BM25 Top1具体食物类比；一次Scientist调用分别生成候选2的多分子证据假设与候选3的参数知识假设。",
        "三个来源": ["BM25 Top1训练具体食物", "至少两个查询分子支持的文献证据假设", "不依赖文献食物字符串的参数知识假设"],
        "冻结": ["DeepSeek模型", "完整查询分子", "5个起始分子", "每分子3条官方证据", "BM25 Top3检索", "Reviewer Prompt和输入", "官方评测器"],
        "对照": "第十一轮K3完整Scientific Agent",
        "探索保留": ["Scientist Top3宏类别oracle>0.577465", "Reviewer宏类别accuracy>0.352113", "Reviewer wins>losses", "具体食物Top3>=2/71", "71/71成功", "无直接宏类别输出"],
        "结论上限": "dev已被多轮自适应复用；通过也只能局部信号，不冻结论文主方法。",
        "API预算": "71次Scientist+71次Reviewer+最多284次固定类别映射",
        "边界": "不读正式test；不增加FlavorDB数据或文献证据；不向Scientist/Reviewer提供宏类别。",
    }
    写_json(output / "冻结实验方案.json", protocol)
    print(json.dumps({"状态": "冻结完成"}, ensure_ascii=False))


def 构造来源路由消息(agent: Any, row: dict[str, Any], anchor: str, evidence_text: str) -> list[dict[str, str]]:
    prompt = (
        "Task:\n"
        "Infer concrete food sources from the FoodPuzzle molecule set. The final output must contain concrete food names, never macro categories.\n\n"
        "FoodPuzzle molecules:\n"
        f"{agent.format_molecules(row)}\n\n"
        "Frozen retrieval anchor:\n"
        f"{anchor}\n\n"
        "Retrieved molecule evidence:\n"
        f"{evidence_text}\n\n"
        "Generate exactly two additional concrete-food hypotheses with separated provenance:\n"
        "1. evidence_hypothesis: Ignore the retrieval anchor as an answer. Use the molecule evidence and the full query. "
        "The rationale and supporting_molecules must identify at least two distinct query molecules that jointly support the food. "
        "Do not choose a food merely because one evidence snippet mentions its name.\n"
        "2. parametric_hypothesis: Ignore food-name strings in the retrieved evidence. Use broader food-chemistry knowledge and the complete molecule combination. "
        "It must differ from both the retrieval anchor and evidence_hypothesis.\n\n"
        "Do not output a macro category, an ingredient class, or a placeholder. Do not output Markdown or extra text.\n"
        "Output JSON only:\n"
        "{\n"
        '  "evidence_hypothesis": {"predicted_food": "...", "supporting_molecules": ["...", "..."], "rationale": "..."},\n'
        '  "parametric_hypothesis": {"predicted_food": "...", "rationale": "..."}\n'
        "}"
    )
    return [
        {"role": "system", "content": "You are a FoodPuzzle Scientist agent that returns only valid JSON."},
        {"role": "user", "content": prompt},
    ]


def 解析来源路由(agent: Any, content: str, query_molecules: list[Any], anchor: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    data = agent.parse_json_object(content)
    if not isinstance(data, dict):
        return None, "JSON解析失败"
    evidence = data.get("evidence_hypothesis")
    parametric = data.get("parametric_hypothesis")
    if not isinstance(evidence, dict) or not isinstance(parametric, dict):
        return None, "缺少两个来源假设"
    ef = str(evidence.get("predicted_food") or "").strip()
    pf = str(parametric.get("predicted_food") or "").strip()
    support = evidence.get("supporting_molecules")
    if not ef or not pf or not isinstance(support, list):
        return None, "来源假设字段不完整"
    query = {规范(x) for x in query_molecules}
    valid_support = []
    for molecule in support:
        text = str(molecule).strip()
        if 规范(text) in query and 规范(text) not in {规范(x) for x in valid_support}:
            valid_support.append(text)
    if len(valid_support) < 2:
        return None, "证据假设不足两个查询分子支持"
    if len({规范(anchor), 规范(ef), 规范(pf)}) != 3:
        return None, "三个来源候选未保持不同具体食物"
    return [
        {"predicted_food": anchor, "rationale": "Frozen BM25 Top-1 training-food analogue.", "来源": "BM25类比"},
        {"predicted_food": ef, "rationale": str(evidence.get("rationale") or "").strip(), "supporting_molecules": valid_support, "来源": "多分子文献证据"},
        {"predicted_food": pf, "rationale": str(parametric.get("rationale") or "").strip(), "来源": "参数知识"},
    ], None


def 执行(output: Path, args: argparse.Namespace) -> None:
    agent = 加载("第十四轮MFP_agent", 根目录 / "code/Only-Deepseek/scientific_agent.py")
    evaluation = agent.load_evaluation_module()
    evaluation.load_local_env_file()
    config = evaluation.resolve_llm_config(args)
    evaluation.require_api_key(config)
    train = 读_jsonl(根目录 / "results/splits/mfp/train.jsonl")
    dev = 读_jsonl(根目录 / "results/splits/mfp/dev.jsonl")
    train_by_id = {str(x["id"]): x for x in train}
    retrieval = agent.load_retrieval_metadata(基线 / "实际检索元数据.jsonl")
    evidence, _ = agent.load_official_evidence(根目录 / "data/collected_evidences/collected_evidences_task1.pkl")
    idf = agent.build_train_idf(train)
    path = output / "来源路由与Reviewer元数据.jsonl"
    done = {str(x["id"]) for x in 读_jsonl(path) if not x.get("错误") and x.get("Reviewer输出")} if path.exists() and args.resume else set()
    for index, row in enumerate(dev, 1):
        rid = str(row["id"])
        if rid in done:
            continue
        demos = agent.resolve_demos(rid, retrieval, train_by_id, 3)
        if len(demos) != 3:
            raise RuntimeError(f"BM25 Top3不完整：id={rid}")
        anchor = str(demos[0]["row"]["actual_food"]).strip()
        starts = agent.select_starting_molecules(row, idf, evidence, 5)
        _, evidence_text, hits = agent.build_evidence_blocks(row, starts, evidence, 3)
        hypotheses = None
        reviewer = None
        error = None
        raw = None
        try:
            raw = evaluation.call_chat_completion(构造来源路由消息(agent, row, anchor, evidence_text), config)
            hypotheses, parse_error = 解析来源路由(agent, raw, row.get("molecules", []), anchor)
            if hypotheses is None:
                raise RuntimeError(parse_error or "来源路由解析失败")
            reviewer_raw = evaluation.call_chat_completion(
                agent.build_reviewer_messages(row, evidence_text, demos, hypotheses), config
            )
            reviewer = agent.parse_reviewer_output(reviewer_raw)
            if reviewer is None:
                raise RuntimeError("Reviewer解析失败")
        except Exception as exc:
            if "Insufficient Balance" in str(exc) or "HTTP error: 402" in str(exc):
                raise
            error = str(exc)
        追加(path, {
            "id": rid,
            "actual_food_for_audit": row.get("actual_food"),
            "BM25锚点": anchor,
            "来源路由假设": hypotheses,
            "Reviewer输出": reviewer,
            "证据屏蔽后答案命中数": hits,
            "Scientist原始响应": raw,
            "错误": error,
        })
        print(f"MFP第十四轮进度: {index}/71 id={rid} {'成功' if not error else '失败'}", flush=True)
    rows = 读_jsonl(path)
    success = {str(x["id"]) for x in rows if not x.get("错误") and x.get("Reviewer输出")}
    print(json.dumps({"成功ID数": len(success), "记录数": len(rows)}, ensure_ascii=False))


def 拆分(output: Path) -> None:
    dev = {str(x["id"]): x for x in 读_jsonl(根目录 / "results/splits/mfp/dev.jsonl")}
    attempts = 读_jsonl(output / "来源路由与Reviewer元数据.jsonl")
    success = {str(x["id"]): x for x in attempts if not x.get("错误") and x.get("Reviewer输出")}
    if len(success) != 71:
        raise RuntimeError("未获得71个完整结果")
    labels = ["BM25锚点候选", "多分子证据候选", "参数知识候选"]
    predictions = {label: [] for label in labels}
    reviewers, diagnostics = [], []
    category_words = {"cereal", "fruit", "essential oil", "plant", "bakery", "fungus", "seed", "dish", "spice", "flower", "nutseed", "beverage", "animal product", "vegetable", "dairy", "fish seafood", "herb", "legume", "meat", "additive"}
    for rid in dev:
        row = success[rid]
        hypotheses = row["来源路由假设"]
        foods = [str(x["predicted_food"]).strip() for x in hypotheses]
        reviewer_food = str(row["Reviewer输出"]["predicted_food"]).strip()
        for label, food in zip(labels, foods):
            predictions[label].append({"id": rid, "predicted_food": food})
        reviewers.append({"id": rid, "predicted_food": reviewer_food})
        gold = 规范(dev[rid]["actual_food"])
        diagnostics.append({
            "id": rid,
            "Scientist_Top1具体食物正确": 规范(foods[0]) == gold,
            "Scientist_Top3具体食物正确": gold in {规范(x) for x in foods},
            "Reviewer具体食物正确": 规范(reviewer_food) == gold,
            "Reviewer选择序号": row["Reviewer输出"].get("selected_hypothesis_index"),
            "直接宏类别输出": 规范(reviewer_food) in category_words,
            "三候选不同数": len({规范(x) for x in foods}),
        })
    for label in labels:
        写_jsonl(output / f"{label}_官方评测输入.jsonl", predictions[label])
    写_jsonl(output / "Reviewer最终预测_官方评测输入.jsonl", reviewers)
    写_jsonl(output / "具体食物诊断.jsonl", diagnostics)
    print(json.dumps({"状态": "拆分完成"}, ensure_ascii=False))


def bootstrap(values: list[float], repeats: int = 10000) -> list[float]:
    rng = random.Random(随机种子)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(repeats))
    return [means[int(0.025 * repeats)], means[int(0.975 * repeats) - 1]]


def 汇总(output: Path) -> None:
    labels = ["BM25锚点候选", "多分子证据候选", "参数知识候选", "Reviewer最终预测"]
    current = {label: {str(x["id"]): x for x in 读_jsonl(output / f"{label}_官方评测逐样本.jsonl")} for label in labels}
    old_labels = ["科学家第1候选", "科学家第2候选", "科学家第3候选", "审查器最终预测"]
    old = {label: {str(x["id"]): x for x in 读_jsonl(基线 / f"{label}_官方类别逐样本.jsonl")} for label in old_labels}
    ids = [str(x["id"]) for x in 读_jsonl(根目录 / "results/splits/mfp/dev.jsonl")]
    details, oracle_gains, reviewer_gains = [], [], []
    for index, rid in enumerate(ids):
        old_oracle = any(old[x][rid]["correct"] for x in old_labels[:3])
        new_oracle = any(current[x][rid]["correct"] for x in labels[:3])
        old_reviewer = bool(old[old_labels[3]][rid]["correct"])
        new_reviewer = bool(current[labels[3]][rid]["correct"])
        oracle_gains.append(float(new_oracle) - float(old_oracle))
        reviewer_gains.append(float(new_reviewer) - float(old_reviewer))
        details.append({"id": rid, "固定分块": index % 5, "K3_oracle": old_oracle, "方法_oracle": new_oracle, "K3_Reviewer": old_reviewer, "方法_Reviewer": new_reviewer})
    diag = 读_jsonl(output / "具体食物诊断.jsonl")
    summary = {
        "样本数": 71,
        "三来源单列宏类别accuracy": [sum(current[x][rid]["correct"] for rid in ids) / 71 for x in labels[:3]],
        "K3_Scientist_Top3宏类别oracle": sum(x["K3_oracle"] for x in details) / 71,
        "方法_Scientist_Top3宏类别oracle": sum(x["方法_oracle"] for x in details) / 71,
        "Scientist_oracle增益": sum(oracle_gains) / 71,
        "Scientist_oracle增益bootstrap_95%区间": bootstrap(oracle_gains),
        "K3_Reviewer宏类别accuracy": sum(x["K3_Reviewer"] for x in details) / 71,
        "方法_Reviewer宏类别accuracy": sum(x["方法_Reviewer"] for x in details) / 71,
        "Reviewer增益": sum(reviewer_gains) / 71,
        "Reviewer增益bootstrap_95%区间": bootstrap(reviewer_gains),
        "Reviewer_wins_losses_ties": [sum(x > 0 for x in reviewer_gains), sum(x < 0 for x in reviewer_gains), sum(x == 0 for x in reviewer_gains)],
        "Reviewer五分块增益": [sum(reviewer_gains[i] for i, x in enumerate(details) if x["固定分块"] == f) / sum(x["固定分块"] == f for x in details) for f in range(5)],
        "Scientist_Top1具体食物命中数": sum(x["Scientist_Top1具体食物正确"] for x in diag),
        "Scientist_Top3具体食物命中数": sum(x["Scientist_Top3具体食物正确"] for x in diag),
        "Reviewer具体食物命中数": sum(x["Reviewer具体食物正确"] for x in diag),
        "直接宏类别输出数": sum(x["直接宏类别输出"] for x in diag),
        "三候选全部不同样本数": sum(x["三候选不同数"] == 3 for x in diag),
    }
    summary["是否通过探索保留"] = bool(
        summary["方法_Scientist_Top3宏类别oracle"] > summary["K3_Scientist_Top3宏类别oracle"]
        and summary["方法_Reviewer宏类别accuracy"] > summary["K3_Reviewer宏类别accuracy"]
        and summary["Reviewer_wins_losses_ties"][0] > summary["Reviewer_wins_losses_ties"][1]
        and summary["Scientist_Top3具体食物命中数"] >= 2
        and summary["直接宏类别输出数"] == 0
        and summary["三候选全部不同样本数"] == 71
    )
    summary["结论"] = "获得局部信号并探索保留，不冻结论文主方法" if summary["是否通过探索保留"] else "未通过并停止"
    写_jsonl(output / "配对审查逐样本.jsonl", details)
    写_json(output / "完整审查结果.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("动作", choices=["准备", "执行", "拆分", "汇总"])
    parser.add_argument("--输出", type=Path, default=默认输出)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm-provider", default="deepseek")
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--llm-base-url", default=None)
    args = parser.parse_args()
    if args.动作 == "准备": 准备(args.输出)
    elif args.动作 == "执行": 执行(args.输出, args)
    elif args.动作 == "拆分": 拆分(args.输出)
    else: 汇总(args.输出)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
