#!/usr/bin/env python3
"""MFP 第十二轮：Top9 检索拆成三个不重叠视角，各自独立生成一个候选。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any


根目录 = Path(__file__).resolve().parents[1]
默认输出 = 根目录 / "results/Only-Deepseek/优化实验/第十二轮/MFP_分视角独立科学家"
基线目录 = 根目录 / "results/Only-Deepseek/优化实验/第十一轮/双任务瓶颈补证/MFP_完整科学家阶段分解_联网重试"
随机种子 = 20260809
视角索引 = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]


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


def 规范(x: Any) -> str:
    return " ".join(str(x or "").strip().lower().split())


def 已完成(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(x["id"]) for x in 读_jsonl(path) if not x.get("error") and x.get("审查器输出")}


def 准备(output: Path) -> None:
    bm25 = 加载("第十二轮MFP_BM25", 根目录 / "code/Only-Deepseek/bm25_icl.py")
    train = 读_jsonl(根目录 / "results/splits/mfp/train.jsonl")
    dev = 读_jsonl(根目录 / "results/splits/mfp/dev.jsonl")
    if {str(x["id"]) for x in train} & {str(x["id"]) for x in dev}:
        raise RuntimeError("train/dev ID 重叠")
    index = bm25.BM25Index(train, "mfp")
    metadata = []
    for row in dev:
        retrieved = index.retrieve(row, 9)
        metadata.append({"id": row["id"], "retrieved": [{"id": x["id"], "rank": x["rank"], "score": x["score"], "actual_food": x["actual_food"], "target_food": x["target_food"]} for x in retrieved]})
    if len(metadata) != 71 or any(len(x["retrieved"]) != 9 for x in metadata):
        raise RuntimeError("Top9 检索不完整")
    protocol = {
        "任务输出": "具体食物名称",
        "主指标": "Reviewer 具体食物映射后的宏类别 accuracy",
        "当前瓶颈": "Scientist Top3 候选缺失正确类别是最大单一错误源；扩大单次长提示的K并未提高Top3 oracle。",
        "历史借鉴": ["第三轮候选多样性在train OOF有正增益，但被当时4/5分块硬门槛拒绝", "第十一轮K=10单调用的Top3 oracle零增益"],
        "唯一改动": "把BM25 Top9按排名拆为(1,4,7)/(2,5,8)/(3,6,9)三个不重叠视角，三次独立Scientist调用各取第1个具体食物，形成三候选。",
        "冻结": ["Scientist Prompt", "Reviewer Prompt", "Reviewer使用原Top3 demonstrations", "DeepSeek模型", "官方证据", "起始分子数", "每分子证据数", "评测器"],
        "对照": ["K=3完整Agent", "K=10单次Scientist"],
        "探索保留": ["Scientist Top3宏类别oracle>0.577465", "Reviewer宏类别accuracy>0.352113", "wins>losses", "71/71解析成功", "无直接宏类别输出", "95%CI和五分块只报告不否决"],
        "论文主方法冻结": "本轮即使探索保留也不直接宣称主方法，仍需更严格的独立证据。",
        "边界": "不读正式test；不使用任务外FlavorDB数据；不输出宏类别；不根据dev中途结果调参。",
    }
    写_json(output / "冻结实验方案.json", protocol)
    写_jsonl(output / "开发集BM25_Top9检索元数据.jsonl", metadata)
    print(json.dumps({"状态": "冻结完成", "样本数": 71}, ensure_ascii=False))


def 执行(output: Path, args: argparse.Namespace) -> None:
    agent = 加载("第十二轮MFP_agent", 根目录 / "code/Only-Deepseek/scientific_agent.py")
    evaluation = agent.load_evaluation_module()
    evaluation.load_local_env_file()
    config = evaluation.resolve_llm_config(args)
    evaluation.require_api_key(config)
    train = 读_jsonl(根目录 / "results/splits/mfp/train.jsonl")
    dev = 读_jsonl(根目录 / "results/splits/mfp/dev.jsonl")
    train_by_id = {str(x["id"]): x for x in train}
    retrieval = agent.load_retrieval_metadata(output / "开发集BM25_Top9检索元数据.jsonl")
    evidence, evidence_info = agent.load_official_evidence(根目录 / "data/collected_evidences/collected_evidences_task1.pkl")
    idf = agent.build_train_idf(train)
    result_path = output / "分视角假设与审查元数据.jsonl"
    done = 已完成(result_path) if args.resume else set()
    for index, row in enumerate(dev, 1):
        row_id = str(row["id"])
        if row_id in done:
            continue
        selected = agent.select_starting_molecules(row, idf, evidence, 5)
        blocks, evidence_text, hits = agent.build_evidence_blocks(row, selected, evidence, 3)
        all_demos = agent.resolve_demos(row_id, retrieval, train_by_id, 9)
        branches, candidates, error = [], [], None
        try:
            for view_no, positions in enumerate(视角索引, 1):
                demos = [all_demos[p] for p in positions]
                content = evaluation.call_chat_completion(agent.build_scientist_messages(row, selected, evidence_text, demos), config)
                hypotheses = agent.parse_hypotheses(content) or []
                if not hypotheses:
                    raise RuntimeError(f"视角{view_no} Scientist解析失败")
                candidates.append(hypotheses[0])
                branches.append({"视角": view_no, "检索排名": [p+1 for p in positions], "demonstration_ids": [str(x["id"]) for x in demos], "完整三候选": hypotheses[:3], "纳入审查器的第一候选": hypotheses[0]})
            reviewer_demos = all_demos[:3]
            reviewer_content = evaluation.call_chat_completion(agent.build_reviewer_messages(row, evidence_text, reviewer_demos, candidates), config)
            reviewer = agent.parse_reviewer_output(reviewer_content)
            if reviewer is None:
                raise RuntimeError("Reviewer解析失败")
        except Exception as exc:
            if "Insufficient Balance" in str(exc) or "HTTP error: 402" in str(exc):
                raise
            reviewer, error = None, str(exc)
        追加(result_path, {"id": row_id, "actual_food_for_audit": row.get("actual_food"), "起始分子": selected, "证据屏蔽后答案命中数": hits, "视角分支": branches, "给审查器的三候选": candidates, "审查器输出": reviewer, "错误": error})
        print(f"MFP进度: {index}/71 id={row_id} {'成功' if not error else '失败'}", flush=True)
    rows = 读_jsonl(result_path)
    print(json.dumps({"总行数": len(rows), "成功数": sum(not x.get("错误") for x in rows), "证据SHA256": evidence_info["sha256"]}, ensure_ascii=False))


def 拆分(output: Path) -> None:
    dev = {str(x["id"]): x for x in 读_jsonl(根目录 / "results/splits/mfp/dev.jsonl")}
    attempts = 读_jsonl(output / "分视角假设与审查元数据.jsonl")
    # 断点重试不覆盖失败记录；审查时每个 ID 只取最后一次成功尝试。
    latest_success = {}
    for attempt in attempts:
        if not attempt.get("错误") and attempt.get("审查器输出"):
            latest_success[str(attempt["id"])] = attempt
    rows = [latest_success[row_id] for row_id in dev if row_id in latest_success]
    if len(rows) != 71:
        raise RuntimeError("未获得71条完整成功结果")
    candidates = {1: [], 2: [], 3: []}; reviewer_rows = []; diagnostic = []
    category_words = {"cereal","fruit","essential oil","plant","bakery","fungus","seed","dish","spice","flower","nutseed","beverage","animal product","vegetable","dairy","fish seafood","herb","legume","meat","additive"}
    for x in rows:
        row_id = str(x["id"]); hypotheses = x["给审查器的三候选"]; reviewer = x["审查器输出"]
        foods = [str(h["predicted_food"]).strip() for h in hypotheses]
        for i, food in enumerate(foods, 1): candidates[i].append({"id": row_id, "predicted_food": food})
        reviewer_food = str(reviewer["predicted_food"]).strip(); reviewer_rows.append({"id": row_id, "predicted_food": reviewer_food})
        gold = 规范(dev[row_id]["actual_food"])
        diagnostic.append({"id": row_id, "Scientist_Top1具体食物正确": 规范(foods[0])==gold, "Scientist_Top3具体食物正确": gold in {规范(f) for f in foods}, "Reviewer具体食物正确": 规范(reviewer_food)==gold, "Reviewer选择序号": reviewer.get("selected_hypothesis_index"), "直接宏类别输出": 规范(reviewer_food) in category_words})
    for i in (1,2,3): 写_jsonl(output / f"Scientist第{i}候选_官方评测输入.jsonl", candidates[i])
    写_jsonl(output / "Reviewer最终预测_官方评测输入.jsonl", reviewer_rows)
    写_jsonl(output / "具体食物诊断.jsonl", diagnostic)
    print(json.dumps({"状态": "拆分完成", "样本数": 71}, ensure_ascii=False))


def bootstrap(values: list[float], repeats: int = 10000) -> list[float]:
    rng = random.Random(随机种子)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(repeats))
    return [means[int(0.025 * repeats)], means[int(0.975 * repeats) - 1]]


def 汇总(output: Path) -> None:
    labels = ["Scientist第1候选", "Scientist第2候选", "Scientist第3候选", "Reviewer最终预测"]
    current = {label: {str(x["id"]): x for x in 读_jsonl(output / f"{label}_官方评测逐样本.jsonl")} for label in labels}
    old_labels = ["科学家第1候选", "科学家第2候选", "科学家第3候选", "审查器最终预测"]
    baseline = {label: {str(x["id"]): x for x in 读_jsonl(基线目录 / f"{label}_官方类别逐样本.jsonl")} for label in old_labels}
    ids = [str(x["id"]) for x in 读_jsonl(根目录 / "results/splits/mfp/dev.jsonl")]
    details, oracle_gains, reviewer_gains = [], [], []
    for index, row_id in enumerate(ids):
        old_oracle = any(bool(baseline[x][row_id]["correct"]) for x in old_labels[:3])
        new_oracle = any(bool(current[x][row_id]["correct"]) for x in labels[:3])
        old_reviewer = bool(baseline[old_labels[3]][row_id]["correct"])
        new_reviewer = bool(current[labels[3]][row_id]["correct"])
        oracle_gains.append(float(new_oracle)-float(old_oracle)); reviewer_gains.append(float(new_reviewer)-float(old_reviewer))
        details.append({"id": row_id, "固定分块": index % 5, "K3_Scientist_oracle": old_oracle, "分视角_Scientist_oracle": new_oracle, "K3_Reviewer正确": old_reviewer, "分视角_Reviewer正确": new_reviewer})
    diagnostics = 读_jsonl(output / "具体食物诊断.jsonl")
    metadata = {str(x["id"]): x for x in 读_jsonl(output / "分视角假设与审查元数据.jsonl") if not x.get("错误")}
    unique_counts = [len({规范(h["predicted_food"]) for h in metadata[row_id]["给审查器的三候选"]}) for row_id in ids]
    fold_gains = [sum(reviewer_gains[i] for i,x in enumerate(details) if x["固定分块"]==fold)/sum(x["固定分块"]==fold for x in details) for fold in range(5)]
    new_oracle_count = sum(x["分视角_Scientist_oracle"] for x in details)
    reviewer_count = sum(x["分视角_Reviewer正确"] for x in details)
    summary = {
        "样本数": 71,
        "三候选单列宏类别accuracy": [sum(current[label][i]["correct"] for i in ids)/71 for label in labels[:3]],
        "K3_Scientist_Top3宏类别oracle": sum(x["K3_Scientist_oracle"] for x in details)/71,
        "分视角_Scientist_Top3宏类别oracle": new_oracle_count/71,
        "Scientist_oracle增益": sum(oracle_gains)/71,
        "Scientist_oracle增益bootstrap_95%区间": bootstrap(oracle_gains),
        "K3_Reviewer宏类别accuracy": sum(x["K3_Reviewer正确"] for x in details)/71,
        "分视角_Reviewer宏类别accuracy": reviewer_count/71,
        "Reviewer增益": sum(reviewer_gains)/71,
        "Reviewer增益bootstrap_95%区间": bootstrap(reviewer_gains),
        "Reviewer_wins_losses_ties": [sum(x>0 for x in reviewer_gains), sum(x<0 for x in reviewer_gains), sum(x==0 for x in reviewer_gains)],
        "Reviewer五个固定分块增益": fold_gains,
        "Reviewer相对本轮Scientist_oracle的选择损失数": new_oracle_count-reviewer_count,
        "三候选平均不同具体食物数": sum(unique_counts)/71,
        "三候选完全重复样本数": sum(x==1 for x in unique_counts),
        "Scientist_Top1具体食物命中数": sum(x["Scientist_Top1具体食物正确"] for x in diagnostics),
        "Scientist_Top3具体食物命中数": sum(x["Scientist_Top3具体食物正确"] for x in diagnostics),
        "Reviewer具体食物命中数": sum(x["Reviewer具体食物正确"] for x in diagnostics),
        "直接宏类别输出数": sum(x["直接宏类别输出"] for x in diagnostics),
    }
    summary["是否通过探索保留"] = bool(summary["分视角_Scientist_Top3宏类别oracle"] > summary["K3_Scientist_Top3宏类别oracle"] and summary["分视角_Reviewer宏类别accuracy"] > summary["K3_Reviewer宏类别accuracy"] and summary["Reviewer_wins_losses_ties"][0] > summary["Reviewer_wins_losses_ties"][1] and summary["直接宏类别输出数"] == 0)
    summary["结论"] = "获得局部信号但暂不准入" if summary["是否通过探索保留"] else "未通过并停止"
    summary["机制审查"] = "按检索位次硬拆分没有产生互补候选；三候选oracle显著下降，因此不能将历史上的‘多样性’简化为三次独立调用。"
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
    {"准备": 准备, "执行": lambda p: 执行(p, args), "拆分": 拆分, "汇总": 汇总}[args.动作](args.输出)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
