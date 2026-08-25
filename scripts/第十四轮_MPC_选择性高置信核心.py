#!/usr/bin/env python3
"""MPC 第十四轮：Scientist 可对不确定分子拒答，只输出有序高置信核心。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


根目录 = Path(__file__).resolve().parents[1]
默认输出 = 根目录 / "results/Only-Deepseek/优化实验/第十四轮/MPC_选择性高置信核心"
第十轮 = 根目录 / "results/Only-Deepseek/优化实验/第十轮/MPC_共识锚定候选效用"
第十三轮 = 根目录 / "results/Only-Deepseek/优化实验/第十三轮/MPC_嵌套OOF后置安全门控"
第七轮H1 = 根目录 / "results/Only-Deepseek/优化实验/第七轮/MPC_当前H1开发集正式口径控制/当前H1开发集预测.jsonl"
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


def 准备(output: Path) -> None:
    protocol = {
        "任务输出": "由冻结H1补足后的exact-N具体缺失分子集合",
        "当前瓶颈": "原始Scientist被迫在不确定时仍生成完整N集合，高置信食品信息与低置信填充混合；第十三轮只能整包接受或回退。",
        "唯一改动": "Scientist不再承担exact-N，只输出任意长度、按置信度降序的具体缺失分子核心；不确定分子允许不输出。",
        "冻结下游": ["第十二轮三项秩特征排序器", "第十三轮嵌套OOF查询级gate", "H1", "共识锁定", "exact-N"],
        "训练请求": "568条exact-profile五折OOF；只发送target_food、partial、N及外折训练侧Top3 demonstrations，不发送当前查询Gold。",
        "训练准入": ["macro和micro具体分子F1均高于第十三轮", "相对H1 wins>losses", "相对H1 losses<=27", "至少3/5折相对第十三轮非负", "568/568 exact-N", "568/568 Scientist调用可评价"],
        "停止条件": "训练任一条件失败即停止，不读dev Gold，不修改Prompt、核心长度、排序器或gate。",
        "开发集准入": ["具体分子macro/micro均高于第十三轮", "相对第十三轮wins>losses", "71/71 exact-N", "官方官能团F1高于第十三轮0.654204"],
        "结论上限": "dev已多轮自适应复用；通过也只能局部信号，不冻结论文主方法。",
        "边界": "不读正式test；不增加FlavorDB数据、证据或中间模型；不以官能团训练Scientist、排序器或gate。",
    }
    写_json(output / "冻结实验方案.json", protocol)
    print(json.dumps({"状态": "冻结完成"}, ensure_ascii=False))


def 演示文本(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for index, row in enumerate(rows, 1):
        blocks.append(
            f"Example {index}:\n"
            f"Target food: {row.get('target_food')}\n"
            f"Known molecules: {', '.join(str(x) for x in row.get('partial_molecules', []))}\n"
            f"Missing molecules observed in this training example: {', '.join(str(x) for x in row.get('missing_molecules', []))}"
        )
    return "\n\n".join(blocks)


def 构造消息(row: dict[str, Any], demos: list[dict[str, Any]]) -> list[dict[str, str]]:
    n = int(row["n"])
    prompt = (
        "Task:\n"
        "You are the Scientist stage of FoodPuzzle MPC. Identify only high-confidence concrete molecules likely missing from the target food.\n\n"
        f"Target food: {row.get('target_food')}\n"
        f"Known molecules: {', '.join(str(x) for x in row.get('partial_molecules', []))}\n"
        f"Final downstream missing-set size: {n}\n\n"
        "Training demonstrations:\n"
        f"{演示文本(demos)}\n\n"
        "A deterministic downstream model, not you, will complete the final list to the required size. "
        "Do not guess merely to reach the final size. Return any number from zero to the final size. "
        "A short or empty list is preferable to uncertain filler. Rank molecules from highest to lowest confidence. "
        "Use the target food and cross-example molecular support; do not copy a molecule solely because it is globally frequent. "
        "Exclude every molecule already present in Known molecules. Use concrete molecule common names only.\n"
        "Do not output functional groups, explanations, confidence numbers, Markdown, or extra text.\n"
        'Output JSON only: {"high_confidence_molecules": ["molecule 1", "molecule 2"]}'
    )
    return [
        {"role": "system", "content": "You are a FoodPuzzle MPC Scientist that returns only valid JSON."},
        {"role": "user", "content": prompt},
    ]


def 解析核心(content: str, row: dict[str, Any], base: Any) -> tuple[list[str] | None, str | None]:
    try:
        data = json.loads(content)
    except Exception:
        match = __import__("re").search(r"\{.*\}", content, __import__("re").DOTALL)
        if not match:
            return None, "JSON解析失败"
        try:
            data = json.loads(match.group(0))
        except Exception:
            return None, "JSON解析失败"
    if not isinstance(data, dict) or not isinstance(data.get("high_confidence_molecules"), list):
        return None, "缺少high_confidence_molecules列表"
    partial = {base.规范(x) for x in row.get("partial_molecules", [])}
    output, seen = [], set()
    for value in data["high_confidence_molecules"]:
        if not isinstance(value, str):
            continue
        text, key = value.strip(), base.规范(value)
        if not text or not key or key in partial or key in seen:
            continue
        output.append(text)
        seen.add(key)
        if len(output) >= int(row["n"]):
            break
    return output, None


def API配置(args: argparse.Namespace) -> tuple[Any, Any]:
    agent = 加载("第十四轮MPC_agent", 根目录 / "code/Only-Deepseek/scientific_agent.py")
    evaluation = agent.load_evaluation_module()
    evaluation.load_local_env_file()
    config = evaluation.resolve_llm_config(args)
    evaluation.require_api_key(config)
    return evaluation, config


def 执行训练Scientist(output: Path, args: argparse.Namespace) -> None:
    base = 加载("第十四轮MPC_base", 根目录 / "scripts/第十二轮_MPC_可复现秩特征候选效用排序.py")
    evaluation, config = API配置(args)
    train = {str(x["id"]): x for x in 读_jsonl(根目录 / "results/splits/mpc/train.jsonl")}
    ledger = {str(x["id"]): x for x in 读_jsonl(第十轮 / "训练OOF检索账本.jsonl")}
    h1 = {str(x["样本编号"]): x for x in 读_jsonl(第十轮 / "训练OOF逐样本.jsonl")}
    if len(train) != 568 or set(train) != set(ledger) or set(train) != set(h1):
        raise RuntimeError("训练OOF冻结输入不完整")
    fold = {rid: int(x["折"]) for rid, x in h1.items()}
    violations = sum(fold[str(d["id"])] == fold[rid] for rid, x in ledger.items() for d in x["retrieved"])
    if violations:
        raise RuntimeError(f"OOF检索跨折违规：{violations}")
    path = output / "训练OOF选择性核心.jsonl"
    done = {str(x["id"]) for x in 读_jsonl(path) if x.get("解析成功") and not x.get("错误")} if path.exists() and args.resume else set()
    ordered_ids = list(train)
    for index, rid in enumerate(ordered_ids, 1):
        if rid in done:
            continue
        row = train[rid]
        demos = [train[str(x["id"])] for x in ledger[rid]["retrieved"]]
        core = None
        raw = None
        error = None
        try:
            raw = evaluation.call_chat_completion(构造消息(row, demos), config)
            core, parse_error = 解析核心(raw, row, base)
            if core is None:
                raise RuntimeError(parse_error or "核心解析失败")
        except Exception as exc:
            if "Insufficient Balance" in str(exc) or "HTTP error: 402" in str(exc):
                raise
            error = str(exc)
        追加(path, {"id": rid, "折": fold[rid], "target_food": row.get("target_food"), "n": row.get("n"), "predicted_molecules": core or [], "解析成功": core is not None, "错误": error, "原始响应": raw})
        print(f"MPC训练OOF Scientist进度: {index}/568 id={rid} {'成功' if core is not None else '失败'}", flush=True)
    rows = 读_jsonl(path)
    success = {str(x["id"]) for x in rows if x.get("解析成功") and not x.get("错误")}
    print(json.dumps({"成功ID数": len(success), "记录数": len(rows), "跨折违规": 0}, ensure_ascii=False))


def 构造训练记录(base: Any, output: Path) -> list[dict[str, Any]]:
    gold = {str(x["id"]): x for x in 读_jsonl(根目录 / "results/splits/mpc/train.jsonl")}
    h1_rows = 读_jsonl(第十轮 / "训练OOF逐样本.jsonl")
    core_rows = {str(x["id"]): x for x in 读_jsonl(output / "训练OOF选择性核心.jsonl") if x.get("解析成功") and not x.get("错误")}
    if len(core_rows) != 568:
        raise RuntimeError("568条训练OOF Scientist结果未全部成功，方法尚不能评价")
    records = []
    for h1 in h1_rows:
        rid = str(h1["样本编号"])
        core = core_rows[rid]["predicted_molecules"]
        records.append({
            "id": rid,
            "n": int(h1["N"]),
            "fold": int(h1["折"]),
            "gold": {base.规范(x) for x in gold[rid]["missing_molecules"]},
            "h1": h1["H1"],
            "icl": core,
            "candidate": base.样本候选(h1["H1"], core),
        })
    return records


def 核心诊断(records: list[dict[str, Any]], old_cores: dict[str, dict[str, Any]], base: Any) -> dict[str, Any]:
    def stats(source: str) -> dict[str, float]:
        hit = count = nonempty = 0
        for record in records:
            values = record["icl"] if source == "new" else old_cores[record["id"]]["predicted_molecules"]
            keys = {base.规范(x) for x in values if base.规范(x)}
            hit += len(keys & record["gold"])
            count += len(keys)
            nonempty += bool(keys)
        return {"总分子数": count, "Gold命中数": hit, "微平均Precision": hit / count if count else 0.0, "非空查询数": nonempty, "平均核心长度": count / len(records)}
    return {"旧强制列表": stats("old"), "新选择性核心": stats("new")}


def 训练审查(output: Path) -> bool:
    base = 加载("第十四轮MPC_base_train", 根目录 / "scripts/第十二轮_MPC_可复现秩特征候选效用排序.py")
    gate_module = 加载("第十四轮MPC_gate", 根目录 / "scripts/第十三轮_MPC_嵌套OOF后置安全门控.py")
    records = 构造训练记录(base, output)
    outputs, models = gate_module.生成嵌套OOF(records, base)
    old = {str(x["id"]): x for x in 读_jsonl(第十三轮 / "训练嵌套OOF逐样本.jsonl")}
    old_cores = {str(x["id"]): x for x in 读_jsonl(第十轮 / "训练OOF_ICL核心.jsonl")}
    details, h1_gains, old_gains = [], [], []
    totals = {"h1_hit": 0, "new_hit": 0, "old_hit": 0, "gold": 0, "h1_pred": 0, "new_pred": 0, "old_pred": 0}
    for record in records:
        result = outputs[record["id"]]
        h1_f1, h1_hit, h1_count = base.f1(record["h1"], record["gold"])
        new_f1, new_hit, new_count = base.f1(result["prediction"], record["gold"])
        old_pred = old[record["id"]]["最终预测"]
        old_f1, old_hit, old_count = base.f1(old_pred, record["gold"])
        h1_gain, old_gain = new_f1 - h1_f1, new_f1 - old_f1
        h1_gains.append(h1_gain); old_gains.append(old_gain)
        totals["h1_hit"] += h1_hit; totals["new_hit"] += new_hit; totals["old_hit"] += old_hit; totals["gold"] += len(record["gold"])
        totals["h1_pred"] += h1_count; totals["new_pred"] += new_count; totals["old_pred"] += old_count
        details.append({"id": record["id"], "折": record["fold"], "N": record["n"], "H1": record["h1"], "选择性核心": record["icl"], "原proposal": result["proposal"], "最终预测": result["prediction"], "决策": result["decision"], "APPLY_PROPOSAL概率": result["probability"], "H1分子F1": h1_f1, "第十三轮分子F1": old_f1, "新方法分子F1": new_f1, "相对H1增益": h1_gain, "相对第十三轮增益": old_gain, "exact_N": new_count == record["n"]})
    n = len(records)
    macro = lambda field: sum(x[field] for x in details) / n
    micro = lambda hit, pred: 2 * totals[hit] / (totals[pred] + totals["gold"])
    fold_gains = [sum(x["相对第十三轮增益"] for x in details if x["折"] == f) / sum(x["折"] == f for x in details) for f in range(5)]
    wins_h1, losses_h1 = sum(x > 0 for x in h1_gains), sum(x < 0 for x in h1_gains)
    old_macro, new_macro = macro("第十三轮分子F1"), macro("新方法分子F1")
    old_micro, new_micro = micro("old_hit", "old_pred"), micro("new_hit", "new_pred")
    passed = bool(new_macro > old_macro and new_micro > old_micro and wins_h1 > losses_h1 and losses_h1 <= 27 and sum(x >= 0 for x in fold_gains) >= 3 and all(x["exact_N"] for x in details))
    summary = {
        "OOF样本数": n,
        "核心诊断": 核心诊断(records, old_cores, base),
        "H1宏平均具体分子F1": macro("H1分子F1"),
        "第十三轮宏平均具体分子F1": old_macro,
        "新方法宏平均具体分子F1": new_macro,
        "相对第十三轮宏平均增益": new_macro - old_macro,
        "相对第十三轮宏增益bootstrap_95%区间": gate_module.bootstrap(old_gains),
        "H1微平均具体分子F1": micro("h1_hit", "h1_pred"),
        "第十三轮微平均具体分子F1": old_micro,
        "新方法微平均具体分子F1": new_micro,
        "相对H1_wins_losses_ties": [wins_h1, losses_h1, sum(x == 0 for x in h1_gains)],
        "相对第十三轮_wins_losses_ties": [sum(x > 0 for x in old_gains), sum(x < 0 for x in old_gains), sum(x == 0 for x in old_gains)],
        "相对第十三轮五折增益": fold_gains,
        "APPLY_PROPOSAL数": sum(x["决策"] == "APPLY_PROPOSAL" for x in details),
        "exact_N样本数": sum(x["exact_N"] for x in details),
        "各外折gate": models,
        "训练OOF是否通过第十四轮准入": passed,
        "开发集状态": "允许执行" if passed else "未通过并停止，不读dev Gold",
    }
    写_jsonl(output / "训练嵌套OOF逐样本.jsonl", details)
    写_json(output / "训练嵌套OOF完整审查结果.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return passed


def 执行开发Scientist(output: Path, args: argparse.Namespace) -> None:
    train_summary = json.loads((output / "训练嵌套OOF完整审查结果.json").read_text(encoding="utf-8"))
    if not train_summary.get("训练OOF是否通过第十四轮准入"):
        raise RuntimeError("训练未通过，禁止执行dev")
    base = 加载("第十四轮MPC_base_dev_call", 根目录 / "scripts/第十二轮_MPC_可复现秩特征候选效用排序.py")
    evaluation, config = API配置(args)
    agent = 加载("第十四轮MPC_agent_dev", 根目录 / "code/Only-Deepseek/scientific_agent.py")
    train = 读_jsonl(根目录 / "results/splits/mpc/train.jsonl")
    dev = 读_jsonl(根目录 / "results/splits/mpc/dev.jsonl")
    bm25 = agent.MPCBM25Index(train)
    path = output / "开发集选择性核心.jsonl"
    done = {str(x["id"]) for x in 读_jsonl(path) if x.get("解析成功") and not x.get("错误")} if path.exists() and args.resume else set()
    for index, row in enumerate(dev, 1):
        rid = str(row["id"])
        if rid in done:
            continue
        demos = [x["row"] for x in bm25.retrieve(row, 3)]
        core = None; raw = None; error = None
        try:
            raw = evaluation.call_chat_completion(构造消息(row, demos), config)
            core, parse_error = 解析核心(raw, row, base)
            if core is None:
                raise RuntimeError(parse_error or "核心解析失败")
        except Exception as exc:
            if "Insufficient Balance" in str(exc) or "HTTP error: 402" in str(exc):
                raise
            error = str(exc)
        追加(path, {"id": rid, "target_food": row.get("target_food"), "n": row.get("n"), "predicted_molecules": core or [], "解析成功": core is not None, "错误": error, "原始响应": raw})
        print(f"MPC开发集 Scientist进度: {index}/71 id={rid} {'成功' if core is not None else '失败'}", flush=True)
    rows = 读_jsonl(path)
    print(json.dumps({"成功ID数": len({str(x['id']) for x in rows if x.get('解析成功') and not x.get('错误')}), "记录数": len(rows)}, ensure_ascii=False))


def 开发集审查(output: Path) -> None:
    train_summary = json.loads((output / "训练嵌套OOF完整审查结果.json").read_text(encoding="utf-8"))
    if not train_summary.get("训练OOF是否通过第十四轮准入"):
        raise RuntimeError("训练未通过，禁止读取dev Gold")
    base = 加载("第十四轮MPC_base_dev", 根目录 / "scripts/第十二轮_MPC_可复现秩特征候选效用排序.py")
    gate_module = 加载("第十四轮MPC_gate_dev", 根目录 / "scripts/第十三轮_MPC_嵌套OOF后置安全门控.py")
    records = 构造训练记录(base, output)
    ranker = base.训练模型(records)
    oof_proposals = {}
    for fold in range(5):
        model = base.训练模型([x for x in records if x["fold"] != fold])
        for record in records:
            if record["fold"] == fold:
                oof_proposals[record["id"]] = base.预测(record, model)
    gate_examples = []
    for record in records:
        proposal = oof_proposals[record["id"]]
        delta = gate_module.命中(proposal, record["gold"], base) - gate_module.命中(record["h1"], record["gold"], base)
        gate_examples.append((gate_module.gate特征(record, proposal, base), delta))
    gate = gate_module.拟合gate(gate_examples)

    dev_gold = {str(x["id"]): x for x in 读_jsonl(根目录 / "results/splits/mpc/dev.jsonl")}
    h1 = {str(x["id"]): x for x in 读_jsonl(第七轮H1)}
    cores = {str(x["id"]): x for x in 读_jsonl(output / "开发集选择性核心.jsonl") if x.get("解析成功") and not x.get("错误")}
    old = {str(x["id"]): x for x in 读_jsonl(第十三轮 / "开发集最终预测.jsonl")}
    if not (len(dev_gold) == len(h1) == len(cores) == len(old) == 71):
        raise RuntimeError("dev冻结输入不完整")
    predictions, details, old_gains = [], [], []
    totals = {"old_hit": 0, "new_hit": 0, "gold": 0, "old_pred": 0, "new_pred": 0}
    for rid, row in dev_gold.items():
        h1_values = h1[rid]["predicted_molecules"]
        core = cores[rid]["predicted_molecules"]
        record = {"id": rid, "n": int(row["n"]), "h1": h1_values, "icl": core, "candidate": base.样本候选(h1_values, core), "gold": {base.规范(x) for x in row["missing_molecules"]}}
        proposal = base.预测(record, ranker)
        features = gate_module.gate特征(record, proposal, base)
        probability = float(gate.predict_proba(np.asarray([features]))[0, 1])
        final = proposal if probability > 0.5 else h1_values
        old_values = old[rid]["predicted_molecules"]
        old_f1, old_hit, old_count = base.f1(old_values, record["gold"])
        new_f1, new_hit, new_count = base.f1(final, record["gold"])
        gain = new_f1 - old_f1
        old_gains.append(gain)
        totals["old_hit"] += old_hit; totals["new_hit"] += new_hit; totals["gold"] += len(record["gold"]); totals["old_pred"] += old_count; totals["new_pred"] += new_count
        predictions.append({"id": rid, "task": "MPC", "target_food": row.get("target_food"), "n": row["n"], "predicted_molecules": final})
        details.append({"id": rid, "N": row["n"], "选择性核心": core, "H1": h1_values, "原proposal": proposal, "最终预测": final, "决策": "APPLY_PROPOSAL" if probability > 0.5 else "KEEP_H1", "APPLY_PROPOSAL概率": probability, "第十三轮分子F1": old_f1, "新方法分子F1": new_f1, "相对第十三轮增益": gain, "exact_N": new_count == int(row["n"])})
    old_macro = sum(x["第十三轮分子F1"] for x in details) / 71
    new_macro = sum(x["新方法分子F1"] for x in details) / 71
    old_micro = 2 * totals["old_hit"] / (totals["old_pred"] + totals["gold"])
    new_micro = 2 * totals["new_hit"] / (totals["new_pred"] + totals["gold"])
    cache = json.loads((根目录 / "results/Only-Deepseek/优化实验/第九轮/MPC_ICL高精度核心与H1精确补全/第九轮独立官能团缓存.json").read_text(encoding="utf-8"))
    cache_keys = {" ".join(str(x).lower().split()) for x in cache}
    missing = sorted({str(m) for row in predictions for m in row["predicted_molecules"] if " ".join(str(m).lower().split()) not in cache_keys})
    passed = bool(new_macro > old_macro and new_micro > old_micro and sum(x > 0 for x in old_gains) > sum(x < 0 for x in old_gains) and all(x["exact_N"] for x in details))
    summary = {"dev样本数": 71, "第十三轮宏平均具体分子F1": old_macro, "新方法宏平均具体分子F1": new_macro, "宏平均增益": new_macro - old_macro, "第十三轮微平均具体分子F1": old_micro, "新方法微平均具体分子F1": new_micro, "相对第十三轮_wins_losses_ties": [sum(x > 0 for x in old_gains), sum(x < 0 for x in old_gains), sum(x == 0 for x in old_gains)], "增益bootstrap_95%区间": gate_module.bootstrap(old_gains), "APPLY_PROPOSAL数": sum(x["决策"] == "APPLY_PROPOSAL" for x in details), "exact_N样本数": sum(x["exact_N"] for x in details), "官能团缓存缺失分子数": len(missing), "缺失分子": missing, "具体分子准入是否通过": passed, "官方评测状态": "允许" if passed else "未通过并停止"}
    写_jsonl(output / "开发集最终预测.jsonl", predictions)
    写_jsonl(output / "开发集具体分子逐样本审查.jsonl", details)
    写_json(output / "开发集具体分子与缓存审查.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def 官方审查(output: Path) -> None:
    local = json.loads((output / "开发集具体分子与缓存审查.json").read_text(encoding="utf-8"))
    if not local.get("具体分子准入是否通过"):
        raise RuntimeError("具体分子准入未通过，禁止官方审查")
    new = {str(x["id"]): x for x in 读_jsonl(output / "开发集官方官能团逐样本.jsonl")}
    old = {str(x["id"]): x for x in 读_jsonl(第十三轮 / "开发集官方官能团逐样本.jsonl")}
    ids = [str(x["id"]) for x in 读_jsonl(根目录 / "results/splits/mpc/dev.jsonl")]
    gains = [float(new[rid]["f1"]) - float(old[rid]["f1"]) for rid in ids]
    new_mean = sum(float(new[rid]["f1"]) for rid in ids) / 71
    old_mean = sum(float(old[rid]["f1"]) for rid in ids) / 71
    passed = bool(new_mean > old_mean and local["具体分子准入是否通过"] and sum(x > 0 for x in gains) > sum(x < 0 for x in gains) and local["exact_N样本数"] == 71)
    summary = {"样本数": 71, "第十三轮官方官能团F1": old_mean, "新方法官方官能团F1": new_mean, "官方增益": new_mean - old_mean, "官方增益bootstrap_95%区间": 加载("第十四轮MPC_gate_review", 根目录 / "scripts/第十三轮_MPC_嵌套OOF后置安全门控.py").bootstrap(gains), "官方wins_losses_ties": [sum(x > 0 for x in gains), sum(x < 0 for x in gains), sum(x == 0 for x in gains)], "具体分子宏平均增益": local["宏平均增益"], "具体分子微平均增益": local["新方法微平均具体分子F1"] - local["第十三轮微平均具体分子F1"], "exact_N样本数": local["exact_N样本数"], "是否通过探索保留": passed, "结论": "获得局部信号并探索保留，不冻结论文主方法" if passed else "未通过并停止"}
    写_json(output / "完整审查结果.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("动作", choices=["准备", "执行训练Scientist", "训练审查", "执行开发Scientist", "开发集审查", "官方审查"])
    parser.add_argument("--输出", type=Path, default=默认输出)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm-provider", default="deepseek")
    parser.add_argument("--llm-model", default="deepseek-v4-flash")
    parser.add_argument("--llm-base-url", default=None)
    args = parser.parse_args()
    if args.动作 == "准备": 准备(args.输出)
    elif args.动作 == "执行训练Scientist": 执行训练Scientist(args.输出, args)
    elif args.动作 == "训练审查": 训练审查(args.输出)
    elif args.动作 == "执行开发Scientist": 执行开发Scientist(args.输出, args)
    elif args.动作 == "开发集审查": 开发集审查(args.输出)
    else: 官方审查(args.输出)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
