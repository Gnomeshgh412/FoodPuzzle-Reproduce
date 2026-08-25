#!/usr/bin/env python3
"""MPC 第十三轮：在冻结秩排序 proposal 上嵌套 OOF 学习 KEEP_H1 后置门控。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


根目录 = Path(__file__).resolve().parents[1]
默认输出 = 根目录 / "results/Only-Deepseek/优化实验/第十三轮/MPC_嵌套OOF后置安全门控"
第十轮 = 根目录 / "results/Only-Deepseek/优化实验/第十轮/MPC_共识锚定候选效用"
第十二轮 = 根目录 / "results/Only-Deepseek/优化实验/第十二轮/MPC_可复现秩特征候选效用排序_既有归一化口径"
第七轮H1 = 根目录 / "results/Only-Deepseek/优化实验/第七轮/MPC_当前H1开发集正式口径控制/当前H1开发集预测.jsonl"
第九轮ICL = 根目录 / "results/Only-Deepseek/优化实验/第九轮/MPC_ICL高精度核心与H1精确补全/原始ICL预测.jsonl"
随机种子 = 20260810


def 加载第十二轮() -> Any:
    path = 根目录 / "scripts/第十二轮_MPC_可复现秩特征候选效用排序.py"
    spec = importlib.util.spec_from_file_location("第十三轮复用第十二轮", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
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


def 准备(output: Path) -> None:
    protocol = {
        "任务输出": "exact-N 具体缺失分子集合",
        "当前瓶颈": "候选级全局排序被无条件转为集合交换，没有估计查询级新增收益与移除H1风险。",
        "历史借鉴": ["第九轮ICL互补候选", "第十轮共识候选效用", "第十二轮macro改善但micro受损", "第十一轮KEEP_H1方向因源码漂移未得评价"],
        "唯一改动": "在冻结第十二轮proposal与H1之上增加查询级APPLY_PROPOSAL/KEEP_H1后置门控。",
        "嵌套OOF": "每个外折的gate训练proposal由其余四折内层OOF生成，外折Gold不进入排序器或gate。",
        "gate特征": ["log1p(N)", "交换数/N", "H1与ICL共识数/N"],
        "gate目标": "proposal相对H1的具体分子净命中是否为正；排除ties；按|净命中差|加权。",
        "gate模型": "LogisticRegression(C=1, solver=liblinear, threshold=0.5, random_state=20260810)；不标准化特征。",
        "冻结": ["第十二轮秩特征排序器", "H1", "ICL", "共识锁定", "exact-N", "exact-profile五折"],
        "训练OOF准入": ["macro和micro具体分子F1均高于H1", "wins>losses", "losses<145", "至少3/5外折非负", "最差折>=-0.01", "568/568 exact-N"],
        "停止条件": "训练OOF任一准入失败即停止，不读dev Gold，不调特征、C、阈值或权重。",
        "边界": "不读正式test；不使用官能团训练；不使用任务外FlavorDB数据；预计0 API。",
    }
    写_json(output / "冻结实验方案.json", protocol)
    print(json.dumps({"状态": "冻结完成"}, ensure_ascii=False))


def 构造训练记录(base: Any) -> list[dict[str, Any]]:
    gold_rows = {str(x["id"]): x for x in 读_jsonl(根目录 / "results/splits/mpc/train.jsonl")}
    h1_rows = 读_jsonl(第十轮 / "训练OOF逐样本.jsonl")
    icl_rows = {str(x["id"]): x for x in 读_jsonl(第十轮 / "训练OOF_ICL核心.jsonl")}
    if not (len(gold_rows) == len(h1_rows) == len(icl_rows) == 568):
        raise RuntimeError("训练冻结输入不完整")
    records = []
    for h1 in h1_rows:
        row_id = str(h1["样本编号"])
        records.append({
            "id": row_id,
            "n": int(h1["N"]),
            "fold": int(h1["折"]),
            "gold": {base.规范(x) for x in gold_rows[row_id]["missing_molecules"]},
            "h1": h1["H1"],
            "icl": icl_rows[row_id]["predicted_molecules"],
            "candidate": base.样本候选(h1["H1"], icl_rows[row_id]["predicted_molecules"]),
        })
    return records


def 命中(prediction: list[str], gold: set[str], base: Any) -> int:
    return len({base.规范(x) for x in prediction} & gold)


def gate特征(record: dict[str, Any], proposal: list[str], base: Any) -> list[float]:
    h1 = {base.规范(x) for x in record["h1"]}
    pred = {base.规范(x) for x in proposal}
    icl = {base.规范(x) for x in record["icl"]}
    n = record["n"]
    return [math.log1p(n), len(pred - h1) / n, len(h1 & icl) / n]


def 拟合gate(examples: list[tuple[list[float], int]]) -> LogisticRegression:
    usable = [(x, delta) for x, delta in examples if delta != 0]
    labels = {delta > 0 for _, delta in usable}
    if len(labels) != 2:
        raise RuntimeError("gate训练数据没有正负两类")
    model = LogisticRegression(C=1.0, solver="liblinear", random_state=随机种子, max_iter=1000)
    model.fit(
        np.asarray([x for x, _ in usable], dtype=np.float64),
        np.asarray([int(delta > 0) for _, delta in usable]),
        sample_weight=np.asarray([abs(delta) for _, delta in usable], dtype=np.float64),
    )
    return model


def 生成嵌套OOF(records: list[dict[str, Any]], base: Any) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    outputs: dict[str, dict[str, Any]] = {}
    models = []
    for outer in range(5):
        inner_examples: list[tuple[list[float], int]] = []
        for inner in range(5):
            if inner == outer:
                continue
            ranker_train = [r for r in records if r["fold"] not in {outer, inner}]
            ranker = base.训练模型(ranker_train)
            for record in records:
                if record["fold"] != inner:
                    continue
                proposal = base.预测(record, ranker)
                delta_hits = 命中(proposal, record["gold"], base) - 命中(record["h1"], record["gold"], base)
                inner_examples.append((gate特征(record, proposal, base), delta_hits))
        gate = 拟合gate(inner_examples)
        outer_ranker = base.训练模型([r for r in records if r["fold"] != outer])
        models.append({"外折": outer, "gate系数": gate.coef_[0].tolist(), "gate截距": gate.intercept_.tolist(), "内层有效样本数": sum(delta != 0 for _, delta in inner_examples)})
        for record in records:
            if record["fold"] != outer:
                continue
            proposal = base.预测(record, outer_ranker)
            features = gate特征(record, proposal, base)
            probability = float(gate.predict_proba(np.asarray([features]))[0, 1])
            apply = probability > 0.5
            outputs[record["id"]] = {"proposal": proposal, "prediction": proposal if apply else record["h1"], "features": features, "probability": probability, "decision": "APPLY_PROPOSAL" if apply else "KEEP_H1"}
        print(f"[MPC嵌套OOF] 外折 {outer + 1}/5 完成", flush=True)
    return outputs, models


def bootstrap(values: list[float], repeats: int = 10000) -> list[float]:
    rng = random.Random(随机种子)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(repeats))
    return [means[int(0.025 * repeats)], means[int(0.975 * repeats) - 1]]


def 审查训练(records: list[dict[str, Any]], outputs: dict[str, dict[str, Any]], models: list[dict[str, Any]], base: Any, output: Path) -> bool:
    details, gains = [], []
    totals = {"h1_hit": 0, "new_hit": 0, "gold": 0, "h1_pred": 0, "new_pred": 0}
    for record in records:
        result = outputs[record["id"]]
        h1_f1, h1_hit, h1_count = base.f1(record["h1"], record["gold"])
        new_f1, new_hit, new_count = base.f1(result["prediction"], record["gold"])
        proposal_f1, proposal_hit, _ = base.f1(result["proposal"], record["gold"])
        gain = new_f1 - h1_f1
        gains.append(gain)
        totals["h1_hit"] += h1_hit; totals["new_hit"] += new_hit; totals["gold"] += len(record["gold"])
        totals["h1_pred"] += h1_count; totals["new_pred"] += new_count
        details.append({"id": record["id"], "折": record["fold"], "N": record["n"], "H1": record["h1"], "原proposal": result["proposal"], "最终预测": result["prediction"], "gate特征": result["features"], "APPLY_PROPOSAL概率": result["probability"], "决策": result["decision"], "H1分子F1": h1_f1, "proposal分子F1": proposal_f1, "最终分子F1": new_f1, "增益": gain, "净命中差": new_hit-h1_hit, "exact_N": new_count == record["n"]})
    h1_macro = sum(x["H1分子F1"] for x in details) / 568
    new_macro = sum(x["最终分子F1"] for x in details) / 568
    proposal_macro = sum(x["proposal分子F1"] for x in details) / 568
    h1_micro = 2 * totals["h1_hit"] / (totals["h1_pred"] + totals["gold"])
    new_micro = 2 * totals["new_hit"] / (totals["new_pred"] + totals["gold"])
    fold_gains = [sum(x["增益"] for x in details if x["折"] == f) / sum(x["折"] == f for x in details) for f in range(5)]
    wins, losses = sum(x > 0 for x in gains), sum(x < 0 for x in gains)
    passed = bool(new_macro > h1_macro and new_micro > h1_micro and wins > losses and losses < 145 and sum(x >= 0 for x in fold_gains) >= 3 and min(fold_gains) >= -0.01 and all(x["exact_N"] for x in details))
    summary = {"OOF样本数": 568, "H1宏平均具体分子F1": h1_macro, "原proposal宏平均具体分子F1": proposal_macro, "门控后宏平均具体分子F1": new_macro, "宏平均增益": new_macro-h1_macro, "宏增益bootstrap_95%区间": bootstrap(gains), "H1微平均具体分子F1": h1_micro, "门控后微平均具体分子F1": new_micro, "wins": wins, "losses": losses, "ties": sum(x == 0 for x in gains), "APPLY_PROPOSAL数": sum(x["决策"] == "APPLY_PROPOSAL" for x in details), "KEEP_H1数": sum(x["决策"] == "KEEP_H1" for x in details), "五折宏平均增益": fold_gains, "exact_N样本数": sum(x["exact_N"] for x in details), "各外折gate": models, "训练OOF是否通过探索准入": passed, "开发集状态": "允许执行" if passed else "训练OOF未通过，按冻结方案停止"}
    写_jsonl(output / "训练嵌套OOF逐样本.jsonl", details)
    写_json(output / "训练嵌套OOF完整审查结果.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return passed


def 训练(output: Path) -> bool:
    base = 加载第十二轮()
    records = 构造训练记录(base)
    outputs, models = 生成嵌套OOF(records, base)
    return 审查训练(records, outputs, models, base, output)


def 开发集(output: Path) -> None:
    train_summary = json.loads((output / "训练嵌套OOF完整审查结果.json").read_text(encoding="utf-8"))
    if not train_summary.get("训练OOF是否通过探索准入"):
        raise RuntimeError("训练OOF未通过，禁止读取dev Gold")
    base = 加载第十二轮()
    train_records = 构造训练记录(base)
    ranker = base.训练模型(train_records)

    # gate 仅使用每条训练样本已有的外折 proposal，不用 in-sample proposal。
    oof_rows = {str(x["id"]): x for x in 读_jsonl(第十二轮 / "训练OOF逐样本.jsonl")}
    gate_examples: list[tuple[list[float], int]] = []
    for record in train_records:
        proposal = oof_rows[record["id"]]["预测分子"]
        delta = 命中(proposal, record["gold"], base) - 命中(record["h1"], record["gold"], base)
        gate_examples.append((gate特征(record, proposal, base), delta))
    gate = 拟合gate(gate_examples)

    # 只有训练通过后才在此处读取 dev Gold。
    dev_gold = {str(x["id"]): x for x in 读_jsonl(根目录 / "results/splits/mpc/dev.jsonl")}
    dev_h1 = {str(x["id"]): x for x in 读_jsonl(第七轮H1)}
    dev_icl = {str(x["id"]): x for x in 读_jsonl(第九轮ICL)}
    if not (len(dev_gold) == len(dev_h1) == len(dev_icl) == 71 and set(dev_gold) == set(dev_h1) == set(dev_icl)):
        raise RuntimeError("dev冻结输入不完整或ID不一致")
    predictions, details = [], []
    total = {"h1_hit": 0, "new_hit": 0, "gold": 0, "h1_pred": 0, "new_pred": 0}
    gains = []
    for row_id in dev_gold:
        h1 = dev_h1[row_id]["predicted_molecules"]
        icl = dev_icl[row_id]["predicted_molecules"]
        record = {"id": row_id, "n": int(dev_gold[row_id]["n"]), "h1": h1, "icl": icl, "candidate": base.样本候选(h1, icl), "gold": {base.规范(x) for x in dev_gold[row_id]["missing_molecules"]}}
        proposal = base.预测(record, ranker)
        features = gate特征(record, proposal, base)
        probability = float(gate.predict_proba(np.asarray([features]))[0, 1])
        apply = probability > 0.5
        final = proposal if apply else h1
        h1_f1, h1_hit, h1_count = base.f1(h1, record["gold"])
        new_f1, new_hit, new_count = base.f1(final, record["gold"])
        proposal_f1, _, _ = base.f1(proposal, record["gold"])
        gains.append(new_f1-h1_f1)
        total["h1_hit"] += h1_hit; total["new_hit"] += new_hit; total["gold"] += len(record["gold"]); total["h1_pred"] += h1_count; total["new_pred"] += new_count
        predictions.append({"id": row_id, "task": "MPC", "target_food": dev_gold[row_id].get("target_food"), "n": record["n"], "predicted_molecules": final})
        details.append({"id": row_id, "N": record["n"], "H1": h1, "原proposal": proposal, "最终预测": final, "gate特征": features, "APPLY_PROPOSAL概率": probability, "决策": "APPLY_PROPOSAL" if apply else "KEEP_H1", "H1分子F1": h1_f1, "proposal分子F1": proposal_f1, "最终分子F1": new_f1, "增益": new_f1-h1_f1, "exact_N": new_count == record["n"]})
    h1_macro = sum(x["H1分子F1"] for x in details)/71
    new_macro = sum(x["最终分子F1"] for x in details)/71
    h1_micro = 2*total["h1_hit"]/(total["h1_pred"]+total["gold"])
    new_micro = 2*total["new_hit"]/(total["new_pred"]+total["gold"])

    cache_path = 根目录 / "results/Only-Deepseek/优化实验/第九轮/MPC_ICL高精度核心与H1精确补全/第九轮独立官能团缓存.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_keys = {" ".join(str(x).lower().split()) for x in cache}
    missing = sorted({str(m) for row in predictions for m in row["predicted_molecules"] if " ".join(str(m).lower().split()) not in cache_keys})
    summary = {"dev样本数": 71, "H1宏平均具体分子F1": h1_macro, "门控后宏平均具体分子F1": new_macro, "宏平均增益": new_macro-h1_macro, "H1微平均具体分子F1": h1_micro, "门控后微平均具体分子F1": new_micro, "wins": sum(x>0 for x in gains), "losses": sum(x<0 for x in gains), "ties": sum(x==0 for x in gains), "APPLY_PROPOSAL数": sum(x["决策"]=="APPLY_PROPOSAL" for x in details), "KEEP_H1数": sum(x["决策"]=="KEEP_H1" for x in details), "exact_N样本数": sum(x["exact_N"] for x in details), "gate系数": gate.coef_[0].tolist(), "gate截距": gate.intercept_.tolist(), "官能团缓存缺失分子数": len(missing), "缺失分子": missing, "官方官能团评测状态": "允许使用已有缓存评测" if not missing else "缓存不完整，停止并重新请权限"}
    写_jsonl(output / "开发集最终预测.jsonl", predictions)
    写_jsonl(output / "开发集具体分子逐样本审查.jsonl", details)
    写_json(output / "开发集具体分子与缓存审查.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def 官方审查(output: Path) -> None:
    """对已完成的官方官能团结果作配对审查，不再调用模型。"""
    新路径 = output / "开发集官方官能团逐样本.jsonl"
    旧路径 = 根目录 / "results/Only-Deepseek/优化实验/第七轮/MPC_当前H1开发集正式口径控制/官方口径逐样本结果.jsonl"
    新 = {str(x["id"]): x for x in 读_jsonl(新路径)}
    旧 = {str(x["id"]): x for x in 读_jsonl(旧路径)}
    dev = 读_jsonl(根目录 / "results/splits/mpc/dev.jsonl")
    分子审查 = json.loads((output / "开发集具体分子与缓存审查.json").read_text(encoding="utf-8"))
    if len(新) != 71 or len(旧) != 71 or {str(x["id"]) for x in dev} != set(新) or set(新) != set(旧):
        raise RuntimeError("官方逐样本结果不完整或ID不一致")

    逐样本, 增益 = [], []
    for index, row in enumerate(dev):
        rid = str(row["id"])
        gain = float(新[rid]["f1"]) - float(旧[rid]["f1"])
        增益.append(gain)
        n = int(row["n"])
        分组 = "N<=50" if n <= 50 else ("51<=N<=100" if n <= 100 else "N>100")
        逐样本.append({
            "id": rid, "N": n, "N分组": 分组, "固定分块": index % 5,
            "H1官方官能团F1": 旧[rid]["f1"], "门控后官方官能团F1": 新[rid]["f1"], "增益": gain,
        })

    分组增益 = {}
    for name in ("N<=50", "51<=N<=100", "N>100"):
        values = [x["增益"] for x in 逐样本 if x["N分组"] == name]
        分组增益[name] = {"样本数": len(values), "平均增益": sum(values) / len(values)}
    分块增益 = []
    for block in range(5):
        values = [x["增益"] for x in 逐样本 if x["固定分块"] == block]
        分块增益.append(sum(values) / len(values))

    胜, 负 = sum(x > 0 for x in 增益), sum(x < 0 for x in 增益)
    旧均值 = sum(float(旧[str(x["id"])]["f1"]) for x in dev) / 71
    新均值 = sum(float(新[str(x["id"])]["f1"]) for x in dev) / 71
    硬条件通过 = bool(
        新均值 > 旧均值
        and 分子审查["门控后宏平均具体分子F1"] >= 分子审查["H1宏平均具体分子F1"]
        and 分子审查["门控后微平均具体分子F1"] >= 分子审查["H1微平均具体分子F1"]
        and 胜 > 负
        and 分子审查["exact_N样本数"] == 71
    )
    summary = {
        "样本数": 71,
        "H1官方官能团F1": 旧均值,
        "门控后官方官能团F1": 新均值,
        "官方官能团F1增益": 新均值 - 旧均值,
        "官方增益bootstrap_95%区间": bootstrap(增益),
        "官方wins_losses_ties": [胜, 负, sum(x == 0 for x in 增益)],
        "官方五分块增益": 分块增益,
        "官方N分组增益": 分组增益,
        "具体分子宏平均F1增益": 分子审查["宏平均增益"],
        "具体分子微平均F1增益": 分子审查["门控后微平均具体分子F1"] - 分子审查["H1微平均具体分子F1"],
        "exact_N样本数": 分子审查["exact_N样本数"],
        "冻结硬条件是否通过": 硬条件通过,
        "N子集审查说明": "冻结方案未预设灾难性下降的数值阈值，因此仅如实报告各N组，不事后新增阈值。",
        "结论": "获得局部信号并探索保留，不冻结论文主方法" if 硬条件通过 else "未通过并停止",
        "结论上限原因": "dev已被多轮自适应复用，官方增益须在未触碰数据或独立重复中验证。",
    }
    写_jsonl(output / "开发集官方配对审查逐样本.jsonl", 逐样本)
    写_json(output / "完整审查结果.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("动作", choices=["准备", "训练", "开发集", "官方审查"])
    parser.add_argument("--输出", type=Path, default=默认输出)
    args = parser.parse_args()
    if args.动作 == "准备": 准备(args.输出)
    elif args.动作 == "训练": 训练(args.输出)
    elif args.动作 == "开发集": 开发集(args.输出)
    else: 官方审查(args.输出)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
