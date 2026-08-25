#!/usr/bin/env python3
"""MPC 第十二轮：仅用冻结 H1/ICL 排名重建可复现的候选效用排序。"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


根目录 = Path(__file__).resolve().parents[1]
默认输出 = 根目录 / "results/Only-Deepseek/优化实验/第十二轮/MPC_可复现秩特征候选效用排序"
第十轮目录 = 根目录 / "results/Only-Deepseek/优化实验/第十轮/MPC_共识锚定候选效用"
第七轮H1 = 根目录 / "results/Only-Deepseek/优化实验/第七轮/MPC_当前H1开发集正式口径控制/当前H1开发集预测.jsonl"
第九轮ICL = 根目录 / "results/Only-Deepseek/优化实验/第九轮/MPC_ICL高精度核心与H1精确补全/原始ICL预测.jsonl"
随机种子 = 20260809


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


def 规范(value: Any) -> str:
    # 与仓库既有 MPC 具体分子口径一致，避免连字符/标点导致假不命中和重复候选。
    text = str(value or "").lower().strip().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^0-9a-z+\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def 去重排名(values: list[Any]) -> tuple[list[str], dict[str, int], dict[str, str]]:
    ordered, ranks, names = [], {}, {}
    for value in values:
        name, key = str(value).strip(), 规范(value)
        if not key or key in ranks:
            continue
        ordered.append(key)
        ranks[key] = len(ordered)
        names[key] = name
    return ordered, ranks, names


def 特征(key: str, h1_rank: dict[str, int], icl_rank: dict[str, int]) -> list[float]:
    return [
        1.0 / h1_rank[key] if key in h1_rank else 0.0,
        1.0 / icl_rank[key] if key in icl_rank else 0.0,
        float(key in h1_rank and key in icl_rank),
    ]


def 样本候选(h1: list[Any], icl: list[Any]) -> dict[str, Any]:
    h1_keys, h1_rank, h1_names = 去重排名(h1)
    icl_keys, icl_rank, icl_names = 去重排名(icl)
    keys = h1_keys + [x for x in icl_keys if x not in h1_rank]
    names = dict(icl_names)
    names.update(h1_names)
    return {"keys": keys, "h1_keys": h1_keys, "h1_rank": h1_rank, "icl_rank": icl_rank, "names": names}


def 训练模型(records: list[dict[str, Any]]) -> LogisticRegression:
    diffs, weights = [], []
    for record in records:
        positives = [k for k in record["candidate"]["keys"] if k in record["gold"]]
        negatives = [k for k in record["candidate"]["keys"] if k not in record["gold"]]
        pair_count = len(positives) * len(negatives)
        if not pair_count:
            continue
        weight = 1.0 / pair_count
        for positive in positives:
            pf = np.asarray(特征(positive, record["candidate"]["h1_rank"], record["candidate"]["icl_rank"]))
            for negative in negatives:
                nf = np.asarray(特征(negative, record["candidate"]["h1_rank"], record["candidate"]["icl_rank"]))
                diffs.append(pf - nf)
                weights.append(weight)
                diffs.append(nf - pf)
                weights.append(weight)
    x = np.asarray(diffs, dtype=np.float32)
    y = np.tile([1, 0], len(diffs) // 2)
    model = LogisticRegression(C=1.0, fit_intercept=False, solver="liblinear", random_state=随机种子, max_iter=1000)
    model.fit(x, y, sample_weight=np.asarray(weights, dtype=np.float32))
    return model


def 预测(record: dict[str, Any], model: LogisticRegression) -> list[str]:
    candidate = record["candidate"]
    n = record["n"]
    consensus = [k for k in candidate["keys"] if k in candidate["h1_rank"] and k in candidate["icl_rank"]]
    consensus.sort(key=lambda k: (candidate["h1_rank"][k], candidate["icl_rank"][k], k))
    if len(consensus) >= n:
        chosen = consensus[:n]
    else:
        residual = [k for k in candidate["keys"] if k not in set(consensus)]
        scores = model.decision_function(np.asarray([特征(k, candidate["h1_rank"], candidate["icl_rank"]) for k in residual]))
        scored = list(zip(residual, scores.tolist()))
        scored.sort(key=lambda p: (-p[1], candidate["h1_rank"].get(p[0], 10**9), candidate["icl_rank"].get(p[0], 10**9), p[0]))
        chosen = consensus + [k for k, _ in scored[: n - len(consensus)]]
    if len(chosen) < n:
        raise RuntimeError(f"候选并集不足 exact-N：id={record['id']}")
    return [candidate["names"][k] for k in chosen]


def f1(prediction: list[Any], gold: set[str]) -> tuple[float, int, int]:
    pred = {规范(x) for x in prediction if 规范(x)}
    hit = len(pred & gold)
    return (2 * hit / (len(pred) + len(gold)) if pred or gold else 1.0), hit, len(pred)


def bootstrap(values: list[float], repeats: int = 10000) -> list[float]:
    rng = random.Random(随机种子)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(repeats))
    return [means[int(0.025 * repeats)], means[int(0.975 * repeats) - 1]]


def 准备(output: Path) -> None:
    protocol = {
        "任务输出": "exact-N 具体缺失分子集合",
        "当前瓶颈": "第十轮候选效用排序有正信号，但历史源码缺失且动态依赖已漂移，无法继续归因。",
        "历史借鉴": ["第九轮 ICL 核心提供互补候选", "第十轮共识锚定与候选效用排序在 train OOF 有稳定正信号", "第十一轮证明不能依赖漂移执行器"],
        "唯一方向": "用冻结 H1/ICL 候选的倒数排名和共识指示重建无外部数据、无动态主版本依赖的成对排序器。",
        "特征": ["1/H1排名", "1/ICL排名", "H1与ICL共识"],
        "冻结": ["H1排名", "ICL排名", "候选并集", "exact-profile五折", "C=1", "liblinear", "exact-N", "共识候选优先保留"],
        "训练准入": ["具体分子macro和micro F1均高于H1", "wins>losses", "至少3/5折非负", "最差折不低于-0.01", "568/568 exact-N"],
        "开发集探索保留": ["官方官能团F1点估计高于H1", "具体分子macro/micro不低于H1", "wins>losses", "71/71 exact-N", "95%CI与固定分块只报告不否决"],
        "停止条件": "训练OOF任一准入失败即停止，不读dev Gold，不调参挽救。",
        "边界": "不读正式test；方法不使用任务外FlavorDB数据；不新增中间模型或API调用。",
    }
    写_json(output / "冻结实验方案.json", protocol)
    print(json.dumps({"状态": "冻结完成"}, ensure_ascii=False))


def 训练(output: Path) -> bool:
    gold_rows = {str(x["id"]): x for x in 读_jsonl(根目录 / "results/splits/mpc/train.jsonl")}
    h1_rows = 读_jsonl(第十轮目录 / "训练OOF逐样本.jsonl")
    icl_rows = {str(x["id"]): x for x in 读_jsonl(第十轮目录 / "训练OOF_ICL核心.jsonl")}
    if len(gold_rows) != 568 or len(h1_rows) != 568 or len(icl_rows) != 568:
        raise RuntimeError("训练冻结输入不完整")
    records = []
    for h1 in h1_rows:
        row_id = str(h1["样本编号"])
        gold = {规范(x) for x in gold_rows[row_id]["missing_molecules"]}
        candidate = 样本候选(h1["H1"], icl_rows[row_id]["predicted_molecules"])
        records.append({"id": row_id, "n": int(h1["N"]), "fold": int(h1["折"]), "gold": gold, "h1": h1["H1"], "candidate": candidate})
    predictions: dict[str, list[str]] = {}
    coefficients = []
    for fold in range(5):
        model = 训练模型([r for r in records if r["fold"] != fold])
        coefficients.append({"折": fold, "系数": model.coef_[0].tolist()})
        for record in records:
            if record["fold"] == fold:
                predictions[record["id"]] = 预测(record, model)
    details, gains = [], []
    total_h1_hit = total_new_hit = total_gold = total_h1_pred = total_new_pred = 0
    for record in records:
        h1_score, h1_hit, h1_count = f1(record["h1"], record["gold"])
        new_score, new_hit, new_count = f1(predictions[record["id"]], record["gold"])
        gain = new_score - h1_score
        gains.append(gain)
        total_h1_hit += h1_hit; total_new_hit += new_hit; total_gold += len(record["gold"])
        total_h1_pred += h1_count; total_new_pred += new_count
        details.append({"id": record["id"], "折": record["fold"], "N": record["n"], "H1": record["h1"], "预测分子": predictions[record["id"]], "Gold数": len(record["gold"]), "H1分子F1": h1_score, "新方法分子F1": new_score, "增益": gain, "exact_N": new_count == record["n"]})
    h1_macro = sum(x["H1分子F1"] for x in details) / len(details)
    new_macro = sum(x["新方法分子F1"] for x in details) / len(details)
    h1_micro = 2 * total_h1_hit / (total_h1_pred + total_gold)
    new_micro = 2 * total_new_hit / (total_new_pred + total_gold)
    fold_gains = [sum(x["增益"] for x in details if x["折"] == f) / sum(x["折"] == f for x in details) for f in range(5)]
    passed = new_macro > h1_macro and new_micro > h1_micro and sum(x > 0 for x in gains) > sum(x < 0 for x in gains) and sum(x >= 0 for x in fold_gains) >= 3 and min(fold_gains) >= -0.01 and all(x["exact_N"] for x in details)
    summary = {"OOF样本数": len(details), "H1宏平均具体分子F1": h1_macro, "新方法宏平均具体分子F1": new_macro, "宏平均增益": new_macro-h1_macro, "宏增益bootstrap_95%区间": bootstrap(gains), "H1微平均具体分子F1": h1_micro, "新方法微平均具体分子F1": new_micro, "wins": sum(x>0 for x in gains), "losses": sum(x<0 for x in gains), "ties": sum(x==0 for x in gains), "五折宏平均增益": fold_gains, "exact_N样本数": sum(x["exact_N"] for x in details), "各折系数": coefficients, "训练OOF是否通过探索准入": passed, "开发集状态": "允许执行" if passed else "训练OOF未通过，按冻结方案停止"}
    写_jsonl(output / "训练OOF逐样本.jsonl", details)
    写_json(output / "训练OOF完整审查结果.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("动作", choices=["准备", "训练"])
    parser.add_argument("--输出", type=Path, default=默认输出)
    args = parser.parse_args()
    if args.动作 == "准备":
        准备(args.输出)
    else:
        训练(args.输出)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
