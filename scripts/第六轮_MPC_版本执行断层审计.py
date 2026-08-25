#!/usr/bin/env python3
"""只读正式元数据并仅加载 MPC train，审计正式 v12 与当前 v15 源码断层。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


项目根目录 = Path(__file__).resolve().parents[1]
默认输出 = 项目根目录 / "results/Only-Deepseek/优化实验/第六轮/MPC_版本执行断层审计.json"


def 哈希(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 加载模块(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("第六轮版本审计代理", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def 读取_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    code_path = 项目根目录 / "code/Only-Deepseek/optimized_agent.py"
    runner_path = 项目根目录 / "scripts/run_optimized_agent.sh"
    train_path = 项目根目录 / "results/splits/mpc/train.jsonl"
    metadata_path = 项目根目录 / "results/Only-Deepseek/optimized-agent/mpc/deepseek-v4-flash/run_metadata.json"
    mfp_summary = 项目根目录 / "results/Only-Deepseek/optimized-agent/mfp/deepseek-v4-flash/evaluation_summary.json"
    mpc_summary = 项目根目录 / "results/Only-Deepseek/optimized-agent/mpc/deepseek-v4-flash/evaluation_summary.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    agent = 加载模块(code_path)
    train = 读取_jsonl(train_path)
    model = agent.MPCStructureModel(train, None, "full", 项目根目录 / "data/raw/flavordb.db", calibrate_residuals=False)
    runner_text = runner_path.read_text(encoding="utf-8")
    deletion_targets = re.findall(r'^\s*"([a-z_]+\.(?:json|jsonl))",?$', runner_text, flags=re.MULTILINE)
    report = {
        "审计名称": "MPC正式v12与当前v15版本执行断层",
        "正式测试逐样本是否读取": False,
        "API调用次数": 0,
        "正式元数据": {
            "方法版本": metadata.get("method"),
            "状态": metadata.get("status"),
            "正式代码哈希": metadata.get("files", {}).get("code", {}).get("optimized_agent.py"),
            "正式运行脚本哈希": metadata.get("files", {}).get("code", {}).get("run_optimized_agent.sh"),
            "正式set_decoder": metadata.get("generation", {}).get("mpc_set_decoder"),
            "正式action_policy": metadata.get("generation", {}).get("mpc_action_policy"),
            "正式离线选择指标": metadata.get("generation", {}).get("mpc_residual_selection_metric"),
        },
        "当前源码": {
            "方法版本": agent.METHOD_VERSION,
            "代码哈希": 哈希(code_path),
            "运行脚本哈希": 哈希(runner_path),
            "训练样本数": len(train),
            "未校准默认residual_policy": model.residual_policy,
            "未校准默认metric_group_policy": model.metric_group_policy,
            "未校准默认dual_gate_policy": model.dual_gate_policy,
            "未校准默认retrieval_action_policy": model.retrieval_action_policy,
            "已有v15离线审计结论": {
                "双门策略是否准入": False,
                "推理决策": "保持H1",
                "v15平均官能团F1增益": 0.00481096,
                "v14平均官能团F1增益": 0.00794142,
                "证据来源": "audit_records/2026-08-01_optimized-agent-v15-dual-gate-offline-audit.md",
            },
            "训练候选宇宙": len(model.training_universe),
            "完整候选目录": len(model.universe),
        },
        "断层判定": {
            "版本不同": metadata.get("method") != agent.METHOD_VERSION,
            "代码哈希不同": metadata.get("files", {}).get("code", {}).get("optimized_agent.py") != 哈希(code_path),
            "运行脚本哈希不同": metadata.get("files", {}).get("code", {}).get("run_optimized_agent.sh") != 哈希(runner_path),
            "当前正式分数能否代表当前源码": False,
        },
        "运行脚本覆盖风险": {
            "同版本同代码才resume": True,
            "版本或代码不同会删除旧正式产物": True,
            "硬编码结果根目录": "results/Only-Deepseek/optimized-agent",
            "检测到的删除目标": deletion_targets,
            "安全决策": "没有单独新结果根目录和用户再次允许前，禁止执行 scripts/run_optimized_agent.sh",
        },
        "冻结正式汇总哈希": {"MFP": 哈希(mfp_summary), "MPC": 哈希(mpc_summary)},
        "结论": (
            "当前0.653867正式MPC是v12产物，不能作为当前v15源码的直接执行结果；"
            "最近离线实验基于当前源码内部H1/Bank，而正式基线来自旧代码，二者存在版本断层。"
        ),
    }
    默认输出.parent.mkdir(parents=True, exist_ok=True)
    默认输出.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
