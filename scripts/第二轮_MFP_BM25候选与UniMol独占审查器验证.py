#!/usr/bin/env python3
"""第二轮 MFP：复用第一轮控制框架，仅将候选召回替换为 BM25。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


项目根目录 = Path(__file__).resolve().parents[1]
基础脚本 = 项目根目录 / "scripts/第一轮_MFP_UniMol独占审查器验证.py"
输出目录 = 项目根目录 / "results/Only-Deepseek/优化实验/第二轮/MFP_BM25候选与UniMol独占审查器"


def main() -> int:
    spec = importlib.util.spec_from_file_location("第二轮MFP基础框架", 基础脚本)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载基础脚本：{基础脚本}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    forwarded = [sys.argv[0], "--候选方法", "bm25", "--输出目录", str(输出目录)]
    if "--仅准备" in sys.argv[1:]:
        forwarded.append("--仅准备")
    sys.argv = forwarded
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
