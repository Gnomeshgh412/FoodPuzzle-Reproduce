#!/usr/bin/env python3
"""MFP 第十三轮：冻结第十二轮九候选池，只替换硬去重集合选择。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any


根目录 = Path(__file__).resolve().parents[1]
默认输出 = 根目录 / "results/Only-Deepseek/优化实验/第十三轮/MFP_九候选硬去重集合选择"
第十二轮 = 根目录 / "results/Only-Deepseek/优化实验/第十二轮/MFP_分视角独立科学家"
K3基线 = 根目录 / "results/Only-Deepseek/优化实验/第十一轮/双任务瓶颈补证/MFP_完整科学家阶段分解_联网重试"
随机种子 = 20260810


def 加载(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"无法加载：{path}")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def 读_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def 写_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise RuntimeError(f"禁止覆盖：{path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def 写_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise RuntimeError(f"禁止覆盖：{path}")
    path.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in rows), encoding="utf-8")


def 追加(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n"); f.flush()


def 规范(x: Any) -> str:
    return " ".join(str(x or "").strip().lower().split())


def 最终成功元数据() -> dict[str, dict[str, Any]]:
    rows = 读_jsonl(第十二轮 / "分视角假设与审查元数据.jsonl")
    return {str(x["id"]): x for x in rows if not x.get("错误") and x.get("审查器输出")}


def 选择三候选(record: dict[str, Any]) -> list[dict[str, Any]]:
    branches = record["视角分支"]
    selected, seen = [], set()
    for branch_index, branch in enumerate(branches, 1):
        for rank, hypothesis in enumerate(branch["完整三候选"], 1):
            key = 规范(hypothesis["predicted_food"])
            if key and key not in seen:
                selected.append({**hypothesis, "来源分支": branch_index, "分支内排名": rank})
                seen.add(key)
                break
    if len(selected) < 3:
        pool = []
        for branch_index, branch in enumerate(branches, 1):
            for rank, hypothesis in enumerate(branch["完整三候选"], 1):
                pool.append((rank, branch_index, hypothesis))
        for rank, branch_index, hypothesis in sorted(pool, key=lambda x:(x[0],x[1],规范(x[2]["predicted_food"]))):
            key = 规范(hypothesis["predicted_food"])
            if key and key not in seen:
                selected.append({**hypothesis, "来源分支": branch_index, "分支内排名": rank})
                seen.add(key)
                if len(selected) == 3: break
    if len(selected) != 3: raise RuntimeError(f"无法选出3个不同具体食物：id={record['id']}")
    return selected


def 准备(output: Path) -> None:
    source = 最终成功元数据()
    if len(source) != 71: raise RuntimeError("第十二轮冻结九候选池不完整")
    selected = [{"id": row_id, "选中三候选": 选择三候选(record)} for row_id,record in source.items()]
    protocol = {
        "任务输出": "具体食物名称",
        "当前瓶颈": "九候选池保留了多样信息，但每分支固定取第一名导致槽位重复与信息丢失。",
        "历史借鉴": ["第三轮MMR式集合去冗余正信号", "第十一轮增大K无效", "第十二轮多调用不等于多样性"],
        "唯一改动": "冻结九候选池，将每分支取第一名改为按分支1→2→3各取本分支排名最高的未选具体食物；不足时按分支内排名、分支序号全池补足。",
        "冻结": ["第十二轮Scientist九候选及rationale", "Scientist Prompt", "Reviewer Prompt", "Reviewer原Top3 demonstrations", "DeepSeek模型", "官方证据", "评测器"],
        "对照": ["K3完整Agent", "第十二轮每分支取第一名"],
        "探索保留": ["Scientist Top3宏类别oracle>0.577465", "Reviewer宏类别accuracy>0.352113", "Reviewer wins>losses", "具体食物Top3>=2/71", "71/71成功且无直接宏类别输出"],
        "结论上限": "dev已多轮自适应复用；通过也只能局部信号/探索保留，不直接冻结论文主方法。",
        "API预算": "0 Scientist + 71 Reviewer + 最多284官方类别映射",
        "边界": "不读正式test；不扩数据/证据；不使用任务外FlavorDB profile；不直接输出宏类别。",
    }
    写_json(output/"冻结实验方案.json",protocol); 写_jsonl(output/"硬去重选中三候选.jsonl",selected)
    print(json.dumps({"状态":"冻结完成","样本数":len(selected)},ensure_ascii=False))


def 执行Reviewer(output: Path, args: argparse.Namespace) -> None:
    agent = 加载("第十三轮MFP_agent", 根目录/"code/Only-Deepseek/scientific_agent.py")
    evaluation=agent.load_evaluation_module(); evaluation.load_local_env_file(); config=evaluation.resolve_llm_config(args); evaluation.require_api_key(config)
    train=读_jsonl(根目录/"results/splits/mfp/train.jsonl"); dev=读_jsonl(根目录/"results/splits/mfp/dev.jsonl"); train_by_id={str(x["id"]):x for x in train}
    source=最终成功元数据(); selected={str(x["id"]):x["选中三候选"] for x in 读_jsonl(output/"硬去重选中三候选.jsonl")}
    retrieval=agent.load_retrieval_metadata(第十二轮/"开发集BM25_Top9检索元数据.jsonl")
    evidence,_=agent.load_official_evidence(根目录/"data/collected_evidences/collected_evidences_task1.pkl"); idf=agent.build_train_idf(train)
    path=output/"Reviewer审查元数据.jsonl"; done={str(x["id"]) for x in 读_jsonl(path) if not x.get("错误") and x.get("Reviewer输出")} if path.exists() and args.resume else set()
    for index,row in enumerate(dev,1):
        rid=str(row["id"])
        if rid in done: continue
        starts=agent.select_starting_molecules(row,idf,evidence,5);_,evidence_text,hits=agent.build_evidence_blocks(row,starts,evidence,3); demos=agent.resolve_demos(rid,retrieval,train_by_id,3)
        hypotheses=[{"predicted_food":x["predicted_food"],"rationale":x["rationale"]} for x in selected[rid]]
        reviewer,error=None,None
        try:
            content=evaluation.call_chat_completion(agent.build_reviewer_messages(row,evidence_text,demos,hypotheses),config); reviewer=agent.parse_reviewer_output(content)
            if reviewer is None: raise RuntimeError("Reviewer解析失败")
        except Exception as exc:
            if "Insufficient Balance" in str(exc) or "HTTP error: 402" in str(exc): raise
            error=str(exc)
        追加(path,{"id":rid,"actual_food_for_audit":row.get("actual_food"),"选中三候选":selected[rid],"证据屏蔽后答案命中数":hits,"Reviewer输出":reviewer,"错误":error})
        print(f"MFP Reviewer进度: {index}/71 id={rid} {'成功' if not error else '失败'}",flush=True)
    rows=读_jsonl(path); success={str(x["id"]):x for x in rows if not x.get("错误") and x.get("Reviewer输出")}; print(json.dumps({"成功ID数":len(success),"尝试记录数":len(rows)},ensure_ascii=False))


def 拆分(output: Path) -> None:
    dev={str(x["id"]):x for x in 读_jsonl(根目录/"results/splits/mfp/dev.jsonl")}; attempts=读_jsonl(output/"Reviewer审查元数据.jsonl"); success={str(x["id"]):x for x in attempts if not x.get("错误") and x.get("Reviewer输出")}
    if len(success)!=71: raise RuntimeError("未获得71个完整Reviewer结果")
    candidates={1:[],2:[],3:[]}; reviewers=[];diagnostics=[]; category_words={"cereal","fruit","essential oil","plant","bakery","fungus","seed","dish","spice","flower","nutseed","beverage","animal product","vegetable","dairy","fish seafood","herb","legume","meat","additive"}
    for rid in dev:
        x=success[rid];foods=[str(h["predicted_food"]).strip() for h in x["选中三候选"]];rf=str(x["Reviewer输出"]["predicted_food"]).strip();gold=规范(dev[rid]["actual_food"])
        for i,f in enumerate(foods,1):candidates[i].append({"id":rid,"predicted_food":f})
        reviewers.append({"id":rid,"predicted_food":rf});diagnostics.append({"id":rid,"Scientist_Top1具体食物正确":规范(foods[0])==gold,"Scientist_Top3具体食物正确":gold in {规范(f) for f in foods},"Reviewer具体食物正确":规范(rf)==gold,"Reviewer选择序号":x["Reviewer输出"].get("selected_hypothesis_index"),"直接宏类别输出":规范(rf) in category_words,"三候选不同数":len({规范(f) for f in foods})})
    for i in (1,2,3):写_jsonl(output/f"Scientist第{i}候选_官方评测输入.jsonl",candidates[i])
    写_jsonl(output/"Reviewer最终预测_官方评测输入.jsonl",reviewers); 写_jsonl(output/"具体食物诊断.jsonl",diagnostics);print(json.dumps({"状态":"拆分完成"},ensure_ascii=False))


def bootstrap(values:list[float],repeats:int=10000)->list[float]:
    rng=random.Random(随机种子);means=sorted(sum(values[rng.randrange(len(values))] for _ in values)/len(values) for _ in range(repeats));return [means[int(.025*repeats)],means[int(.975*repeats)-1]]


def 汇总(output: Path) -> None:
    labels=["Scientist第1候选","Scientist第2候选","Scientist第3候选","Reviewer最终预测"];cur={l:{str(x["id"]):x for x in 读_jsonl(output/f"{l}_官方评测逐样本.jsonl")} for l in labels};oldlabels=["科学家第1候选","科学家第2候选","科学家第3候选","审查器最终预测"];old={l:{str(x["id"]):x for x in 读_jsonl(K3基线/f"{l}_官方类别逐样本.jsonl")} for l in oldlabels};ids=[str(x["id"]) for x in 读_jsonl(根目录/"results/splits/mfp/dev.jsonl")]
    details=[];og=[];rg=[]
    for idx,rid in enumerate(ids):
        oo=any(old[l][rid]["correct"] for l in oldlabels[:3]);no=any(cur[l][rid]["correct"] for l in labels[:3]);orr=bool(old[oldlabels[3]][rid]["correct"]);nr=bool(cur[labels[3]][rid]["correct"]);og.append(float(no)-float(oo));rg.append(float(nr)-float(orr));details.append({"id":rid,"固定分块":idx%5,"K3_oracle":oo,"方法_oracle":no,"K3_Reviewer":orr,"方法_Reviewer":nr})
    diag=读_jsonl(output/"具体食物诊断.jsonl");fold=[sum(rg[i] for i,x in enumerate(details) if x["固定分块"]==f)/sum(x["固定分块"]==f for x in details) for f in range(5)]
    summary={"样本数":71,"三候选单列宏类别accuracy":[sum(cur[l][i]["correct"] for i in ids)/71 for l in labels[:3]],"K3_Scientist_Top3宏类别oracle":sum(x["K3_oracle"] for x in details)/71,"方法_Scientist_Top3宏类别oracle":sum(x["方法_oracle"] for x in details)/71,"Scientist_oracle增益":sum(og)/71,"Scientist_oracle增益bootstrap_95%区间":bootstrap(og),"K3_Reviewer宏类别accuracy":sum(x["K3_Reviewer"] for x in details)/71,"方法_Reviewer宏类别accuracy":sum(x["方法_Reviewer"] for x in details)/71,"Reviewer增益":sum(rg)/71,"Reviewer增益bootstrap_95%区间":bootstrap(rg),"Reviewer_wins_losses_ties":[sum(x>0 for x in rg),sum(x<0 for x in rg),sum(x==0 for x in rg)],"Reviewer五分块增益":fold,"Scientist_Top1具体食物命中数":sum(x["Scientist_Top1具体食物正确"] for x in diag),"Scientist_Top3具体食物命中数":sum(x["Scientist_Top3具体食物正确"] for x in diag),"Reviewer具体食物命中数":sum(x["Reviewer具体食物正确"] for x in diag),"直接宏类别输出数":sum(x["直接宏类别输出"] for x in diag),"三候选全部不同样本数":sum(x["三候选不同数"]==3 for x in diag)}
    summary["是否通过探索保留"]=bool(summary["方法_Scientist_Top3宏类别oracle"]>summary["K3_Scientist_Top3宏类别oracle"] and summary["方法_Reviewer宏类别accuracy"]>summary["K3_Reviewer宏类别accuracy"] and summary["Reviewer_wins_losses_ties"][0]>summary["Reviewer_wins_losses_ties"][1] and summary["Scientist_Top3具体食物命中数"]>=2 and summary["直接宏类别输出数"]==0)
    summary["结论"]="获得局部信号并探索保留，不冻结论文主方法" if summary["是否通过探索保留"] else "未通过并停止"
    写_jsonl(output/"配对审查逐样本.jsonl",details);写_json(output/"完整审查结果.json",summary);print(json.dumps(summary,ensure_ascii=False,indent=2))


def main()->int:
    p=argparse.ArgumentParser();p.add_argument("动作",choices=["准备","执行Reviewer","拆分","汇总"]);p.add_argument("--输出",type=Path,default=默认输出);p.add_argument("--resume",action="store_true");p.add_argument("--llm-provider",default="deepseek");p.add_argument("--llm-model",default="deepseek-v4-flash");p.add_argument("--llm-base-url",default=None);a=p.parse_args();{"准备":准备,"执行Reviewer":lambda x:执行Reviewer(x,a),"拆分":拆分,"汇总":汇总}[a.动作](a.输出);return 0


if __name__=="__main__":raise SystemExit(main())
