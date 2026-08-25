# ECCA 真实 PubMed 与反先验候选生成长任务阶段审查

日期：2026-08-15  
审查性质：探索阶段检查点；外部调用额度阻塞后的离线收口

## 审查范围

- 冻结 dev 71 条；未读 test；
- 真实 PubMed 候选条件检索；
- 固定候选关系评分、置换负对照和实体遮蔽 LLM verifier；
- 两种开放 Scientist 候选生成及候选位置审查；
- PubMed 相邻食物实体发现；
- POM/UniMol 在生成支持、排序和证据审计位置的边界；
- MPC P4/POM 历史产物完整性；
- 未修改正式 Scientist/Optimised Agent。

## 关键审计结论

1. 候选条件 PubMed 检索在 401 个候选上取得正确候选关系语言覆盖 1.0，但错误候选仍为 0.46985。覆盖不是关系区分度。
2. 关系文章计数的 category accuracy 为 0.38028，表面高于 Reviewer 的 0.33803，但 exact 从 1/71 降到 0/71。该方案属于类别捷径，停止。
3. 实体遮蔽 LLM verifier 的 20 条筛选 exact 净胜为 0，按预注册停止，不进行全量扩展。
4. `idf-role-diverse` 生成 20 条无 exact 命中，候选碰撞率 0.5625，停止。
5. `anti-prior-contrastive` 生成将碰撞率降至 0.275，支持分子精度升至 0.85924，并独占命中 `Soy milk`；与历史候选并集后同一 20 条 exact 命中从 1 增至 2。它是正向筛选信号，不是正式增益。
6. POM/UniMol 重排没有增加 exact 命中；保留为判别分子/结构族/证据一致性审计器，不作食物事实判定器。
7. PubMed 相邻实体发现独占找到 `Soybean`，但排序太低，Top-12/20 不增。由此提出候选前、分子先行的 MEED，而不是继续强化候选后 verifier。
8. 报告引用的 4 个 P4/UniMol-all 正式结构化评价产物当前缺失；P4 只能标为文档支持、产物不完整。POM demand/supply 失败 JSON 存续，本轮不训练新 MPC。

## 合规性

- 预测阶段未使用 FlavorDB food–molecule membership；gold/category 只用于排序冻结后的评价。
- PubMed 是唯一新增外部信息源；POM/UniMol 是已授权预训练表示。
- 未读独立 test；未改任务与评价；未改正式 Agent。
- 真实 API 的全 71 条扩展因平台用量限制被硬拒绝。未尝试替代 endpoint、替代密钥或其他外部来源规避限制。

## 阶段决定

- 停止：候选条件 PubMed verifier、当前规则关系分数、当前实体遮蔽 verifier、角色多样化生成、当前相邻实体排序。
- 保留待全量确认：反先验对比候选生成。
- 下一正式候选：MEED（Molecule-first Evidence-grounded Entity Discovery）。它必须先提高全 71 exact recall@K，再在固定候选上证明 relation verifier 的 exact 净胜，并超过匹配置换证据。
- 在上述门槛通过前，不进入 `optimised-agent.py`。

## 可复核位置

- 汇总报告：`exploration-experiments/reports/ECCA长任务全面探索与正式组合报告.md`
- 机器检查点：`exploration-experiments/results/ecca/ecca_long_task_checkpoint.json`
- 完整性审计：`exploration-experiments/results/ecca/mpc/p4_artifact_integrity_audit.json`
- 代码：`exploration-experiments/code/ecca/`
