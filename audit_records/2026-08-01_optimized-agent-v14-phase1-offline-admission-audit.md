# Optimized Agent v14 Phase 1 离线准入审查

## 结论

v14 Phase 1 实现和离线检查已完成，但 **未获准进入正式运行**。

这不是因为方向没有信号：高召回动作库和 Top-5 Scientist 均显示了明确正的可恢复空间。未通过的原因是，当前的自动执行器仍然无法充分识别会造成小幅损失的动作，未满足预先规定的胜负比准入标准。因此，代码保持 `KEEP_H1`，不会将该模块带入正式结果。

## 已实现的方法

- MPC 保留冻结的 H1 作为安全基线。
- 从 H1 边界构造原子级单交换 Action Bank，不允许生成候选库以外的分子。
- 使用训练集检索估计联合后验 `q(group, |GoldGroups| | query)`。
- 根据 Bayes 期望宏平均官能团 F1 为动作排序，Scientist 只保留固定 Top-5。
- 执行器最多接受一个单交换，严格保持 exact-N。
- OOF 外层折按完整分子谱精确聚类，相同谱不会跨折。
- 执行阈值使用留一外层折的交叉拟合，不使用 held-out 折标签。
- MPC 路径不加载 UniMol；MFP 保持已有 UniMol 路径。

## MPC 离线 OOF 结果

| 指标 | 结果 |
|---|---:|
| 训练查询 | 568 |
| 完整谱簇 | 377 |
| 五折样本量 | 114 / 114 / 114 / 113 / 113 |
| Action Bank Oracle 宏 F1 增益 | +0.04421160 |
| Scientist Oracle@5 宏 F1 增益 | +0.03539127 |
| Top-5 对 Bank Oracle 的捕获率 | 80.049753% |
| 交叉拟合执行器宏 F1 增益 | +0.00794142 |
| paired bootstrap 95% 下界 | +0.00345332 |
| 胜 / 负 | 93 / 59 |
| 改变查询数 | 152 |
| 折 0 增益 | +0.00770978 |
| 折 1 增益 | +0.00319624 |
| 折 2 增益 | +0.00671559 |
| 折 3 增益 | +0.01187780 |
| 折 4 增益 | +0.01026259 |
| 准入 | 否 |

## 为什么不准入

预注册规则同时要求：

1. Bank Oracle 为正；
2. Scientist Oracle@5 至少保留 Bank Oracle 的一半；
3. 交叉拟合增益的 bootstrap 下界大于零；
4. 胜数至少为负数的两倍；
5. 无系统性负收益折；
6. Scientist Oracle 捕获率高于 v13 的 22.97%。

v14 满足除第 4 项外的所有要求。93 胜 / 59 负说明它的平均收益来自较大的正收益，但对负收益事例的选择性仍不足。在不改变预注册规则的情况下，不能事后放宽门槛并宣称成功。

## 第一性原理判断

MPC 的决策难点已经从“候选召回”进一步收窄为“行动可验证性”：

- Action Bank 能产生明显好于 H1 的候选，所以不再是“Scientist 从未提出好答案”。
- Top-5 保留 80.05% Oracle 空间，所以主要问题也不是 Top-K 召回。
- 自动 Top-1 门控仍造成 59 个负收益查询，真正瓶颈是判断某个局部替换在当前食物语境下是否值得执行。
- 因此下一步不应继续扩大 Bank，也不应重新引入 UniMol；应对 Top-5 动作做独立、可弃权的证据验证。

## 与文献的关系

- Waegeman et al. (JMLR 2014) 支持使用联合标签-基数后验而非独立标签阈值来优化期望 F1。
- Recall-then-Verify (ACL 2022) 支持先高召回生成候选，再对有限候选进行独立验证；当前结果表明 Recall 阶段已经足够强，Verify 阶段是下一瓶颈。
- LEVER (ICML 2023) 支持将候选的生成与外部可验证信号分离，而不是让生成器用同一内部分数自证。
- Selective classification 与 conformal risk-control 方向支持在证据不足时保留 H1/弃权，但不应在未满足校准假设时冒充严格 conformal 保证。

## 建议的下一步

v14 Phase 2 只处理当前 Top-5，不再改变 H1 和 Action Bank：

1. Reviewer 对每个原子动作独立输出 `ACCEPT / REJECT / ABSTAIN`。
2. Reviewer 只看规范化的食物语境、动作的官能团增删账本和类型化证据，不看 Scientist 的自然语言说服。
3. Reviewer 不能创建新分子，只能为 Bank 内动作提供验证分数。
4. Executor 依然对 `KEEP_H1` 与被验证动作做风险门控；无充分证据时弃权。
5. 继续使用完整谱聚类 OOF，分别报告 Bank Oracle、Scientist Oracle@5、Reviewer 过滤后胜/负和 Final 增益。

Phase 2 的成功标准不是“使 Reviewer 总是改答案”，而是在保持已有正收益的同时显著减少 59 个负收益查询。

## 工程检查

- `optimized_agent.py` 和 `run_optimized_agent.sh` 版本均为 `optimized_agent_v14`。
- Python 语法检查通过；Shell 语法检查通过。
- MPC `check-only` 通过，71/71 查询均保证 exact-N，计划 Reviewer API 调用为 0。
- MPC 映射的 UniMol occurrence 为 0，符合“MPC 不使用 UniMol”。
- MFP `check-only` 通过，53,617/53,617 分子 occurrence 均映射 UniMol，候选 ledger 固定为 30。
- 未调用 API，未创建或覆盖正式结果。

