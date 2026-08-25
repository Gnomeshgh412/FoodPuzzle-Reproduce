# MPC v14 设计：Counterfactual Expected-F1 Recall–Verify Agent

- 日期：2026-07-31
- 范围：Only-Deepseek Scientist–Reviewer 主路线
- 性质：历史实证、数据审计、第一性原理与文献共同约束的方法设计
- 本轮操作：只读检查 v1–v13 记录与数据，查阅论文，执行 train-only 离线压力验证，形成设计记录
- 本轮未执行：正式代码修改、模型 API、正式预测、正式评测、结果覆盖、结果目录创建、Git 更新或提交
- MFP：冻结现有 UniMol 路线，本设计只改变 MPC
- MPC：继续不使用 UniMol

## 1. 为什么需要重新定义下一版

当前问题已经可以分成三个不同层次：

1. **Candidate bank**：正确局部动作是否存在于候选空间；
2. **Scientist proposal**：Scientist 是否把正确动作放入短名单；
3. **Reviewer selection**：Reviewer 是否能从短名单中识别正收益动作。

过去将三层合并成“生成 H1/H2/H3，再由 Reviewer 选一个完整集合”，导致：

- 候选召回不足和 Reviewer 选错无法区分；
- 三个完整列表高度相关，oracle 本身很低；
- Reviewer 面对几十到几百个分子的长列表，无法定位一次局部差异；
- 同模型、同证据的自审没有新增可判定信息；
- 最终 F1 下降时无法确定失败发生在哪一层。

下一版不再把“多生成几个完整答案”视为 Scientist 能力，而是将最小可证伪
单位定义为：

`action = remove one molecule + add one molecule`

同时始终保留：

`A0 = KEEP_H1`

## 2. 历史实证约束

### 2.1 必须保留的模块

跨 v1–v13 重复出现正证据：

- occurrence/cooccurrence H1；
- BM25/IDF/profile retrieval；
- exact-N 确定性执行；
- 只在 H1 边界进行少量交换；
- 弱通道允许退化为零；
- train-only OOF 决定是否准入；
- 功能团集合覆盖能够改善官方 MPC 指标。

### 2.2 必须拒绝的模块

跨版本重复出现负证据：

- raw UniMol similarity 接管 MPC 排序；
- 将 profile 外候选当成确定负例；
- occurrence、retrieval、结构、感知的固定全局融合；
- Reviewer 在没有候选级独立证据时强制选择；
- Reviewer/Fusion 自由改写完整集合；
- exact-molecule F1 代替 functional-group F1；
- 根据当前 71 条测试错误设计 N 桶或特殊名称规则。

### 2.3 Scientist 与 Reviewer 的既有上限

v10：

| 阶段 | MPC F1 |
|---|---:|
| H1 | 0.6725 |
| H2 | 0.6584 |
| H3 | 0.6613 |
| H1/H2/H3 oracle | 0.6811 |
| Reviewer | 0.6576 |
| Fusion | 0.6574 |

这说明两种失败同时存在：

1. 三个完整候选的 oracle 相对 H1 只有约 `+0.0086`，Scientist 候选上限不足；
2. Reviewer 将潜在正上限变成了实际负收益。

v13：

- `N+30` 候选池覆盖 96.49% 的 gold functional groups；
- 568 条中 286 条存在正收益单交换；
- 边界单交换 oracle 约 `+0.0480`；
- v13 实际约 `+0.00527`，只取得 oracle 的 10.97%；
- 102 wins、86 losses、380 ties。

所以正确动作通常存在于原始 bank，但没有被可靠转化为最终动作。

## 3. 数据集与复现边界

### 3.1 当前不是官方 exact MPC 输入

公开 `MPC_tasks.jsonl` 缺少论文定义中的 official：

- `partial_molecules`
- `n`

当前重构为：

`partial = FlavorDB full profile - public missing_molecules`

`n = len(public missing_molecules)`

因此当前工作是 FlavorDB-derived reconstructed MPC。它可以用于同一协议下的
方法比较，但绝对分数不能当作论文表格的同条件复现。

### 3.2 观测机制高度偏向大比例缺失

train missing ratio 四分位数：

- 0.8750
- 0.8925
- 0.8952

v10 使用 15%、35%、60% 为主的随机遮蔽，与真实重构任务约 89% 缺失严重
不一致。这会让训练阶段学到“小修补已知 profile”，而测试要求“从极少 partial
恢复大 missing set”。

下一版第一阶段不再生成大量任意遮蔽样本。真实重构 row 是主要训练单位；
若以后加入增强，遮蔽率必须从 train-only 经验分布采样，并作为独立消融。

### 3.3 原 grouped OOF 实际没有按 food 形成群组

568 条 train row 对应 568 个唯一规范化 food。此前按 `target_food` grouped
OOF 实际接近普通随机 OOF，不能阻止相似 profile 跨折。

train 完整 profile 审计：

- 568 rows；
- 377 个唯一 exact full profiles；
- 204 rows 属于重复 exact-profile group；
- 最大 exact-profile group 有 62 rows。

当前 split 的跨集合相似性：

- test 中 23/71 的 exact full profile 已在 train 出现；
- test 到 train 的最近 full-profile Jaccard 中位数约 0.9459；
- 42/71 的最近 Jaccard 不低于 0.8；
- 39/71 不低于 0.9。

这些事实说明：

1. 随机 split 上的绝对结果混合了 profile 记忆与真正泛化；
2. 所有方法仍可在同一 split 下公平比较；
3. 方法选择必须采用 profile-cluster grouped OOF；
4. 论文陈述必须限定为 reconstructed benchmark；
5. 不修改现有 `split_data`，只升级方法内部的训练验证协议。

### 3.4 exact-profile grouped OOF 压力验证

本轮将相同 full profile 的 row 固定在同一折，并按 group size 平衡五折。

| 指标 | 结果 |
|---|---:|
| H1 mean F1 | 0.891448 |
| v13 mean F1 | 0.904826 |
| v13 gain | **+0.013379** |
| single-swap oracle gain | **+0.058252** |
| oracle capture ratio | **22.97%** |
| wins / losses / ties | 119 / 54 / 395 |

五折 v13 gain：

- fold 0: `-0.000302`
- fold 1: `+0.042815`
- fold 2: `-0.000037`
- fold 3: `+0.019213`
- fold 4: `+0.005183`

结论：

- v13 的 metric-aligned 方向不是完全重复 profile 泄漏造成的；
- 但收益由少数 fold 主导，仍没有达到稳定准入；
- profile 隔离后 oracle 仍有 `+0.0583`，下一版有足够空间；
- 下一版必须把 exact-profile grouped OOF 作为最低协议，并增加
  near-duplicate stress split。

## 4. FoodPuzzle MPC 的第一性原理

### 4.1 真正的决策对象

输入：

- food `f`
- observed partial molecule set `O`
- required missing molecule count `N`

输出：

- exactly `N` molecules `S`

正式效用：

`F1(groups(S), groups(Gold))`

因此模型不应只学习：

`P(molecule occurs | food, partial)`

还必须学习：

`P(functional group is required | food, partial, N)`

以及集合条件动作效用：

`delta(action | S) = F1(groups(S - remove + add), GoldGroups)
                     - F1(groups(S), GoldGroups)`

### 4.2 为什么独立 group threshold 不够

F1 是非可分解集合指标。一个 group 的价值依赖：

- gold group set 的总基数；
- 当前预测已经覆盖哪些 group；
- remove 是否删除独占 group；
- add 是否引入 false-positive groups；
- 多个 group 的相关性。

Waegeman et al. 的 F-measure 决策分析表明，用 Hamming 或 subset loss 等
代理目标可能产生高 F1 regret，Bayes-optimal F1 需要联合信息：

https://www.jmlr.org/papers/v15/waegeman14a.html

对应 MPC，下一版不再只预测独立 `P(group)`，而估计：

`q(g, s | x) = P(g in GoldGroups, |GoldGroups| = s | x)`

给定某个动作后的确定性预测 group set `P_action`，可计算：

`ExpectedF1(action)
 = sum_{g in P_action} sum_s [2 / (|P_action| + s)] * q(g, s | x)`

这将正式 F1 直接带入动作排序，并显式考虑 gold group cardinality。

### 4.3 为什么 set-conditioned marginal utility 必须进入 Scientist

Contextual Submodular Prediction 研究的是固定大小集合，同时考虑单项质量与
集合多样性的预测：

https://proceedings.mlr.press/v28/ross13b.html

本项目中：

- occurrence/retrieval 是单项相关性；
- 新增需求 group 是覆盖增益；
- 重复 group 是冗余；
- 删除独占 group 是损失；
- exact-N 是基数约束。

这也解释 v5 为什么出现“基础排序 0.5739，集合解码 0.6363”的明确阶段
提升：集合互补思想本身有效，失败的是弱 H1 和后续 Verifier。

## 5. v14 总体架构

建议名称：

**Counterfactual Expected-F1 Recall–Verify Agent**

```text
Frozen occurrence H1
        ↓
Deterministic Action Bank (high recall, Top-M)
        ↓
Scientist (Top-K proposals + typed claims)
        ↓
Independent Metric Reviewer (verify or abstain)
        ↓
Risk-controlled deterministic executor
        ↓
Exact-N final set
```

### 5.1 Stage A：冻结 H1

H1 只使用历史已验证的：

- occurrence frequency；
- cooccurrence；
- food/partial profile retrieval；
- PU-aware pairwise ordering；
- exact-N Top-N。

要求：

- 单独输出 H1；
- H1 的训练逻辑、随机种子和 hash 固定；
- 后续模块只能返回 `KEEP_H1` 或有限局部交换；
- 任何 Agent 失败均回退 H1。

### 5.2 Stage B：高召回 Action Bank

不是直接生成 H2/H3 完整集合，而是在 H1 边界产生 `M` 个原子动作：

`Ai = (remove_i, add_i)`

remove 来源：

- H1 边界；
- occurrence margin 小；
- group redundancy 高；
- retrieval support 弱；
- 删除后不应轻易损失独占高需求 group。

add 来源：

- occurrence `N+30` 边界；
- IDF/profile retrieval residual；
- train-only food-conditioned candidates；
- 有确定性 FlavorDB group mapping；
- 有明确 provenance。

Action Bank 排序只要求高召回，不负责最终决定。对每个 action 记录：

- H1 cutoff margin；
- add/remove occurrence 与 retrieval；
- groups added；
- groups removed；
- groups lost after removal；
- redundant groups；
- false-positive group risk；
- deterministic mapping coverage；
- claim type；
- evidence provenance。

Bank 的核心指标：

- `Bank Oracle@M`
- positive-action recall
- oracle functional-group F1 gain

只有 Bank Oracle 足够高，才进入 Scientist/Reviewer。

### 5.3 Stage C：GFM-style Group–Cardinality Scientist

用 train-only、profile-cluster cross-fitting 估计：

`q(g, s | food, partial, N, H1 coverage)`

建议采用低容量、可审计的两路估计：

1. **Retrieval posterior**
   - BM25/IDF/profile neighbors；
   - exact-profile cluster 逆频率加权；
   - neighbor relevance 与 rank discount；
   - 输出 group-cardinality joint counts。

2. **Sparse conditional posterior**
   - food tokens；
   - partial molecule/group indicators；
   - 连续 N、partial size、missing ratio；
   - H1 predicted group coverage；
   - 低容量正则化模型。

两路只允许通过 inner OOF 融合，不固定全局平均。

对每个 action 计算：

- ExpectedF1；
- Expected delta-F1 relative to H1；
- fold/bootstrap uncertainty；
- probability of positive utility；
- proposed claim type。

Scientist 从 Action Bank 输出固定 Top-K，但不能生成 Bank 外分子。

Scientist 必须单独报告：

- `Scientist Oracle@K`
- 相对 Bank Oracle 的 recall；
- Top-1 gain；
- Top-K 中 positive action coverage；
- proposal diversity；
- 按 profile cluster 和 N 连续分层的稳定性。

### 5.4 Stage D：PU / exposure-aware 学习

当前 missing set 是已观测 positive，不代表 profile 外候选是真 negative。

SAR PU Learning 强调 positive 的被观测概率可能依赖属性，不能假定完全随机：

https://proceedings.mlr.press/v94/bekker18a.html

Missing-label multilabel work同样指出不完整标签需要专门风险处理：

- https://proceedings.mlr.press/v32/yu14.html
- https://proceedings.mlr.press/v119/cabannnes20a.html

对应 v14：

- hidden/missing molecules 是 positive；
- partial molecules 是 observed positive，但不能作为 missing action positive；
- profile 外分子是 unlabeled；
- 不构造“低频、零检索、低相似 = negative”；
- action supervision 直接使用 train gold functional-group delta-F1；
- propensity 只用于校正不同 profile/group 被观测机会；
- 相同 full-profile cluster 逆频率加权，避免重复 profile 主导 posterior。

第一版不需要复杂深网。低容量 pairwise/action utility model更符合 568 条
训练样本的规模。

### 5.5 Stage E：Independent Metric Reviewer

Reviewer 不再比较三个完整集合，而逐 action 审查。

Scientist 与 Reviewer 的信息必须不完全相同：

Scientist 主要看到：

- posterior；
- retrieval；
- H1 boundary；
- action expected-F1。

Reviewer 主要看到独立可执行 ledger：

- FlavorDB functional groups；
- remove 后 group multiplicity；
- add 后新增与额外 groups；
- candidate-level typed evidence；
- provenance；
- entity normalization / mapping status。

Reviewer 输入必须：

- action A/B/C 匿名化；
- 顺序按固定随机种子打乱；
- 不显示 H1/H2/H3、模型置信权威标签；
- 不显示 gold；
- 不显示测试指标。

Reviewer 输出固定 schema：

```json
{
  "action_id": "A3",
  "claim_type": "occurrence | functional_replication",
  "verified_added_groups": [],
  "unsupported_claims": [],
  "contradictions": [],
  "evidence_ids": [],
  "verdict": "ACCEPT | REJECT | ABSTAIN"
}
```

Reviewer 不得：

- 生成新分子；
- 改写完整集合；
- 强制选择；
- 把 sensory similarity 当 occurrence；
- 在没有独立依据时给 ACCEPT。

Recall-then-Verify 将多答案逐项召回和验证分开，避免联合生成时不同答案之间
相互干扰：

https://aclanthology.org/2022.acl-long.128/

LEVER 的关键不是“再问一次模型”，而是 verifier 获得可执行结果，再与生成
概率结合：

https://proceedings.mlr.press/v202/ni23b.html

无外部反馈的 intrinsic self-correction 可能退化，而 CRITIC 的改善依赖工具
反馈：

- https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html
- https://proceedings.iclr.cc/paper_files/paper/2024/hash/fef126561bbf9d4467dbb8d27334b8fe-Abstract-Conference.html

因此同一个 DeepSeek 可以承担两个角色，但 Reviewer 必须获得独立 ledger，
且上下文隔离；单纯改变角色提示词不算独立验证。

### 5.6 Stage F：Risk-controlled deterministic executor

最终执行器比较：

- `A0 = KEEP_H1`
- Scientist Top-K 中被 Reviewer 接受的 actions

执行条件：

1. estimated delta-F1 > 0；
2. one-sided lower confidence bound > 0；
3. Reviewer 为 ACCEPT；
4. deterministic ledger 无映射/基数/重复错误；
5. action 未移除受保护的独占需求 group；
6. exact-N 检查通过。

否则：

`KEEP_H1`

Selective prediction 将准确性与覆盖率显式权衡：

https://proceedings.mlr.press/v130/gangrade21a.html

Conformal Risk Control 可控制一族单调决策规则的期望损失：

https://openreview.net/pdf?id=33XGfHLtZg

本项目第一阶段可以继续使用 paired bootstrap 和 one-sided gate；CRC 只在
明确构造单调阈值族、独立校准集和适合的损失后作为增强，不能只贴名称。

### 5.7 为什么 v14 先只允许一次交换

profile-group OOF：

- one-swap oracle gain：`+0.05825`

此前边界 oracle：

- one swap：`+0.04802`
- three greedy swaps：`+0.06100`

第一次交换已包含多步 oracle 的大部分可达收益。当前瓶颈是选不准第一步，
不是交换预算不足。

因此 v14 第一阶段：

- 只允许一次交换；
- 先证明 Scientist Oracle@K 与 Reviewer 净贡献；
- 多步必须在第一次成功后重新计算集合状态并重新校准；
- 多步作为后续消融，不进入初始主方法。

## 6. 数据与验证协议

### 6.1 Nested profile-cluster OOF

替代当前 food-name grouped OOF：

1. 用 train row 的完整 profile signature 构造 exact-profile groups；
2. 外层 5 折按 group size 平衡；
3. 所有模型、posterior、阈值只在其余外层折训练；
4. 内层 OOF 选择 K、M、正则化、gate；
5. 外层只报告，不反向调参；
6. 对 Jaccard >= 0.9 的 near-duplicate clustering 做压力测试；
7. exact-profile OOF 与 near-duplicate stress 必须同时记录。

### 6.2 不使用测试集设计方法

当前 71 test 已被多版本反复查看，只能继续作为固定历史比较集，不能再用于：

- 选择 N 阈值；
- 选择 action features；
- 调 Reviewer prompt；
- 选择 K/M；
- 决定某个分子或 group 规则。

方法冻结后，正式 test 只运行一次。此前单独抽出的 holdout 也只能在完整冻结
后使用，不能边看边改。

### 6.3 逐阶段指标

每一折必须同时保存/汇报：

1. H1 F1；
2. Bank Oracle@M；
3. Scientist Oracle@K；
4. Scientist Top-1；
5. Reviewer selected action；
6. final executor；
7. Bank-to-Scientist recall loss；
8. Scientist-to-Reviewer selection regret；
9. wins / losses / ties / abstains；
10. exact-N、重复、partial 泄漏、unmapped；
11. small-profile 与大-profile稳定性；
12. exact-profile 与 near-duplicate stress 差异。

这样才能回答：

- Scientist 没给正确动作；
- 还是给了但 Reviewer 没选出；
- 还是执行器门控过度保守。

## 7. 预注册准入条件

### 7.1 Action Bank

- Bank Oracle@M 在每折均高于 H1；
- mean oracle gain 有足够空间；
- functional-group coverage 不依赖测试 cache；
- exact-profile 与 near-duplicate stress 方向一致。

### 7.2 Scientist

- Scientist Oracle@K 捕获至少一半 Bank oracle 可达增益；
- Top-1 相对 H1 的 paired bootstrap lower bound > 0；
- 五折无系统性负收益；
- wins 至少为 losses 的两倍；
- 中等 N 不能像 v13 一样近似零或负；
- oracle capture ratio 明显高于 v13 的 22.97% profile-group结果。

### 7.3 Reviewer

只有 Scientist Oracle@K 通过后才调用 API。

- 冻结 Scientist slate 后单独评测；
- Reviewer 相对 Scientist Top-1 必须有正增量；
- Reviewer regret 显著低于 v10；
- Reviewer losses 不得多于 wins；
- 若不通过，Reviewer 保留解释用途，不进入预测路径。

### 7.4 Final

- final 相对冻结 H1 的 lower bound > 0；
- exact-N 100%；
- 无重复与 partial 泄漏；
- 不读取 LLM functional-group evaluation cache；
- MPC 不使用 UniMol；
- 不允许某一个 fold 单独贡献全部收益。

这些是工程准入标准，不应伪装成对未知真实分布的严格统计证明。

## 8. 实现顺序

### Phase 1：完全离线，不调用 DeepSeek

1. 冻结 H1；
2. 实现 profile-cluster nested OOF；
3. 构造 Action Bank；
4. 训练 group-cardinality joint posterior；
5. 计算 ExpectedF1 与 Top-K；
6. 报告 Bank Oracle@M、Scientist Oracle@K、Top-1；
7. 不修改正式结果。

若 Scientist 没有通过准入，停止，不进入 Reviewer。

### Phase 2：Reviewer check-only / train-only

1. 冻结 Phase 1 code、slate、prompt；
2. Reviewer 只审固定 action；
3. 使用独立 chemical/evidence ledger；
4. 评估 Reviewer regret 与净贡献；
5. 不读取正式 test。

### Phase 3：冻结后的正式运行

仅在 Phase 1 和 Phase 2 均通过后：

- 记录 hash；
- 覆盖既有 optimized-agent 路径；
- 运行 MFP 与 MPC；
- 与 v2、v9、ICL、原 Agent、v12 比较；
- test 结果不用于现场改参数。

## 9. 消融设计

主方法冻结后再运行：

1. H1 only；
2. H1 + Action Bank Top-1；
3. `P(group)` 独立阈值 vs `q(group, cardinality)` ExpectedF1；
4. no profile-cluster weighting；
5. no PU/exposure correction；
6. no Reviewer；
7. Reviewer without independent ledger；
8. no risk gate；
9. retrieval-only posterior vs sparse-only posterior vs OOF fusion；
10. one swap vs conditional second swap。

UniMol 不属于 MPC v14 消融。它留在 MFP；如果未来重新研究 MPC-UniMol，
必须作为完全独立路线，与 v14 冻结后单变量比较。

## 10. 创新点

### 10.1 Metric-aligned counterfactual hypothesis

将 MPC Scientist 的“假设”从三个完整分子列表改为可证伪的局部反事实动作，
并直接以 functional-group expected-F1 排序。

### 10.2 Group-cardinality joint decision

不是把官能团独立分类后阈值化，而是估计 group 与 gold group cardinality 的
联合量，直接服务非可分解 F1。

### 10.3 Recall–verify decomposition with executable chemistry feedback

明确分离：

- Bank 是否召回；
- Scientist 是否提出；
- Reviewer 是否选择；
- Executor 是否安全执行。

Reviewer 得到 FlavorDB group ledger 和 typed evidence，而不是只看 Scientist
的自然语言理由。

### 10.4 Profile-cluster robust validation

针对 reconstructed FoodPuzzle 中大量相同/近重复 full profiles，采用
profile-cluster nested OOF，并公开普通随机 split 与 cluster-stress 的差异。

这些创新都来自任务结构和已观察失败，不依赖测试样本特例。

## 11. 最终判断

下一版不应继续尝试：

- 更复杂的 H1/H2/H3 完整列表；
- 更长 Reviewer prompt；
- 更多 Agent 层；
- 更大的候选目录；
- 更高 swap budget。

应该解决的首要问题是：

> 在已经具有 96.49% gold-group 覆盖的边界候选中，如何让 Scientist 以高
> Oracle@K 提出真正有正 delta-F1 的局部动作，并让 Reviewer 借助独立化学
> ledger 只接受可验证动作。

因此 v14 的正确路线为：

> **冻结强 H1 + high-recall counterfactual Action Bank +
> group-cardinality ExpectedF1 Scientist + independent Metric Reviewer +
> risk-controlled exact-N executor。**

在 Phase 1 证明 Scientist 以前，不调用 Reviewer；在 Reviewer 单独证明正
增量以前，不进入正式 test。
