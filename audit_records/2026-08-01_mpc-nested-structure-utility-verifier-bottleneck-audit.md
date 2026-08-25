# MPC 严格嵌套结构效用验证器瓶颈审计

> **后续限定（2026-08-01）**：本文关于“Action Bank 绝对容量不是瓶颈”的表述过强。后续全量 OOF 搜索空间审查发现，当前 Bank20 在“末尾 12 remove × 40 add × 单步”这一受限邻域内确实已捕获约 98.2% Oracle，但名义 Bank20 平均只有 7.78 个有效终态；放宽到全 H1 remove、100 add 和两步搜索后，训练侧 Oracle 仍可由约 `+0.04421` 增至 `+0.05866`–`+0.06396`。因此本文结论只适用于“当前 Bank 内 verifier 很弱”，不能用于否定 Scientist 搜索空间瓶颈。详见 `2026-08-01_mpc-scientist-search-space-bottleneck-and-literature-audit.md`。

## 1. 审计目的

本轮不修改正式 Agent，也不调用 API。目标只有一个：验证“使用训练查询真实 missing molecules 构造结构 F1 监督，再训练一个低容量 action verifier”能否合法识别 held-out MPC 查询中的有益动作。

此前已经确认 Top-20 Action Bank 的事后正式 Oracle 约为 `+0.0459`，因此 Scientist 并非完全没有生成更好答案。尚未解决的是：推理时能否在不知道 Gold 的情况下识别这些答案。

## 2. 严格无泄漏协议

- 外层：固定 5 折 OOF；每次只评估一个完全 held-out 外层折。
- 内层：外层训练数据再做 4 折交叉拟合，生成 Ridge 的训练记录。
- nuisance model、训练标签和特征均不得读取外层 Gold。
- 训练标签：仅对训练查询使用其真实 missing molecules，计算候选动作的结构集合 F1 增益。
- 推理特征：22 个推理时可得的通用特征，包括合法训练邻居结构 posterior、候选 rank、集合大小、add/remove 结构边际、检索支持和候选间结构 Jaccard 等。
- 模型预先固定为 `StandardScaler + Ridge(alpha=1.0)`，不搜索模型、超参数或阈值。
- 正式 evaluation cache 只在外层预测全部冻结后用于事后评估。
- 没有写入中间预测、测试结果或正式结果目录。

外层共有 568 个查询，折大小为 `114/114/114/113/113`；其中 507 个查询能够按当前正式审计口径重构并评估动作。

## 3. 核心结果

### 3.1 代理目标是否可学习

| 指标 | 结果 |
|---|---:|
| 预测值与训练定义的结构效用相关性 | **0.557099** |
| 结构效用 RMSE | **0.074942** |

低容量模型能够学到相当一部分结构代理目标。因此，失败不能简单归因于“Ridge 太弱”或“结构特征完全没有信息”。

### 3.2 学到的分数是否能识别正式 MPC 收益

| 指标 | 结果 |
|---|---:|
| 与正式 action gain 的相关性 | **0.068344** |
| 正收益动作 ROC-AUC | **0.441063** |
| 正收益动作 PR-AUC | **0.366972** |
| 正收益动作 prevalence | **0.301288** |

这是本轮最关键的结果。模型能预测结构代理目标，却几乎不能预测正式动作收益；ROC-AUC 甚至低于随机排序的 0.5。说明可学习的主要是结构代理中容易预测、但对最终选择不关键的部分，而不是“当前食物应该增加或删除哪些分子”这一条件化反事实效用。

### 3.3 查询级执行收益

| 执行范围 | 平均正式 F1 增益 | bootstrap 95% 下界 | 正/负/平 |
|---|---:|---:|---:|
| 原 Top-5 | **+0.00662245** | +0.00338308 | 80 / 43 / 384 |
| Top-20 Bank | **+0.00610728** | +0.00273932 | 88 / 41 / 378 |

该收益与此前合法结构 posterior 的 `+0.00750665` 处于同一量级，并没有形成新突破。扩大到 Top-20 仍未改善，说明增加候选后，噪声最大的候选也更容易被错误高估。

### 3.4 候选召回与选择差距

| 指标 | 平均正式 F1 增益 |
|---|---:|
| verifier 重排后的 Top-5 Oracle | **+0.02846320** |
| 完整 Top-20 Bank Oracle | **+0.04590472** |
| 重排 Top-5 对 Bank Oracle 的捕获率 | **62.00%** |

Scientist 的 Bank 中依然存在大量可改善动作，但 verifier 没有把它们稳定送入前列。真正瓶颈是候选动作的 query-conditioned utility identification，而不是 Action Bank 的绝对容量。

### 3.5 折间稳定性

| 外层折 | Top-5 executor | Bank20 executor | 重排 Top-5 Oracle |
|---|---:|---:|---:|
| 0 | +0.002254 | -0.000306 | +0.007087 |
| 1 | +0.013510 | +0.014468 | +0.060949 |
| 2 | +0.005598 | +0.004250 | +0.010582 |
| 3 | +0.010433 | +0.008987 | +0.042335 |
| 4 | +0.001694 | +0.003721 | +0.023634 |

收益高度依赖折 1 和折 3，折 0 的 Bank20 执行甚至为负。这进一步否定了将当前 verifier 作为稳定通用模块的合理性。

## 4. 真正瓶颈

### 4.1 不是结构“供给”不可表示

SMILES/SMARTS、UniMol 或结构官能团可以描述候选分子具有什么。结构代理目标相关性达到 0.557，已经证明这部分信息能够被模型吸收。

### 4.2 是食物条件化“需求”与动作交互不可辨识

MPC 的决策量是：

`ΔF1(action | target food, observed partial set, evidence)`

而当前 22 个特征主要刻画：

`structure(candidate proposal)` 与 `training-neighbor marginal demand`

它们没有充分建模“这个食物在当前已观察集合条件下，具体缺少什么”与“这一个 add/remove 动作如何改变整个集合 F1”的交互。结构监督能被预测，却没有转化为正式 action ranking，正是这种 demand–supply interaction 缺失的直接证据。

### 4.3 点式回归优化了错误的统计问题

Ridge 最小化跨所有动作的均方误差，但 Reviewer 实际面对的是“同一查询内部，从多个高度相关动作中选出相对最优且优于 no-op 的动作”。整体相关性并不保证 query 内排序正确，也不保证集合 F1 改善。当前 `0.557 -> 0.068` 的坍缩说明，继续加深点式回归器或添加更多结构特征没有充分依据。

### 4.4 为什么十二轮优化仍难突破

过去多个版本反复改变候选生成、Reviewer、融合、结构表示和门控，但多数变化仍在改善候选“看起来是否合理”，没有获得新的、可部署的 food-conditioned action utility 信号。由于正式收益由 Reviewer 的相对选择决定，供给侧表示继续增强只能产生有限或不稳定收益。这不是单个 prompt 的偶然失败，而是信息与目标错配。

## 5. 文献支撑及其边界

1. **F1 不能由任意代理替代。** Zhang et al.（ICML 2020）证明需要对多标签 F-measure 校准的 surrogate；Bao & Sugiyama（AISTATS 2020）同样强调 F-measure/Jaccard 这类非可分指标需要校准效用，而不是普通逐样本回归。因此，结构相似度或结构 F1 代理可学习，并不推出正式 MPC F1 可改善。
   - <https://proceedings.mlr.press/v119/zhang20w.html>
   - <https://proceedings.mlr.press/v108/bao20a.html>

2. **候选选择应按查询建模。** Rudin & Wang（AISTATS 2018）指出凸代理可能造成较差的排序近似，并研究直接 rank/rerank；Tewari & Chaudhuri（ICML 2015）把泛化问题定义在 query-level subset ranking。它们支持将一个查询的动作集合视为一个训练单元，而不是把所有动作混在一起做点式回归。
   - <https://proceedings.mlr.press/v84/rudin18a.html>
   - <https://proceedings.mlr.press/v37/tewari15.html>

3. **高召回必须配合独立验证，但验证信号必须有效。** Recall-then-Verify（ACL 2022）支持先扩大候选召回，再逐候选验证；本轮 Bank Oracle 也证明 recall 有价值。但当前 verifier 的正式正收益 ROC-AUC 只有 0.441，因此不能仅凭该框架名称假设验证器有效。
   - <https://aclanthology.org/2022.acl-long.128/>

4. **允许保留 no-op/拒绝动作。** Post-hoc learning-to-defer（NeurIPS 2022）支持在基础系统之上单独学习何时采用替代决策。对 MPC，这对应 Reviewer 只有在预测增益可靠为正时才改写 H1；它不能修复弱 utility signal，但能减少 selector's curse。
   - <https://proceedings.neurips.cc/paper_files/paper/2022/hash/bc8f76d9caadd48f77025b1c889d2e2d-Abstract-Conference.html>

上述论文提供的是方法设计原则，不是对 FoodPuzzle 提升的保证。尤其不能用“换成 listwise 网络”掩盖输入中缺乏 food-conditioned demand 的事实。

## 6. 下一步优化方向与取舍

### 6.1 不进入正式代码的内容

- 当前 Ridge structure-utility verifier；
- 结构 posterior 直接 Bank→Top-5 重排；
- 继续添加 UniMol/多构象作为 MPC 主信号；
- 更深的 pointwise 模型；
- 在同一 71 个正式样本上继续调门控阈值。

### 6.2 唯一仍有充分依据的下一次审计

设计一个**查询条件化、评价目标对齐的反事实 Reviewer**，但先做离线可行性审计，不直接进入正式版本：

1. **训练单元改为 query action list。** 每个训练查询的 no-op、add、remove、swap 候选组成一个列表；只比较同一查询内的动作。
2. **目标改为训练侧真实集合效用。** 仅使用训练查询 Gold，按与正式任务一致的冻结集合语义构造 `ΔF1` 或 pairwise preference；外层 held-out Gold 永不进入特征、标签构造或阈值选择。
3. **显式建模 demand–supply interaction。** query 表示必须包含 partial profile、目标食物 evidence/retrieval context；action 表示包含 proposed set diff；模型学习二者交互，而不是只输入候选结构汇总量。
4. **使用 query 内 pairwise/listwise 目标。** 首先判断 action 是否优于 no-op，再在正候选中排序；避免整体 MSE 被大量平局和查询间尺度差异主导。
5. **保留 no-op，并严格交叉拟合置信门。** 只有预测增益的内层校准下界为正才采用 Reviewer 修改，否则保持 Scientist H1。
6. **UniMol在 MPC 中只作为可选 supply-side 消融。** 必须先证明没有 UniMol 的 demand–action verifier 能工作，再测试结构表示是否带来额外增益；不能让 UniMol承担其无法提供的食物需求信息。

这条路线的创新点不是“再加一个模型”，而是把 Reviewer 从文本合理性裁判改为**评价校准、查询条件化、可拒绝的反事实集合决策器**。该设计仍保持 Scientist–Reviewer 主结构，也适用于一般 set-completion 任务，而非针对 FoodPuzzle 样本 ID 或特定规则调参。

## 7. 预注册停止条件

在任何正式 API 运行前，下一审计应预注册：

- 严格 nested OOF；
- 与 no-op 的 query 内排序 AUC/accuracy；
- Top-5/Top-20 的 query-level regret 与 Oracle capture；
- 五折收益方向和 bootstrap 下界；
- 不允许在正式 71 样本上选择模型、阈值或 feature set；
- 若 query 内正收益识别仍不高于随机，或平均收益仍未显著超过现有合法约 `+0.0075`，停止离线 Reviewer 堆叠，承认在现有离线 evidence 下 demand 不可辨识，不再继续迭代版本号。

## 8. 当前决策

- 本轮 verifier **失败，不进入正式代码**；
- 真正瓶颈已定位为 `food-conditioned action utility identification`，不是 UniMol 或结构表示不足；
- 下一步只审计 query-conditioned、evaluation-aligned、pairwise/listwise 且可拒绝的 Reviewer；
- 在该审计通过前，不再新增正式 Optimized-Agent 版本，不调用 API，不覆盖正式结果。
