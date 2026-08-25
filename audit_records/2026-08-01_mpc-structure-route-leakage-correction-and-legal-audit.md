# MPC 结构路线标签泄漏更正与无泄漏重算

## 1. 更正结论

在对 Bank→Top-5 多目标重排序进行第一性原理复核时，发现前两轮结构执行器审计存在目标泄漏：

- 对 held-out 查询的候选 proposal 计算结构官能团集合；
- 随后直接与该查询真实 missing molecules 的 Gold 官能团集合计算 `ΔF1`；
- 再使用该 `ΔF1` 选择 held-out 查询动作。

推理时不知道 missing molecules，也就不知道 Gold 官能团集合。完整谱聚类 OOF 只能保证候选模型没见过 held-out 行，不能让 held-out 标签成为合法特征。因此此前 `+0.0169` 的执行器增益属于 Oracle 辅助选择，不是可部署方法结果。

本轮主动撤销：

- “冻结结构执行器通过准入”；
- “结构执行器只有 16 个负动作”；
- “结构执行器捕获 59%–74% Top-5 Oracle”；
- 直接根据该 Gold-conditioned structure gain 设计 Top-5 重排序。

原审计文件没有删除，而是在文件顶部加入醒目的撤销说明，以保留完整研究轨迹。

## 2. 合法结构信号的定义

无泄漏版本对每个 held-out 查询只使用其他四折：

1. 使用其他折食物谱建立检索索引；
2. 检索与当前 partial profile 相近的训练谱；
3. 将这些训练谱的 missing molecules 通过冻结 SMILES/SMARTS 转为结构官能团集合；
4. 根据检索权重估计 `P(functional group, gold group cardinality | query)`；
5. 对 H1 和 proposal 计算 expected structural F1；
6. 以 expected-F1 差作为合法动作分数；
7. Gold 和 evaluation cache 只在动作冻结后评测。

该流程在推理时所需信息全部可获得，也不读取 evaluation cache、held-out missing molecules 或 UniMol。

## 3. 无泄漏动作级结果

| 指标 | Gold-conditioned Oracle 结构分数 | 合法训练折 posterior 分数 |
|---|---:|---:|
| 与 formal-audit action gain 相关 | 0.622344 | **0.144400** |
| ROC-AUC | 0.723759 | **0.511987** |
| PR-AUC | 0.628692 | **0.426651** |
| 正分动作 precision | 70.31% | **51.20%** |

合法分数的 ROC-AUC 接近随机排序。它仍能通过“是否大于零”过滤一部分动作，但几乎不能在多个候选之间可靠排序。

这说明：

- SMARTS 可以可靠描述候选分子“含有什么”；
- 但训练邻居 posterior 不能可靠回答当前食物“需要什么”；
- 前者不能替代后者。

UniMol 同样只增强分子表示，不能凭空提供食物条件下的缺失真值，因此这个结果再次说明不应把 UniMol放回 MPC 主链。

## 4. 无泄漏查询级结果

在相同 510 个可重构正式口径的查询上：

| 方法 | 平均 F1 增益 | bootstrap 95% 下界 | 正/负/平 |
|---|---:|---:|---:|
| 合法结构 posterior，在原 Top-5 中执行 | **+0.00750665** | +0.00421132 | 89 / 51 / 370 |
| 合法结构 posterior，直接在 Top-20 中执行 | **+0.00654752** | +0.00302481 | 91 / 71 / 348 |

扩大可选择集合反而增加负动作并降低平均收益。这是典型的 selector’s curse：当评分噪声很大时，从更多候选中取最大值会放大估计误差，而不是接近 Oracle。

合法结果与此前 v14 DB 代理执行器的约 `+0.00794` 同一量级，没有形成新的显著突破。因此当前结构 posterior 不能进入正式 Optimized-Agent。

### 折级增益

| 折 | Top-5 合法执行器 | Top-20 合法执行器 |
|---|---:|---:|
| 0 | +0.00611508 | +0.00611508 |
| 1 | +0.01337156 | +0.00667410 |
| 2 | +0.00251958 | +0.00321986 |
| 3 | +0.00740248 | +0.00932706 |
| 4 | +0.00815591 | +0.00772122 |

虽然五折均为正，但收益小，Top-20 也没有稳定优于 Top-5。

## 5. 合法 Bank→Top-5 重排序结果

将 Top-20 按合法 expected structural F1 重排后取 Top-5，再使用 cache 仅计算事后 Oracle：

| Top-5 构造 | Top-5 Oracle |
|---|---:|
| 原 Scientist Top-5 | +0.02813141 |
| 合法结构 posterior 重排 Top-5 | **+0.03126907** |
| 完整 Top-20 Bank Oracle | +0.04594515 |

合法重排只增加 `+0.00313766`，捕获 Bank Oracle 的 68.06%，没有达到预注册的 `>= +0.038` / 82.7% 准入线。

因此当前 Bank→Top-5 结构重排序路线正式判定：**失败，不进入代码实现。**

## 6. 哪些结论仍然有效

以下结果不依赖泄漏，继续成立：

1. 正式评测存在 `LLM-FG(prediction) vs DB-FG(gold)` 的不对称目标；
2. DB 对称代理与正式动作收益相关性较低；
3. 冻结结构官能团与 cache 的平均分子级 Jaccard 高于 DB 字符串解析；
4. Top-20 Bank 的事后正式 Oracle 约为 `+0.04595`；
5. 原 Scientist Top-5 的事后正式 Oracle 约为 `+0.02813`；
6. 官方离线 evidence 对具体 add molecule 的直接覆盖不足；
7. 结构表示本身不是 MPC 最终瓶颈，食物条件下的 action utility 才是。

Oracle 只能说明候选集合中存在好动作，不能说明存在合法可部署的信息去识别它们。

## 7. 第一性原理解释

一个 MPC 动作需要估计：

`P(required functional groups | target food, partial profile, evidence)`

SMILES/SMARTS 只能确定：

`functional groups(candidate molecule)`

动作效用同时依赖需求与供给。我们改善了“供给”测量，却没有获得更准确的“需求”信息。直接使用 Gold 会让需求看似已知，从而制造虚假收益；移除 Gold 后，合法 posterior 的动作相关性立刻降到 0.144。

这解释了为什么不断强化 UniMol、结构相似度或候选官能团仍无法突破 MPC：它们都主要描述候选分子，而不是目标食物缺失分子的条件分布。

## 8. 文献支撑与适用边界

### 标签泄漏

Kaufman et al. 对 data leakage 的核心定义是：模型使用了预测时不可获得的信息，导致离线估计过于乐观。held-out Gold 即使没有用于拟合候选模型，只要进入当前样本的动作选择，仍然属于直接目标泄漏。

Kaufman et al., *Leakage in Data Mining: Formulation, Detection, and Avoidance*, ACM TKDD 2012：

<https://doi.org/10.1145/2382577.2382579>

### F-measure 需要校准代理

Waegeman et al. 证明随意使用 Hamming/subset 等代理优化 F-measure 可能产生很高 regret；ICML 2020 的 Zhang et al. 进一步构造了对多标签 F-measure 校准的 surrogate。这支持下一步必须学习一个与集合 F1 有明确关系的训练侧动作目标，而不能把分子结构相似度或未校准 expected score直接当作 F1。

- <https://www.jmlr.org/papers/v15/waegeman14a.html>
- <https://proceedings.mlr.press/v119/zhang20w.html>

### 多样性只解决召回，不解决效用识别

MMR 和 DPP 的文献支持在有限集合中兼顾质量与多样性；Recall-then-Verify 支持先高召回再独立验证。但这些方法都要求存在有意义的 relevance/quality 或 verifier 信号。当前合法结构 score 的 ROC-AUC 只有 0.512，因此仅增加 diversity 不能解决 verifier 缺乏信息的问题。

- MMR：<https://aclanthology.org/X98-1025/>
- Structured DPP：<https://proceedings.neurips.cc/paper/2010/hash/1f50893f80d6830d62765ffad7721742-Abstract.html>
- Recall-then-Verify：<https://aclanthology.org/2022.acl-long.128/>

## 9. 下一条仍然合法的研究路线

如果继续 Scientist–Reviewer 主线，下一步不能直接按 held-out Gold 打分，而应建立训练侧监督的反事实动作模型：

1. 对训练查询生成 OOF Action Bank；
2. 只对“训练标签”使用该训练查询真实 missing molecules；
3. 用冻结结构官能团为训练动作定义 add/remove/集合 F1 utility 标签；
4. 训练低容量 action verifier，输入只能是推理可得的 query、retrieval、structure 和 add/remove 特征；
5. 对外层 held-out 查询只输出预测 utility，绝不能读取其 Gold；
6. formal cache 只在所有预测冻结后一次性评价；
7. 目标应直接面向集合级 F1 或使用有理论校准关系的 surrogate，而不是分别拟合互不协调的局部标签；
8. 如果该严格 stacked OOF verifier 仍不能达到显著收益，则在“不联网、不使用评测器反馈”的约束下，应承认 action utility 不可辨识上限。

这与 v15 不完全相同：v15 的训练标签来自错位的 DB 字符串代理；新路线若开展，训练标签必须来自冻结结构本体，并在 held-out 推理中只使用模型预测。

## 10. 当前决策

- 撤销此前结构执行器准入；
- 不实现结构 Bank→Top-5 直接重排序；
- 不调用正式 API；
- 不修改现有 Agent；
- 不重新调 SMARTS；
- 保留结构官能团层作为训练标签构造工具，而不是 held-out Oracle；
- 下一步是否审计“训练侧结构 F1 监督的 stacked OOF action verifier”，必须由用户另行批准。

