# MPC v14 单瓶颈与文献审查

## 审查目标

本轮严格遵循“一次只解决最大瓶颈”的迭代逻辑：

1. 定量定位目前损失最大的阶段；
2. 判断该阶段是否存在合法、独立且可泛化的观测信号；
3. 只设计一个最小优化动作；
4. 在离线 OOF 收益不明显时立即停止，不进入 API 实验。

本轮未修改代码、未调用 API、未运行正式任务。

## 1. 最大瓶颈在哪里

### 1.1 收益损失链

| 阶段 | OOF 宏平均官能团 F1 增益 |
|---|---:|
| v13 广义单交换 Oracle | +0.0582518 |
| v14 Action Bank Oracle | +0.0442116 |
| v14 Scientist Oracle@5 | +0.03539127 |
| v14 交叉拟合执行器 | +0.00794142 |

从 Action Bank 到 Top-5 损失 `0.00882033`；从 Top-5 到实际执行损失 `0.02744985`。后者是前者的约 3.11 倍，也占 Action Bank Oracle 总空间的 62.09%。

**因此当前最大瓶颈是 Top-5 内的动作判断，不是继续扩大候选召回。**

### 1.2 568 个完整谱聚类 OOF 查询的阶段分解

| 类型 | 查询数 |
|---|---:|
| Action Bank 中没有任何正收益动作 | 306 |
| Bank 有正动作，但 Top-5 漏掉 | 12 |
| Top-5 有正动作，但 Top-1 为非正 | 88 |
| Top-1 为正，但不是 Top-5 中最优 | 44 |
| Top-1 就是 Top-5 Oracle | 118 |

Bank 可改进的 262 个查询中，只有 12 个属于 Top-5 召回问题；132 个属于 Top-5 已经包含有用动作、但顺序或门控判断错误。

Top-5 内候选动作的成对次序检查中，3,059 个可比较动作对有 953 个次序错误，错序率为 31.15%。当前期望 F1 分数适合召回，但不足以担任精确 verifier。

## 2. 59 个负收益动作的共性

以正式脚本相同的 `PYTHONHASHSEED=0` 重算，交叉拟合门控选中 93 个正动作、59 个负动作和 4 个平局。

### 2.1 负动作

| 特征 | 数量 | 占 59 个负动作的比例 |
|---|---:|---:|
| 没有增加任何 Gold 官能团 | 43 | 72.88% |
| 增加了非 Gold 官能团 | 35 | 59.32% |
| 删除了 Gold 官能团 | 34 | 57.63% |
| 同时错加且破坏性删除 | 10 | 16.95% |
| 纯粹“加入 Gold 且不删 Gold” | 0 | 0% |

### 2.2 正动作

| 特征 | 数量 | 占 93 个正动作的比例 |
|---|---:|---:|
| 增加至少一个 Gold 官能团 | 82 | 88.17% |
| 纯粹“加入 Gold 且不删 Gold” | 60 | 64.52% |
| 删除 Gold 官能团 | 2 | 2.15% |

这个分解显示，单一最大的可解释错误不是“对整个方案评分不准”，而是没有显式回答两个不对称的反事实问题：

1. **Add necessity**：新增分子是否带来当前食物确实需要、且 H1 尚未充分覆盖的官能团？
2. **Remove safety**：删除分子会不会丢失当前 Gold 中不可被其他 H1 分子补偿的官能团？

当前 expected-F1 对完整 proposal 作一个标量评分，将“加的价值”和“删的危害”压缩在一起，导致两类证据相互抵消。

## 3. 官方离线 evidence 能否直接解决

568 个训练查询均有至少一段食物级文本，但关系与分子链接如下：

| 指标 | 结果 |
|---|---:|
| 有至少一个直接分子 occurrence 链接的查询 | 110 / 568（19.37%） |
| 直接链接分子总次数 | 175 |
| occurrence support 片段 | 490 |
| sensory replication support 片段 | 429 |
| functional role support 片段 | 150 |
| contradiction 片段 | 5 |
| ambiguous 片段 | 4,604 |

因此，官方 evidence 可以作为高精度的可选信号，但无法作为覆盖所有 Top-5 动作的主验证器。如果只让 LLM Reviewer 重读这些 evidence，大部分查询仍然没有与 add/remove 动作直接相关的外部反馈。

## 4. 文献支撑与适用边界

### FoodPuzzle 原文

FoodPuzzle 的 MPC 评测是对预测分子集和 Gold 分子集分别抽取官能团后计算 F1，所以动作的 add/remove 官能团账本与任务目标一致。原文同时报告了搜索空间初始化错误 32%、认识性幻觉 26%和错误解读来源 20%，并指出 Reviewer 只是从 Scientist 的三个假设中选择或拒绝。这支持保留 Scientist–Reviewer 逻辑，但不能证明“使用相同证据再读一次”会带来改善。

原文：<https://arxiv.org/html/2409.12832>

### Recall-then-Verify（ACL 2022）

该工作将多答案召回与每个候选的独立验证分开，避免多个候选在一次联合生成中相互干扰。这与“对 Top-5 的每个 atomic action 单独验证”直接对应。

<https://aclanthology.org/2022.acl-long.128/>

### Double Retrieval and Ranking（Findings of EACL 2023）

该工作指出，只按原始问题相关性检索支持证据是次优的；第二阶段应使用“问题 + 待验证答案”作为查询。对 MPC 的启示是：验证证据必须以 `(food, partial profile, add molecule, remove molecule)` 为条件，而不能只使用 food-level evidence。

<https://aclanthology.org/2023.findings-eacl.130/>

### LEVER（ICML 2023）

LEVER 的 verifier 不只看生成程序本身，还利用执行结果这个外部信号，再与生成概率融合排序。对 MPC 的边界是：我们没有真实化学实验结果，因此不能宣称拥有等价的 execution verifier；但可以使用不泄漏的 OOF 食物–分子条件统计和官能团增删账本作为可观测反馈。

<https://proceedings.mlr.press/v202/ni23b.html>

### SelectiveNet（ICML 2019）

SelectiveNet 将预测和拒绝决策联合优化，并明确区分风险与覆盖率。对 MPC 的启示不是直接搬用深网络，而是将 `KEEP_H1` 视为合法 abstain 动作，在验证信号不足时不强制交换。

<https://proceedings.mlr.press/v97/geifman19a.html>

### LLM 内在自我修正与 CRITIC（ICLR 2024）

Huang et al. 发现在没有外部反馈时，LLM 的内在自我修正可能无效甚至降低推理性能；CRITIC 的改善则依赖工具交互式外部反馈。这两项结果共同否定了“只增加一层相同模型、相同证据的 Reviewer”作为当前主优化动作。

- <https://openreview.net/forum?id=IkmD3fKBPQ>
- <https://openreview.net/forum?id=Sx038qxjek>

## 5. 第一性原理结论

一个交换动作的实际价值是两项反事实效用的差：

`Action utility = utility(add | food, partial, H1) - harm(remove | food, partial, H1)`

当前 v14 的 Scientist 分数是一个整体 proposal expected-F1，适合生成高召回 Top-5；它不能同时作为自己的 verifier。要继续改善，verifier 必须：

- 以每个 atomic action 为单位；
- 分开预测 add necessity 和 remove safety；
- 使用与 Scientist 整体期望 F1 不同的动作特异信号；
- 证据不足时 abstain/KEEP_H1；
- 不能生成 Bank 外分子；
- 不使用 UniMol，因为本瓶颈是食物条件下的动作真值判断，不是分子结构表征。

## 6. 唯一建议的下一优化动作

### v14 Phase 2-A：低容量反事实双门验证器

本阶段先不调用 LLM Reviewer。只在 Top-5 内对每个动作建立两个低容量 OOF 模型：

1. `P(add_is_necessary | query, action-specific support)`；
2. `P(remove_is_safe | query, remaining-set coverage)`。

特征只允许使用通用信号：

- add/remove 官能团增删账本；
- 删除后剩余 H1 对被删官能团的覆盖度；
- 由邻近训练谱交叉拟合得到的 add 支持和 remove 风险；
- 官方 evidence 中与具体 add/remove 分子直接链接的 relation，缺失时显式标记 missing；
- Scientist 分数只作为一个 prior，不允许它独立通过验证。

Executor 只在两个门均通过且交叉拟合风险门槛通过时执行一次交换，否则 `KEEP_H1`。

### 为什么先不用 LLM Reviewer

当前需要先证明“现有可观测信号能否分离 93 个正动作和 59 个负动作”。如果低容量 verifier 在严格 OOF 下都不能降低负动作，那么使用同一 evidence 的 LLM Reviewer 更缺乏可证明的改善来源。

### 离线准入标准

Phase 2-A 必须在相同的完整谱聚类五折 OOF 下同时满足：

1. 最终宏 F1 增益显著高于 v14 Phase 1 的 `+0.00794142`；
2. bootstrap 95% 下界大于零；
3. 负收益动作数显著少于 59；
4. 不通过事后放宽阈值获得收益；
5. 固定 `PYTHONHASHSEED=0`，重复运行得到一致结果；
6. 未通过时保持 H1，不调用 API，不覆盖正式结果。

## 7. 当前路线决策

- 不继续扩大 Action Bank。
- 不增加 UniMol 或多构象 UniMol。
- 不直接新增一个读相同上下文的 LLM Reviewer。
- 不同时修改 Scientist、Reviewer、Action Bank 和 Executor。
- 只验证“add necessity + remove safety”这一个单瓶颈解法。

