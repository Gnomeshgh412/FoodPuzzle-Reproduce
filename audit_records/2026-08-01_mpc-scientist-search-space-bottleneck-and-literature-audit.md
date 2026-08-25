# MPC Scientist 搜索空间瓶颈与文献审查

## 1. 审查目标

本轮遵循新的优化顺序：在 Scientist 的候选召回获得显著提升以前，不继续优化 Reviewer。

需要回答的不是“把 Top-20 再排好一点是否有收益”，而是：

1. 当前名义 Top-20 是否真的包含 20 个有效、不同的动作；
2. 收益损失来自 Top-K 截断，还是动作空间本身过窄；
3. `remove` 只允许 H1 末尾 12 项是否合理；
4. `add` 只查看边界后 40 项是否足够；
5. 一步交换是否构成 Scientist 的主要上限；
6. 哪种改动同时具有数据证据、第一性原理和文献依据。

本轮未修改正式代码、未调用 API、未覆盖结果。

## 2. 对此前表述的更正

Top-20 事后正式 Oracle `+0.0459` 不能证明 Scientist “经常得到正确结果”。在此前 507 个可重构查询中，只有 187 个查询存在改善候选，占 36.9%；其余 63.1% 没有改善。

本轮尝试进一步用现有 DeepSeek functional-group cache 重构全训练集的正式口径搜索空间，但只有 47/568 个查询的整个 H1 都已有缓存映射，519 个查询缺少至少一个 H1 分子的 LLM 映射。47 条是严重选择偏置子样本，其数值不进入本审查结论，也不用于选方法。

因此下面的全量搜索诊断采用训练侧对称 FlavorDB 官能团 F1。它适合回答“化学集合搜索空间是否受限”，但不能冒充官方 `LLM-FG(prediction) vs DB-FG(gold)` 正式指标。

## 3. 无泄漏诊断协议

- MPC train 568 条；
- 依据完整 profile signature 聚类后做 5 折 OOF；
- 折大小：114 / 114 / 114 / 113 / 113；
- 每折只用其余四折拟合 H1、检索和 group-cardinality posterior；
- held-out Gold 只在所有候选冻结后计算 Oracle；
- 不读取测试标签；
- 不调用 API；
- MPC 不使用 UniMol；
- 指标：对称 FlavorDB functional-group set macro-F1。

## 4. 当前 Scientist 的真实有效宽度

当前 `_build_v14_action_bank`：

- 只允许删除 H1 最后最多 12 个分子；
- 只允许加入 occurrence 边界后的前 40 个分子；
- 只生成一次 swap；
- 按 FlavorDB 官能团集合签名去重；
- 最多保留 20 个动作。

实际结果：

| 指标 | 结果 |
|---|---:|
| 名义 Bank 上限 | 20 |
| 平均实际动作数 | **7.7835** |
| 平均唯一官能团 proposal 数 | **7.7835** |

因此“Top-20”多数情况下并不是 20 个候选。去重本身没有错；问题是受限的 remove/add 组合只能形成平均约 7.8 个不同的终态。

## 5. K 曲线：当前 Bank 内排序是不是主问题

基线对称 FG-F1 为 `0.898085`。

| K | Oracle 增益 | 至少有一个正动作的查询 | 覆盖率 |
|---:|---:|---:|---:|
| 1 | +0.021886 | 162 | 28.52% |
| 3 | +0.027115 | 206 | 36.27% |
| 5 | +0.035391 | 250 | 44.01% |
| 10 | +0.041248 | 260 | 45.77% |
| 20 | **+0.044212** | **262** | **46.13%** |

当前受限原始空间（末尾 12 remove × 边界 40 add）的一步 Oracle 为 `+0.045025`，264 个查询可改善。Bank20 已捕获：

- `0.044212 / 0.045025 = 98.19%` 的该空间 Oracle；
- 262/264 个可改善查询；
- Top-5 又捕获 250/262 个 Bank 可改善查询。

结论：**在当前受限动作空间内部，Bank20 排序和截断已经不是主要 Scientist 瓶颈。**继续调 expected-F1 的 Top-20 排序，最大剩余空间只有约 `+0.00081`。

## 6. 放宽不同搜索约束后的 Oracle

| 搜索空间 | Oracle 增益 | 可改善查询 | 覆盖率 |
|---|---:|---:|---:|
| 当前：末尾 12 remove × 40 add，1 步 | +0.045025 | 264 | 46.48% |
| 全 H1 remove × 40 add，1 步 | **+0.052570** | **311** | **54.75%** |
| 全 H1 remove × 100 add，1 步 | **+0.058661** | **324** | **57.04%** |
| 全 H1 remove × 40 add，greedy 2 步 | **+0.063963** | 311 | 54.75% |
| 全 H1 remove × 40 add，greedy 3 步 | **+0.067003** | 311 | 54.75% |

补充计数：

- 放开 remove 位置在 100 个查询上带来额外收益；
- add 边界从 40 扩至 100 在 80 个查询上进一步提高 Oracle；
- 当前原始空间有正动作而 Bank20 完全漏掉的查询只有 2 个；
- Bank20 有正动作而 Top-5 漏掉的查询只有 12 个。

### 边际收益分解

1. 全 H1 remove：`+0.007545`；
2. 40→100 add：再增加 `+0.006090`；
3. 1→2 步：在全 H1×40 下增加 `+0.011393`；
4. 2→3 步：仅再增加 `+0.003040`。

两步搜索的额外空间最大，而第三步已经明显递减。当前证据支持深度 2，不支持继续无界增加编辑预算。

这里的 greedy 2/3 步只有在第一步已经正收益时继续，因此没有增加“存在正候选”的查询数，只提高了这些查询的可达改善幅度。它没有检验“先小幅下降、第二步才总体改善”的非贪心路径；后者搜索空间更大、风险也更高，当前不应直接加入。

## 7. 真正的 Scientist 瓶颈

当前最大瓶颈不是 Top-20 内排序，而是三个相互关联的搜索约束：

1. **删除位置偏置**：默认只有 H1 末尾 12 项可删除，但集合 F1 的错误成员不一定处于 occurrence 排名末端；
2. **加入候选边界过窄**：前 40 项之外仍有互补官能团候选；
3. **单步局部性**：一个 swap 无法纠正 H1 中多个协同错误，且每一步后集合覆盖与冗余都会变化。

第一性原理上，MPC Scientist 应搜索：

`Y = argmax_{|Y|=n} Utility(Y | food, partial profile, evidence)`

而当前实现近似搜索：

`Y = H1 - one_of(last_12) + one_of(next_40)`

后者只是前者极小的局部邻域。平均只有 7.8 个有效动作正是该约束的直接表现。

这与 FoodPuzzle 原文错误分析中的“Inappropriate Initialization of Search Space（32%）”一致：论文也发现错误搜索空间初始化是最大单项错误来源，而不是只归因于最终 Reviewer。

原文：<https://arxiv.org/html/2409.12832>

## 8. 文献支撑与适用边界

### 8.1 Multiple Choice Learning：Scientist 应优化 best-of-K

Guzmán-Rivera et al.（NeurIPS 2012）把多假设生成直接建模为 multiple structured outputs，并以“后续组件能够从多个候选中取得最好结果”的损失训练，而不是先训练单一 MAP 模型后取相似的 M-best。

对 MPC 的对应关系是：Scientist 的训练目标应是降低 `min_k loss(Y_k, Gold)`，即提高 Oracle@K 和正候选覆盖，而不是让 20 个候选都追随同一个 expected-F1 排名。

<https://proceedings.neurips.cc/paper_files/paper/2012/hash/cfbce4c1d7c425baf21d6b6f2babe6be-Abstract.html>

### 8.2 Learning to Search：多步动作必须看到状态变化和 rollout loss

Chang et al.（ICML 2015）的 LOLS 将结构化预测定义为状态、动作、转移和终态损失，并用 rollout 后的 task loss 给不同动作构造 cost-sensitive 监督。

对 MPC 的对应关系是：第二次 swap 不能复用第一步的静态分数；必须在更新后的集合状态上重新计算覆盖、冗余、add/remove 边际和终态 F1 训练代价。

<https://proceedings.mlr.press/v37/changb15.html>

该论文不保证本项目的小数据一定学好，也不允许测试 Gold 参与 rollout。MPC 只能在训练查询内构造 cost-to-go，并做严格 nested OOF。

### 8.3 Direct Loss Minimization：不能继续用无关代理训练搜索

Hazan et al.（NeurIPS 2010）强调结构化预测应围绕任务损失进行 loss-adjusted inference，而不是假定普通 surrogate 必然对应最终结构指标。

对 MPC 的对应关系是：训练侧 Scientist 应使用冻结的集合 F1/Jaccard 终态损失生成 cost-augmented actions；结构相似度、occurrence 或 retrieval 只能作为特征和 proposal prior。

<https://proceedings.neurips.cc/paper/2010/hash/ca8155f4d27f205953f9d3d7974bdd70-Abstract.html>

### 8.4 多样化结构输出：候选应覆盖不同错误模式

Prasad et al.（NeurIPS 2014）研究在指数级结构空间中寻找兼顾质量与多样性的候选子集；Diverse Beam Search（AAAI 2018）也说明普通 beam 容易产生轻微变体，而分组多样性约束可以覆盖不同模式。

- <https://proceedings.neurips.cc/paper_files/paper/2014/hash/8fcd0c8d0e4335895172454b51bcc506-Abstract.html>
- <https://ojs.aaai.org/index.php/AAAI/article/view/12340>

对 MPC 可采用“added groups、removed groups、edit path”上的质量–多样性选择，但不能照搬它们的近似保证。MPC 的 F1 含有 false-positive 代价和分母变化，并非单调次模效用；remove/swap 也使状态非单调。因此多样性只能作为固定预算下提高覆盖的机制，不能替代任务损失审计。

## 9. 下一版 Scientist：受控两步多选择搜索

建议先实现和审计一个完全离线的 **Cost-Augmented Diverse Two-Step Scientist**，不接 Reviewer。

### 9.1 固定 H1 锚点

- 保留 H1 作为 `NO_CHANGE`；
- 所有候选均从 H1 出发；
- 每个终态必须 exact-N、无重复、不得包含 partial molecules。

### 9.2 扩展但不无界的原子动作空间

- remove 覆盖全部 H1，而非最后 12 项；
- add pool 固定到训练侧多来源候选的前 100，并保留来源标签；
- 来源包括 occurrence、IDF/containment retrieval 和少量 direct evidence；
- UniMol不进入 MPC 候选生成或排序。

为控制计算量，不按单一总分过早裁剪，而按动作类型分层保留：

- boundary removal；
- low-support removal；
- functionally redundant removal；
- retrieval-supported add；
- occurrence-supported add；
- group-complement add。

这是一种通用的 stratified proposal policy，不使用样本 ID 或测试桶规则。

### 9.3 深度固定为 2 的状态化搜索

第一步产生若干互补 beam state；第二步在每个新状态上重新计算：

- 当前官能团覆盖；
- 被删除分子的独占组；
- 新增组和额外组；
- candidate support；
- predicted cardinality-conditioned expected F1。

不复用初始 H1 的静态 action score。深度 3 暂不进入主方法，因为本轮 Oracle 显示其额外空间只有约 `+0.0030`。

### 9.4 Scientist 的目标改为 best-of-K

训练查询中，使用 Gold 只构造训练动作的终态集合损失；外层 held-out Gold 永远不可见。

固定输出：

- 内部 Action Bank：20 个终态；
- Scientist Slate：从 Bank 中选 5 个质量–多样性候选；
- 一个显式 `KEEP_H1`。

候选选择优化：

`best-of-K task loss + diversity regularization`

多样性定义只使用通用的：

- added-group Jaccard；
- removed-group Jaccard；
- edit-path overlap；
- proposal-set Jaccard。

### 9.5 暂不赋予执行权

新 Scientist 只生成候选，不直接修改正式输出。只有 Scientist 离线准入后，才冻结候选并进入 Reviewer 研究。

## 10. 预注册准入条件

使用同一 exact-profile clustered nested OOF，至少报告：

1. 当前 Bank20 与新 Bank20 的 paired Oracle 差；
2. 正候选 query coverage；
3. Bank20 与 Scientist Slate5 的 Oracle；
4. 每折增益；
5. 有效唯一终态数；
6. 一步/两步、末尾12/全remove、add40/add100、多样性开关消融；
7. near-duplicate profile stress test。

准入条件预先固定为：

- 新 Bank20 相对当前 Bank20 的 paired bootstrap 95% 下界大于 0；
- 新 Bank20 捕获至少 85% 的“全 H1×40 add、两步 greedy”Oracle；
- 新 Bank20 覆盖至少 90% 的该扩展空间正查询；
- Scientist Slate5 捕获至少 80% 的新 Bank20 Oracle；
- 五折相对当前 Bank20 不出现系统性负方向；
- 所有候选满足 exact-N 和无泄漏约束。

按本轮上限换算，参考目标约为：

- Bank20 Oracle 至少约 `0.85 × 0.063963 = +0.05437`；
- 正查询覆盖至少约 `0.90 × 311 = 280` 条；
- 这些只是训练 OOF 准入线，不是正式测试提升承诺。

如果固定 K 下不能显著超过当前 `+0.04421 / 262 queries`，说明扩展空间无法被合法 proposal policy 压缩到有限候选，停止该路线，不进入 Reviewer。

## 11. 当前结论

Scientist 的最大可操作瓶颈已定位为：

> **受限 remove/add 支持与单步局部搜索，使名义 Top-20 平均退化为约 7.8 个有效终态。**

当前 Bank 在自己的狭窄空间内已经接近穷尽，因此继续调 Top-20 排序不是主要方向。下一步应先做“全 H1 remove + 扩展 add pool + 深度 2 状态化搜索 + best-of-K 多样性目标”的离线 Scientist，并严格冻结 Reviewer。

本轮没有证明该方法一定提升正式 MPC；它证明的是该方向拥有比继续调 Bank 排序更大的合法训练侧 Oracle 空间，并且与原论文的 search-space initialization 错误、结构化学习搜索和多假设输出文献相符。

## 12. 冻结实现与 OOF 准入结果

在获得用户批准后，实现了一个**尚未接入正式推理**的实验方法
`_build_v16_scientist_bank`。本轮只改 Scientist 候选生成，MFP、MPC
Reviewer、正式输出路径和 MPC 无 UniMol 策略均保持不变；没有调用 API，
也没有读写正式测试结果。

固定实现为：

- 全 H1 可删除；
- 100 个训练侧多来源 addition；
- 第一步 quality-diverse beam 为 8；
- 深度为 2，并在新状态重算功能团集合；
- 内部 Bank20、Scientist Slate5；
- 默认 H1 由控制器保留，Oracle 统计中的零增益选项等价于 `KEEP_H1`；
- Gold 仅在 held-out 候选全部冻结后用于计算事后 Oracle。

这里需要如实限定：当前实现是预注册思想的**确定性质量–多样性 proposal
policy**，还不是端到端学得的 Multiple Choice / best-of-K loss 模型。先用它
检验有限 Bank 能否压缩扩展搜索空间；若连候选准入都失败，则没有理由进入
Reviewer 或 API 阶段。

### 12.1 协议

- 数据：568 条 MPC train 查询；
- 划分：exact full-profile clustered 5-fold OOF，折大小
  `114 / 114 / 114 / 113 / 113`；
- 每折只在其余折拟合候选排序和功能团 posterior；
- 指标：FlavorDB 固有功能团的 symmetric set F1，仅作训练侧结构审计，
  不是论文正式 LLM 功能团映射分数；
- 参数在运行前冻结，没有查看部分折结果后调参；
- 初版实现因逐候选重建整个分子集合在 `n` 中位数 83、最大 351 时过慢；
  随后只做了语义等价的增量功能团计数优化，并从头重跑完整协议。

### 12.2 总体结果

| 指标 | 当前 v14/v15 Scientist | 新实验 Scientist |
|---|---:|---:|
| H1 平均结构 F1 | 0.89808496 | 0.89808496 |
| Bank20 Oracle 增益 | +0.04421160 | **+0.05250439** |
| 正收益查询 | 262 / 568 | **276 / 568** |
| Slate5 Oracle 增益 | +0.03539127 | **+0.04234027** |
| Slate5 正收益查询 | 未单独重算 | 244 / 568 |
| 新 Bank 相对旧 Bank 增益 | - | **+0.00829280** |
| paired bootstrap 5% lower bound | - | **+0.00360298** |

新 Bank 平均含 19.89 个唯一动作，Slate 固定为 5；平均原始唯一一步终态
34.67、两步终态 131.36。Bank 中一步动作共 4,458 个、两步动作共
6,840 个。所有候选均满足 exact-N 且无重复分子。

### 12.3 对预注册门槛的逐项判定

| 门槛 | 结果 | 判定 |
|---|---:|---|
| 新 Bank 对旧 Bank paired bootstrap 下界 > 0 | +0.00360298 | 通过 |
| 捕获 greedy-2 `+0.06396308` 的至少 85% | **82.0855%** | **失败** |
| 覆盖 311 个正查询中的至少 90%（至少 280） | **276 / 311 = 88.7460%** | **失败** |
| Slate5 捕获新 Bank 至少 80% | **80.6414%** | 通过 |
| exact-N、无重复、无泄漏 | 全部满足 | 通过 |

因此本轮严格判定为：**不准入正式 MPC，不进入 Reviewer，不调用 API。**

### 12.4 折级稳定性

| 折 | 当前 Bank Oracle | 新 Bank Oracle | 差值 | 新 Slate5 |
|---:|---:|---:|---:|---:|
| 0 | 0.01902381 | 0.03239202 | +0.01336821 | 0.02492726 |
| 1 | 0.07605667 | 0.06033986 | **-0.01571681** | 0.05299991 |
| 2 | 0.02437012 | 0.04537844 | +0.02100833 | 0.03822114 |
| 3 | 0.05985144 | 0.06877061 | +0.00891917 | 0.04959568 |
| 4 | 0.04187262 | 0.05581275 | +0.01394013 | 0.04605359 |

总体均值改善不能掩盖第 1 折的明显退化。这说明当前 posterior 驱动的
quality-diverse beam 在某类完整食物谱上会过早剪掉旧 Bank 的强候选；扩大
原始支持空间是有效的，但“用同一个有偏质量分数把巨大空间压回 20 个”仍是
瓶颈。

## 13. 本轮学到的真正结论

1. **搜索支持扩张有效，但幅度不足以单独解决 Scientist。** Bank Oracle
   增加约 `+0.00829`，证明全 remove、宽 add 和两步状态并非无效；但仍未达到
   事先规定的机制门槛。
2. **主要剩余损失发生在 beam 压缩，而不是 Slate5。** Slate5 已保留新 Bank
   的 80.64%，恰好通过；新 Bank 本身只保留扩展 greedy-2 上限的 82.09%。
3. **固定 expected-F1 排名仍会造成 selector bias。** 第 1 折退化说明训练侧
   posterior 在分布变化时会把错误的第一步送入 beam；第二步再强也无法恢复
   被第一步剪掉的路径。
4. **不能为了过线继续在同一 OOF 上调 beam/权重。** 这会把机制审计变成
   train-fold 调参并放大过拟合。下一轮必须提出新的、文献支持的 Scientist
   学习问题和新的冻结准入协议，再获得批准后实施。

截至本记录结束，正式方法版本仍为 `optimized_agent_v15`；实验 v16 builder
保留在代码中但明确禁用，正式结果目录未被覆盖。

## 14. v16 失败后的全链路瓶颈分解

上一节根据 Bank20 结果推测“第一步 beam 过早剪枝”是剩余瓶颈。
该推测只是从最终 Bank 反推，没有直接观察 beam 压缩前的两步终态。
因此本轮保持 v16 参数完全冻结，在同一 568 条 exact-profile
clustered 5-fold OOF 上增加如下反事实观测点：

1. 当前窄一步空间：末尾 12 remove × 40 add；
2. 全 H1 remove × 40 add；
3. 全 H1 remove × 100 add；
4. Gold 只在事后选第一步的 greedy-2 Oracle；
5. v16 固定 beam8 展开后、尚未压成 Bank20 的全部两步终态；
6. Bank20；
7. Slate5。

所有 Gold 都只在上述候选被冻结后计算事后 Oracle，不进入候选特征、
posterior、beam 或 Bank 选择。

### 14.1 各层可达空间

| 层级 | 平均 Oracle 增益 | 存在正候选的查询 |
|---|---:|---:|
| 末尾 12 remove × add40，一步 | +0.04502509 | 264 / 568 |
| 全 H1 remove × add40，一步 | +0.05257036 | 311 / 568 |
| 全 H1 remove × add100，一步 | +0.05866790 | 324 / 568 |
| Oracle-first greedy-2，add100 | **+0.07505259** | 324 / 568 |
| 固定 beam8 的未压缩两步终态 | **+0.07126342** | **320 / 568** |
| v16 Bank20 | +0.05250439 | 276 / 568 |
| v16 Slate5 | +0.04234027 | 244 / 568 |

这个分解否定了“第一步 beam 是当前最大瓶颈”的推测。固定 beam8
已保留 `320 / 324 = 98.77%` 的扩展空间正查询，平均 Oracle 也保留
`0.071263 / 0.075053 = 94.95%`。候选在进入 Bank20 前大部分仍然存在。

### 14.2 边际收益与 regret

| 动作/压缩层 | 平均影响 |
|---|---:|
| 末尾 12 remove → 全 H1 remove | **+0.00754527** |
| add40 → add100 | **+0.00609754** |
| 一步 → Oracle-first 两步 | **+0.01638469** |
| Oracle-first 与固定 beam 之间的正向 regret | 0.00387721 |
| 未压缩 beam 终态 → Bank20 的正向 regret | **0.01875902** |
| Bank20 → Slate5 的正向 regret | 0.01016412 |

`Bank compression regret / first-beam regret ≈ 4.84`。因此当前最大、最先需要
处理的 Scientist 瓶颈是：

> **约 131 个有效两步终态被压缩成 20 个候选时，当前“单一
> expected-F1 质量排名 + 几何 Jaccard 多样性”丢失了大量对不同潜在
> Gold 模式有用的候选。**

### 14.3 为什么不是简单的排名器失效

在全部一步 action 上，合法 predicted expected-F1 与真实增益的：

- action-level ROC-AUC：`0.7911`；
- query-level Spearman 平均：`0.6365`；
- 正 action 只占 `15.86%`。

这表明分数并非完全无信号，但它解决的是“哪个动作的平均效用高”，
而 Bank 真正需要的是“固定 20 个动作联合覆盖多少个可能真相”。
独立排名后取 Top-20 不会自动优化这个联合目标。

低 H1 结构 F1 的查询上，平均 Spearman 仅 `0.4945`，但这些查询的
Oracle-first 两步增益高达 `+0.14976`；它们既是潜力最大的查询，也是单分数
排名最不可靠的查询。H1 真实 F1 在推理时不可知，因此不能用它做
bucket rule；只能通过合法的检索场景分歧度表示这种不确定性。

特别地，上轮退化的第 1 折并非没有好候选：

- Oracle-first 两步：`+0.10737`；
- 固定 beam 未压缩：`+0.09822`；
- v16 Bank20：`+0.06034`。

该折主要丢失也发生在 Bank 压缩，而不是生成阶段。

## 15. 从第一性原理重新定义 Scientist Bank

### 15.1 需要优化的不是 20 个独立分数

对查询 `x`，设候选终态集为 `A(x)`，Bank 为 `S ⊆ A(x)`且
`|S| ≤ 20`，真实未知功能团集合为 `G`。Scientist 的目标应是：

`E_G [ max_{a ∈ S} max(0, F1(a, G) - F1(H1, G)) ]`

而不是：

`sum_{a ∈ S} E_G[F1(a, G)]`。

前者是 best-of-K 联合效用；后者会选出许多针对同一高概率模式的
近重复候选。任意 Jaccard 距离只能让它们“形式不同”，不能保证它们
“对不同可能真相有用”。

### 15.2 保留后验的多模态，不把它压成边缘平均

当前 cardinality-conditioned posterior 将检索到的多个训练谱压成功能团边缘
概率。这会丢失“哪些功能团共同出现”的相关性，并产生一个可能不对应
任何真实食物谱的平均模式。

下一设计应直接保留训练侧检索到的完整 residual functional-group
sets 作为离散潜在场景 `C={c_1,...,c_m}`，其权重 `w_j` 只来自合法的
检索相似度和训练侧先验。不使用 held-out Gold、测试 ID 或 UniMol。

### 15.3 任务效用驱动的场景最大覆盖

对任意候选 `a` 和合法场景 `c_j`，定义：

`u(a,c_j)=max(0, F1(groups(a),c_j)-F1(groups(H1),c_j))`

然后选择 Bank：

`F(S)=sum_j w_j max_{a ∈ S} u(a,c_j)`

`F(S)` 是非负加权的 maximum-coverage/facility-location 型目标。在场景和
utility 被冻结后，它对候选集 `S` 是单调子模的：新增一个候选时，
随着已有 Bank 变大，能新覆盖的场景只会减少。因此在 `|S|≤20` 的基数
约束下可使用逐步最大边际收益的 greedy selector。

这里的“多样性”不再是任意结构距离，而是：

> 新候选是否能提高当前 Bank 尚未覆盖的某个合法潜在真相的
> MPC 终态 F1。

它直接对齐 Scientist Oracle@K，同时保留 exact-N 和功能团评测定义。

## 16. 文献支撑与严格适用边界

### 16.1 直接支持主路线的文献

1. **Multiple Choice Learning, NeurIPS 2012**
   ([Guzmán-Rivera et al.](https://proceedings.neurips.cc/paper_files/paper/2012/hash/cfbce4c1d7c425baf21d6b6f2babe6be-Abstract.html))
   明确区分“单模型的 M-best MAP”和“直接学习多个结构化输出”，并以
   多输出任务损失优化后续阶段可获得的最好候选。这直接支持
   MPC Scientist 应优化 best-of-K，而不是单分数 Top-K。
2. **Submodular Meets Structured, NeurIPS 2014**
   ([Prasad et al.](https://proceedings.neurips.cc/paper_files/paper/2014/hash/8fcd0c8d0e4335895172454b51bcc506-Abstract.html))
   专门研究从巨大结构化输出空间中选择高质量、多样的有限 proposal
   subset，并使用子模边际收益贪心扩充。它支持“候选集联合效用”这一
   数学形式。
3. **A Class of Submodular Functions for Document Summarization, ACL 2011**
   ([Lin & Bilmes](https://aclanthology.org/P11-1052/))
   展示了如何用子模目标在有限预算下平衡重要性和覆盖。它不直接解决
   MPC，但支持“有预算的代表性子集选择”。
4. **Direct Loss Minimization for Structured Prediction, NeurIPS 2010**
   ([Hazan et al.](https://proceedings.neurips.cc/paper/2010/hash/ca8155f4d27f205953f9d3d7974bdd70-Abstract.html))
   指出结构化任务中通用 surrogate 可与真实任务损失错位，并研究
   loss-adjusted inference 下的直接任务损失优化。它支持使用终态功能团 F1
   改善量定义 scenario utility，而不是继续训练 molecule-level correctness。

### 16.2 只能作次要支持的文献

- **Structured DPP, NeurIPS 2010**
  ([Kulesza & Taskar](https://proceedings.neurips.cc/paper/2010/hash/1f50893f80d6830d62765ffad7721742-Abstract.html))
  支持对结构化候选集建模排斥性/多样性，但 DPP kernel 中的“不相似”
  本身不保证 MPC 任务有用。当前 Jaccard 失败已证明不能单独依赖
  无监督几何多样性，因此本轮不以 DPP 为主方法。
- **Differentiable Top-k Classification, ICML 2022**
  ([Petersen et al.](https://proceedings.mlr.press/v162/petersen22a.html))
  支持训练目标应考虑 top-k 而非只有 top-1，但其标准设定是固定类别空间的
  classification，不能直接代替可变结构候选集的 MPC Bank 目标。
- **Conformal Prediction Sets with Limited False Positives, ICML 2022**
  ([Fisch et al.](https://proceedings.mlr.press/v162/fisch22a.html))
  支持在覆盖与错误候选数之间做受控交换，但需要独立校准设定和可交换性。
  当前只有 568 条训练查询，且 Bank20 预算已固定，因此不应在下一轮同时
  引入 conformal adaptive-K；可留作以后的风险控制扩展。
- **LOLS, ICML 2015**
  ([Chang et al.](https://proceedings.mlr.press/v37/changb15.html))
  支持状态化搜索和终态任务损失，但本轮已发现第一步 beam 仅丢失
  `0.00388`，因此继续优化搜索 policy 不是当前优先项。

### 16.3 与 FoodPuzzle 原论文的对齐

FoodPuzzle 将 MPC 定义为给定食物、部分分子和精确缺失数量的集合补全，
并以预测与 Gold 的功能团集合 F1 评价
([FoodPuzzle, Sec. 3](https://arxiv.org/html/2409.12832))。本设计不改变输入、输出、
exact-N 或评测定义。

原论文的错误分析把 `32%` 错误归于不恰当的搜索空间初始化，但该数字来自
50 个随机错误的人工分类，不能被解读为对本子模目标的实验证明。它只说明
“候选范围与起点很重要”；本项目的 568 条 OOF 分解才是 Bank 压缩瓶颈的
直接证据。原论文同时报告 epistemic hallucination `26%` 和错误解读证据 `20%`；
它们更直接对应以后的 Reviewer/证据阶段，不应用来解释当前已定位的
Scientist Bank 压缩损失。

## 17. 下一轮单一优化动作与预注册门槛

下一轮只替换 v16 的 Bank selector，不同时修改：

- raw one/two-step candidate generator；
- beam8；
- add100 和全 H1 remove；
- Scientist Slate5；
- Reviewer；
- MFP；
- MPC 无 UniMol 政策。

候选生成后，使用上述训练侧完整场景覆盖目标贪心选择 Bank20。
`KEEP_H1` 保留为 Bank 外的零风险回退，不占用 20 个提案槽位。

为避免再次在同一 OOF 上调到过线，下一轮实现前必须冻结：

- scenario 数量和检索方法；
- scenario weight 归一化；
- utility 严格使用 positive delta functional-group F1；
- Bank 预算 20；
- 贪心 tie-break；
- 不使用 sample ID、Gold bucket、正式 cache 或 UniMol。

预注册准入线：

1. 新 Bank20 相对 v16 Bank20 (`+0.05250439`) 的 paired bootstrap 95%
   下界大于 0；
2. 新 Bank20 至少保留固定 beam 未压缩 Oracle `+0.07126342` 的
   **90%**，即至少约 `+0.06414`；
3. 覆盖固定 beam 320 个正查询的至少 **90%**，即至少 288 条；
4. 五折中不再出现类似第 1 折 `-0.01572` 的大幅退化；
5. 对目标食物名、完整谱和 near-duplicate profile 进行 grouped stress
   test，防止检索场景泄漏；
6. 若 Bank 通过，才在同一冻结 Bank 上审计 Slate5；若 Bank 失败，不进入
   Slate/Reviewer/API。

这一设计的创新点不是“再加一个多样性项”，而是将 MPC Scientist
的有限候选压缩改写为：

> **在合法、多模态的训练侧功能团场景上，直接最大化终态 MPC
> task-utility 的 best-of-K 子模覆盖。**

它针对本轮最大的 Bank compression regret，同时有 Multiple Choice
Learning、结构化子模 proposal selection 和 direct task-loss 文献支撑，
且不改变 FoodPuzzle MPC 的任务设定。
