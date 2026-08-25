# optimized-agent v12 失败归因与 MPC 路线重置审查

- 日期：2026-07-30
- 范围：Only-Deepseek Scientist–Reviewer 路线，重点审查 MPC v1–v12
- 本轮操作：只读检查正式结果、逐样本评测、候选/Reviewer 元数据、数据重构代码、既有审查记录与原始论文；未修改正式代码，未调用模型 API，未覆盖结果，未启动 v13，未执行 Git 操作
- 上游记录：
  - `2026-07-30_optimized-agent-v1-v10-longitudinal-audit.md`
  - `2026-07-30_optimized-agent-v11-design-and-offline-admission.md`
  - `2026-07-30_mfp-unimol-and-mpc-scientist-reviewer-route-audit.md`
  - `2026-07-30_optimized-agent-v12-action-review-implementation-and-check-only.md`

## 1. 核心结论

十二个版本均未让 MPC 稳定超过早期版本，不再能解释为偶然波动或某个阈值未调好。当前 MPC 路线存在三个结构性错位：

1. **优化目标错位**：候选排序、OOF action gate 和 Reviewer 证据规则主要优化“精确缺失分子是否出现”，正式指标却计算预测分子集合与真实集合的**官能团集合 F1**。
2. **Reviewer 信息与权限错位**：Scientist 与 Reviewer 基本共享同一模型、同一候选和同一证据，没有可靠的外部判定信号；v12 又把感官/功能证据排除为有效依据，Reviewer 最终只能保守弃权。
3. **版本选择与强主干错位**：后续版本没有真正冻结、回归验证 v7/v9 的强 H1，而是反复改写基础排序。新增模块经常背负了基础主干已经下降的结果。

因此，下一步不应直接实现 v13，也不应继续增加 Reviewer、Fusion、UniMol 或多 Agent 层数。应先重置 MPC 的学习目标和离线准入协议。

## 2. v1–v12 的纵向证据

历史正式结果来自已保存审查记录；旧结果文件曾被覆盖，因此不把无法从当前文件重新计算的数值伪装为新复算。

| 版本 | MPC F1 | 关键事实 |
|---|---:|---|
| v1 | 0.6783 | 简单 MPC 决策已较强 |
| v2 | **0.6819** | 当前历史最高 |
| v3 | 0.6737 | MFP 改善，MPC 回落 |
| v4 | 0.6747 | 仍接近强基线 |
| v5 | 0.6269 | 基础排序 0.5739；集合解码升至 0.6363；Verifier 再降至 0.6269 |
| v6 | 0.6231 | 全局多通道融合继续伤害 MPC |
| v7 | 0.6721 | H1 单独约 0.67475；Reviewer 最终约 0.67207 |
| v8 | 未正式运行 | retrieval residual 小样本 OOF 为正；raw UniMol H3 显著为负 |
| v9 | 0.6737 | 当时最均衡；exact-N 71/71 |
| v10 | 0.6574 | H1 约 0.6725；H2/H3/Reviewer/Fusion 将其逐步降低 |
| v11 | 未正式运行 | 离线设计通过不等于正式测试提升 |
| v12 | **0.6539** | Reviewer 零动作；最终值实际上就是新 H1 |

可重复的规律不是“Agent 越复杂越好”，而是：

- v1–v4 的简单 MPC 排序长期处于 0.674–0.682；
- v5 的集合互补解码是少数明确的正阶段贡献；
- v7、v10 的 Reviewer 均伤害强 H1；
- v8/v10 的 raw 或全局 UniMol 对 MPC 为负；
- v12 的 Reviewer 完全没有改变答案；
- 后续性能下降主要来自基础候选/排序被改写，以及错误目标上的门控。

## 3. v12 正式结果与真实执行路径

### 3.1 正式指标

`results/Only-Deepseek/optimized-agent/mpc/deepseek-v4-flash/evaluation_summary.json`：

- samples: 71
- Precision: 0.6117529
- Recall: 0.7097336
- F1: 0.6538672
- IoU: 0.5118680
- zero-F1: 2
- exact-N violations: 0
- unmapped predicted molecules: 39
- failed functional-group predictions: 39

### 3.2 Scientist–Reviewer 没有产生任何预测增量

对 `hypotheses_metadata.jsonl` 的核验：

- 71 个样本；
- 10 个样本触发 Scientist/Reviewer；
- 10 个 Reviewer 均选择 `ABSTAIN`；
- 0 个 action 被接受；
- 0 个 Reviewer swap 被执行；
- 最终均走 occurrence H1 或 H1 safe fallback。

因此，v12 的 0.6539 不能称为 action-review Agent 的效果；它是 v12 新 H1 的效果。

这也证明 v12 并未实现设计文档中“恢复 v7/v9 强 H1”的关键前提。历史 v7/v10 H1 约为 0.6748/0.6725，而当前 v12 H1 只有 0.6539。

### 3.3 门控选择了错误的风险区域

按 `n` 分层重算：

| n 区间 | 样本数 | 平均 F1 | 平均 Precision | 平均 Recall | 平均 gold FG 数 | 平均 predicted FG 数 |
|---|---:|---:|---:|---:|---:|---:|
| `n <= 5` | 19 | 0.4569 | 0.4504 | 0.4679 | 8.53 | 9.32 |
| `6 <= n <= 20` | 4 | 0.3205 | 0.2630 | 0.4104 | 8.50 | 13.75 |
| `n > 20` | 48 | 0.7596 | 0.7047 | 0.8304 | 22.75 | 26.71 |

`n` 与 F1 的 Pearson 相关系数约为 0.680。

但 10 个 action-gated 样本的 `n` 分别为：

`94, 97, 100, 104, 108, 118, 119, 173, 191, 228`

即门控全部落在大集合，而小/中集合的 23 个主要错误样本一个都没有进入审查。

其原因来自 v12 的 OOF 目标与动作定义：

- 目标是 hidden exact-molecule set F1，而不是正式 functional-group F1；
- action 只允许一次 add/remove；
- 大 `n` 样本提供更多共现与检索支持，更容易达到高阈值；
- 但一次交换对大集合 exact-set F1 的影响约为 `O(1/n)`；
- 正式评测对 71 个样本做宏平均，每个样本权重相同，而不是按 `n` 加权。

这使 gate 学到“在容易的大集合上做极小、统计稳定的动作”，而不是“在会显著影响宏平均指标的样本上纠错”。

### 3.4 action OOF 的正收益不能外推到正式指标

v12 check-only 报告：

- threshold: 0.90
- changed OOF queries: 41
- wins/losses: 31/2
- mean hidden exact-set F1 gain: 0.00049068
- bootstrap lower bound: 0.00032735

这说明 action 对它自己的训练目标有微小正收益，但不说明它对 functional-group F1 有益。数值本身也只有约 `5e-4`，远小于版本间正常波动和当前相对 v2 的约 0.028 差距。

## 4. 官方任务语义中存在两个不同目标

FoodPuzzle 论文对 MPC 的形式定义是：

`food + partial molecules + n -> missing molecules`

但正式评价不是 exact molecule F1，而是对预测和真实分子集合分别提取官能团集合，再计算 F1。论文明确说这样做是为了“优先考虑化学功能而非严格结构相似性”：

https://arxiv.org/html/2409.12832

论文错误分析还指出，模型不应只列出天然存在的分子，而应识别能够模仿或重建目标风味的分子；同时又警告不能把“coffee-like”之类的感官描述误读为“该分子天然存在于 coffee”。

这意味着必须区分两类合法但不同的 claim：

1. **occurrence claim**：分子真实存在于目标食品；
2. **functional replication claim**：分子能贡献或重建目标风味功能，但不必天然存在。

v12 将第二类证据整体排除，只允许 occurrence support 触发动作。这能减少“把感官相似误当天然存在”的错误，却也让 Agent 没有任何渠道优化官方强调的功能等价性。

正确做法不是放宽所有感官证据，而是做**类型一致的证据验证**：

- occurrence evidence 只能支持 occurrence claim；
- flavor-function/sensory evidence 只能支持 functional replication claim；
- Reviewer 必须检查 claim type 与 evidence relation 是否一致；
- 最终解码器再根据任务目标决定两类候选的权重。

## 5. 数据重构进一步放大了目标不一致

公开 `MPC_tasks.jsonl` 只有 food 和 `missing_molecules`，没有论文定义中正式输入所需的 `partial_molecules` 与 `n`。

当前 `code/reconstruct_mpc_data.py` 的重构方式是：

`partial = FlavorDB full profile - public missing_molecules`

`n = len(public missing_molecules)`

代码和 README 已正确标注这不是 official exact reproduction。其后果是：

1. `partial` 是利用 FlavorDB 完整 profile 与 gold missing 集合反推得到的确定性补集，而不是官方未公开的缺失过程；
2. 不同样本的 `n` 跨度极大，导致大集合与小集合具有完全不同的统计难度；
3. 当前结果只能作为 **FlavorDB-derived reconstruction** 的结果，不能把绝对分数直接与论文表格作同条件比较；
4. 在同一 71 样本上迭代十二次，会逐渐适应这一重构机制，而不一定提升通用 MPC 方法。

这并不使当前任务无效，但要求后续方法与论文中明确写出重构边界，并采用冻结验证协议。

## 6. 文献证据与本项目的对应关系

### 6.1 任务损失必须直接进入结构化预测

Direct Loss Minimization 指出，结构化任务常有专用指标，常见 surrogate 并不保证任务损失最优：

https://proceedings.neurips.cc/paper/2010/hash/ca8155f4d27f205953f9d3d7974bdd70-Abstract.html

F-measure 研究进一步表明：

- F1 是非线性指标；
- Hamming/subset 等替代损失可能对 F1 产生很高 regret；
- F1 最大化需要考虑成本不对称、阈值和标签依赖。

https://proceedings.neurips.cc/paper_files/paper/2014/hash/5c0314ec1b57fcd36bbb013f3f025868-Abstract.html

https://www.jmlr.org/papers/v15/waegeman14a.html

对应本项目：不能再用 exact-molecule action gain 作为 functional-group F1 的替代目标。

### 6.2 MPC 是固定基数的集合预测，不是独立 Top-N 分类

Contextual Submodular Prediction 将“固定大小的集合/列表，同时考虑单项质量与多样性”作为结构化预测问题：

https://proceedings.mlr.press/v28/ross13b.html

对应本项目：

- occurrence relevance 提供单项质量；
- 新增官能团覆盖提供集合边际收益；
- 重复官能团需要冗余惩罚；
- exact-N 是基数约束；
- v5 的子模集合解码曾把阶段 F1 从 0.5739 提升至 0.6363，是仓库内与该理论一致的正证据。

### 6.3 同模型自审不能替代独立验证

TACL 2024 的系统审查指出，纯提示式内生自我纠错通常没有稳定收益；可靠外部反馈或专门训练的纠错模型是成功的重要条件：

https://aclanthology.org/2024.tacl-1.78/

EMNLP 2024 的正结果也依赖一个可遮蔽、可重新预测的关键条件作为验证信号，而不是让模型自由评价自己的整套答案：

https://aclanthology.org/2024.emnlp-main.714/

对应本项目：

- Scientist 与 Reviewer 同源且共享信息时，Reviewer 没有新的可判定事实；
- “再思考一次”不构成 verifier；
- Reviewer 只有在获得独立检索证据、确定性化学检查、或训练出的 metric-aware utility 时才有合理的纠错基础；
- 否则 Reviewer 应保留为解释/证据审计层，而不是主预测增益层。

### 6.4 拒答门控不能只做事后阈值

SelectiveNet 说明拒答应与预测风险和覆盖率联合优化，而不是在预训练模型置信度上附加一个任意阈值：

https://proceedings.mlr.press/v97/geifman19a.html

对应本项目：v12 的 0.90 gate 对 exact-set F1 有效，但没有针对 functional-group risk/coverage 训练，因而选择了错误样本区域。

### 6.5 十二次版本选择已经构成选择偏差风险

Cawley 与 Talbot 指出，在有限样本上反复优化模型选择指标会过拟合选择过程，其影响可以与算法间真实性能差异同量级：

https://www.jmlr.org/papers/v11/cawley10a.html

对应本项目：后续必须停止用同一 71 样本的正式结果指导模块取舍。此前单独抽出的测试批次可以用作最终盲测，但方法选择必须只用 train 内 grouped OOF 或固定 dev。

## 7. UniMol 在下一条 MPC 路线中的正确位置

UniMol 的原始贡献是通用 3D 分子表征，预训练覆盖约 2.09 亿构象：

https://mlanthology.org/iclr/2023/zhou2023iclr-unimol/

它能够表示单个分子的结构和几何，但不直接包含：

`P(molecule occurs | food, partial profile, processing, database observation)`

因此：

- 不再把 UniMol cosine、partial centroid 或 set compatibility 当作出现概率；
- 不因项目创新要求而强迫 UniMol 进入 MPC 主干；
- 若保留 UniMol，只允许它承担**分子内禀功能表示**：
  - 预测/补全分子的官能团或结构功能；
  - 估计两个候选在功能空间中的冗余；
  - 在缺少确定性结构映射时提供 group-role embedding；
- 必须与 RDKit/SMARTS 官能团、Morgan fingerprint、无 UniMol版本进行同一 OOF 消融；
- 只有 metric-aligned OOF 对 functional-group F1 有稳定增益，UniMol 才进入 MPC。

多构象 UniMol只能降低构象表示方差，不能修复“食品出现关系”和“官能团 F1”之间的目标错位，所以目前不应优先实施。

## 8. 第一性原理重建：下一条 MPC 方法路线

### 8.1 不再从 v12 继续补丁

先构造三个明确隔离的层：

#### A. Food-conditioned candidate retriever

目标：在合法候选目录中高召回可能出现或可能贡献目标风味的分子。

输入：

- target food；
- partial molecules；
- n；
- train-only occurrence/co-occurrence；
- typed evidence；
- retrieval demonstrations。

输出：候选及两种分开的置信度：

- `p_occurrence`;
- `p_functional_replication`.

#### B. Functional-group demand estimator

目标：估计真实 missing set 需要覆盖的官能团分布/集合，而不是直接猜每个精确分子。

训练目标必须来自 train 内 missing sets 的官能团映射；不能读取测试 gold 或评测缓存。

可采用：

- FlavorDB/RDKit/SMARTS 的确定性内禀官能团映射；
- 缺映射时才比较 UniMol adapter；
- N 作为连续条件输入，而不是硬编码 small/medium/large 测试规则。

#### C. Exact-N metric-aware set decoder

在候选集合中选择恰好 N 个分子，目标同时包含：

- food/occurrence relevance；
- 预计官能团需求覆盖；
- 新官能团的边际增益；
- 重复官能团冗余惩罚；
- 证据类型一致性；
- unmapped/不可评测风险惩罚。

这是对 v5 有效集合解码思想的恢复，但基础排序必须使用冻结强 H1，且训练/准入直接比较 functional-group F1。

### 8.2 Scientist–Reviewer 的新职责

保留论文逻辑，但不让 Reviewer承担它无法完成的数学优化：

- Scientist：提出候选或少量局部 action，并为每个 claim 标注类型；
- Reviewer：验证证据是否真的支持相应 claim，检测“感官相似 -> 天然存在”的错误推理；
- deterministic decoder：根据通过验证的候选和 group utility 生成 exact-N 最终集合；
- Reviewer 不再从三个高度相关的完整长列表中任选一个；
- 没有独立证据或可检查条件时必须 abstain，且 abstain 不应被包装成性能贡献。

## 9. 下一阶段准入与停止条件

在任何正式 v13 运行之前，必须先完成以下离线门槛：

1. **冻结强 H1 控制**  
   当前代码必须能独立输出并回归验证同一套 H1；若无法从现有代码复现历史 v7/v9 H1，就重新建立一个冻结、版本化的 control，而不是声称“已恢复”。

2. **目标对齐**  
   所有 decoder、gate 和阈值选择都以 train-only grouped OOF 的 functional-group F1/IoU 为准；exact molecule F1 只能作为诊断指标。

3. **候选与解码分离**  
   分别报告 candidate recall、H1、group-aware decoder、Reviewer 后结果，禁止只报告最终值。

4. **成对稳定性**  
   要求多数 fold 同向、paired bootstrap 下界为正、small-N 与 large-N 都不出现系统性崩溃。

5. **Reviewer 增量准入**  
   Reviewer 必须在冻结候选与 decoder 后单独证明正增量；否则保留为解释审计层，不进入预测路径。

6. **UniMol 单变量准入**  
   与 deterministic functional groups、Morgan/RDKit 和 no-UniMol 对照；若没有独立正增量，MPC 正式方法不使用 UniMol。

7. **盲测保护**  
   不再根据当前 71 个正式测试样本的错误调权。之前单独抽出的批次只在方法完全冻结后使用一次。

8. **停止规则**  
   如果 metric-aware decoder 在 train OOF 上仍不能超过冻结 H1，说明当前候选信息不足；此时应转向新的可靠证据/候选召回，而不是继续增加 Agent 层。

## 10. 最终判断

v12 没有改善并不是因为 Reviewer 过于保守这一件事。更根本的原因是：

- v12 的基础 H1 已比历史强 H1 低约 0.019；
- action gate 优化了错误的 exact-molecule目标；
- gate 将所有审查预算分配给本来表现较好的大 N 样本；
- Reviewer 没有独立验证信息；
- Reviewer 的证据规则排除了能够服务官方功能指标的功能性 claim；
- 十二轮在同一小测试集上的自适应迭代增加了选择偏差。

所以路线应从“让 Scientist–Reviewer 更会选分子”改为：

> **metric-aligned candidate retrieval + functional-group demand estimation + exact-N set decoding；Scientist–Reviewer负责 typed evidence verification，而不是替代结构化集合优化。**

这仍然没有脱离 FoodPuzzle 的 MFP/MPC 任务，也保留了原论文 Scientist–Reviewer 的科学证据审查逻辑；改变的是让每个模块只负责它拥有信息、能够被验证的部分。
