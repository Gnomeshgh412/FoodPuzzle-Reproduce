# Optimized Agent v11 设计与离线准入审查

- 日期：2026-07-30
- 路线：Only-Deepseek Scientist–Reviewer
- 范围：v11 方法设计、历史模块取舍、MPC 离线准入
- 本轮操作：只读分析已有代码、结果与缓存；未修改正式代码，未调用 API，未运行正式任务
- 上游记录：`2026-07-30_optimized-agent-v1-v10-longitudinal-audit.md`

## 1. 本轮要回答的问题

本轮不是继续为当前测试集调参，而是回答四个方法问题：

1. v5–v10 中哪些模块有可重复的正证据？
2. 为什么 v10 的 H2、H3 和 Reviewer 理论合理但实际有害？
3. 如何保留 Scientist–Reviewer 结构，同时避免 Reviewer 破坏 H1？
4. 如何让 UniMol 成为通用的结构残差专家，而不是数据集特定规则或装饰性特征？

## 2. 证据边界

### 2.1 当前仍可直接核验的证据

仓库当前保留的是 v10 的正式结果：

- MFP：34/71，Accuracy 0.4789。
- MPC：Precision 0.6248，Recall 0.7004，F1 0.6574，IoU 0.5153。
- MPC exact-N：71/71。
- `hypotheses_metadata.jsonl`：71 条，其中 52 条生成 H1/H2/H3 并调用 Reviewer/Fusion，19 条直接采用结构置信结果。

### 2.2 历史证据

v1–v9 的正式文件已被覆盖，因此只能采用此前已经读取并核验后保存的纵向审查记录。关键节点为：

- v2 MPC F1 0.6819：历史最高。
- v5 子模集合解码将阶段 F1 从 0.5739 提升至 0.6363，但 Verifier 又降至 0.6269。
- v7 H1 约 0.6748，最终约 0.6721；Reviewer 净负贡献。
- v8 retrieval residual 的 small-N OOF 增益约 +0.0205；raw UniMol H3 显著有害。
- v9 MFP 35/71，MPC F1 0.6737：当前最均衡版本。

历史结果只用于模块方向判断，不用于重新选择当前测试样本的参数。

## 3. 新增的离线事实

### 3.1 真实观测机制与 v10 人工掩码不匹配

从重建划分直接统计：

| 划分 | 样本数 | 平均缺失率 | 中位缺失率 | 最小–最大缺失率 | 平均已知分子数 |
|---|---:|---:|---:|---:|---:|
| Train | 568 | 0.8687 | 0.8925 | 0.7500–0.9078 | 9.03 |
| Test | 71 | 0.8593 | 0.8925 | 0.7500–0.8996 | 8.58 |

v10 每个训练食品包含一条 task-shaped query，同时又加入 15%、35%、60% 三种随机缺失 query。结果是每条符合真实任务的 query 被三条明显更容易、观测更完整的 query 稀释。

问题不只是“掩码比例选错”，而是训练风险主要由与部署条件不同的 query 决定。v11 必须以训练集自身的真实观测机制为主，额外增强只能保持同一观测分布。

### 3.2 v10 各假设的真实阶段贡献

使用已有 DeepSeek 功能团缓存，对 H1/H2/H3 和最终结果进行只读重算。该分析没有发出任何 LLM 请求。

| 阶段 | 功能团 Macro-F1 |
|---|---:|
| H1 occurrence Top-N | **0.6725** |
| H2 UniMol-conditioned set energy | 0.6584 |
| H3 structure-seeded set energy | 0.6613 |
| Reviewer 选定的 base | 0.6576 |
| Fusion 最终输出 | 0.6574 |

在 52 个被审查样本上，相对 H1：

| 阶段 | 平均变化 | 改善 | 损害 | 持平 |
|---|---:|---:|---:|---:|
| H2 | -0.0192 | 6 | 16 | 30 |
| H3 | -0.0153 | 10 | 15 | 27 |
| Reviewer base | -0.0204 | 4 | 14 | 34 |
| Fusion final | -0.0206 | 3 | 13 | 36 |

结论：

- v10 的主要性能损失发生在 H1 之后。
- Fusion 没有修复 Reviewer，反而略微继续下降。
- H2/H3 不是始终错误，但平均质量不足，不能作为完整集合与 H1 平权竞争。

### 3.3 当前三候选的 oracle 上限不足以支撑显著突破

如果事后对每条样本从 H1/H2/H3 中选择功能团 F1 最高者：

- H1 全集 F1：0.6725。
- 三候选 oracle F1：0.6811。
- 理论上限增益：约 +0.0085。

这只是不可实现的测试集 oracle，且仍仅接近 v2 的 0.6819。因此：

> 单独把 Reviewer 换成更聪明的选择器，最多只能恢复到历史最高附近；要获得显著提升，必须先生成比当前 H2/H3 更有信息的局部候选。

### 3.4 Reviewer 的置信度不是可靠的增益指标

52 个被审查样本中：

- Reviewer 选择 H2 48 次、H3 2 次、H1 2 次。
- 35 条没有候选级 evidence；这些样本的 Reviewer base 相对 H1 平均下降约 0.0232。
- 17 条具有候选级 evidence，仍平均下降约 0.0146。
- confidence ≥ 0.8 的 14 条接近零增益，但仍是 2 改善、4 损害、8 持平。
- confidence < 0.8 的 38 条平均下降约 0.0279。

所以“LLM 自报高置信”不能承担门控职责。Reviewer 的权限必须由训练内可验证的预期增益门控，而不是由自然语言置信度决定。

## 4. 第一性原理分解

MPC 的输入是目标食品、少量已知分子和缺失数量 N；输出是恰好 N 个分子。其困难可分成三个互不等价的问题。

### 4.1 候选召回

正确缺失分子首先必须进入候选池。食品共现、相似食品 profile residual 和分子频率对此最直接。UniMol 只描述分子结构，无法凭结构单独恢复“某食品是否包含该分子”的事实。

### 4.2 边界排序

候选池进入之后，需要区分 H1 截断边界附近的候选。这里结构、属性需求和相似食品残差可能有效，因为候选已经具有食品相关先验。

### 4.3 集合决策

官方 MPC 指标是预测集合所覆盖的功能团，而不是精确分子命中率。独立排序可能选择大量功能相似分子，因此适度的集合互补有理论价值；但如果为了多样性替换高置信 occurrence 核心，又会损失 Recall。

由此得到权限顺序：

> 食品相关性决定候选资格；结构和属性只解决边界残差；集合互补只在边界候选中优化；Reviewer 只审核具体交换；Fusion 只执行已批准动作。

## 5. 文献支撑与适用边界

### 5.1 Positive–Unlabeled 学习

[nnPU（NeurIPS 2017）](https://proceedings.neurips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html)指出，将未标注数据直接视为负例会造成风险估计和过拟合问题，并提出非负风险估计。

对 MPC 的对应关系：

- 完整训练 profile 中的已知成员是正例。
- 某分子没有出现在某个 profile 中，并不能证明其不存在，只能视为未标注。
- v10 的 hard corruption 可以用于构造“困难未标注候选”，不能直接获得确定负标签。

适用边界：

- v11 不应声称严格满足经典 SCAR 假设；食品数据库的记录概率很可能依分子而变化。
- 因此首版应采用低容量 PU residual 和分组 OOF，而不是高容量端到端网络。

### 5.2 选择性预测与拒答

[Selective Classification via One-Sided Prediction（AISTATS 2021）](https://proceedings.mlr.press/v130/gangrade21a.html)将拒答明确建模为准确率与覆盖率之间的取舍；[Structured Abstain（COLT 2022）](https://proceedings.mlr.press/v178/nueve22a.html)进一步说明结构化输出也需要可拒答机制。

对 MPC 的对应关系：

- Reviewer 不是必须覆盖全部样本。
- `ABSTAIN` 的含义是保留 H1，而不是输出为空。
- Reviewer 的目标应是“在可验证的少数交换上获得正增益”，而不是提高审查覆盖率。

### 5.3 集合互补

[Contextual Submodular Prediction（ICML 2013）](https://proceedings.mlr.press/v28/ross13b.html)为同时考虑单项质量与集合多样性的预测提供了方法依据。

对 MPC 的对应关系：

- v5 的阶段结果已经证明功能互补可能提高功能团 F1。
- 但论文的理论并不支持把任意 diversity 项无条件加入当前任务。
- v11 的互补项必须是单调、受限、局部的，且不能移除 H1 高置信核心。

### 5.4 UniMol 的角色

[Uni-Mol（ICLR 2023）](https://openreview.net/pdf?id=6K2RM6wVqKu)提供通用 3D 分子表征，但下游任务仍需要任务适配。

对 MPC 的对应关系：

- frozen UniMol embedding 是分子结构先验。
- 食品—分子共现不是 UniMol 预训练目标。
- raw cosine、partial centroid cosine 或“结构相似即共现”没有方法保证，且 v8/v10 已给出负证据。
- 合理用法是学习“在食品相关候选已经成立的条件下，结构信息是否支持一次局部交换”。

### 5.5 LLM Reviewer 偏差

[PORTIA（EMNLP 2024）](https://aclanthology.org/2024.emnlp-main.621/)指出 LLM 对成对候选存在位置偏差；[Humans or LLMs as the Judge?（EMNLP 2024）](https://aclanthology.org/2024.emnlp-main.474/)展示了包括权威暗示在内的 Judge 偏差。

对 MPC 的对应关系：

- 不向 Reviewer 暴露 H1/H2/H3、UniMol expert、set-energy 等策略身份。
- 候选动作使用中性 A/B 编号。
- 不让 Reviewer 比较三个长达几十或上百项的完整集合。
- Reviewer 只审查少量 remove/add 交换，并允许拒答。

## 6. v11 总体结构

### 6.1 MFP：冻结 v9 路线

v11 不修改 MFP 的方法逻辑。正式实现时恢复并固定 v9 已验证的：

- 直接宏观类别输出；
- 稀疏 occurrence；
- 类别条件 UniMol 集合适配；
- class-aware grouped OOF 融合；
- 固定候选 Reviewer。

MFP 只在最终完整运行时做回归验证，不参与本轮 MPC 选择。

### 6.2 MPC H1：稳定的食品相关性主干

H1 使用 v7/v9 已验证、v10 中仍达到 0.6725 的 occurrence 主干：

- 分子频率；
- 与 partial profile 的 cooccurrence；
- 相似训练食品的 profile support；
- 低容量 query-conditioned 排序；
- exact-N；
- 高置信核心锁定。

H1 是默认输出。所有辅助模块的“零预算”都必须精确退化回 H1。

### 6.3 MPC H2：检索残差专家

H2 不再重新生成完整集合，而是产生局部交换建议：

1. 仅从训练集检索相似食品。
2. 按食品或原始 profile 分组做 OOF，避免同源 profile 同时出现在拟合和验证侧。
3. 估计候选在相似 profile 中相对 partial 的条件 residual support。
4. 只比较 H1 截断边界内外的候选。
5. 输出若干 `(remove, add, gain, provenance)`，不输出自由集合。

准入目标是训练内 grouped OOF 的隐藏分子集合指标，而不是测试集功能团 F1。

### 6.4 MPC H3：UniMol 条件交换残差专家

H3 的训练单位从“候选是否属于 profile”改为“结构信息是否支持一次边界交换”。

每个训练样本：

1. 按真实任务分布构造 partial/hidden。
2. 运行冻结 H1。
3. 从 H1 边界产生 remove/add 对。
4. 使用隐藏集合判断该交换是改善、损害还是中性。
5. 只用训练数据学习低容量 residual scorer。

建议特征：

- `H1(add) - H1(remove)`：食品相关性代价；
- frozen UniMol 的 add/remove embedding 差；
- add/remove 与 partial-set 结构摘要的条件交互；
- add/remove 的结构离群程度；
- 训练内分子属性与预测缺失属性需求的匹配差；
- H2 residual support 差；
- 候选覆盖与名称可映射性。

明确禁止：

- raw UniMol cosine 直接进入全局 Top-N；
- UniMol 单独扩大候选宇宙；
- 使用测试集功能团缓存训练 H3；
- H3 无视 H1 margin 替换高置信候选；
- 在单构象残差无正 OOF 增益前引入多构象。

### 6.5 局部集合互补

吸收 v5 的有效思想，但改变作用范围。

候选集合分为：

- `core`：H1 高置信部分，锁定；
- `boundary_in`：H1 中允许移除的边界项；
- `boundary_out`：H1 外允许加入的边界项。

局部目标：

`utility(swap) = relevance_residual + λ × attribute_complement_gain - μ × uncertainty`

其中：

- relevance residual 来自 H2/H3；
- attribute complement 只使用 molecule-intrinsic FlavorDB 属性或训练内预测属性；
- λ、μ 仅由 grouped OOF 选择；
- 零预算必须是候选之一；
- 一次只接受正效用交换，并始终保持 exact-N。

这一设计保留“集合级互补”创新，但不允许其重写完整集合。

### 6.6 Selective Reviewer

Scientist 输出的是结构化交换 dossier，而不是三个完整答案：

- partial profile 摘要；
- 中性编号的 remove/add；
- H1 margin；
- H2/H3 是否独立支持；
- molecule-intrinsic 属性；
- 外部 evidence 是否明确提及候选；
- provenance 和冲突。

Reviewer 输出：

- `ACCEPT_A` / `ACCEPT_B` / `ABSTAIN`；
- 对应 action ID；
- 证据字段引用；
- 置信度仅作记录，不直接决定权限。

Reviewer 调用前由训练内 meta-gate 判断：

- OOF 预测交换净增益必须为正；
- 至少两个相互独立的支持来源，或者一个明确候选级 evidence；
- H1 margin 不得超过 OOF 学得的安全区间；
- 当前 action 不得涉及 core；
- 不满足则直接 `ABSTAIN`。

Reviewer 输出后还要通过一致性校验：

- action ID 必须存在；
- 引用证据必须能在 dossier 中定位；
- 不得生成新分子；
- 不一致则回退 H1。

### 6.7 Deterministic Fusion

Fusion 不再调用 LLM 做第二次自由判断。它只：

- 按已批准 action ID 执行交换；
- 去重；
- 排除 partial 中已知分子；
- 检查候选 provenance；
- 验证 exact-N；
- 任一检查失败则回退 H1。

Scientist–Reviewer 结构仍然保留；被取消的是“无证据的自由重写权限”，不是 Reviewer 本身。

## 7. 模块离线准入表

| 模块 | 当前证据 | v11 状态 | 进入正式运行前的条件 |
|---|---|---|---|
| v7/v9 occurrence H1 | 多版本稳定正证据；v10 单独 0.6725 | 直接准入 | exact-N 与回归检查通过 |
| Retrieval residual H2 | v8 small-N OOF 正增益；v9 有效 | 条件准入 | grouped OOF 平均正增益，多数折同向 |
| 当前 v10 set-energy H2 | 正式结果 -0.0141 | 拒绝 | 不沿用 |
| Raw/centroid UniMol | v8 显著负；v10 接近随机 | 拒绝 | 不沿用 |
| Task-adapted UniMol swap residual | 方法合理但尚无实证 | 待验证 | grouped OOF 正增益且伤害数不超过改善数 |
| v5 全局子模 decoder | 阶段有效但主干弱 | 拒绝全局使用 | 仅提取局部互补项 |
| 局部属性互补 | 有历史阶段证据 | 待验证 | 不降低 exact-molecule OOF，属性覆盖正增益 |
| 当前强制 Reviewer | 4 改善、14 损害 | 拒绝 | 不沿用 |
| Selective Reviewer | 有拒答与 Judge 偏差文献支撑 | 待验证 | OOF meta-gate 正效用；否则覆盖率为 0 |
| LLM Fusion | v10 未修复 Reviewer | 拒绝 | 改为确定性执行器 |
| 多构象 UniMol | 尚未解决目标错配 | 暂缓 | 单构象 residual 稳定准入后再做独立对照 |

## 8. 训练与验证协议

### 8.1 数据使用

- 不修改现有 `split_data`。
- v11 的模型选择只使用当前 MPC train。
- test 只在版本冻结后运行一次。
- 额外 holdout 暂不参与反复调参，保留用于后续独立核查。

### 8.2 Grouped OOF

- 以原始食品/profile 为 group。
- 同一完整 profile 派生出的所有 mask 必须位于同一 fold。
- 训练 mask 的缺失率从 train 的经验分布采样，或直接采用其 task-shaped partial/missing。
- 辅助 mask 不得让低缺失率 query 数量压过 task-shaped query。

### 8.3 通用准入指标

模块选择不读取官方功能团评测缓存。训练内指标采用：

1. hidden molecule exact-set F1/Recall；
2. H1 边界交换的净正确成员变化；
3. molecule-intrinsic attribute coverage；
4. exact-N、重复、partial 泄漏；
5. 每折方向与 bootstrap 区间；
6. 对 H1 的改善/损害样本计数。

测试集功能团 F1 只用于冻结版本的最终外部评测。

### 8.4 防止迭代过拟合

- 每个模块预先登记准入条件。
- 未通过 OOF 的模块不调用 API、不进入正式版本。
- 不按测试集 N、样本 ID、食品名或 CID 编写规则。
- 不因一次 test 下降立即针对错误样本加特例。
- 每次正式结果与审查均保存在 `audit_records/`。

## 9. v11 实现顺序

正式实现应拆成四个可回退阶段，但仍保存在现有代码文件中：

1. 恢复/固定 v9 MFP 和 MPC H1。
2. 重启 grouped OOF retrieval residual，验证 H2。
3. 实现低容量 UniMol swap residual 与局部互补，验证 H3。
4. 实现 selective Reviewer 和 deterministic Fusion。

每一阶段都必须保留 `budget=0 / ABSTAIN -> H1` 的严格退化路径。

## 10. 正式运行前的硬性检查

只有全部满足才允许正式 MPC：

- H1 能在当前代码中独立复现约 0.6725 的既有离线阶段结果。
- H2 grouped OOF 平均增益为正，且多数折方向一致。
- H3 grouped OOF 平均增益为正；若失败则 v11 以 `H3 disabled` 运行，不为形式完整强行保留。
- H2/H3 的 OOF oracle 明显高于 H1，证明 Reviewer 存在可选择空间。
- meta-gate 的 OOF 净效用为正。
- Reviewer 可以全量 `ABSTAIN`，且此时输出逐字等于 H1。
- Fusion 不调用 LLM，所有输出 exact-N。
- 不读取功能团评测缓存。
- 不生成新的多构象 embedding。

## 11. 成功标准

### 11.1 方法成功

优先判断：

- H2/H3 在训练内 grouped OOF 中稳定提供互补增益；
- Reviewer 只在可验证动作上介入；
- UniMol 的贡献来自 task-adapted structural residual；
- 所有弱模块都能退化为零；
- 方法不依赖 FoodPuzzle 测试身份规则。

### 11.2 正式结果成功

冻结后希望：

- MPC 超过 v9 的 0.6737；
- 至少接近并尽可能超过 v2 的 0.6819；
- Precision 与 Recall 不通过极端牺牲一方换取；
- exact-N 71/71；
- 相对 H1 的 Reviewer/Fusion 净贡献非负；
- MFP 保持 v9 的 35/71 附近，不因 MPC 修改回退。

“显著提升”不能只由 71 条上的单点差值宣称，需要配对 bootstrap 区间和额外留出样本方向共同支持。

## 12. 最终决策

v11 的核心不是增加第四个打分器，而是重建权限与学习目标：

> 用食品相关性 H1 保证召回，用检索 H2 和任务适配 UniMol H3 生成局部交换，用受限集合互补改善功能覆盖，用训练内 meta-gate 决定是否值得审查，用可拒答 Reviewer 验证证据，最后由确定性 Fusion 保证 exact-N。

当前准入结论：

- 可以立即实现：H1 恢复、经验观测机制、grouped OOF 框架、H2 局部残差、确定性 Fusion。
- 必须先 OOF 后决定：UniMol swap residual、局部集合互补、Selective Reviewer。
- 明确不进入 v11：v10 全局 set-energy、raw UniMol 全局排序、强制 Reviewer、LLM Fusion、多构象 UniMol。

下一步应先修改正式代码以实现 v11 的训练内 `--check-only` 准入检查；检查通过后向用户汇报，再申请正式 API 运行许可。
