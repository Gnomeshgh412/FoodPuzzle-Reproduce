# Optimized Agent v1–v10 纵向审查

- 日期：2026-07-30
- 范围：Only-Deepseek Scientist–Reviewer 主路线
- 任务：MFP、MPC
- 性质：历史结果与模块演化审查
- 注意：v1–v9 正式结果曾被后续版本覆盖；本文中的旧版本指标来自此前已经完成的审查记录。v10 指标及逐阶段元数据仍可由当前结果文件直接核验。

## 1. 证据等级

本文将证据分为三级，避免将历史记忆与当前文件混为一谈。

- A：当前仓库仍存在正式结果、元数据或逐样本记录，可直接复核。
- B：旧结果已被覆盖，但此前任务中曾读取正式文件并完成核验，当前可从任务记录恢复指标与审查结论。
- C：只完成训练内 OOF、`--check-only` 或方法设计，没有正式测试结果。

## 2. v1–v10 结果演化

| 版本 | MFP | MPC F1 | 状态 | 主要结论 |
|---|---:|---:|---|---|
| v1 | 20/71 | 0.6783 | B | 早期 MPC 较强，MFP 较弱 |
| v2 | 19/71 | **0.6819** | B | 历史最高 MPC，但 MFP 最弱之一 |
| v3 | 23/71 | 0.6737 | B | MFP 改善，MPC 略降 |
| v4 | 23/71 | 0.6747 | B | UniMol/结构检索方向可行，但决策接口存在问题 |
| v5 | 32/71 | 0.6269 | B | MFP 大幅提升；MPC 因伪负例、全局融合和 Verifier 明显下降 |
| v6 | **34/71** | 0.6231 | B | MFP 继续提升；MPC Precision 上升但 Recall 下降 |
| v7 | 33/71 | 0.6721 | B | PU occurrence 主干恢复 MPC；Reviewer 仍为负贡献 |
| v8 | 未正式运行 | 未正式运行 | C | H2 小 N 有益；直接 UniMol H3 在 OOF 中显著有害，因此未正式运行 |
| v9 | **35/71** | **0.6737** | B | 当前 MFP 历史最佳；MPC 与 ICL 基本持平且 exact-N 100% |
| v10 | 34/71 | 0.6574 | A | 全局集合能量与强制 Reviewer 选择破坏了 H1 |

基线参考：

| 方法 | MFP | MPC F1 | MPC exact-N |
|---|---:|---:|---:|
| Zero-shot | 15/71 | 0.5355 | 62/71 |
| BM25-ICL | 17/71 | 0.6758 | 50/71 |
| 原 Scientist–Reviewer Agent | 21/71 | 0.6012 | 43/71 |
| 独立 Multi-Agent | 30/71 | 0.6565 | 71/71 |

## 3. 不能只看最终版本的原因

历史结果显示，MFP 和 MPC 的有效机制明显不同。

- v1–v4：MPC 已经达到约 0.674–0.682，但 MFP 只有 19–23 条正确。
- v5–v6：MFP 提升至 32–34 条，MPC 却下降至约 0.623–0.627。
- v7：撤销有害的 MPC 全局融合后，MPC 立即恢复至 0.6721。
- v9：在 v7 强主干上加入更谨慎的功能团需求与受约束融合，MPC 达到 0.6737。
- v10：再次允许集合模型和 Reviewer 大范围改写后，MPC 降至 0.6574。

因此，下一版不能从 v10 单向继续增加模块；必须恢复 v7/v9 已验证的强主干，并吸收 v5/v6 中被最终指标掩盖的有效局部模块。

## 4. 各阶段值得保留的经验

### 4.1 v1–v4：较强的 MPC 候选生成与简单决策

可确认的经验：

- 早期版本没有复杂的全局集合能量和大规模自由改写，MPC F1 反而最高。
- v4 的 MPC F1 0.6747，已经非常接近 ICL。
- v4 的主要不足不是 MPC 候选能力，而是 MFP 决策接口：
  - UniMol 原始相似度高度饱和；
  - 正确类别经常出现在更深候选中；
  - Reviewer 的选择被最终控制器忽略；
  - 内部宏观类别又被转回代表食品名称，产生二次映射误差。

应吸收：

- MPC 的强基础候选排序不应被复杂 Agent 无条件覆盖。
- MFP 应直接输出宏观类别，并允许受控 Reviewer 修正。
- UniMol 必须经过任务适配，不能直接把预训练余弦相似度当作任务概率。

### 4.2 v5：集合覆盖有效，但基础排序和 Verifier 有害

v5 的逐阶段官能团 F1：

| 阶段 | F1 |
|---|---:|
| 基础排序 | 0.5739 |
| 加入子模集合解码 | 0.6363 |
| Verifier 最终输出 | 0.6269 |

重要发现：

- 子模集合解码并非整体失败，它将功能团 F1 提高约 0.0624。
- 它虽然降低精确分子命中，却改善了官方功能团指标，说明“相关性 + 功能互补性”的集合思想有效。
- v5 的主要失败来自所谓“保守伪负例”：
  - 从低频、零检索支持、低结构相似候选中选最容易的负例；
  - 模型被 frequency 与 cooccurrence 支配；
  - top-300 中大量概率饱和；
  - 71 条结果只使用 332 种分子，而 ICL 使用 728 种。
- Verifier 使 F1 从 0.6363 降到 0.6269。

应吸收：

- 保留集合级功能互补思想，但只能作为强 occurrence 排序后的局部重排。
- 未标注候选不能作为确定负例。
- Reviewer/Verifier 不能在候选召回不足时承担“凭推理找回答案”的职责。

### 4.3 v6：MFP 结构适配有效，MPC 多通道全局融合仍不稳定

v5 → v6：

| 指标 | v5 | v6 |
|---|---:|---:|
| MFP | 32/71 | 34/71 |
| MPC Precision | 0.5952 | 0.6324 |
| MPC Recall | 0.6713 | 0.6298 |
| MPC F1 | 0.6269 | 0.6231 |

有效经验：

- MFP 的类别条件 UniMol 集合表示、训练内 OOF 融合和固定候选 Reviewer 使结果继续提升。
- MPC 候选覆盖与名称映射改善，未映射次数从 41 降至 7。

失败经验：

- 开放候选空间和多通道结构/感知融合提高 Precision，却替换掉大量能贡献 Recall 的候选。
- 54/71 样本触发 Reviewer，45 条发生边界调整，修改过于频繁。
- occurrence、retrieval、UniMol、感知通道固定全局融合，弱模态可以破坏强模态。

应吸收：

- MFP 的类别条件结构适配器值得保留。
- MPC 多通道应采用“强主干 + 可退化为零的残差”，不能固定全局融合。

### 4.4 v7：最关键的恢复版本

v7 的核心变化：

- PU occurrence ranker 成为唯一全局主排序。
- 检索与 UniMol/感知只能做边界局部交换。
- 交换预算由五折 OOF 选择，并允许自动退化为 0。
- exact-N 由确定性控制器保证。

结果：

- MFP：33/71。
- MPC：Precision 0.6372，Recall 0.7184，F1 0.6721，IoU 0.5295。
- 71/71 exact-N。

最重要的阶段归因：

- H1 occurrence Top-N 单独 F1：约 0.67475。
- Scientist–Reviewer 最终 F1：约 0.67207。
- 5 次审查中：1 次改善、3 次下降、1 次不变。

因此 v7 已验证的核心贡献是：

1. occurrence 主排序；
2. OOF 残差预算；
3. 弱通道可退化为零；
4. exact-N 硬约束；
5. MFP 和 MPC 使用任务特定决策头。

未验证甚至被否定的是：

- MPC Reviewer 能稳定改善预测；
- raw UniMol residual 可以直接改善分子共现排序。

### 4.5 v8：没有正式运行，但离线失败信息非常有价值

v8 尝试构造真正独立的 H3 UniMol set-compatibility expert，并加入 Scientist → Reviewer → Fusion。

OOF 结果：

- H2 retrieval 在 small-N 上有效：
  - F1 增益约 +0.0205；
  - 8 次改善、1 次下降。
- H3 UniMol set compatibility：
  - small-N：约 -0.1749，1 次改善、52 次下降；
  - medium-N：约 -0.0047；
  - large-N：约 -0.0030。

正确决策是没有为了三假设形式完整而正式运行 v8。

应吸收：

- Retrieval residual 是真实有效的辅助通道。
- “UniMol 与 partial set 的结构兼容性”仍不能直接等同于食品分子共现。
- OOF 科学门控应当决定模块是否进入正式系统。
- 三层 Agent 只有在三个候选确实互补时才有意义。

### 4.6 v9：当前最值得恢复的组合

v9 结果：

- MFP：35/71，Accuracy 0.4930。
- MPC：Precision 0.6357，Recall 0.7221，F1 0.6737，IoU 0.5306。
- 71/71 exact-N。

相对原 Agent：

- MPC F1 +0.07245；
- 配对 bootstrap 95% CI 为正；
- exact-N 违规从 28 降为 0。

相对 ICL：

- F1 只低约 0.0021；
- 差异不显著；
- ICL 有 21/71 exact-N 违规，而 v9 为 0。

v9 的有效组合：

- v7 occurrence 主干；
- retrieval residual；
- UniMol/FlavorDB 用于功能团需求和候选属性，而不是 raw cosine 全局排序；
- 受限 H3；
- Reviewer 只在极少数不确定样本触发；
- Fusion 只能执行控制器允许的交换。

但 v9 仍有问题：

- 只有 4 条进入三层 Agent；
- Reviewer 两次错误选择 H3，Fusion 没有纠正；
- Precision 低于 ICL，主要通过更高 Recall 与 ICL 持平；
- 不透明名称 `CID 644104` 引发功能团映射污染。

### 4.7 v10：通用化尝试中的正确动机与错误实现

正确动机：

- 移除基于测试指标的 N 分桶策略；
- 用随机遮蔽学习通用集合补全；
- 采用等基数受损集合；
- 使用 exact-N 局部搜索；
- Reviewer 做反事实审查。

失败原因：

- 真实 MPC 平均缺失比例约 86.9%，训练却主要使用 15%、35%、60% 遮蔽。
- 集合外候选被当作负例，违反 PU 数据语义。
- UniMol raw similarity 对真实缺失分子的判别 AUC 约 0.508。
- 集合能量实际上被共现、频率、检索特征主导。
- 52 个 Reviewer 样本中只有 17 个有候选级 evidence。
- Reviewer 48/52 次选择 H2，实际只带来 4 次改善和 14 次下降。
- H1 单独约 0.6725，最终被改写为 0.6574。

应吸收：

- 通用遮蔽集合补全的定义是正确方向。
- exact-N 局部搜索和反事实动作接口可以保留。
- 训练观测机制、PU 风险和 Reviewer 证据条件必须重做。
- 当前全局 set-energy decoder 应停用。

## 5. 跨版本稳定规律

### 5.1 MFP 的稳定增益来源

MFP 从 v1 的 20/71 提升到 v9 的 35/71，主要来自：

- 直接预测宏观类别；
- 稀疏出现特征；
- 类别条件 UniMol 集合表示；
- class-aware OOF 融合；
- 固定候选空间；
- Reviewer 的受控覆盖。

v7 审查曾发现：

- 结构控制器单独正确 26 条；
- Reviewer 最终正确 33 条；
- 10 次错误转正确、3 次正确转错误；
- Reviewer 净增 7 条。

这说明 Reviewer 在单标签 MFP 上确实能发挥作用，因为它只需在少量宏观类别之间做语义判别。

### 5.2 MPC 的稳定增益来源

跨版本反复有效：

- occurrence/cooccurrence 主排序；
- 相似食品检索与 profile residual；
- OOF 决定残差交换预算；
- exact-N 硬约束；
- 高置信核心锁定；
- 只在边界候选中进行少量交换；
- 弱辅助通道允许退化为零。

### 5.3 MPC 的稳定失败来源

跨版本反复有害：

- 让 raw UniMol similarity 接管全局排序；
- 将 sensory similarity 当成自然存在证据；
- 把未标注候选当成确定负例；
- 固定全局融合所有通道；
- Reviewer 在缺少候选级独立证据时强制作答；
- Reviewer/Fusion 大范围改写完整集合；
- 为功能团覆盖过度牺牲 occurrence 核心；
- 用当前测试集选择 N 分桶、阈值或输出名称。

## 6. 下一版不应简单称为 v10 修复

下一版应当重建为 v7/v9 强主干上的保守残差系统，而不是继续在 v10 全局集合能量上补丁。

建议结构：

### 6.1 MFP：冻结 v9 路线

- 保留类别条件 UniMol 集合适配器。
- 保留训练内 OOF 类别融合。
- 保留固定 top-3 Scientist 审计。
- Reviewer 只在候选内选择。
- 暂不修改 MFP，避免同时改变两个任务。

### 6.2 MPC H1：恢复 v7/v9 occurrence 主干

- H1 必须是默认 exact-N 基础集合。
- 保留 frequency、cooccurrence、retrieval residual、query-conditioned 低容量特征。
- 不使用 v10 set energy 覆盖 H1。

### 6.3 MPC H2：强化历史上有效的 retrieval residual

- 从相似训练食品的完整 profile 中构造条件缺失后验。
- 使用 train-only BM25/profile neighbor。
- 对近重复 profile 做分组 OOF，防止检索记忆造成乐观估计。
- 只修改 H1 截断边界附近候选。

### 6.4 MPC H3：改为任务适配的 UniMol 残差，而不是 raw similarity

UniMol 提供：

- 候选分子的冻结 3D 表示；
- 候选与 partial set 的低容量条件交互；
- 候选结构离群风险；
- 候选属性与需求的匹配；
- 与 H1/H2 边界候选的残差判断。

它不直接提供：

- “这个分子一定出现在该食品中”的结论；
- 未经任务适配的共现概率；
- 独立推翻 occurrence 主干的权限。

### 6.5 训练机制：经验缺失率 + PU-aware

- 从真实 MPC 观测机制估计遮蔽比例，重点覆盖约 75%–95% 缺失。
- 保留少量低缺失率样本作为辅助，而不是主训练分布。
- 完整 profile 内的隐藏分子是正例。
- profile 外候选是未标注样本，不是确定负例。
- 使用 nnPU/SAR 风险或低容量 pairwise PU 排序。

### 6.6 Reviewer：盲化、可拒答、只审查交换

- 不显示 H1/H2/H3、set energy 等带权威暗示的策略名。
- 随机化 A/B 顺序。
- 只判断具体 remove/add swap。
- 没有候选级 evidence 时必须 `ABSTAIN`。
- Reviewer 不得生成候选池外分子。
- `ABSTAIN` 或输出不一致时回退 H1。

### 6.7 Fusion：确定性执行器

- 不再承担第二次自由语义判断。
- 只执行被批准的局部交换。
- 去重、过滤 partial、校验 provenance。
- 始终保持 exact-N。

## 7. UniMol 与多构象取舍

目前不应立即生成多构象文件。

原因：

- 当前失败首先是 UniMol 目标适配错误，而不是构象方差不足。
- raw single-conformer similarity 的 AUC 接近随机，多构象平均无法自动创造共现语义。
- 应先证明“单构象 + task-aligned adapter”在 grouped OOF 中有稳定正增益。
- 之后仅替换表征做单构象/多构象受控对照。

多构象进入主方法的前提：

- 其他模型、掩码、候选和 Reviewer 全部冻结；
- 多个划分方向一致；
- 改善来自 H3 的候选判别或局部 swap，而不是偶然改变评测映射。

## 8. 下一版本准入门槛

任何新模块进入正式 71 条测试前，应满足：

1. 只用训练集 grouped OOF 选择参数。
2. 模块相对 H1 的平均增益为正。
3. 多数折方向一致。
4. 改善样本数大于损害样本数。
5. H2/H3 Oracle 相对 H1 有足够上限；Oracle 无增益时不运行 Reviewer。
6. Reviewer 能实现正的候选增益，而不是只输出高置信解释。
7. exact-N、无重复、无 partial 泄漏全部通过。
8. 不读取功能团评测缓存。
9. 不使用测试集 N 分桶、CID、名称或样本 ID 规则。
10. test 结果只用于冻结版本后的最终比较。

## 9. 文献依据

- Uni-Mol：预训练 3D 分子表示需要下游任务适配。  
  https://mlanthology.org/iclr/2023/zhou2023iclr-unimol/
- Deep Sets：集合函数的置换不变性。  
  https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
- Set Transformer：集合元素交互与集合级表示。  
  https://proceedings.mlr.press/v97/lee19d.html
- nnPU：正例—未标注学习中的非负风险估计。  
  https://proceedings.neurips.cc/paper_files/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html
- SAR PU Learning：正样本选择机制不能被忽略。  
  https://proceedings.mlr.press/v94/bekker18a.html
- SelectiveNet：预测系统应在高风险样本上拒答。  
  https://proceedings.mlr.press/v97/geifman19a.html
- Contextual Submodular Prediction：相关性与集合多样性联合优化。  
  https://proceedings.mlr.press/v28/ross13b.html
- F-measure Bayes optimality：F1 不能由任意可分解代理目标可靠替代。  
  https://www.jmlr.org/papers/v15/waegeman14a.html
- LLM Judge bias：候选顺序和表达形式会影响 Reviewer。  
  https://aclanthology.org/2024.acl-long.511/
- Model-selection overfitting：有限验证集上的反复选择会造成选择偏差。  
  https://www.jmlr.org/papers/v11/cawley10a.html

## 10. 最终判断

历史版本并不是“旧版本全部失败、v10 最接近正确答案”。

更准确的认识是：

- v5/v6 找到了 MFP 的有效结构适配方式，也证明集合功能互补对 MPC 指标有价值。
- v7 找到了 MPC 最可靠的 occurrence 主干、OOF 残差门控和 exact-N 控制。
- v8 证明 retrieval residual 值得保留，同时否定 raw UniMol set compatibility。
- v9 是当前最均衡的正式组合，尤其适合作为下一版的恢复起点。
- v10 提供了通用集合补全、反事实动作和局部搜索的接口，但其训练机制与 Reviewer 权限需要大幅收缩。

下一版应当是：

> v9 强主干 + v5 的有限集合互补思想 + v8 的检索残差 + 任务适配 UniMol + 经验缺失率 PU 学习 + 可拒答 Reviewer + 确定性 Fusion。

它不是把所有历史模块简单相加，而是只保留跨版本出现过正证据的模块，并把仍未验证的 UniMol 与 Reviewer 限定为可退化的残差通道。
