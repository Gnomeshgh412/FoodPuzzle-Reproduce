# UniMol 在 MPC 中的边界条件化重新设计

- 日期：2026-07-30
- 状态：设计冻结，尚未正式运行
- 目标：验证 UniMol 能否通过通用、任务条件化的接口对 MPC 产生训练内正收益

## 1. 当前负结果不代表 UniMol 对 MPC 无效

此前被否定的是：

- raw UniMol cosine 全局排序；
- candidate 与 partial centroid 的简单相似度；
- UniMol/属性集合能量重写完整 H1；
- pointwise structural ranker 独立预测食品分子共现。

这些接口都隐含了“结构相似即可推出食品共现”，但 UniMol 的预训练目标并不包含食品—分子关系。

分子预训练并不保证所有小数据下游任务受益。NeurIPS 2022 的系统研究发现，自监督分子预训练在多种小数据设置中没有稳定显著优势：

https://proceedings.neurips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html

## 2. 文献支持的正确方向

### 2.1 轻量任务适配与上下文条件化

Pin-Tuning 使用上下文感知的轻量 adapter 适配预训练分子编码器，避免小样本下全量微调：

https://openreview.net/forum?id=859DtlwnAD

MIPT 进一步强调预训练表示和下游目标之间存在噪声与错配，需要任务特定适配和噪声抑制：

https://proceedings.mlr.press/v267/chen25cu.html

### 2.2 冻结 UniMol 需要学习投影接口

3D-MoLM 没有直接使用 frozen UniMol 距离，而是学习 Q-Former，将3D分子表示投影到具体下游语义空间：

https://openreview.net/pdf?id=xI4yNlkaqh

### 2.3 多构象是第二阶段，不是目标错配的补丁

MARCEL 证明构象集合能够改善多种分子与反应属性任务，但没有证明其能够直接预测食品共现：

https://openreview.net/forum?id=NSDszJ2uIV

因此应先证明单构象的任务条件化接口有效，再独立评估多构象。

## 3. 新 H3：Boundary-Conditional UniMol Residual Adapter

UniMol 不再生成完整集合，只评估 H1 边界交换：

`remove B from H1, add A from boundary-out`

训练标签直接表示该交换是否改善隐藏集合，而不是候选是否一般性地“像”partial。

### 3.1 训练样本

对每个 MPC train query：

1. 用食品相关性 H1 生成 exact-N。
2. 在 H1 尾部选择可移除候选。
3. 在 H1 外部边界选择可加入候选。
4. 根据 train hidden set 标记交换为 beneficial、harmful 或 neutral。
5. 只使用边界动作，不构造完整集合伪负例。

### 3.2 上下文条件化特征

- `z_add - z_remove`；
- 与 partial-set UniMol centroid 的条件交互；
- 与 H1-set UniMol centroid 的条件交互；
- add/remove 的食品相关性 margin；
- retrieval residual 差；
- molecule-intrinsic 属性需求差；
- functional-demand residual 差；
- 结构离群程度和可映射性。

UniMol 表示冻结，使用降维后的单构象 embedding；swap scorer 为低容量正则化模型。

### 3.3 边际属性辅助信号

属性目标不是“最终集合是否含常见功能团”，而是：

> 该交换是否补充当前集合尚缺的 molecule-intrinsic 属性。

这避免大型隐藏集合中常见功能团概率全部饱和。属性只来自 FlavorDB 分子固有信息，不读取官方 LLM 功能团缓存。

精确成员改善是主目标；属性改善只处理成员变化中性或提供辅助特征，不能覆盖明显有害的成员交换。

## 4. 推理权限

- H1 核心锁定。
- H3 只考虑 H1 截断边界内外。
- 每次选择 swap scorer 最高且效用为正的交换。
- 最大预算由 grouped OOF 在0、1、2中选择。
- 预算0必须逐项返回 H1。
- Reviewer 只能审核已经通过 OOF 的交换，允许 `ABSTAIN`。

## 5. 准入协议

只使用 MPC train grouped OOF：

- 平均 hidden-molecule set F1 增益为正；
- paired bootstrap 95% 下界为正；
- wins > losses；
- 多数 fold 为正；
- exact-N 全部通过；
- 不使用 test 功能团指标进行选择。

对照：

1. H1；
2. raw/pointwise UniMol residual；
3. boundary-conditional UniMol swap adapter；
4. boundary-conditional adapter + marginal attribute signal。

只有3或4通过准入，UniMol 才进入正式 MPC。

## 6. 多构象决策

本轮继续使用现有 `unimol_embeddings.npz`。

只有单构象 boundary adapter 通过后，才生成独立多构象表示，并冻结：

- H1；
- retrieval H2；
- swap训练数据；
- OOF折；
- Reviewer；
- 所有预算候选。

这样才能把增益归因给多构象，而不是其他同时变化的模块。

## 7. 失败时的科学结论

如果新的 context-conditioned swap adapter 仍没有正 OOF 增益，应得出的结论是：

> 在当前样本规模、partial观测量和数据库标签条件下，UniMol结构表示不能稳定改善 MPC 的食品条件集合补全。

此时不能为创新点强行启用 UniMol；UniMol 可以继续作为 MFP 的有效组成，或转入独立的多构象研究，而不是污染 MPC 主结果。
