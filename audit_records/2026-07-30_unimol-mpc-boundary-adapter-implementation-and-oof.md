# UniMol MPC 边界适配器：实现与 OOF 准入结论

- 日期：2026-07-30
- 方法版本：`optimized_agent_v11`
- 状态：代码与离线检查完成；未正式运行、未调用 API

## 1. 本轮实现

在 `code/Only-Deepseek/optimized_agent.py` 中实现了
Boundary-Conditional UniMol Residual Adapter：

1. H1 仍由食品条件的频率、共现和检索相关性产生。
2. UniMol 只判断 H1 截断边界上的 `remove B + add A`。
3. 训练边界使用逐食品 leave-one-out，避免检索回当前食品自身。
4. 交换特征包含：
   - frozen UniMol 的 `z_add-z_remove`；
   - 与 partial centroid、H1 centroid 的交互；
   - H1 分数与检索支持差；
   - 分子固有属性需求差；
   - 功能团需求覆盖变化；
   - UniMol 映射状态。
5. scorer 使用标准化后的低容量正则 Logistic Regression。
6. 只有严格提高隐藏成员集合的交换为正类；harmful 和 neutral
   都是负类。
7. grouped five-fold OOF 决定预算 0、1 或 2；未通过则逐项回退 H1。

没有读取 test label、官方 LLM 功能团缓存或正式评测结果。

## 2. 首轮实现偏差及修正

首轮错误地丢弃 neutral swap，并使用类别平衡。这样训练出的
decision threshold 不能表示“严格正收益”，导致：

- budget 1 changed queries：552/568；
- mean hidden-set F1 gain：-0.01049152；
- bootstrap 95% lower bound：-0.01628940；
- wins/losses：72/159；
- positive folds：2/5。

该结果首先证明门控实现错误，不能用于判断 UniMol 本身。

修正为“所有非严格改善均为负类”、保留自然基率并标准化特征后：

- budget 1 changed queries：0/568；
- budget 2 changed queries：0/568；
- mean gain：0；
- structural selected budget：0。

这说明严格概率阈值下模型没有学到可迁移的正交换，而不是再次造成
负收益。安全回退工作正常。

## 3. 同轮对照

食品条件 retrieval residual 仍通过准入：

- selected budget：1；
- changed queries：166/568；
- mean hidden-set F1 gain：+0.00190750；
- bootstrap 95% lower bound：+0.00014320；
- wins/losses：50/25；
- positive folds：3/5。

local attribute complementarity 继续被拒绝：

- budget 1 mean gain：-0.01188606；
- positive folds：0/5。

因此当前 MPC 可部署策略仍是：

`H1 + one grouped-OOF-admitted retrieval residual + zero structural residual`

## 4. 离线完整性

MPC check-only：

- 71/71 exact-N；
- structural budget=0；
- 无 API 调用；
- PASS。

MFP check-only：

- 71 个测试样本完成结构检查；
- UniMol 映射 53617/53617 次分子出现；
- PASS。

## 5. 科学结论

当前结果不能支持“没有任何方法能让 UniMol 对 MPC 有正作用”，但已经
排除了三类接口：

1. raw/global UniMol similarity；
2. pointwise structural candidate ranker；
3. 单构象、线性、边界条件化的严格正收益 swap adapter。

共同瓶颈不是 UniMol 表示缺失，而是 MPC 标签为
`food context -> molecule occurrence`。同一结构在不同食品中的出现概率
主要由食品来源、生化过程和数据记录机制决定；冻结分子表示本身没有
这些条件变量。文献也只支持“任务适配可能有效”，不保证在目标错配和
小样本下产生增益：

- NeurIPS 2022，分子预训练在小数据下游任务中并非稳定有效：
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html
- Pin-Tuning，强调上下文感知轻量适配：
  https://openreview.net/forum?id=859DtlwnAD
- MIPT，强调预训练—下游错配与噪声抑制：
  https://proceedings.mlr.press/v267/chen25cu.html
- 3D-MoLM，通过学习接口连接 frozen UniMol，而非直接使用距离：
  https://openreview.net/pdf?id=xI4yNlkaqh

## 6. 下一步取舍

不应正式运行当前 UniMol structural H3，也不应为了“展示创新点”降低
准入阈值。

若继续研究 UniMol 对 MPC 的正作用，下一项可证伪实验应是独立的
多构象表示对照：

- 冻结 H1、retrieval H2、OOF folds、swap 标签和模型容量；
- 只替换单构象向量为构象集合表示；
- 同样执行严格 grouped OOF；
- 只有出现正 mean gain、正 bootstrap lower bound、wins>losses 和多数
  positive folds 才准入。

如果多构象仍为零或负，应停止把 UniMol 作为 MPC 主增益模块；更合理的
论文叙述是 UniMol 对 MFP 有效，而 MPC 的可靠提升来自食品条件检索和
受控集合决策。这样的任务异质性结论比强行让一个模块覆盖两个任务更
可信。
