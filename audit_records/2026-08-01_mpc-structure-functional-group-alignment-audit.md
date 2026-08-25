# MPC 冻结结构官能团感知层目标对齐审计

> **重要更正（2026-08-01）**：本记录的分子级结构–cache 对齐结果有效；但动作级 `structure_gain` 直接使用了 held-out 查询的 Gold 官能团，因此只能视为不可部署 Oracle，不能作为推理策略。第 5–10 节关于合法执行器和准入的结论已撤销。完整更正与无泄漏重算见 `2026-08-01_mpc-structure-route-leakage-correction-and-legal-audit.md`。原记录保留用于追踪审计过程，不得引用其中动作级收益作为方法结果。

## 1. 审计目的与边界

本轮只检验一个问题：从 SMILES / 分子图确定官能团的通用、确定性结构感知层，是否比当前 FlavorDB 字符串解析代理更接近正式 MPC 对预测分子的 LLM 官能团表示。

本轮严格遵守以下边界：

- 不修改 Optimized-Agent；
- 不调用任何模型 API；
- 不写入或覆盖正式结果；
- 不使用 71 条正式测试集设计 SMARTS；
- 不使用 functional-group evaluation cache 训练模型或修改规则；
- SMARTS 和程序化结构规则在读取 cache 评分结果前一次性冻结；
- cache 只作为不可部署的一次性诊断 Oracle；
- 结果无论好坏都不进行第二轮规则调整。

因此，本轮结论只能是“结构代理是否值得进入下一阶段”，不能宣称已经提高正式 MPC F1。

## 2. 环境与数据可行性

- 审计环境：项目本地 `.venv-chem-audit`；
- RDKit：`2025.03.6`；
- FlavorDB `molecules + molecules_all` 唯一规范名称/SMILES：25,197；
- RDKit 成功解析：25,197 / 25,197；
- SMILES 解析失败：0；
- H1 结构完整覆盖：568 / 568 个训练查询。

这排除了“结构信息覆盖不足”这一工程性阻碍。

## 3. 冻结的官能团定义

正式词表共有 53 个标签。本轮用 45 个直接语义 SMARTS 和 7 个确定性分子图规则覆盖其中 52 个标签；`derivative` 没有唯一、可复核的结构定义，因此预先排除。

### 3.1 直接 SMARTS 标签

| 标签 | 冻结 SMARTS |
|---|---|
| thiocarboxylic | `[CX3](=[SX1])[OX2H,OX1-,SX2H,SX1-]` |
| sulfone | `[SX4](=[OX1])(=[OX1])([#6])[#6]` |
| hydroxy | `[OX2H]` |
| sulfonic | `[SX4](=[OX1])(=[OX1])[OX2H,OX1-]` |
| alcohol | `[OX2H]-[C;!$(C=O)]` |
| ketone | `[#6][CX3](=[OX1])[#6]` |
| hydroxyhetarene | `[OX2H]-[a;!#6]` |
| amine | `[NX3;!$(N-C(=O));!$(N-S(=O)=O)]` |
| trialkylamine | `[NX3H0]([C])([C])[C]` |
| carboxylic | `[CX3](=[OX1])[OX2H1,OX1-]` |
| alkyne | `[CX2]#[CX2]` |
| ketene | `[CX2]=[CX2]=[OX1]` |
| anhydride | `[CX3](=[OX1])[OX2][CX3](=[OX1])` |
| acetal | `[CX4]([OX2][#6])([OX2][#6])` |
| amide | `[CX3](=[OX1])[NX3]` |
| carbonitrile | `[CX2]#[NX1]` |
| (alkylamine) | `[NX3H2]-[C;!a]` |
| imide, | `[NX3]([CX3](=[OX1]))[CX3](=[OX1])` |
| enol | `[CX3]=[CX3][OX2H]` |
| halide | `[F,Cl,Br,I]` |
| phenol | `[OX2H]-c` |
| sulfoxide | `[SX3](=[OX1])([#6])[#6]` |
| aldehyde | `[CX3H1](=[OX1])[#6]` |
| thioether | `[SX2]([#6])[#6]` |
| hydroperoxide | `[OX2H]-[OX2]` |
| ester | `[CX3](=[OX1])[OX2][#6]` |
| isothiocyanate | `[NX2]=[CX2]=[SX1]` |
| alpha-aminoacid | `[NX3]-[CX4H1]-[CX3](=[OX1])[OX2H1,OX1-]` |
| dialkylamine | `[NX3H1]([C])[C]` |
| thiol | `[SX2H]` |
| ammonium | `[NX4+]` |
| arylthiol | `[SX2H]-c` |
| thioacetal | `[CX4]([SX2][#6])([SX2][#6])` |
| alpha-hydroxyacid | `[OX2H]-[CX4H1]-[CX3](=[OX1])[OX2H1,OX1-]` |
| acid | `[CX3](=[OX1])[OX2H1,OX1-]` |
| sulfanyl | `[SX2H]` |
| alkylthiol | `[SX2H]-[C;!a]` |
| alkene | `[CX3]=[CX3]` |
| ether | `[OD2]([#6])[#6]` |
| sulfenic | `[#6][SX2][OX2H]` |
| carbonyl | `[CX3]=[OX1]` |
| nitrite | `[OX2]-[NX2]=[OX1]` |
| halogen | `[F,Cl,Br,I]` |
| chloride | `[Cl]` |
| oxo(het)arene | `[a]-[CX3]=[OX1]` |

### 3.2 程序化结构标签

- `cation`：至少一个原子具有正形式电荷；
- `salt`：分子具有多个 disconnected fragments，或同时存在正负形式电荷；
- `aromatic`：存在芳香原子；
- `aryl`：存在芳香原子；
- `aliphatic`：存在非芳香碳原子；
- `aliphatic/aromatic`：同时具有芳香原子和非芳香碳；
- `heterocyclic`：至少一个环包含非碳原子。

这些规则是对正式标签名称的直接结构语义解释，不包含食物名称、查询 ID、Gold 分子、出现频率、检索邻居或 cache 输出。

## 4. 分子级一次性对齐结果

结构表示与 DeepSeek 评测 cache 共同覆盖 1,141 个分子：

| 指标 | DB 字符串代理 | 冻结结构代理 |
|---|---:|---:|
| 完全一致率 | 8.15% | 4.73% |
| 平均 Jaccard | 0.348690 | **0.453050** |
| 中位 Jaccard | 0.333333 | **0.428571** |

完全一致率下降并不与平均重叠改善矛盾。结构规则会系统输出 `aromatic`、`aliphatic`、`carbonyl` 等可同时成立的多标签，而 LLM cache 的标签详略不完全一致，因此严格集合相等更难；但平均及中位 Jaccard 均明显提高。对最终集合 F1，标签重叠程度比“整组完全一致”更相关。

## 5. 动作级目标对齐结果

按照 v14/v15 相同的完整谱聚类五折 OOF，重新生成：

- 568 个训练查询；
- 2,105 个 Scientist Top-5 原子交换动作；
- 其中 1,953 个动作同时具有完整 cache 和结构覆盖，占 92.78%；
- cache 完整覆盖 H1 的查询为 511 / 568；
- 结构完整覆盖 H1 的查询为 568 / 568。

为保证公平，以下 DB 和结构代理均只比较同一批 1,953 个动作：

| 指标 | DB 字符串代理 | 冻结结构代理 | 变化 |
|---|---:|---:|---:|
| 与 formal-audit action gain 的 Pearson 相关 | 0.389503 | **0.622344** | **+0.232841** |
| 代理正动作数 | 506 | 357 | -149 |
| 其中正式正动作 | 255 | 251 | -4 |
| 代理正动作 precision | 50.40% | **70.31%** | **+19.91 pct** |
| 正动作识别 ROC-AUC | 0.687217 | **0.723759** | **+0.036541** |
| 正动作识别 PR-AUC | 0.525519 | **0.628692** | **+0.103173** |
| 正式正动作率 | 21.15% | 21.15% | — |

结构代理保留了几乎相同数量的正式正动作（251 vs 255），同时过滤掉大量假阳性动作。这正对应此前 MPC 的最大瓶颈：Scientist Top-5 中有正确动作，但 Reviewer / Executor 无法可靠识别哪些交换真的改善正式官能团 F1。

### 5.1 符号混淆矩阵

DB 字符串代理：

| DB \\ Formal-audit | 负 | 正 | 平 |
|---|---:|---:|---:|
| 负 | 310 | 119 | 552 |
| 正 | 98 | 255 | 153 |
| 平 | 43 | 39 | 384 |

冻结结构代理：

| Structure \\ Formal-audit | 负 | 正 | 平 |
|---|---:|---:|---:|
| 负 | 344 | 109 | 276 |
| 正 | **29** | **251** | 77 |
| 平 | 78 | 53 | 736 |

最关键变化是：代理正动作中的正式负动作由 98 降到 29，下降 70.41%。这比继续调 Reviewer 阈值更接近实际错误来源。

## 6. 是否通过预设目标对齐门槛

上一轮预设的最低门槛是：

1. action gain 相关性明显高于约 `0.382`；
2. proxy-positive 的正式正收益率明显高于约 `47.99%`；
3. 不读取 cache 作为模型特征；
4. 结构覆盖不能成为新瓶颈。

本轮结果：

| 门槛 | 结果 |
|---|---|
| action gain correlation | 通过：0.622344 |
| positive-action precision | 通过：70.31% |
| cache leakage | 通过：cache 只用于一次性事后诊断 |
| 结构覆盖 | 通过：25,197/25,197 SMILES、568/568 H1 |

因此，**冻结结构官能团感知层通过“值得进入下一阶段”的目标对齐门槛**。

它尚未通过“接入正式 Optimized-Agent”的准入门槛，因为本轮没有评估完整的查询级执行策略、固定阈值、宏平均 F1 增益及 paired bootstrap 下界。

## 7. 为什么这不是恢复 UniMol

本轮使用的是确定性 2D 分子图/SMARTS 官能团识别，不是 UniMol embedding，也不依赖三维构象。

- UniMol回答的是“两个分子的连续结构/性质表征是否相似”；
- 当前感知层回答的是“某个分子是否明确含有固定离散官能团”；
- MPC 正式指标直接定义在后者上。

所以结果支持的是“让 MPC 使用任务指标相容的结构化学工具”，而不是“UniMol 对 MPC 突然有效”。MFP 继续使用 UniMol；MPC 继续不把 UniMol放回主链。

## 8. 文献支撑

Ertl 2017 提出了从分子图按通用规则自动识别官能团的方法，说明结构官能团识别具有独立于 FoodPuzzle 的化学信息学基础：

<https://doi.org/10.1186/s13321-017-0225-z>

RDKit 官方提供 SMARTS 子结构匹配与官能团层级接口，本轮采用的技术机制与该通用路线一致：

- <https://www.rdkit.org/docs/source/rdkit.Chem.FunctionalGroups.html>
- <https://www.rdkit.org/docs/Cookbook.html#functional-group-with-smarts-queries>

目标对齐结果也符合 objective mismatch 的一般结论：下游性能取决于代理目标是否对下游决策有用，而不只是代理自身是否容易优化：

<https://proceedings.mlr.press/v120/lambert20a.html>

## 9. 必须保留的限制

1. cache 只覆盖 511/568 个 H1、1,953/2,105 个动作；动作对齐结果可能存在覆盖选择偏差。
2. cache 是 LLM 评测器的历史输出，不是化学真值；提高对 cache 的一致性不等同于提高科学真实性。
3. 规则由正式 53 类标签的文字语义构造，虽然没有根据 cache 结果迭代，但仍与评测词表天然相关；这属于 metric-aligned method，而非完全任务无关的分子表征。
4. 同一训练 OOF 已被此前多个版本自适应观察；本轮只能作机制筛选，不能作最终泛化声明。
5. 当前尚未证明“按结构代理选择动作”在查询级宏 F1 上有正收益。

## 10. 下一步唯一建议

下一步只实现一个独立的**离线执行策略审计**，仍不修改正式 Agent：

1. 冻结本轮 52 类规则，不允许再改；
2. 在 Scientist Top-5 内用结构 ΔF1 排序；
3. 将 `KEEP_H1` 作为显式拒绝动作；
4. 阈值只能在训练侧外层折以外的数据选择；
5. 在完整谱聚类 stacked OOF 下报告查询级宏 F1 增益、wins/losses、paired bootstrap 下界和 cache 覆盖敏感性；
6. 预先要求相对 v14 的收益下界大于 0，且正式负动作数显著少于 59；
7. 未通过则停止结构感知层，不调 SMARTS、不增加 Reviewer；
8. 通过后才提交接入现有 Scientist–Reviewer 的具体设计，并再次请求批准。

本轮不授权上述执行策略审计自动开始。
