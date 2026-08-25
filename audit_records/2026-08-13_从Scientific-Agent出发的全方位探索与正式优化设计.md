# 从 Scientific Agent 出发的全方位探索与正式优化设计

> 日期：2026-08-13  
> 状态：强基线探索终稿 + 创新机制探索阶段报告  
> 研究对象：`code/Only-Deepseek/scientific_agent.py`、`optimized_agent.py` 及既有 14 轮优化资产  
> 边界：未运行正式 test，未将正式 test 内容或结果用于探索；未修改任何 Agent 源码；临时代码与中间结果仅位于 `/tmp`。

> 追加阶段（2026-08-13）：前述 M2/P2/P4 已重新定位为强基线。本报告从第 12 节继续记录创新机制探索，目标不是给常规检索/融合换名字，而是实际检验正负证据、观测过程、双任务一致性、条件结构关系和集合级解码是否产生超越强基线的独立收益。

## 0. 结论先行

本轮不是对 `optimized_agent.py` 继续叠模块，而是把历史试错拆成可证伪机制，再从 `scientific_agent.py` 重新设计。当前得到两条明显正向、机理清楚、值得进入正式实验的主线：

1. **MFP：实例检索之后做宏类别证据聚合，再在获胜类别内输出具体食物。**
   - 用官方近似类别映射复核后，单个训练实例 BM25 的 20 组完整-profile grouped OOF 平均宏类别准确率为 **0.37452**。
   - 固定的“单实例证据 + 类别内 Top-8 实例支持”聚合为 **0.43369**，提升 **+0.05917**；20/20 个重复分组均为正。
   - 冻结 dev 代理准确率由 **0.40845（29/71）** 提到 **0.43662（31/71）**。另一个更偏聚合的版本可到 0.45070，但不是 OOF 最优，不应据 dev 追选。
   - 这比继续做开放式 LLM 食物生成、分视角多调用、成对/三元分子共现或大规模监督适配更可靠。
2. **MPC：把任务改写为“部分观测检索完整训练原型 → 标签传播”，并与 H1、ICL 做三源条件化候选效用排序。**
   - 新的跨视角、交叉归一化 soft-BM25 标签传播，grouped OOF 具体分子宏/micro 约 **0.72113/0.79915**；dev 为 **0.73810/0.79528**，全部 exact-N。
   - H1 + ICL + 新检索源的 HistGradientBoosting exact-N 解码，在全候选池 grouped OOF 相对 H1 提升约 **+0.04501**，dev **+0.04375**；加入冻结 UniMol 后 OOF **+0.04817**、dev **+0.03735**。固定 depth=3 时，无 UniMol 的 0.5N/1N 池为 **+0.05424/+0.05279**；含 UniMol 为 **+0.06048/+0.05333**，其中含 UniMol 1N 的 bootstrap 95% CI **[+0.04328,+0.06366]**。
   - 新 R 单独已强于旧三源模型；复杂融合不再默认优于 R，必须作为并列主竞争方案。

因此，正式优化不应复刻当前 **8583 行**的单体 `optimized_agent.py`。建议从 Scientist Agent 的数据与调用骨架出发，重建两个相互独立、可消融的任务管线：

- MFP：`legal query → grouped-BM25 → category evidence aggregation → concrete-food decoder → optional bounded Reviewer`；
- MPC：`legal query → H1 / ICL / prototype propagation → candidate utility → exact-N decoder → optional triple-consensus core`。

## 1. 研究问题与口径

### 1.1 第一性原理分解

MFP 的形式不是“精确识别训练集中同一个食物实体”，而是：

\[
X_{molecules}\rightarrow food\ name\rightarrow macro\ category
\]

当前检索器直接取最高分训练食物，相当于只用一个嘈杂实例代表整个类别。若多个同类实例都与查询相似，它们提供的是重复但独立的类别证据。合理决策应比较类别的稳健聚合证据，而不是只比较全库最大值。

MPC 的形式是已知集合、目标食物和精确缺失数条件下的集合补全：

\[
(food, X_{partial}, N)\rightarrow Y_{missing},\quad |Y|=N
\]

数据中 partial 是完整食品谱的稀疏观测。最近的完整训练 profile 因而天然是标签传播原型；H1、ICL、原型检索各自含不同误差。真正的问题不是再生成更多集合，而是估计每个候选分子在当前查询中的条件效用，并做 exact-N 约束解码。

### 1.2 指标标记

报告严格区分三类数字：

- **grouped OOF 具体分子指标**：MPC 的本地真实缺失分子 F1；可用于机制筛选，但不是正式官能团指标。
- **本地 MFP 宏类别代理**：用冻结 FlavorDB 食物—类别映射评估；与官方 LLM 类别映射目标相近，但仍需正式 evaluator 确认。
- **正式指标**：MFP 的官方近似 LLM category mapping accuracy；MPC 的官方近似 LLM functional-group F1。只有已存在正式审计缓存的结果才称正式结果。

## 2. 协议审计：为什么必须从原始版本重建

### 2.1 `scientific_agent.py` 的可保留骨架

- 输入/输出 JSONL、resume、元数据账本和错误处理清楚；
- MFP/MPC Scientist—Reviewer 调用边界明确；
- MPC prompt 暴露的当前查询字段合法；
- 训练 demonstrations 含训练标签，属于正常 ICL；
- 输出解析器可复用。

### 2.2 必须修复的协议/目标问题

1. MFP `build_evidence_blocks` 读取当前行 `actual_food` 来遮蔽证据。即便意图是防泄漏，推理程序仍依赖当前 gold；这不是可部署的因果输入路径。应改为与当前答案无关的预清洗，或在正式主方案中完全移除该证据路由。
2. MFP Reviewer 被允许返回 “food source or category”，且不被强制限制在 Scientist 三个具体食物候选内。输出空间与任务协议不一致。
3. MPC 后处理只去重、去 partial、超长截断；短于 N 时不补齐。历史原始版本有 28/71 条 exact-N 失败。
4. 两个任务都把 LLM 同时承担候选召回和最终选择，导致无法识别瓶颈。

### 2.3 `optimized_agent.py` 不适合作为继续增量开发的基底

- 单文件包含大量已退役版本、适配器、Bank、门控和 ledger，因果边界难审计；
- MFP 当前类别适配器用按类别轮转的五折选择融合权重，没有隔离重复完整 profile；
- MFP Chamfer 权重源码注释明确为 reconstructed dev 选择；
- MPC 当前源码与历史最佳正式 v12 不是同一个被冻结实现；
- 多个离线目标（具体分子、数据库官能团、缓存官能团、Reviewer 机会）被先后叠加，造成目标漂移。

本轮因此只把现有实现当作“机制和失败档案”，不把它当作新版本的类继承树。

## 3. 数据结构与污染风险

- MFP train/dev 为 567/71；train 只有 376 个唯一完整 profile，dev 53 个。
- MPC train/dev 为 568/71；train 377 个唯一完整 profile，dev 53 个。
- 两个任务的 dev 均有 **28/71** 行的完整 profile 在 train 中出现。
- train 内最大重复 profile 约 62–63 行。

因此随机行 OOF 会跨折记忆同一 profile。本轮主筛选均使用完整 profile 分组；对最强 MFP 机制又做了 20 个分组种子。dev 只作冻结方向检查，不作正式超参数选择。

还检查了 UniMol 资产覆盖：MFP/MPC train/dev 的查询分子均为 **100% 向量覆盖**。UniMol 效果有限不能归因于 OOV，而应归因于结构相似与食品归属/条件缺失关系之间的任务差距。

## 4. MFP 全方向探索

### 4.1 宽筛范围

统一 grouped OOF 框架内至少比较了 61 个静态评分/融合方法，并额外探索：

- overlap、IDF overlap、Jaccard、Dice、query/candidate coverage、F0.5/F1/F2；
- BM25 的长度归一与参数变体；
- 候选冲突/额外分子惩罚；
- 稀有分子子查询与高频 stop 策略；
- 稀有 pair/triple IDF；
- 冻结 UniMol mean/spread 集合表示；
- 线性融合、RRF、多检索器候选并集；
- 确定性 query dropout；
- masked-profile 自监督 Logistic、pairwise 与 HistGB；
- 类别 max/mean/top-k mean/log-sum-exp/rank-decay 聚合；
- 标签无关的检索源路由。

一个早期 dropout 实现曾产生大于 1 的非法统计，已定位为累计变量复用错误并整组废弃；报告只保留修正后的结果。

### 4.2 单实例检索与候选召回

代表性结果：

| 方法 | grouped OOF Top1 | dev Top1 | OOF Top5 类别 oracle | dev Top5 类别 oracle |
|---|---:|---:|---:|---:|
| BM25 `b=0.5` | 0.38448 | 0.40845 | 0.67549 | 0.73239 |
| UniMol mean+spread | 约 0.337 | 低于 BM25 | — | — |
| masked-profile pairwise | 0.35802 | 0.39437 | — | — |
| pair IDF | 0.25926 | 0.22535 | — | — |
| triple IDF | 0.21869 | 0.16901 | — | — |
| 修正 dropout keep=.8/reps=8 | 0.38977 | 0.40845 | — | — |

候选并集的价值比硬融合更稳定：

| 候选集合 | grouped OOF 类别 oracle | dev 类别 oracle | 平均候选数 |
|---|---:|---:|---:|
| BM25 Top5 | 0.67549 | 0.73239 | 5 |
| BM25 Top5 + UniMol Top2 | **0.71076** | **0.78873** | 小于等于 7 |
| 上式 + dropout Top2 | 0.72840 | 0.78873 | 6.44 |
| 五专家并集 | 0.71605 | 0.70423 | 约 7 |

结论：UniMol 适合增加候选覆盖，不适合未经条件化就线性主导分数；增加更多弱专家会稀释候选质量。

### 4.3 新发现：类别证据聚合

类别聚合将每个类别内相似度最高的若干训练实例作为“支持证据”，再与单实例分数融合。它只使用训练标签和查询分子，最终仍输出该类别内的一个具体食物，符合输出协议。

20 个完整-profile 分组种子结果（按官方近似类别映射复核；旧近似映射只错分了 Mustard，方向和增益结论不变）：

| 决策 | 平均 OOF | 最差 | 最好 | dev |
|---|---:|---:|---:|---:|
| 单实例 BM25 | 0.37452 | 0.36155 | 0.38801 | 0.40845 |
| Top-8 类证据，融合权重 .4 | **0.43369** | **0.41799** | 0.45679 | **0.43662** |
| 仅在最高分并列时启用同一聚合 | 0.40926 | 0.39506 | 0.42857 | **0.46479** |

Top-3、Top-5、Top-8 多种权重在 20/20 个分组种子都优于各自的单实例基线。按 OOF 冻结应选 Top-8 / 0.4，而不能根据 dev 追选 0.6 或其他高 dev 组合。

在一个固定分组上，Top-8 / 0.4 的配对增益为 **+0.05996**，bootstrap 95% CI **[+0.02822,+0.09171]**，59 改对 / 25 改错，五折增益均为正。按谱长度分层，`<=50`、`51–100`、`>100` 分子的增益分别约 +0.0773、+0.0226、+0.0765，不是单一长度层驱动。

重复证据压力测试：以具体食物或 `(profile,food)` 去重不改变当前结果，因为 Top-8 中没有重复具体食物；只按完整 profile 去重后，10 个分组种子的 OOF仍约 **0.41905** 对单实例 **0.37266**，dev 0.42254 对 0.40845。说明主机制成立，但 raw 版本的部分额外增益确实来自不同食物共享同一谱所提供的类别证据。正式实验需同时报告 raw 与 profile-dedup。

#### 泛化边界与选择性聚合

冻结 dev 的总增益不能被当成均匀泛化：

- 28 条 train 中已有完整 profile 的 dev 行：0.2143 → 0.3214，5 改对/2 改错；
- 43 条未见完整 profile：0.5349 → 0.5116，1 改对/2 改错；
- 训练频次 >30 的常见 gold 类：0.4727 → 0.5273；频次 6–30 的类：0.1875 → 0.1250。

这与 grouped OOF 的五折正向并不矛盾，但说明 dev 的收益偏向重复/常见分布，M2 尚不能宣称解决真正新 profile 泛化。

进一步发现 BM25 大量最高分并列：固定 grouped OOF 中 200/567 条并列，其基线准确率仅 0.160，类别聚合为 0.250；唯一最高分的 367 条为 0.5014 → 0.5450。dev 的并列层 31/71 为 0.2258 → 0.3548，而唯一最高分层 40/71 为 0.550 → 0.500。因此增加一个无需训练、只在 Top 分数并列时启用聚合的保守变体：OOF **0.41270**、dev **0.46479**。它低于全量聚合的 OOF，但在 dev 更稳，且有明确的失效针对性；应作为正式预注册的 M2b，而非根据 dev 替换 M2。

沿这一发现继续扫标准化 `Top1−Top2` 间隔门控：阈值 0.2 时在一个固定 grouped OOF 上以 72.1% 覆盖率执行聚合，准确率 **0.43563**，略高于全量聚合 0.43034；dev 为 **0.45070**。再做 20 个分组种子，单实例/全量聚合/间隔门控均值为 **0.37390/0.43404/0.43589**，门控相对基线 20/20 为正，但仅 10/20 次超过全量聚合。它说明不确定度路由值得保留，却没有证据取代更简单的全量聚合；阈值仍须 nested OOF 后才可准入。

加入小权重 UniMol residual 后，平均 OOF 没有超过无 UniMol 的 0.43369；因此它不进入 MFP 主分数，只保留候选扩展消融。

### 4.4 LLM 历史行为审计

| 方向 | 结果 | 机制判断 |
|---|---|---|
| 固定三候选内 Reviewer | 第一轮相对固定 Top1 +7 净正确；第二轮相对固定 Top1 +2/71，但不稳定 | 封闭选择有潜力 |
| UniMol 独占 Reviewer | 相对无 UniMol仅 +2/71、+1/71，bootstrap 下界均不大于 0 | 不能单独准入 |
| 开放生成具体食物 | 具体食物 0/71；宏类别 oracle -0.0845 | 开放召回失败 |
| 低熵 10 分子开放起点 | 具体食物仍 0；宏类别 oracle继续下降 | 不是上下文长度问题 |
| Top10 检索给 Scientist | Top3 oracle +0；Reviewer +0 | 更多上下文无效 |
| 三次分视角调用 | Top3 oracle -0.15493 | 视角拆分不产生互补 |
| 九候选硬去重 | Top3 oracle -0.01408；Reviewer -0.02817 | 多样性不等于质量 |
| 网页食物词证据桥 | OOF宏类别覆盖 -0.06526 | 文本共现不等于归属关系 |

本轮计划的新 LLM 机制包含 direct、support/conflict、elimination、pairwise tournament、counterfactual，并设计前后顺序反转检查。第一次 240 次调用全部因沙箱 DNS 失败；申请外部调用时，安全审查指出会把仓库衍生的分子和候选 dossier 发给外部 DeepSeek，因缺少用户对外部数据传输的明确知情授权而拒绝。该失败不计为模型实验结果，也没有绕过安全限制。

由历史证据可得：未来若做 LLM 实验，只应在高召回、具体食物、封闭候选集内做可拒绝选择；不再开放发明候选，不让 Reviewer直接输出宏类别。

## 5. MPC 全方向探索

### 5.1 宽筛范围

225 个检索/传播配置覆盖：

- 全局缺失分子频率；
- partial 对 source partial / source full profile；
- overlap、IDF、Jaccard、query coverage、BM25；
- hard Top1/3/5/10/20/50；
- source missing 与 source full 的标签传播；
- softmax temperature 0.1/0.5/1/2/5；
- global + retrieval 混合；
- target-food token 检索；
- 与历史 H1、ICL 的两源/三源 consensus、RRF、候选效用模型；
- Logistic / HistGB、UniMol 特征、硬核心、查询 gate 与 exact-N 解码。

### 5.2 原型检索传播

最佳低复杂度检索器之一是跨视角 `soft full-profile BM25`：查询 partial 与训练完整 profile 匹配，**IDF 按训练 partial 分布估计、长度基准也取训练 partial**，再 soft 权重传播训练缺失分子标签。其最佳宽筛配置为 `b=.25, temperature=1`。这不是标准同域 BM25，报告前期称“full-profile BM25”不够准确，现统一改名“partial-stat/full-doc cross-normalized BM25”。

| 方法 | OOF macro | OOF micro | dev macro | dev micro | exact-N |
|---|---:|---:|---:|---:|---:|
| 全局频率 | 明显更低 | — | — | — | 可补齐 |
| cross-normalized BM25 `b=.25,T=1` | **0.72113** | **0.79915** | **0.73810** | **0.79528** | 568/568；71/71 |
| temperature=.5 | 0.70239 | 0.78683 | 0.73888 | 0.78233 | 全部 |
| target-food token retrieval | 0.3853 | 0.4653 | 较差 | 较差 | — |

2×2 IDF/长度基准消融表明：partial-IDF 是关键；`partial-IDF/partial-avg` OOF 0.70192（旧 b=.75）显著高于 `full-IDF/full-avg` 0.65844，调到 b=.25 后达 0.72113。它的含义是：候选完整谱可以很长，但查询词的辨识度必须由实际可观测的 partial 过程定义。检索候选 gold recall 随训练近邻数上升：Top1/3/5/10/20/50 为 0.617/0.747/0.792/0.850/0.890/0.940。

新的条件参数启发实验显示，固定 `b=.25,T=1` 仍是最强单配置；若按已知 N 分层选择 `(b,T)`，宽筛 OOF 可由 **0.72113** 到 **0.72337**，dev 由 **0.73810** 到 **0.75105**。但每层领先第二名只有约 0.0019–0.0097，而且选择与报告使用了同一 OOF，属于乐观上界。它只进入“嵌套选择候选”，不替代固定 P2。

### 5.3 三源候选效用

定义：H=历史 H1，I=食品条件 ICL，R=新原型传播。修正版 R 的全尾部候选量很大，来源精度必须按查询条件看，不能再用一个无条件均值概括。例如 HIR 三源一致在 `N=21–100, partial=10–15` 时精度 0.9914，在 `N=4–20, partial=3–9` 时仅 0.2188；同层的 R-only 精度通常低于 0.04。R 的主要价值是强排序和向 H/I 候选提供独立证据，不是把 R 尾部候选全部视为可信。

| exact-N 解码器 | OOF macro | OOF micro | 相对 H1 macro | 胜/负 | dev 相对 H1 |
|---|---:|---:|---:|---:|---:|
| H1 | 0.66457 | 0.77487 | — | — | — |
| R 单独 | **0.72026** | **0.79860** | **+0.05569** | 211/127 | +0.04412 |
| 三源简单 consensus | 0.70214 | 0.78811 | +0.03757 | 153/122 | +0.04244 |
| 三源 Logistic C=.1 | 0.70112 | 0.79934 | +0.03655 | 154/42 | +0.05169 |
| 三源 HistGB | 0.70958 | 0.80246 | +0.04501 | 188/62 | +0.04375 |
| 三源 HistGB + UniMol | **0.71274** | **0.80415** | **+0.04817** | 199/51 | +0.03735 |

所有上述组合的五个外层折为正。强 R 单独在宏指标上已超过三源 HistGB；HistGB主要提高 micro 并减少 R 的部分错误。正式实验不能只跑融合模型，必须把 R 单独列为共同主方案。

#### 候选池压缩

R 的尾部几乎覆盖全部候选，但 R-only 精度极低。把候选池截为 `H ∪ I ∪ R[:kN]` 后，固定含 UniMol 的 HistGB 得到：

| R 截断 | 平均候选池 | grouped OOF 相对 H1 | 95% CI/折稳定性 | dev 相对 H1 |
|---|---:|---:|---|---:|
| 0.5N | 约 92 | **+0.06048** | 五折正；一折 +0.1563 | **+0.05038** |
| 1N | 约 100 | +0.05333 | **[+0.04328,+0.06366]**，五折正 | +0.04505 |
| 2N | 约 164 | +0.05621 | 五折正 | +0.04562 |
| 3N | 约 235 | +0.05340 | **[+0.04368,+0.06362]**，五折正 | +0.04612 |
| 5N | 约 382 | +0.05298 | 五折正 | +0.03828 |
| 全池 | 约 1500 | +0.05326 | 五折正 | +0.03486 |

修正 R 后，池压缩相对全池的额外收益变成温和但一致的计算—统计优势，而不是旧结果所显示的巨大跃升。固定 depth=3 时 0.5N 在 OOF 和 dev 都最好；进一步联合扫树深/迭代数时，1N、depth=4、100 iterations 的 OOF 增益达到 +0.06145，但这是多重搜索后的最高点，不能当成无偏估计。正式实验应将倍率和树复杂度放进外层训练折内的嵌套选择；1N 可作固定低成本控制。

去掉 5 个 UniMol 特征后，同一 depth=3 的 0.5N/1N/全池 OOF 增益分别为 **+0.05424/+0.05279/+0.05088**，五折仍全正。这说明池压缩本身是正向机制。配对比较中，UniMol 在 0.5N 上另加 **+0.00624**，query bootstrap 95% CI **[+0.00262,+0.01031]**，但只有 3/5 折正且主要由一个折驱动；在 1N 仅 **+0.00054**，CI **[-0.00163,+0.00287]**。所以 UniMol 有候选边界依赖的弱正信号，不能升级为默认主线。

### 5.4 共识不是全局规则

三源一致精度高度条件化：

- `N=21–100, partial=10–15`：0.9914；
- `N>100, partial=10–15`：0.9615；
- `N>100, partial>15`：0.8633；
- `N=3, partial=1–2`：0.7632；
- `N=4–20, partial=3–9`：0.2188。

对 Logistic，三源一致锁从 +0.03569 小幅到 +0.03644；对 HistGB 则从 +0.05088 微降到 +0.05071。硬锁所有两源一致把收益压到约 +0.0005。正确做法是让 `N`、partial 数、来源位次、来源交集和检索强度共同决定效用；三源锁最多作为预注册消融，不是默认规则。

### 5.5 UniMol 的正确边界

- 旧 compressed Bank 上，UniMol Reviewer 相对无 UniMol仅带来约 +0.00150 官能团代理差，CI 跨 0，并降低具体分子指标；不准入。
- 新三源候选模型中，UniMol 候选到 partial 的 mean/max/std 和到 H1 的 mean/max 是小幅独立特征：均值增量很小，但显著减少损失查询。
- UniMol 不应缩小合法候选宇宙；没有向量的分子也必须保留。
- 不应从 UniMol 几何直接推断官方功能团 gold；结构只作为 molecule-local frozen feature。

在修正版 1N 候选模型上，FlavorDB 数据库官能团代理相对 H1 提升约 **+0.02525**（21 个查询改善、14 个下降）。历史官方近似缓存能覆盖约 97.8% 的预测分子，但仍有 135 个唯一分子待映射；把缺失缓存当空集合得到的 +0.01908 有偏，不能称正式结果。该结果只说明具体分子收益有一部分可能传递到功能团层，正式 P4/P5 仍必须调用同一 evaluator 验证。

### 5.6 Gate 结果

简单查询 gate 使用 N、partial/N、H/I/R 交集和 proposal 改动率。修正 R 后，某个 HistGB gate（阈值 .4）在宽筛 OOF 由 ungated 的约 +0.0450 到 +0.0484，并把 dev 的损失查询降到 0，但 dev 均值也从约 +0.0438 降到 +0.0382。这个阈值是在多组 gate 中观察到的，仍有选择偏差。结论不是“gate 无效”，而是它可能提供风险—收益 Pareto 点；正式阶段应先比较 ungated exact-N 主模型，再以预注册阈值报告风险—覆盖曲线。

## 6. 失败方向总账

### 6.0 对最初承诺清单的覆盖核对

符号：✅ 本轮或仓库既有同协议实验已实际运行并复核；🟡 有历史证据/离线替代，但本轮未完成新的外部 LLM 对照；⛔ 因协议或安全边界停止。设计文字不算“已探索”。

| 原承诺机制族 | 覆盖 | 实际证据 |
|---|---|---|
| MFP molecule BM25 / IDF-Jaccard / asymmetric coverage / 长度归一 | ✅ | 61+ 静态评分统一 grouped OOF |
| MFP 稀有分子、高频抑制、query 子集/dropout | ✅ | rare4–32、stop10–75、修正 dropout 多 keep/reps |
| MFP 多相似样本食物/类别聚合 | ✅ | max/mean/top-k/LSE/rank decay；20 分组种子；去重压力测试 |
| MFP pair/triple、支持减冲突 | ✅ | pair/triple IDF 与多档 conflict penalty，均未胜主线 |
| MFP 轻量 Logistic/pairwise/tree 排序 | ✅ | masked-profile 自监督三类模型，均低于 BM25 |
| MFP 开放 LLM、低熵起点、新食物候选 | ✅ | 第七/八轮；exact-food 0/71，类别覆盖下降 |
| MFP 独立评分、两两锦标赛、正反方、顺序偏差 | ⛔ | 已构造 5 提示 × 正反顺序 × 24 样本；外部数据发送未获明确授权，0 次成功调用，不伪称完成 |
| MFP Reviewer 封闭选择、UniMol Reviewer | ✅ | 第一/二轮与第十一至十三轮历史响应逐样本复核 |
| MFP 动态专家路由 | ✅/🟡 | 标签无关三源路由 OOF 0.3298，失败；LLM 路由未新调用 |
| MPC 频率、partial/full profile BM25、加权投票、N 动态特征 | ✅ | 225 个检索传播配置与候选效用特征 |
| MPC Top1/3/5/10/20/50 残差与深度噪声 | ✅ | 候选 recall 曲线及 0.5N–全池压缩 |
| MPC LLM 完整集合/补充候选 | ✅/🟡 | 历史 Scientist、食品条件 ICL；未做新的多次采样/分块调用 |
| MPC 来源共识与条件可靠性 | ✅ | H/I/R 七种来源组合、N/partial 分层 |
| MPC constrained beam / 贪心交换 / 核心补全 / rerank | ✅ | 历史 Bank/beam/local swap + 本轮 consensus/exact-N candidate rerank |
| MPC 规则/线性/树 gate | ✅ | 历史多实例/嵌套 gate + 本轮新 proposal gate；均不胜 ungated 主模型 |
| MPC LLM 原子 swap Reviewer | 🟡 | 历史局部 Reviewer总体负；本轮没有外部新调用 |
| UniMol MFP/MPC | ✅ | 全覆盖审计、候选扩展、Reviewer、candidate utility、损失查询分析 |

严格说，本轮已全面覆盖非 API 机制族与仓库已有 LLM 资产，但**没有完成新的外部 LLM 提示机制实验**。原因不是负结果，而是外部传输授权边界；终稿不得把它描述为已实验。

### MFP 暂停

- 开放式 LLM 具体食物生成；
- 仅增加上下文/Top10；
- 分视角多调用和硬多样性；
- 网页食物词共现证据；
- pair/triple 分子共现直接评分；
- 大容量监督对比/自监督模型替代 BM25；
- UniMol 线性强融合；
- 标签无关查询级路由（OOF 低于 BM25）。

### MPC 暂停

- target-food token 单独检索；
- R-only 候选直接补全；
- 两源一致全部硬锁；
- 继承旧查询 gate；
- 旧 Bank 内动作 Reviewer；
- 用数据库对称官能团作为官方目标替代；
- 用 UniMol 结构邻近直接等价功能团效用。

## 7. 建议的正式实验组合

### 7.1 MFP 冻结矩阵

共同协议：完整-profile grouped nested OOF；固定分组文件；dev 只运行一次；输出必须为具体食物；Reviewer不得越出候选；当前 gold不进入任何输入处理。

| ID | 组件 | 目的 |
|---|---|---|
| M0 | 清洁版 Scientist Agent BM25 + 原提示 | 可复现原始控制 |
| M1 | BM25 `b=.5/.75` 单实例、无 LLM | 检索控制 |
| M2a（主） | BM25 + 全量类别 Top-8 聚合，固定 OOF 权重 .4，类内最高 BM25 具体食物输出 | 检验 OOF 最强发现 |
| M2b（保守） | 仅当 BM25 Top 分数并列时启用同一类别聚合 | 针对低置信并列、保护唯一 Top1 |
| M3 | M2a/M2b + BM25 Top5/UniMol Top2 候选并集，但类别分数不加 UniMol | 检验 UniMol 召回贡献 |
| M4 | M2a 的 Top3/Top5/Top8 与 profile-dedup 预注册消融 | 聚合规模/重复证据敏感性 |
| M5（需外发授权） | M2/M3 的封闭 Top3 具体食物 Reviewer；support/conflict + 顺序反转 | 检验 LLM 选择增量 |

正式准入以 M2 相对 M1 的 grouped OOF 下界 >0、至少 4/5 折非负、dev 不出现大幅反转、正式 MFP evaluator 正增益为条件。

### 7.2 MPC 冻结矩阵

共同协议：完整-profile grouped nested OOF；候选训练只用外层训练折；统一名称归一；剔除 partial；所有方法 100% exact-N；同时报告具体分子 F1 和正式功能团 F1。

| ID | 组件 | 目的 |
|---|---|---|
| P0 | 清洁版原 Scientist—Reviewer | 原始控制 |
| P1 | 历史冻结 H1 | 强控制 |
| P2（共同主方案） | partial-stat/full-doc cross-normalized BM25 原型传播 | 新机制单独贡献，当前 macro 最强 |
| P3 | H1 + ICL + R 简单 consensus/RRF | 无监督融合控制 |
| P4（共同主方案） | H/I/R 候选效用 HistGB + exact-N；R 倍率在外层训练折内从 {0.5,1,2,3,5} 嵌套选择 | 融合与去噪，重点比较 micro/风险 |
| P5 | P4 + 冻结 UniMol 5 个集合相似特征；固定 1N 对照 | UniMol 增量 |
| P6 | P5 + 仅三源一致软/硬核心消融 | 高置信核心贡献 |
| P7 | P5 + 预注册 gate | 风险—覆盖曲线，不作为默认主模型 |

优先冻结 HistGB 深度、迭代数、L2 和 R 候选池倍率；不得根据 dev 或官方 cache逐项追选。

### 7.3 正式优先级与预期效应

| 优先级 | 组合 | 当前低成本证据 | 正式决策重点 |
|---|---|---|---|
| A | M2a 类别 Top-8 聚合 | OOF +0.05917；20/20 分组种子正；dev +0.02817 | 官方类别 evaluator 是否仍正向 |
| A | P2 固定 cross-normalized R | 对 H1 OOF +0.05569；dev +0.04412 | 具体分子与官方功能团同时报告 |
| A | P4 固定 0.5N/1N HistGB（无 UniMol） | OOF +0.05424/+0.05279；五折均正 | nested OOF 后是否仍超过 R-only |
| B | M2b 并列或间隔门控 | 并列门控 dev 最强；间隔 .2 固定 OOF 略胜全量 | 阈值须嵌套，比较风险—覆盖 |
| B | P5 P4 + UniMol | 0.5N OOF 另加约 +0.00624；1N 仅 +0.00054；全池约 +0.00317，dev 全池反降约 0.00641 | 只认 paired CI 和正式功能团增益 |
| B | N 条件 `(b,T)` | 非嵌套宽筛 OOF +0.00224、dev +0.01295 | 仅在 nested 选择复现后准入 |
| C | 三源一致核心 / gate | 核心近零；gate 可能减少损失但降低 dev 均值 | 风险消融，不作为默认方案 |
| C | 封闭 LLM Reviewer | 历史最多小幅正且不稳定；新调用未授权 | 仅在候选已冻结且获外发授权后运行 |

这里的 A/B/C 是实验优先级，不是已经确认的最终排名；所有数值仍是 train grouped OOF 或 dev 代理，不替代正式 evaluator。

## 8. 文献依据与本项目映射

1. FoodPuzzle 任务定义与 Scientist/Reviewer 框架：[FoodPuzzle: The What and Why of Food Prediction](https://arxiv.org/abs/2409.12832)。本项目近似复现其关键输入输出，但公开信息不足以声称逐行官方复现。
2. 集合输入应保持置换不变：[Deep Sets](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)。支持 MFP 类别内证据聚合与 MPC 查询级集合特征，而非按任意输入顺序建模。
3. 先召回再排序：[Retrieve and Re-Rank](https://aclanthology.org/W18-5504/) 与 [Double Retrieval and Ranking](https://aclanthology.org/2023.findings-eacl.130/)。对应本轮把候选召回和最终选择拆开。
4. 检索增强应把外部/训练记忆作为条件证据：[Retrieval-Augmented Generation](https://proceedings.neurips.cc/paper_files/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)。本项目进一步表明结构化标签传播可以在生成前承担主要召回。
5. 精确基数应作为解码约束：[Predict and Constrain](https://proceedings.mlr.press/v80/brukhim18a.html)。支持 MPC 训练候选效用、解码时强制 exact-N，而不是让 LLM 自愿满足 N。
6. F1 最优决策不等价于逐标签 0.5 阈值：[Bayes-optimality of F-measure Maximizers](https://www.jmlr.org/papers/v15/waegeman14a.html)。支持在已知 N 下做集合级排序/解码。
7. 条件集合生成：[Conditional Set Generation](https://aclanthology.org/2022.emnlp-main.324/)。支持把 MPC 视为条件集合预测，而非独立分子分类。
8. UniMol 的角色是预训练 3D 分子表示：[Uni-Mol](https://openreview.net/forum?id=6K2RM6wVqKu)。其分子内结构表征不能自动提供食品归属关系，故应冻结并作为局部特征/召回补充。
9. 分子表征的任务适配差异：[Characterizing pretrained and task-adapted molecular representations](https://openreview.net/forum?id=zV0gv4a4Yj)。支持必须用本任务消融验证表示价值，而非因模型先进就默认有效。
10. 选择性预测应报告风险—覆盖，而非只报 gate 后均值：[Selective Classification for Deep Neural Networks](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)。
11. 没有外部反馈的 LLM 自我纠错可能恶化：[Large Language Models Cannot Self-Correct Reasoning Yet](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html)。与本项目 Reviewer 多次未转化 oracle 一致。
12. 长上下文中关键信息位置会影响利用：[Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)。支持不把 Top10/大量证据堆给 Scientist，而用紧凑候选 dossier。
13. 模型选择本身可造成过拟合：[On Over-fitting in Model Selection and Subsequent Selection Bias](https://www.jmlr.org/papers/v11/cawley10a.html)。支持 grouped nested OOF、冻结矩阵和 dev 单次使用。

## 9. 建议的新代码架构（设计，不在本轮实现）

```text
scientific_agent_clean.py
├── protocol.py          # schema、合法字段、exact-N、输出约束
├── splits.py            # profile group、nested OOF、污染检查
├── mfp/
│   ├── bm25.py
│   ├── category_pool.py
│   ├── concrete_decoder.py
│   └── bounded_reviewer.py
├── mpc/
│   ├── h1.py
│   ├── prototype_retrieval.py
│   ├── candidate_features.py
│   ├── exact_n_decoder.py
│   └── selective_gate.py
├── unimol_features.py   # 只读冻结向量；缺失向量不删候选
└── ledgers.py           # 每个阶段可复现实验账本
```

每个组件只承担一个因果职责，所有候选生成、排序和执行均输出单独账本。旧 `optimized_agent.py` 只作为回归对照，不再复制其版本分支和退役代码。

## 10. 下一步执行顺序

1. 冻结本报告的 MFP M0–M4、MPC P1–P6 配置与 profile groups；
2. 在新文件中实现“清洁原始版”，先完成协议单元测试；
3. 复现 M1/P1，确保与本地控制一致；
4. 分别实现 M2 和 P2，不接 LLM；
5. 实现 P4、P5 与 exact-N，跑 nested grouped OOF；
6. 只对通过 OOF 的固定组合运行一次 dev 和正式 evaluator；
7. 如用户明确授权把候选 dossier 发送给外部模型，再执行 M5 的小规模、顺序平衡 LLM 实验；
8. 只有正式 evaluator 也正向时，才合并为新 Agent 候选版本。

### 10.1 是否还需要“大量实验”

需要，但应是**结构化实验**，不是继续无边界堆模块。建议预算为：

1. 本地机制阶段：MFP 运行 M1/M2a/M2b/M4 的 20 个 grouped seeds；MPC 运行 P1–P6 的固定 5 个 outer groups，并在每个 outer-train 内做 inner 选择。这里计算量大但无 API 成本。
2. 稳健性阶段：对最终两条主线各做 paired bootstrap、N/profile-frequency/seen-unseen 分层、候选覆盖和 exact-N 审计；只保留跨折方向稳定者。
3. 正式代理阶段：只把 A 级固定组合送入同一官方近似 evaluator，先跑一轮全 dev；不得用 evaluator 结果继续逐参数搜索。
4. LLM 机制阶段：若获得外部传输授权，只比较原始 Scientist、封闭 direct Reviewer、support/conflict、顺序反转四个预注册条件；每个样本保持相同候选和调用预算。
5. 最终确认阶段：选择一个 MFP 和至多两个 MPC 组合，用独立重复或保留集确认；失败则回退到 M2a/P2，而不是把所有 B/C 模块拼起来。

大量实验能找到方向，但只有“机制假设 → 低成本宽筛 → grouped/nested 复核 → 冻结 → 正式 evaluator”的漏斗能把探索收益和选参噪声区分开。本轮已经完成前两级并对主方向做了部分第三级复核；下一阶段的核心不是再增加 100 个任意变体，而是把 A 级组合做成可复现实现并完成嵌套验证。

## 11. 完整性、边界与复现审计

- 本轮目标计时 **10943 秒（3 小时 2 分 23 秒）** 后进入终稿校验，满足至少 3 小时的探索要求；其中包括仓库审计、宽筛、修错重跑、组合、分层、bootstrap、文献核验和报告整理，并非以等待凑时长。
- 已阅读两份工作文档和 `audit_records` 现有 **33** 份审计文件；本报告共约 440 行，早期广覆盖报告保留为过程快照，以本终稿为准。
- `scientific_agent.py` SHA-256：`4ebaeaf438ac8fde02af5f2af1e4db6fda6f94244766fa6b462fd159f8e5d686`；`optimized_agent.py` SHA-256：`a3cfffeee0e914334553ff4891fa136340e463e0e08a7dcf1adc7ade48f6cb1f`。两者与探索开始时一致，且最终定向 diff 为空。
- 未修改 Agent 源码，未运行正式 test；仓库原有大量 staged/unstaged/untracked 状态均未清理或覆盖。本轮仓库内唯一有意写入的是审计报告，所有预实验脚本仍在 `/tmp`。
- 新 LLM 机制实验为 **0 次成功外部调用**，原因是外部发送仓库衍生 molecule/candidate dossier 未取得明确知情授权；历史 LLM 缓存可审计结果与新调用被严格分开。
- 功能团缓存覆盖不足以构成无偏正式比较：数据库代理与缺失当空的缓存诊断均已降级标注。P4/P5 必须在正式阶段使用同一官方近似 evaluator。

最终结论的置信层级：M2a、P2、无 UniMol 的固定 P4 是进入正式实验的 A 级方案；MFP 间隔门控、UniMol P5、N 条件参数为 B 级嵌套消融；硬共识、gate、外部 LLM Reviewer 为 C 级风险/机制实验。没有理由把所有正向模块再次堆成一个单体版本。

## 12. 创新机制追加探索（阶段完成）

### 12.1 研究标准

前 11 节的类别聚合、交叉归一化检索和候选树模型从此只作为强基线。一个方向要进入“创新候选”，至少必须满足：有区别于普通 pooling/retrieval/reranking 的可证伪机制；所有统计在训练折内估计；在相同 grouped OOF 上提供强基线之外的增量；能解释何时有效、何时失败；复杂度增加与增益相称。概念新颖但没有实验增量的方案只记录为假设。

### 12.2 MFP 正—负—歧义证据：首轮结果

首个原型在每个训练折内估计类别条件分子发生率与其余类别背景发生率，以收缩 log-odds 分解支持、负向和跨类别歧义，再叠加到 Top-8 类别聚合。固定 grouped OOF 强基线为 0.44092（本脚本的确定性分组与前文 20-seed 均值口径不同），dev 为 0.43662。

| 机制 | OOF 相对强基线 | dev 相对强基线 | 判断 |
|---|---:|---:|---|
| signed log-odds，权重 .05 | +0.00529 | **-0.05634** | 明显反转，不准入 |
| signed log-odds，权重 .1 | +0.00176 | -0.07042 | 淘汰 |
| positive-only，权重 .1 | -0.00529 | -0.04225 | 淘汰 |
| negative-only，权重 .1 | 0 | +0.01408（仅 1 条净改善） | 样本过少，不构成证据 |
| support/conflict 联合 .2/.2 | -0.01940 | -0.02817 | 淘汰 |

第一性原理修正：谱中“没有记录某分子”不等于该分子对类别构成反证；数据库覆盖、实验检测和任务遮蔽共同造成 missing-not-at-random。全局正负词典把观测缺失误当成生物/食品排斥，因此不能成为 MFP 创新。若后续保留负证据，只允许经过观测过程校准后选择性使用。

### 12.3 MPC 观测过程感知残差传播：首轮结果

将训练原型拆成其任务中显示的 partial 与 missing，比较只传播 missing、传播 full，以及按分子遮蔽倾向、源查询遮蔽比例和混合残差加权。固定 P2 为 OOF/dev macro **0.72113/0.73810**。

| 传播标签 | OOF macro | dev macro | 结论 |
|---|---:|---:|---|
| source missing（P2） | 0.72113 | 0.73810 | 强基线 |
| source full residual | 0.71792 | 0.74221 | OOF 降、dev 升 |
| missing + 0.25×source observed | **0.72225** | **0.74064** | 小幅双正向 |
| missing + 0.50×source observed | 0.72215 | 0.74013 | 小幅双正向 |
| inverse-propensity missing | 0.72077 | 0.73730 | 无效 |
| source/query mask-ratio compatibility | 0.71404 | 0.74212 | 明显 OOF/dev 分歧 |

这支持“训练 missing 不是完整食品残差的无偏样本”，但不支持目前的粗粒度 propensity 模型。0.25 混合的增益仅约 +0.00112/+0.00254，暂为机制信号而非正式候选。新的方向是以伪遮蔽恢复直接估计条件补全能力，而不是按总体遮蔽比例重权重。

### 12.4 MFP–MPC 伪遮蔽循环一致性：首轮结果

对 MFP 查询确定性遮住 20%/40%/60% 分子，以剩余 anchor 在每个候选类别内部检索，再计算该类别对 probe 的恢复能力；这一分数与强类别聚合融合，不使用 gold food。

| 循环分数权重 | OOF 相对强基线 | dev 相对强基线 |
|---:|---:|---:|
| .01 | +0.01764 | 0 |
| .02 | **+0.02293** | -0.01408 |
| .05 | +0.01587 | -0.05634 |
| .10 | -0.00529 | -0.09859 |
| 1.0 | -0.09700 | -0.14085 |

循环信息与 BM25 明显不完全同源，但尺度极不稳定：极小权重有 OOF 信号，稍大即破坏泛化。这不足以称为已成功的联合能量模型。下一步不直接校准类别，而把 anchor→probe 恢复能力用于训练原型可靠性，再传播 missing residual，以判断问题来自“循环机制无效”还是“类别级融合失准”。

原型可靠性版本在修复“预测集合必须剔除当前 partial”后，P2 正确基线回到 OOF/dev 0.72113/0.73810。最佳伪遮蔽 coverage 权重 .01 的 OOF 为 0.72108（-0.00005），dev 0.73966；权重增大时 OOF持续下降。结论：伪遮蔽是有用的机制诊断，但暂时不能直接重权原型。此前未剔除 partial 的中间输出已判无效，不进入结果。

### 12.5 exact-N 集合级解码与条件 UniMol：首轮结果

固定无 UniMol 的 0.5N HistGB 候选效用后，不再独立取 Top-N，而按已选集合逐步解码。测试三类集合项：冻结 UniMol 的已选集合结构相似度、FlavorDB 功能团新增覆盖，以及覆盖减结构冗余。OOF 基线具体分子 macro 0.71881、数据库功能团代理 0.91874。

| 集合项 | molecule macro 增量 | DB 功能团代理增量 | 初判 |
|---|---:|---:|---|
| UniMol 结构凝聚，权重 .2 | +0.00491 | **+0.00301** | 双指标正，需稳健性复核 |
| 功能覆盖，权重 .01 | +0.00534 | +0.00087 | molecule 正，代理很小 |
| 覆盖−结构冗余，权重 .02 | **+0.00569** | +0.00062 | 当前 molecule 最好 |
| 功能覆盖，权重 .2 | +0.00523 | -0.00196 | 过强多样性伤害代理 |

重要的新解释：在这个任务里，UniMol 的正向方向不是“尽量多样”，而可能是**条件结构凝聚**——同一食品的缺失集合在结构空间存在局部簇，强行排斥相似候选反而破坏集合。该发现与此前“UniMol 作为点式特征不稳定”不同：结构关系更适合在已选集合条件下参与解码。当前仅为单一 OOF 扫描，必须补 paired CI、五折稳定性和 dev 转移后才能升级。

稳健性复核否定了直接升级：结构凝聚 .2 的 query-bootstrap CI 为 [+0.00169,+0.00865]，但五折增益约 `[+0.00006,+0.02433,-0.00026,-0.00012,+0.00043]`，dev -0.00009；覆盖−冗余 .02 的 CI 为 [+0.00248,+0.00944]，但约 +0.02678 来自单一折，dev -0.00027。query bootstrap 把同一 profile/折结构当成独立样本，因此给出了过度乐观区间。集合关系目前只能作为**异质作用域线索**，下一步必须定位可观测条件并做跨折门控，不能称为稳定创新收益。

来源条件 UniMol 通道进一步比较候选到 `H∩R`、`I∩R`、`H∩I∩R` 与 R 前沿的 mean/max 相似及相对 partial 的对比。0.5N OOF 相对无 UniMol为 +0.00600，CI [+0.00183,+0.01049]，但五折约 `[+0.00436,+0.02408,-0.00342,+0.00162,+0.00329]`，dev -0.00182；旧 5 个 UniMol 特征在同口径 dev 反而 +0.00226。合并全部关系为 OOF +0.00650、dev -0.00078。更复杂的结构关系没有消除折异质性，暂不准入。

作用域分层揭示集合项的收益几乎集中于 `N≤3`：覆盖−结构项在该层 +0.04348（9 改善/0 下降），N4–20 仅 +0.00151，N21–100 +0.00047，N>100 -0.00010；等价地 partial≤2 为 +0.02248，partial≥10 约 0。由查询特征 cross-fit 的效应 gate 仍保留 +0.00553、12 改善/1 下降，但单折主导未消除。新的可证伪假设是：集合关系仅是低基数组合歧义消解器，直接用任务先验 `N≤3` 验证，而不再全局应用。

直接 `N≤3` 验证表面上总体 +0.00528、低 N 层 +0.04348，9 改善/0 下降；但全部改善只在同一个外层折，dev 的 9 条低 N 完全不变，而且结构、功能覆盖、二者组合产生完全相同预测。这说明它不是三种集合机制的独立证据，而是单个重复 profile 簇附近的边界扰动。该方向降级为数据结构诊断，不进入创新候选。

### 12.6 潜在完整 profile—遮蔽过程去卷积

由单折异常产生的新假设：同一 full profile 的多行不是独立食品原型，而是潜在完整谱的多次 partial/missing 遮蔽。首版将重复 profile 完全折叠，再从重复遮蔽估计每个分子的 missing 概率。结果：均匀潜在 profile 传播 OOF/dev 0.71362/0.72488，低于 P2 0.72113/0.73810；mask probability 为 0.70186/0.72374，更差。重复频率既造成分组依赖，也携带真实采样/遮蔽分布，不能简单删除。

因此继续把模型拆成两个连续指数：profile 频率权重 `freq^γ` 表示潜在谱的经验先验，mask probability 指数 `p_mask^α` 表示条件遮蔽通道；原始行传播与完全去重位于这一二维空间的不同端点。只有中间指数在 OOF/dev 同时超过 P2，才支持“去卷积”创新。

二维宽筛出现首个同向候选：`γ=.75, α=.25` 的 OOF macro/micro **0.72429/0.80201**，dev **0.74171/0.79871**；P2 为 0.72113/0.79915、0.73810/0.79528。附近 `γ=.75, α=.5/.75/1` 也保持较高 OOF，说明不是单个浮点尖峰。初步机制是：重复 profile 用次线性频率表达潜在食品谱先验，mask probability 只做弱校正；完全去重丢失采样先验，线性重复又过度计数。

配对复核通过初步门槛：相对 P2，`γ=.75, α=.25` OOF macro **+0.00316**，query-bootstrap 95% CI **[+0.00170,+0.00471]**，104 改善/40 下降；五折约 `[-0.00003,+0.00336,+0.00375,+0.00551,+0.00323]`。dev **+0.00361**，CI **[+0.00003,+0.00825]**，12 改善/4 下降。分层为 N≤3 0、N4–20 +0.00605、N21–100 +0.00341、N>100 +0.00262；dev 的 N≥21 同向。`α=.5` 得到接近结果，形成小平台。

这仍是低成本探索效应而非最终确认，但它是目前首个同时具备新机制、相同协议独立增量、近邻参数平台、近全折同向和 dev 同向的创新候选。暂名 **Latent Profile–Mask Deconvolution Retrieval（LPMD-R）**：先聚合同一潜在完整谱，再以次线性经验先验和弱遮蔽通道传播 residual。

直接把 P4 的 R 源替换为 LPMD 后，相对原 P4 的 OOF **-0.00266**，CI [-0.00878,+0.00343]，五折一近零、四负；dev +0.00478 但 CI 跨零。LPMD 的独立排序收益会被 HistGB 的来源秩/交集特征部分吸收，直接堆叠不是正确组合。正式定位应是独立 P2 竞争方法，或只将 P2–LPMD 分歧作为候选不确定度特征，不能宣称和 P4 可加。

P2–LPMD 分歧特征也未改善 P4：OOF -0.00024，CI [-0.00167,+0.00128]；dev +0.00123但与 OOF 反向。由此停止继续向树模型堆 LPMD 特征，保留方法因果边界。

### 12.7 MFP 潜在 profile 与保守循环裁决：负结论

对 MFP 对称地把相同 full profile 折叠为潜在 profile，保留类别软分布、标签熵和次线性频率。二维/Top-k 宽筛最佳仍相对强类别聚合 OOF **-0.01235**、dev **-0.04225**。同谱多食品/多类别不是可直接平均掉的标签噪声，具体实体证据必须保留。

最后将循环恢复限制为仅在 BM25 单实例类别与类别聚合冲突时二选一，避免全类别尺度融合。221 条 OOF 冲突中，无置信阈值为 -0.04056；最保守测试仍 -0.02646，dev至多为 0 增益。循环恢复系统性偏向单个相似 profile，无法裁决宏类别聚合。

因此本轮对 MFP 得到的是有价值的负结论：静态谱上的 log-odds 正负证据、潜在 profile 软标签和 MFP→MPC 伪遮蔽循环均不能稳定超过 Top-8 类别聚合。若追求真正 MFP 创新，需要新的监督信号、外部食品语义或端到端联合学习，并必须在额外授权/资源下单独立项；当前不能凭概念将其列为已发现创新。

### 12.8 创新探索阶段性结论与正式候选

| 方向 | 实际结果 | 阶段结论 |
|---|---|---|
| MFP 正/负/歧义证据 | 最佳 OOF +0.00529但 dev -0.05634 | 淘汰全局静态证据 |
| MFP–MPC 循环一致性 | 小权重 OOF最高 +0.02293，dev反向；保守裁决仍负 | 信息存在但不可校准，不准入 |
| MPC 粗 propensity / mask compatibility | 最佳只约 +0.001，或 OOF下降 | 不准入 |
| MPC 伪遮蔽原型可靠性 | OOF约 0，dev弱正 | 仅诊断 |
| exact-N 集合级结构/覆盖 | OOF表面 +0.0057，单折主导，dev约 0/负 | 降级为异质性发现 |
| 条件 UniMol 来源关系 | OOF +0.0060但两折负、dev -0.0018 | 不准入；UniMol保留 B 级消融 |
| **LPMD-R** | 对 P2 OOF +0.00316、dev +0.00361；近全折同向、邻域平台 | **唯一创新正式候选** |
| LPMD + P4 | OOF -0.00266 | 不可堆叠，作为独立检索方案 |

建议在原正式矩阵中新增：

| ID | 组件 | 定位 |
|---|---|---|
| P2-L（创新主候选） | LPMD-R，固定 `γ=.75, α=.25`，exact-N | 与 P2/P4 并列而非叠加 |
| P2-Lα | `α∈{0,.25,.5}` 固定消融，γ=.75 | 验证弱 mask 通道贡献 |
| P2-Lγ | `γ∈{.5,.75,1}` 固定消融，α=.25 | 验证次线性 profile prior |
| P2-L0 | 完全 profile 去重 `γ=0,α=0` | 证明不是普通去重 |

正式验证必须使用 profile-grouped nested OOF；`γ,α` 不得再根据 dev 选择。主比较同时报告 macro/micro molecule F1、exact-N、N 分层与官方近似功能团 F1。P2-L 若正式功能团指标不正，仍不能仅凭 molecule F1 合并。

### 12.9 新增文献边界

- MNAR 识别通常需要对缺失机制施加额外假设；简单逆倾向并不会自动去偏：[Semiparametric Inference for Nonmonotone Missing-Not-at-Random Data](https://www.tandfonline.com/doi/abs/10.1080/01621459.2020.1862669)。这与本轮粗 propensity 失败一致，也约束了 LPMD-R 的表述：它利用特定重复遮蔽结构，不声称解决一般 MNAR。
- MNAR tensor completion 的典型路线是先估计 propensity，再做加权恢复：[TenIPS](https://proceedings.mlr.press/v130/yang21d/yang21d.pdf)。本项目的经验结果表明，缺乏可识别结构时强 IPW 不稳定，弱收缩 mask 通道更合适。
- 多实例多标签学习用于处理 bag/instance 与标签歧义：[Multi-instance multi-label learning in the presence of novel class instances](https://proceedings.mlr.press/v37/pham15.html)。MFP 潜在 profile 失败说明这里的同谱多标签不能由朴素 bag 平均解决。
- 模型选择偏差仍适用于本轮二维搜索；因此 P2-L 只是待正式验证候选，不能把 query-bootstrap CI 当最终证明：[Cawley & Talbot](https://jmlr.org/papers/v11/cawley10a.html)。

### 12.10 本轮完整性记录

- 创新追加探索目标计时约 **4883 秒（1 小时 21 分）**，实际覆盖 12 个新临时实验脚本、五条预定机制主线及其衍生反证/组合；耗时实验的首版集合解码因复杂度不合理被主动中止并以预计算矩阵重写。
- 本轮没有修改 `scientific_agent.py` 或 `optimized_agent.py`，没有运行正式 test，没有新增 Agent 实现；仅更新本报告，所有预实验代码仍在 `/tmp`。
- 中间发现并作废两类错误口径：伪遮蔽传播未剔除当前 partial 的旧输出；集合解码逐候选重复计算 UniMol 导致不可接受复杂度。报告仅保留修正版结果。
- 阶段结论不是“创新已经正式成立”：LPMD-R 已从广泛探索中获得进入正式实验的资格；MFP 三条创新设想与其他 MPC 结构/循环设想均未通过准入，不应包装成贡献。

## 13. UniMol 全应用位置创新探索（2026-08-13 追加）

### 13.1 本轮问题不是“要不要用 UniMol”，而是“结构信息应该进入哪一个因果位置”

前述探索只足以说明 raw UniMol 余弦不宜直接接管食品语义。本轮进一步把可能接口拆为：

1. **表示层**：raw、中心化、正则白化、去枢纽、食品谱共现自监督适配；
2. **分子层**：候选对 partial、H/I/R 来源、画像、已选集合的局部关系；
3. **集合层**：centroid、mean+spread、方向 Chamfer、稀有分子覆盖、双向覆盖、图扩散、集合凝聚/多样性；
4. **检索层**：MFP 食物 profile 召回、MPC 潜在 full-profile 召回、候选并集；
5. **生成层**：MPC 遮蔽概率、latent profile 传播、exact-N 集合解码；
6. **决策层**：结构类别原型、BM25 residual、来源条件 residual、不确定度门控；
7. **跨模态层**：向 LLM 提供结构证据、结构—文本 projector；
8. **训练层**：冻结表示、低秩任务适配、对比学习、atom/token 级微调、多构象聚合。

评价仍使用 full-profile grouped 五折 OOF 和已复用 dev；本轮不读取或运行正式 test，不修改 Agent 源码。全部实现位于 `/tmp/unimol_*.py`。宽筛中的最优点只用于发现机制；凡出现单折主导、OOF/dev 反向或仅 oracle 提高，均不升级为实际收益。

### 13.2 资产与表示几何：raw cosine 存在严重各向异性

现有 NPZ 为 1,777×512，全部 finite，raw L2 norm 的中位数约 24.61。新增几何审计发现：

| 诊断 | 结果 |
|---|---:|
| 随机分子对 raw cosine 均值 | **0.95776** |
| 随机分子对 raw cosine 中位数 | **0.96754** |
| 全体单位向量均值的范数 | **0.97869** |
| covariance participation-ratio effective rank | **9.14 / 512** |
| 前 8/32/128 主方向解释方差 | 66.48% / 88.02% / 97.68% |

因此 raw cosine 的动态范围极窄，“0.98 比 0.96 更相似”不能未经校准地解释为强结构证据。历史 raw mean/max/attention 的弱效应，一部分可能不是 UniMol 没信息，而是公共方向、hubness 和集合均值稀释共同造成。

资产身份审计还发现：1,777 个唯一名称只对应 1,639 个唯一 SMILES；114 个重复 SMILES 组覆盖 252 个名称，138 个重复项与组内首项的向量逐元素相同。它们可能是异名、盐/立体命名折叠或相同 canonical 结构。正式实现必须：

- 以 canonical SMILES 为结构身份，名称只作显示别名；
- grouped OOF 同时报告 full-profile group 与 canonical-SMILES cluster 敏感性；
- 不把同一结构的多个名称当作独立结构支持票；
- 补齐 checkpoint hash、UniMol commit、构象生成/随机种子、RDKit 与 Python 环境；当前 NPZ 可读取但不足以独立再生。

正则白化使用 centered PCA 后按方差缩放，并以中位尺度的 0.25 作 floor，避免低方差方向爆炸。它提高了一些候选召回，但并未稳定提高独立 Top-1，说明几何修正是必要条件而非食品语义对齐的充分条件。

### 13.3 MFP：结构更适合产生异质候选，不适合直接裁决类别

#### 13.3.1 集合核与表示变体

本轮实际比较了 raw/centered/32、64、128、256 维正则白化 centroid，以及 BM25 Top-30 内的 query→document Chamfer、IDF-Chamfer、双向 Chamfer、rare-8 覆盖、阈值覆盖和最大分子对，并做局部 z-score 融合。

- 独立 raw centroid OOF/dev Top-1 约 0.309/0.324；centered 约 0.312/0.282；最佳白化变体 OOF有所提高但 dev 下降。
- BM25 Top-30 内小权重双向 Chamfer 在该探索口径 OOF 比 BM25 高约 0.023，但 dev 低约 0.042；方向 Chamfer、rare-anchor 和 coverage 的迁移同样不稳。
- 结论不是“set kernel 没用”，而是候选食品的完整长谱与输入谱之间并非一一对应的结构运输问题；强制 Chamfer/OT 会把大量共享结构当成食品特异证据。

因此 Sinkhorn/完整 OT 没有继续做大规模权重搜索：当前低成本 Chamfer 已否定“更精确集合匹配自然改善食品语义”的前提，而 OT 还会强加质量守恒并增加 O(|Q||D|) 以上计算。它只适合未来在稀有 anchors、unbalanced OT 和明确的检测概率权重下重新立项。

#### 13.3.2 去各向异性扩大候选覆盖，但没有解决选择器

| 候选集合 | OOF 类别 oracle | dev 类别 oracle | 平均候选规模 |
|---|---:|---:|---:|
| BM25 Top5 + raw UniMol Top5 | 0.75661 | 0.84507 | 约 8.88 / 8.52 |
| BM25 Top5 + white-64 Top5 | **0.78307** | 0.84507 | 约 8.33 / 8.39 |
| BM25 Top5 + white-128 Top5 | 0.77072 | **0.87324** | 约 8.35 / 8.20 |

白化 64 维主要改善 OOF，128 维主要改善 dev，不能根据两者分别选最优维度；但它们共同证明 raw Top2 不是 UniMol 候选召回的上限。正式候选生成应把维度/正则放进外层训练折选择，或预先固定 white-64 为低成本主消融、raw 为控制。

随后用强类别聚合分数在 BM25+white 候选类别内裁决，最佳 OOF约 +0.0106 至 +0.0123，却全部来自单折，dev 为 0 或负。直接把结构类别 Top1/Top-k 原型以 0.01–0.4 融入类别分数也同样单折化。因此：

> UniMol 已明确提高“正确类别是否出现在候选池”的上限，但当前不存在经验证的确定性选择器把该 headroom 转成稳定 MFP accuracy。

#### 13.3.3 食品谱共现自监督任务适配

为了不再假设预训练化学距离等于食品共现距离，本轮只用训练食品 profile 内部的分子集合、完全不使用类别 gold，估计白化空间内的 within-profile scatter，再做低秩 metric；同时测试 CSLS 风格密度修正以抑制 hub。

| 方法 | 独立 OOF/dev Top-1 | BM25 Top5 + 该专家 Top5 oracle OOF/dev |
|---|---:|---:|
| profile-self-supervised metric | 0.33157 / 0.36620 | 0.79012 / 0.81690 |
| metric + CSLS | 0.32099 / 0.30986 | **0.80071** / 0.81690 |

小权重 BM25+metric 的 OOF约比 BM25高 0.0123，dev 不变；CSLS 则 OOF负、dev最多净改善 1/71。候选 oracle 的确超过 raw，但 OOF/dev 对不同几何修正的偏好不一致。它是一个真实创新方向——**Food-context Adapter on Frozen UniMol（FCA-U）**——但目前只获得“候选源研究资格”，没有获得主排序资格。

正式 FCA-U 不应直接全参数微调 UniMol，而应先比较：

1. frozen UniMol + 32/64 维线性 adapter；
2. 同一 canonical SMILES 的多名称为等价约束；
3. 同一 latent full profile 内分子为软正例，跨 profile 高频分子不作硬负例；
4. leave-one-profile-out 的 masked-set retrieval loss；
5. raw、white、FCA-U、FCA-U+CSLS 四个固定候选源；
6. 只评价候选 recall/oracle 与固定 blind Reviewer，不用 dev 选维度。

这比“用类别标签微调一个分类器”更符合当前数据：食品谱多标签、同分子跨类别、观测 MNAR，硬 supervised contrastive 容易制造假负例。

### 13.4 MPC：结构应作用于“检索完整生成上下文”，而不是直接规定缺失分子相似/多样

#### 13.4.1 P2 的 full-profile structural residual

在严格复现 P2（OOF macro/micro 0.72113/0.79915，dev 0.73810/0.79528）的前提下，把 partial centroid 与训练原型的 partial/full centroid 分别融合到检索权重：

| 结构接口 | OOF macro 增量 | bootstrap 95% CI | 五折增量 | dev 增量 |
|---|---:|---:|---|---:|
| white full-profile，w=.10 | +0.00532 | [+0.00202,+0.00913] | [-.00020,+.02846,-.00037,-.00169,+.00027] | +0.00180 |
| white full-profile，w=.25 | +0.00242 | [-.00302,+.00802] | [+.00020,+.00736,-.00143,+.00199,+.00401] | +0.00272 |
| white partial-profile，w=.05 | +0.00004 | 跨 0 | 近 0 | -0.00021 |

w=.10 的 query bootstrap 被单折 profile cluster 夸大；w=.25 的点估计更跨折、dev 同向但 CI 跨 0。真正的机制信号是 **full-profile compatibility > partial-profile similarity**：UniMol 应帮助判断“当前 partial 来自哪类完整结构上下文”，而不是再匹配一个同样被遮蔽的 partial。

#### 13.4.2 LPMD 强基线上的结构画像检索

先前一版脚本曾错误地对 lexical profile score 做 query 内 z-normalization，使基线降到 0.70808；该中间结果全部作废。修正后 base 精确回到 LPMD-R OOF/dev macro 0.72429/0.74171。

| LPMD + 结构接口 | OOF 增量 | CI | 五折 | dev 增量 |
|---|---:|---:|---|---:|
| white full-profile，w=.25 | **+0.00224** | [-.00326,+.00787] | 4 正 1 负 | **+0.00604** |
| white full-profile，w=.10 | +0.00083 | 跨 0 | 4 正 1 负 | +0.00191 |
| raw full-profile，w=.50 | +0.00021 | 跨 0 | 2 正 3 负 | +0.01826 |

不能因 raw w=.50 的 dev 高而选择它；OOF几乎为零。white w=.25 在两个 split 同向且四折正，可作为 **LPMD-U（结构条件潜在画像）B 级消融**，但尚不足以替代 P2-L。

#### 13.4.3 明确失败的 MPC 应用位置

相对同一修正 LPMD 基线：

- 结构单独检索 full profile：raw 约 -0.137、white 约 -0.066 OOF，淘汰；
- 候选分子向 partial 凝聚：white w=.25 约 -0.073 OOF，淘汰；
- 候选分子远离 partial 的“互补”：white w=-.25 约 -0.086 OOF，淘汰；
- 候选图 5-NN 分数扩散：权重 .1 约 -0.0046，.25 约 -0.0184，淘汰；
- 画像内按结构近邻平滑 mask probability：约 +0.00069、CI 跨 0、dev +0.00026，信息不足；
- 来源条件 H∩R/I∩R/HIR 特征：前述 OOF约 +0.0060但两折负、dev -0.0018，不准入；
- exact-N 结构凝聚：前述表面 +0.0049但单折主导、dev约 0/负，不准入；
- 结构多样性与功能团覆盖代理：不能稳定迁移，且 DB functional group 不是官方 evaluator，不能作为准入依据。

第一性原理解释：MPC 的 gold 是“同一食品中因遮蔽而缺失”的集合，不是“与 partial 最相似/最不相似”的集合。结构只描述供给空间，完整食品画像与遮蔽过程才描述条件需求。因此正确分解应是：

`partial lexical evidence + weak full-profile structural compatibility → latent profile posterior → mask channel → exact-N`。

而不是：

`candidate-to-partial cosine → missing probability`。

### 13.5 从所有结果形成的 UniMol 应用位置矩阵

| 应用位置 | 已测结果 | 当前等级 | 后续动作 |
|---|---|---|---|
| raw molecule cosine | 严重各向异性；直接排序弱 | D | 仅作控制，不再默认使用 |
| centered/regularized whitening | 改善部分候选覆盖，Top-1不稳 | B | 固定 white-64 与 raw 配对消融 |
| MFP centroid/mean+spread | 低于 BM25 | C | 仅异质候选源 |
| MFP Chamfer/rare/coverage/局部重排 | OOF/dev 反向 | D | 停止全局融合 |
| MFP white/FCA-U/CSLS 候选扩展 | oracle 明显提高 | **B** | 进入候选召回正式消融 |
| MFP 结构类别原型/类别融合 | 单折收益，dev约 0 | C | 不进入主分数 |
| MFP 封闭 LLM Reviewer + 结构账本 | 历史仅 +1/+2、CI不过线 | C | 只有固定候选、顺序随机、回退控制下再测 |
| MPC candidate-to-partial 凝聚/互补 | 显著负 | D | 淘汰 |
| MPC full-profile structural retrieval | P2/LPMD 均有弱同向信号 | **B** | P2-U、LPMD-U 固定消融 |
| MPC mask 概率结构平滑 | 约 0 | C | 不进入正式主矩阵 |
| MPC H/I/R source-local features | 折异质且 dev负 | C | 仅机制附录 |
| MPC candidate graph diffusion | 明确负 | D | 淘汰 |
| MPC exact-N 结构凝聚/多样性 | 单折 cluster artifact | C | 不准入 |
| UniMol→功能团代理 | 数据库代理非官方 gold | C | 只作解释，不作训练/选择 |
| frozen UniMol→LLM 文本 | 缺乏对齐 projector | C | 需独立跨模态数据项目 |
| atom/token-level、多构象 UniMol | 当前 NPZ 不支持 | U（未测） | 需重生成资产后单独立项 |

### 13.6 新的创新设计，而不是把 UniMol 再堆成五个统计量

#### U1：FCA-U（Food-context Adapter on Frozen UniMol）

冻结 UniMol，只学习低秩 adapter，使同一 latent full profile 内分子成为带置信的软正例；正例权重由 profile 频率次线性收缩，高频跨食品分子不作负例。训练目标是 masked-set→full-profile retrieval，而非食物类别分类。输出只作为 MFP/MPC 的异质 profile 候选源。

创新点不在“用了对比学习”，而在把食品共现、重复 profile 与遮蔽过程联合定义为弱监督，避免当前 raw 3D 几何与食品条件语义错位。主要风险是假负例和 profile 重复泄漏，必须在 full-profile outer group 内训练 adapter。

#### U2：LPMD-U（Structure-conditioned latent-profile posterior）

保持 LPMD 的 `freq^.75 × p_mask^.25` 生成通道，只在 latent-profile posterior 中加入低权重 white full-profile compatibility；结构不触碰候选分子效用、不做图扩散、不改变 mask probability。建议固定 `w=.25`，与 w=0、raw w=.25、white partial w=.25 比较。

这是目前因果位置最清楚的 UniMol-MPC 设计，但探索效应只有 OOF +0.00224、dev +0.00604且 CI 跨 0，因此是 B 级而非创新主结果。

#### U3：Canonical-structure evidence budgeting

把同一 canonical SMILES 的名称折叠为一个结构节点；MFP 候选预算在 BM25、white-64、FCA-U 三个源之间按“新类别/新结构簇覆盖”分配，而非机械 Top5+Top5。目标是在相同 7–9 个候选预算下最大化跨源 novelty，防止同结构异名重复消耗 Reviewer 上下文。

该设计由本轮 138 个重复向量直接启发，尚未获得实际 accuracy，只能列为下一探索方向。评价必须同时报告候选数、唯一类别数、唯一 canonical 结构簇数和类别 oracle。

#### U4：结构—语言桥接只能作为独立项目

若希望 Scientist/Reviewer 真正“读懂”UniMol，不能把 512 个浮点或几个余弦写进 prompt 就称跨模态。3D-MoLM 使用专门 Q-Former、molecule-text matching/contrastive/captioning 将冻结 3D encoder 对齐到语言空间。本项目缺少食品—分子结构文本对齐语料，当前可做的只有结构证据账本；训练 projector、atom token 或 LoRA 需要单独数据预算和泄漏审计，不进入近期优化。

### 13.7 建议进入正式实验的 UniMol 子矩阵

在原正式矩阵上增加以下**相互独立**的消融，不与所有模块一次性堆叠：

| ID | 方案 | 固定比较 | 准入标准 |
|---|---|---|---|
| M3-R | BM25 Top5 + raw UniMol Top2/5 | M2 类别主分数不加 UniMol | 候选 recall/oracle 与预算 |
| M3-W | BM25 Top5 + white-64 Top5 | raw、white-128 仅消融 | nested grouped OOF 中稳定提高候选覆盖 |
| M3-F | BM25 Top5 + FCA-U Top5 | FCA-U 在每个 outer train 内拟合 | 固定 blind selector 后 accuracy 也必须正 |
| P2-U | P2 + white full-profile w=.25 | w=0、partial-profile、raw | 五折/cluster bootstrap、macro/micro 同报 |
| P2-LU | LPMD-R + white full-profile w=.25 | P2-L、raw、w=.1 | 结构项 paired CI 与官方 FG 同向才准入 |
| P5-old | P4 + 历史 5 UniMol features | 保留旧边界控制 | 不因新方案删除历史对照 |

正式阶段的选择规则：

- white 维度、adapter rank、结构权重不得看 dev 后固定；放进 inner folds 或在方案前固定；
- 同时做 query bootstrap、full-profile cluster bootstrap、各外层折和 canonical-SMILES cluster sensitivity；
- MFP oracle 只证明候选 headroom，主结论必须来自固定 selector 的 exact output；
- MPC 必须 exact-N、剔除 partial、报告 molecule macro/micro 与同一官方近似 functional-group evaluator；
- 若 P2-U/P2-LU 只在一个 profile cluster 增益，回退 P2/P2-L；若 M3-W/F 只提高 oracle 而 selector 不提高，则只保留检索工程结果。

### 13.8 文献支撑与边界

- [Uni-Mol（ICLR 2023）](https://openreview.net/forum?id=6K2RM6wVqKu)以 209M 构象预训练 3D 分子模型，并为下游任务设计 finetuning；这支持“冻结表示仍需任务接口”，不支持 raw cosine 自动等价食品归属。
- [3D-MoLM（ICLR 2024）](https://openreview.net/forum?id=xI4yNlkaqh)通过 Q-Former 和 matching/contrastive/captioning 对齐冻结 3D encoder 与文本；它直接约束了本报告的跨模态表述：几个余弦统计不是结构—语言对齐。
- [Deep Sets（NeurIPS 2017）](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)给出 permutation-invariant set modeling 的基本形式，也讨论 set expansion；它支持未来学习条件集合函数，但不意味着均值池化足够。
- [Sinkhorn Distances（NeurIPS 2013）](https://proceedings.neurips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport.pdf)提供熵正则 OT 的高效集合比较工具；本轮 Chamfer 阴性说明若引入 OT，必须先明确 unbalanced mass、稀有 anchor 和观测权重，不能仅用更昂贵距离替换均值。
- [MolCLR](https://arxiv.org/abs/2102.10056)说明分子表示可通过自监督对比目标适配；FCA-U 借鉴“自监督适配”原则，但正例来自食品 profile 共现，因而必须额外处理食品多标签和假负例。
- [WhitenedCSE（ACL 2023）](https://aclanthology.org/2023.acl-long.677/)及[表示各向同性分析（ACL Findings 2023）](https://aclanthology.org/2023.findings-acl.778/)说明预训练表示的各向异性会影响相似度，并可由 whitening/contrastive dynamics 改变；它们是跨领域几何依据。本报告没有声称 NLP 数值可直接迁移到分子空间，而是用本地随机余弦/effective-rank 实测确认了同类问题。

### 13.9 本轮 UniMol 探索的最终结论

1. **最大的新发现是表示几何，不是某个新统计量**：当前 raw UniMol 空间随机余弦约 0.958、有效秩约 9.14，过去 raw cosine 阴性实验不能代表所有经校准的 UniMol 接口。
2. **MFP 的可靠作用仍是候选召回**：白化与食品谱自监督 metric 可进一步提高类别 oracle，但固定选择器没有稳定把它转成 accuracy；因此进入正式矩阵的是候选消融，不是主分数。
3. **MPC 的正确位置是 latent full-profile posterior**：P2 与 LPMD 上都出现弱同向信号；candidate-to-partial、图扩散、直接凝聚/互补、mask 平滑和集合多样性均失败或不稳。
4. **没有一个 UniMol 组合目前超过 LPMD-R 的创新证据等级**：P2-U/LPMD-U 是 B 级正式消融，FCA-U 是新探索候选；不能把 dev 最优 raw 权重或单折 white 增益包装成成功。
5. **下一步确实需要实验，但不是无边界大量试参**：应围绕 M3-W/M3-F、P2-U、P2-LU 做 nested grouped OOF、cluster bootstrap 和固定 dev 确认。大量实验可以发现作用位置；只有预注册的组合与独立确认才能证明优化。

本轮再次确认两个 Agent 文件 hash 未变：Scientist `4ebaeaf438ac8fde02af5f2af1e4db6fda6f94244766fa6b462fd159f8e5d686`，Optimized `a3cfffeee0e914334553ff4891fa136340e463e0e08a7dcf1adc7ade48f6cb1f`。未运行正式 test；仓库内只追加本报告，临时实验仍位于 `/tmp`。
