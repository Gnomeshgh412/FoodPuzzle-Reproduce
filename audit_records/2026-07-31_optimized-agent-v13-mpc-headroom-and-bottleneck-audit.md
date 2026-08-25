# optimized-agent v13：MPC 可达上限与瓶颈分解审查

- 日期：2026-07-31
- 范围：Only-Deepseek optimized-agent v13
- 本轮权限：只读离线审查，并在既有 `audit_records/` 中记录结论
- 本轮未执行：模型 API、正式预测、正式评测、结果覆盖、代码修改、结果目录创建、Git 更新或提交
- 上游记录：
  - `2026-07-30_optimized-agent-v12-failure-and-mpc-direction-reset-audit.md`
  - `2026-07-30_optimized-agent-v13-metric-aligned-implementation-and-oof.md`

## 1. 审查问题

v13 的 train-only grouped OOF 平均 functional-group F1 增益只有约
`+0.0053`。本轮不再假设“收益低是因为候选不足”，而是将完整链路拆成：

1. occurrence H1 基线本身还有多少可修复空间；
2. 当前 `N+30` 边界池是否包含可修复所需的官能团；
3. 如果知道金标准官能团，1/2/3 次交换能够达到多少；
4. v13 的无金标准 demand estimator 实际取得多少；
5. 瓶颈主要属于候选召回，还是需求估计与动作决策。

审查遵循与 v13 相同的 5 折 grouped OOF 划分：

- 568 个训练样本；
- 按规范化 `target_food` 分组；
- 每折只用其余四折建立 occurrence、retrieval 与官能团先验；
- 不读取测试标签、测试身份或 LLM 评测缓存；
- MPC 不使用 UniMol；
- 官能团来自本地 FlavorDB molecule-local `functional_groups`；
- 指标为逐样本 functional-group set F1 的宏平均。

## 2. Oracle 定义与边界

### 2.1 Operational boundary oracle

以同折 occurrence H1 的前 `N` 个分子为基线，新增候选只允许来自当前
`N+30` occurrence 边界池。使用金标准官能团计算每一步最优的单分子
remove/add，连续执行最多 3 步。

这是逐步 greedy oracle，不是对所有 2/3 交换组合的穷举全局最优。因此：

- 一步结果是精确的单交换 oracle；
- 两步、三步结果是保守的可达下界；
- 只用于测量方法空间，不进入训练或正式预测。

### 2.2 Full-catalog oracle

新增候选扩展到 FlavorDB 合法候选目录。由于正式指标只比较官能团集合，
具有相同官能团签名的分子等价，因此 oracle 对唯一官能团签名搜索，不依赖
具体分子名称。

它回答“扩大候选池最多还能增加多少”，而不是提出可部署算法。

## 3. 总体结果

### 3.1 基线、v13 与 oracle

| 方法 | Mean FG-F1 | 相对 H1 增益 |
|---|---:|---:|
| 同折 occurrence H1 | 0.904596 | — |
| v13 demand + 1 swap | 0.909866 | **+0.005269** |
| `N+30` 边界 oracle，1 swap | 0.952616 | **+0.048020** |
| `N+30` 边界 greedy oracle，2 swaps | 0.963227 | **+0.058631** |
| `N+30` 边界 greedy oracle，3 swaps | 0.965598 | **+0.061002** |
| 全目录 oracle，1 swap | 0.965243 | **+0.060647** |
| 全目录 greedy oracle，2 swaps | 0.979622 | **+0.075026** |
| 全目录 greedy oracle，3 swaps | 0.984021 | **+0.079425** |

核心结论：

1. v13 实际只取得边界单交换 oracle 收益的约
   `0.005269 / 0.048020 = 10.97%`；
2. 当前边界池的一次 oracle 已有 `+0.0480`，所以“单次交换天然没有空间”
   这一解释不成立；
3. 将候选从边界池扩大到全目录，一次 oracle 只额外增加约 `+0.0126`；
4. 在现有边界池内，需求估计与动作决策损失约 `+0.0428`，明显大于候选池
   扩展空间；
5. 无条件增加交换次数不是答案：oracle 知道金标准时才随预算增长，而 v13
   的 budget 2/3/5 已在上一轮 OOF 中因不稳定被拒绝。

### 3.2 候选覆盖

| 覆盖指标 | 结果 |
|---|---:|
| `N+30` 边界池对 gold functional groups 的平均覆盖 | **96.49%** |
| 全目录对 gold functional groups 的平均覆盖 | **100%** |
| `N+30` 边界池对 gold molecule identity 的平均召回 | 72.57% |
| OOF gold molecules 中无 FlavorDB group 映射的出现次数 | 242 |

官能团覆盖 96.49%，但精确分子召回只有 72.57%，说明：

> MPC 的多对一官能团等价性非常强。当前任务不要求恢复同一个分子身份，
> 大部分所需官能团已经可以由边界池中的替代分子覆盖。

因此下一版不应重新把 exact-molecule recall 当作主要目标，也不应因为
exact molecule 没进候选池就盲目扩大检索。

### 3.3 动作覆盖

- 568 条 OOF 样本中，286 条存在严格正收益的边界单交换；
- v13：102 wins、86 losses、380 ties；
- v13 的两条负收益 fold 仍然存在：
  - fold 0：`-0.001681`
  - fold 3：`-0.008738`

这同时暴露两个问题：

1. **动作召回不足**：有 286 条可通过一次交换改善，但 v13 只在 102 条上
   实现正收益；
2. **动作精度不足**：v13 产生 86 条负收益，说明 demand estimator 不能可靠
   判断“是否应改”和“应补哪个官能团”。

当前瓶颈不是 Reviewer 层数，而是 Reviewer 之前没有足够可靠的、与正式
functional-group F1 对齐的独立决策信号。

## 4. 按 N 分层

训练 OOF 的 `n` 四分位数为 17、83、105。

| N 分层 | 样本数 | H1 F1 | v13 gain | 边界 oracle 1-swap gain | 边界 oracle 3-swap gain |
|---|---:|---:|---:|---:|---:|
| `n <= 17` | 143 | 0.800443 | **+0.019306** | **+0.123941** | **+0.147994** |
| `18 <= n <= 83` | 168 | 0.903574 | **-0.000091** | +0.034779 | +0.048760 |
| `84 <= n <= 105` | 117 | 0.978247 | +0.000509 | +0.008993 | +0.012254 |
| `n > 105` | 140 | 0.950657 | +0.001343 | +0.018976 | +0.027575 |

最重要的事实是：

- small-N 是最大可改善区域；
- v13 在 small-N 有正收益，但只取得其单交换 oracle 的约 15.6%；
- `18–83` 区域明明存在 `+0.0348` 单交换空间，v13 却近似零收益；
- 大 N 的基线已经接近饱和，强行修改的风险高、边际收益低。

这支持“按预测风险选择性修改”，但不支持手工记忆测试集 N 桶。可部署方法
应使用样本自身的不确定性、边界 margin 和 group-demand posterior，而不是
测试集阈值。

## 5. OOF 与当前正式测试结果的分布警告

只为解释历史正式结果，读取了已经存在的 v12
`evaluation_details.jsonl`，没有重新使用测试标签训练或调参。

| 数据 | N 四分位数 |
|---|---|
| train OOF | 17 / 83 / 105 |
| 正式 test | 5 / 83 / 104 |

正式 test 的最低四分位明显更小。v12 正式结果按同一 OOF 分层边界统计：

- `n <= 17`：F1 约 0.4294
- `18–83`：F1 约 0.7230
- `84–105`：F1 约 0.7603
- `n > 105`：F1 约 0.7740

这说明历史正式 MPC 的主要失败集中在 small-N，而 test 的 small-N 比例与
难度高于训练 OOF。它也解释了为什么 0.90 左右的训练 OOF 绝对 F1 不能
外推为正式测试绝对 F1。

限制：

- 当前 formal details 属于 v12，不是 v13；
- 可以用于错误分析，不能用于挑选 v13 超参数；
- v13 是否提升仍必须由一次冻结后的正式运行决定。

## 6. 第一性原理诊断

functional-group set F1 只关心最终官能团并集。对任意一次交换，真正需要
估计的是：

`expected delta F1 = value(newly covered needed groups)
                    - cost(removed unique needed groups)
                    - cost(new false-positive groups)`

v13 当前主要从检索邻居的 missing profiles 估计 group demand，但它没有
充分建模：

1. 当前 H1 已经覆盖了什么；
2. 某个待删除分子是否独占一个重要官能团；
3. 新分子同时引入多少非需求官能团；
4. demand posterior 的不确定性；
5. “不修改”相对每个动作的风险。

所以 v13 找到了正确的优化对象，却仍使用了过弱的决策器。其问题不是
“官能团思路错误”，而是把 group prevalence/retrieval support 当成了
action utility 的充分统计量。

## 7. 文献方法与本项目的适用关系

### 7.1 Expected-F1 / multilabel decision

Waegeman et al., *On the Bayes-optimality of F-measure Maximizers*
（JMLR 2014）说明，F-measure 的最优决策不能通过独立标签阈值简单得到，
需要直接考虑联合输出和 F-measure 风险：

https://www.jmlr.org/papers/v15/waegeman14a.html

对应本项目：下一版应预测 group posterior 后直接比较每个局部交换的
expected F1，而不是按 group demand 分数贪心补充。

### 7.2 Contextual submodular set prediction

Ross et al., *Learning Policies for Contextual Submodular Prediction*
（ICML 2013）为同时考虑单项价值、集合覆盖和冗余提供了方法基础：

https://proceedings.mlr.press/v28/ross13b.html

对应本项目：官能团并集具有覆盖和边际收益结构。候选价值必须依赖当前已选
集合，不能只由分子自己的 retrieval/group score 决定。

### 7.3 Selective prediction / abstention

Gangrade et al., *Selective Classification via One-Sided Prediction*
（AISTATS 2021）以及结构化 abstention 工作强调风险与覆盖率的显式取舍：

https://proceedings.mlr.press/v130/gangrade21a.html

对应本项目：只有当动作的保守 expected-F1 下界高于 0 时才允许修改；
其余样本保留 H1。目标不是最大修改覆盖率，而是保证净收益。

### 7.4 Independent verification

关于 LLM intrinsic self-correction 的实证显示，没有外部反馈的自我纠错
经常无效；CRITIC 和 Chain-of-Verification 则强调独立工具/证据反馈：

- https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html
- https://proceedings.iclr.cc/paper_files/paper/2024/hash/fef126561bbf9d4467dbb8d27334b8fe-Abstract-Conference.html
- https://aclanthology.org/2024.findings-acl.212/

对应本项目：Scientist 可以提出 remove/add action；Reviewer 必须读取独立的
FlavorDB group ledger、训练侧统计和 typed evidence，审计该 action 的新增、
丢失与误增官能团。让同一模型只凭自然语言“再想一次”不能解决 demand
estimator 的结构性误差。

## 8. 下一版方向

### 8.1 保留

- MFP 的 UniMol 路径不变；
- MPC 不使用 UniMol；
- occurrence H1；
- `N+30` 边界候选池；
- exact-N；
- FlavorDB 确定性官能团 ledger；
- grouped OOF 与 selective abstention。

### 8.2 不做

- 不把候选池先扩大到全目录；
- 不无条件增加 swap budget；
- 不恢复 exact-molecule objective；
- 不加入更多同质 Reviewer；
- 不按正式 test 的样本或 N 桶手工调规则；
- 不因 oracle 使用 gold groups 而把 oracle 逻辑带入推理。

### 8.3 应实现的通用模块

1. **Set-conditioned Group Demand Scientist**
   - 输入 food、partial profile、H1 已覆盖 groups、训练侧检索统计；
   - 输出每个 group 的 posterior 与不确定性；
   - 显式预测 missing、already-covered 和 likely-false-positive 三类状态。

2. **Counterfactual Action Generator**
   - 只在 H1 边界生成少量 remove/add；
   - 对每个动作记录新增 groups、删除后丢失 groups、引入的额外 groups；
   - 动作特征与集合状态绑定，而不是只描述候选分子。

3. **Independent Metric Reviewer**
   - 使用独立 FlavorDB ledger 验证 action 声明；
   - 估计动作 expected delta-F1 及置信下界；
   - 无独立支持则 abstain，不得自由重写完整集合。

4. **Risk-controlled Decoder**
   - 先比较 H1 与所有单交换动作；
   - 只有 OOF 校准的保守效用下界为正才执行；
   - 第二次交换必须在第一次交换后的新集合状态上重新估计，并单独通过门控；
   - 不能使用固定 N 桶作为推理规则。

## 9. 最终判断

v13 收益低，不是因为 MPC 没有优化空间，也不是因为当前候选池严重缺失。

更准确的结论是：

> `N+30` 候选池已经包含绝大多数所需官能团，单交换也有明显 oracle
> 空间；但当前 retrieval-smoothed demand estimator 只能取得约 11% 的
> 可达收益，并伴随较高的负动作率。下一阶段应从“扩大候选/增加 Agent”
> 转向“集合条件下的 group posterior、反事实动作效用和选择性独立验证”。

因此 v13 不建议直接进入正式运行。先实现并通过新一轮 train-only grouped
OOF 准入，最低要求应包括：

1. 五折均不为负，或预注册的置信下界稳定为正；
2. 相对 v13 明显提高 oracle capture ratio；
3. losses 显著少于 wins，而不是靠大量修改换取微弱均值；
4. small-N 和中等 N 均有正收益；
5. 不读取正式 test 标签、不使用 N 桶手工规则；
6. MPC 继续不使用 UniMol。
