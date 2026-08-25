# MFP UniMol 归因与 MPC Scientist–Reviewer 路线重审

- 日期：2026-07-30
- 项目路线：Only-Deepseek Scientist–Reviewer
- 审查范围：MFP 中 UniMol 的作用、MPC 中 UniMol 的取舍、下一阶段 Scientist–Reviewer 优化
- 本轮操作：只读审查代码、正式结果、v1–v11 历史记录与相关文献；未修改正式代码，未调用 API，未运行正式实验，未修改结果，未执行 Git 操作
- 上游记录：
  - `2026-07-30_optimized-agent-v1-v10-longitudinal-audit.md`
  - `2026-07-30_optimized-agent-v11-design-and-offline-admission.md`
  - `2026-07-30_unimol-mpc-boundary-conditional-redesign.md`
  - `2026-07-30_unimol-mpc-boundary-adapter-implementation-and-oof.md`
  - `2026-07-30_optimized-agent-v11-implementation-and-check-only.md`

## 1. 本轮结论

### 1.1 MFP

UniMol 对 MFP 有效具有明确的任务层解释：

1. MFP 是“分子集合到单一宏观食品类别”的压缩分类。
2. UniMol 提供的分子结构和三维构象表示与输入侧信息直接对齐。
3. 多分子集合聚合能够累积类别共有结构模式并平均单分子噪声。
4. 输出空间较小，Reviewer 只需在少数宏观类别之间进行判别。

但现有结果只能证明“包含 UniMol 的完整 MFP 系统稳定提升”，尚不能把全部增益严格归因给 UniMol。最终仍需在冻结版本上进行 `full` 与 `no_unimol` 的单变量消融。

### 1.2 MPC

当前证据不支持继续把 UniMol 作为 MPC 主增益模块：

- raw/global UniMol similarity 已失败；
- partial-centroid/set compatibility 已失败；
- 全局 UniMol-conditioned set energy 已失败；
- 单构象、线性、边界条件化 swap adapter 在 grouped OOF 中没有正收益，最终预算为 0；
- 多构象尚未测试，但它只能解决构象方差，不能解决“结构表征不包含食品出现关系”的目标错配。

下一阶段 MPC 主线暂时舍弃 UniMol 的预测和决策权限，转向优化原 Scientist–Reviewer 结构。UniMol 可保留为独立消融研究，但不进入下一正式候选的 MPC 排序、门控和 Reviewer 提示。

### 1.3 Scientist–Reviewer

下一阶段不应继续采用：

> Scientist 审计三套完整集合，Reviewer 从 H1/H2/H3 中选择一个完整答案。

应改成：

> Scientist 负责高召回候选和候选级证据档案；Reviewer 逐个验证候选或局部交换；确定性执行器应用通过验证的少量交换并保证 exact-N。

这仍然保留原论文的 Scientist–Reviewer 逻辑，只是将角色权限与 MPC 的多答案集合补全性质重新对齐。

## 2. 证据边界与当前仓库状态

### 2.1 当前正式结果与当前代码不是同一版本

当前 `results/Only-Deepseek/optimized-agent/.../run_metadata.json` 记录的正式结果仍是：

- `method=optimized_agent_v10`
- MFP：34/71，Accuracy 0.4789
- MPC：Precision 0.6248，Recall 0.7004，F1 0.6574，IoU 0.5153

当前 `code/Only-Deepseek/optimized_agent.py` 则是：

- `METHOD_VERSION = "optimized_agent_v11"`

v11 只完成了代码与 check-only 离线准入，没有正式 API 结果。因此后续不能把 v11 的设计或 OOF 结果写成正式测试提升。

### 2.2 v11 已确认的训练内事实

v11 grouped OOF：

| 通道 | OOF 结果 | 准入 |
|---|---:|---|
| H1 occurrence/cooccurrence 主干 | 默认基础集合 | 是 |
| Retrieval residual，budget=1 | hidden-set F1 +0.0019075；bootstrap 下界 +0.0001432；50 wins / 25 losses | 是 |
| UniMol boundary residual | 严格门控后 0 次交换、0 增益 | 否 |
| Local attribute complementarity | F1 -0.0118861；0/5 positive folds | 否 |

这说明当前可部署的候选结构实际为：

`H1 + 最多一次 retrieval boundary swap`

而不是三个互补且同等有效的完整假设。

### 2.3 历史正式结果提供的上限证据

v10 已有逐阶段重算：

| 阶段 | MPC 功能团 F1 |
|---|---:|
| H1 | 0.6725 |
| H2 | 0.6584 |
| H3 | 0.6613 |
| Reviewer 选择后 | 0.6576 |
| Fusion 后 | 0.6574 |
| H1/H2/H3 测试 oracle | 0.6811 |

三候选 oracle 相对 H1 只有约 +0.0085。即使 Reviewer 完美选择，也难以产生显著突破。因此候选生成上限必须先提高，不能把主要希望放在 Reviewer 提示词上。

## 3. 为什么 UniMol 对 MFP 有效

### 3.1 信息方向

MFP 可写为：

`set of molecule structures -> food macro-category`

MPC 可写为：

`food + partial molecule set -> exact missing molecule set`

两者不是互逆映射。

对于 MFP，分子骨架、官能团、原子局部环境和三维构象共同形成食品类别的统计化学指纹。UniMol 在约 2.09 亿个分子构象上预训练，面向三维分子表征和下游分子性质任务，因此其表示与 MFP 输入方向相符：

https://openreview.net/pdf?id=6K2RM6wVqKu

对于 MPC，真正需要估计的是：

`P(candidate occurs | target food, observed molecules, evidence)`

而不是：

`P(candidate is structurally similar | observed molecules)`

食品中的真实出现关系还受到来源、生化过程、加工方式和数据库记录机制影响。冻结 UniMol 表示没有直接学习这些条件变量。

### 3.2 集合聚合降低噪声

MFP 不是依赖一个分子，而是聚合一个分子集合。若某类食品在结构、官能团或风味相关分子属性上存在分布性规律，则：

- 类别共有信号随分子数量累积；
- 单个异常分子和构象误差被平均；
- 类别条件原型比单分子最近邻更稳定。

这解释了类别条件 UniMol set representation 比 raw cosine 更合理。

### 3.3 低熵输出降低审查难度

MFP 只需在有限宏观类别中选择一个标签。历史 v7 审查中：

- 结构控制器正确 26 条；
- Reviewer 最终正确 33 条；
- 10 次错误转正确，3 次正确转错误；
- Reviewer 净增 7 条。

Reviewer 面对的是少量类别之间的语义判别，而不是 MPC 中数十至数百个候选的开放集合决策。

### 3.4 不能把系统提升全部归因于 UniMol

MFP 的提升同时包含：

- 直接宏观类别输出；
- 稀疏 occurrence 特征；
- 类别条件 UniMol 集合表示；
- class-aware grouped OOF 融合；
- 固定候选空间；
- 受控 Reviewer override。

因此下一次 MFP 消融至少需要：

1. 固定数据、随机种子、候选、Reviewer 和评测；
2. 只关闭 UniMol 特征；
3. 比较整体准确率、逐类别变化和 Reviewer 净贡献；
4. 不根据测试错误重新选择融合权重。

分子预训练并不保证所有小数据下游任务都受益，任务相关性不足还可能产生负迁移：

- NeurIPS 2022，分子预训练收益并不稳定：  
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html
- Communications Chemistry 2024，任务相关性不足会造成负迁移：  
  https://www.nature.com/articles/s42004-024-01169-4

## 4. 为什么当前 MPC Scientist–Reviewer 仍然受限

### 4.1 Scientist 当前仍在审计完整集合

当前 v11 代码仍要求 Scientist：

- 恰好审计 H1、H2、H3；
- 对三个完整 exact-N proposal 分别给出 support/conflicts；
- 再由 Reviewer 选择完整 base hypothesis。

问题是 v11 的 H3 预算已经为 0，H3 实际退化为 H1；有效差异主要只剩一次 retrieval swap。要求 LLM 审计三个完整集合会：

- 放大大量共同元素，稀释真正的边界差异；
- 将相关或重复候选伪装成多种独立假设；
- 增加无关上下文和解释性噪声；
- 让 Reviewer 仍在做全局集合选择，而不是验证一个可证伪动作。

### 4.2 Reviewer 的证据约束目前主要停留在 Prompt

当前解析器能机械检查：

- Reviewer 只能选择 eligible hypothesis；
- 可以 `ABSTAIN`；
- 不能生成候选池外分子；
- 输出必须符合 JSON 和 exact-N。

但它不能机械检查：

- strengths/conflicts 是否引用真实 evidence ID；
- 引用内容是否真的支持对应候选；
- Reviewer 是否把气味相似误当作真实出现；
- 每个被接受交换是否同时有 add 支持和 remove 反证；
- 证据是否相互独立。

因此当前是“提示模型要依据证据”，还不是“系统验证模型确实依据了证据”。

### 4.3 当前证据门控存在语义过宽

`format_mpc_evidence()` 当前将以下关系都加入 `supported_linked`：

- `occurrence_support`
- `sensory_replication_support`
- `functional_role_support`

随后只要分歧候选属于 `typed_evidence_linked`，就可能触发 Reviewer。

但：

- 气味相似只能说明感官替代或相似；
- 功能作用只能说明该分子可能贡献某类香气；
- 二者都不能证明该分子真实存在于目标食品。

虽然 Prompt 提醒 Reviewer 区分这些关系，门控代码本身仍把它们共同视为候选支持。这与当前任务的 occurrence 语义不一致。

### 4.4 关闭 UniMol 会意外改变候选宇宙

当前 MPC 初始化逻辑在有 UniMol embeddings 时，将 embedding 中的分子加入 `display_names` 和候选 universe。

因此简单使用 `no_unimol` 不只会删除结构特征，还可能把候选宇宙从：

`training molecules + UniMol mapped molecules`

改变为：

`training molecules`

这会把“去掉 UniMol 表征”与“缩小候选空间”混为一个消融，无法进行公平归因。

下一版如果 MPC 舍弃 UniMol，候选宇宙应独立来自：

- FlavorDB 合法分子目录；
- 训练 profile 中的分子；
- 可规范化并可评测的 evidence-linked 分子。

候选合法性不能依赖是否存在 UniMol embedding。

### 4.5 当前检索支持仍较粗糙

`_build_retrieved_support()` 当前主要使用：

- 0.35 × food token overlap；
- 0.65 × partial-profile Jaccard；
- top-k profile；
- 每个候选取邻居中的最大支持。

局限：

- 食品名称词面重合不等于食品语义或来源相近；
- 普通 Jaccard 没有降低高频公共分子的权重；
- max aggregation 允许一个偶然邻居决定支持；
- 没有显式建模多个独立邻居的共识；
- 没有区分 evidence relevance 与对最终验证的 utility。

当前 retrieval residual 的 OOF 增益为正但很小，说明方向有效、接口仍有改进空间。

## 5. 文献审查与对 MPC 的对应关系

### 5.1 Recall-then-Verify

[Answering Open-Domain Multi-Answer Questions via a Recall-then-Verify Framework（ACL 2022）](https://aclanthology.org/2022.acl-long.128/)指出，一次性联合生成多个答案会使某个答案是否生成受到其他答案证据的干扰；将不同答案分开召回和验证能够更充分利用证据。

对 MPC 的对应关系：

- Scientist 先扩大候选召回；
- 每个候选或交换独立验证；
- 不让 Reviewer 直接比较三个很长的完整集合。

### 5.2 内生自我纠错的限制

[Large Language Models Cannot Self-Correct Reasoning Yet（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html)发现，没有外部反馈的 intrinsic self-correction 经常无效甚至降低表现。

这与本项目 v7、v9、v10 中 Reviewer 的持续负收益一致。Scientist 与 Reviewer 使用同一 DeepSeek 模型时，二者可能共享相同知识缺口，角色名不同并不能自动产生独立信息。

### 5.3 外部、可核查反馈

[CRITIC（ICLR 2024）](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fef126561bbf9d4467dbb8d27334b8fe-Abstract-Conference.html)强调工具交互和外部反馈对可靠纠错的重要性。

本项目暂不引入联网搜索，但仍可使用可核查的本地外部信号：

- train-only profile retrieval；
- occurrence/cooccurrence 统计；
- 官方 evidence 的候选级 occurrence 关系；
- FlavorDB molecule catalog 和固有属性；
- OOF 记录的同类交换历史效用。

### 5.4 独立验证问题

[Chain-of-Verification（Findings of ACL 2024）](https://aclanthology.org/2024.findings-acl.212/)将验证问题独立回答，以降低原始答案对验证过程的偏置。

对 MPC 的对应关系：

Reviewer 不应先看到“哪套是主答案”再解释它，而应分别回答：

1. 有什么证据支持加入候选 B？
2. 有什么证据反对保留候选 A？
3. 证据是否直接表示 occurrence？
4. 若没有足够证据，是否应该 abstain？

### 5.5 科学证据验证

[SciFact（EMNLP 2020）](https://aclanthology.org/2020.emnlp-main.609/)把科学验证分成证据检索、支持/反驳判断和 rationale 定位。

对 MPC 的对应关系：

- 候选出现应被写成原子 claim；
- evidence 必须绑定 candidate ID 和 evidence ID；
- 关系必须是 SUPPORT、REFUTE 或 INSUFFICIENT；
- rationale 必须能回指原始 evidence，而不是自由生成解释。

### 5.6 验证器需要客观结果信号

[LEVER（ICML 2023）](https://proceedings.mlr.press/v202/ni23b.html)利用输入、候选程序和执行结果训练 verifier，再与生成概率结合重排序。

MPC 没有程序执行结果，但 train grouped OOF 中有 hidden molecules，可把历史局部交换是否改善 hidden-set F1 作为 Reviewer/meta-gate 的客观效用标签。LLM 的自报 confidence 不能替代这个标签。

### 5.7 从相关性到验证效用

[From Relevance to Utility（Findings of EMNLP 2023）](https://aclanthology.org/2023.findings-emnlp.422/)指出，用于验证的证据检索应优化 verifier 实际获得的效用，而不只是一般相关性。

对 MPC 的对应关系：

- 不只检索“看起来相似的食品”；
- 应训练内评估某邻居或证据是否真正提高候选验证正确率；
- retrieval score 与 verifier utility 应分别记录。

### 5.8 结构化拒答

[Structured Output Learning with Abstention（ICML 2018）](https://proceedings.mlr.press/v80/garcia18a.html)表明，结构化输出也可以显式建模拒答。

在 MPC 中，`ABSTAIN` 不意味着输出空集合，而是：

> 不授权任何 Reviewer 交换，确定性回退到 H1。

## 6. 下一阶段方法：Evidence-Grounded Recall–Verify–Revise

### 6.1 总体结构

下一候选版本继续保持 Scientist–Reviewer，而不是转向本对话之外的 Multi-Agent：

1. H1 产生稳定 exact-N 基础集合。
2. Recall Scientist 产生扩大后的边界候选池。
3. Dossier Builder 为每个候选或交换绑定结构化证据。
4. Reviewer 对原子候选/交换做 SUPPORT、REFUTE、INSUFFICIENT 判断。
5. OOF utility gate 决定 Reviewer 的结论是否有执行权限。
6. Deterministic Reviser 执行少量合法交换并保证 exact-N。

### 6.2 H1：冻结已验证主干

保留：

- frequency；
- partial-conditioned cooccurrence；
- train-only profile retrieval；
- exact-N；
- 去重和 partial 排除。

H1 是默认答案，不再由完整集合 Reviewer 随意覆盖。

### 6.3 Recall Scientist：优化召回而不是直接生成最终集合

候选来源应保持相互可区分：

- occurrence prior；
- IDF/BM25 加权的 partial-profile retrieval；
- food-name/类别检索；
- 多邻居 profile residual 共识；
- cooccurrence expansion；
- 官方 evidence 中可规范化的 occurrence-linked molecule；
- LLM 候选扩展，仅作为未验证候选，必须映射到合法 molecule catalog。

Scientist 输出的不是三套完整集合，而是边界 action：

```json
{
  "action_id": "A001",
  "remove_candidate_id": "M012",
  "add_candidate_id": "M087",
  "candidate_sources": ["retrieval", "cooccurrence"],
  "evidence_ids": ["E03"],
  "retrieval_support": {
    "neighbor_count": 4,
    "independent_food_count": 3,
    "weighted_support": 0.71
  },
  "conflicts": [],
  "h1_margin_cost": 0.018
}
```

### 6.4 Candidate-level evidence dossier

每个 add/remove 候选至少记录：

- occurrence prior；
- cooccurrence mean/max；
- retrieval rank 与加权支持；
- 支持邻居数量和独立食品数量；
- H1 cutoff margin；
- direct occurrence evidence IDs；
- sensory/functional evidence IDs，但明确标记为非 occurrence；
- conflict/insufficient 状态；
- molecule catalog 映射状态；
- 来源 provenance。

证据关系不能再压缩成一个 `typed_evidence_linked` 布尔量。

### 6.5 Reviewer：验证交换，不选择长集合

Reviewer 输入：

- 一个匿名 A/B 局部交换；
- add 和 remove 的独立 dossier；
- 不显示 H1/H2、retrieval expert 等策略身份；
- 不显示未经校准的自报“专家权威”。

Reviewer 输出：

```json
{
  "action_id": "A001",
  "add_verdict": "SUPPORT|REFUTE|INSUFFICIENT",
  "remove_verdict": "SUPPORT|REFUTE|INSUFFICIENT",
  "cited_evidence_ids": ["E03"],
  "occurrence_grounded": true,
  "final_verdict": "ACCEPT|REJECT|ABSTAIN"
}
```

执行器必须机械验证：

- action ID 存在；
- evidence ID 存在且绑定对应候选；
- `occurrence_grounded=true` 时至少有一个 direct occurrence source；
- sensory/functional evidence 不能单独触发 ACCEPT；
- Reviewer 不能生成新候选；
- 任一检查失败即 ABSTAIN。

### 6.6 OOF utility gate

Reviewer 不是最终权威。训练内 grouped OOF 学习：

`P(local swap improves hidden set | dossier features, reviewer verdict)`

输入只包含通用特征：

- retrieval 共识；
- cooccurrence 差；
- occurrence prior 差；
- H1 margin；
- evidence relation 和独立来源数；
- Reviewer verdict；
- 是否涉及高置信 core。

不包含：

- test ID；
- test food 特例；
- released evaluation cache；
- CID 特例；
- 根据当前 71 条结果选择的 N 分桶。

只有 OOF 预测效用为正、且预登记准入条件通过的 action 才能执行。

### 6.7 Deterministic Reviser

确定性执行：

- 锁定 H1 高置信 core；
- 只修改 boundary；
- 每次 remove/add 保持等基数；
- 最大预算包含 0，并由 grouped OOF 选择；
- 去重；
- 排除 partial；
- 校验 molecule catalog；
- 任一失败回退 H1；
- 始终 exact-N。

## 7. 在实现 Reviewer 前必须完成的候选上限审查

Reviewer 能否产生显著提升，取决于正确候选是否已经进入 Scientist 的候选池。

下一次只读/离线准入需要计算：

1. `hidden-molecule Recall@N`
2. `Recall@(N+10)`
3. `Recall@(N+30)`
4. `Recall@2N`
5. candidate-pool exact-molecule oracle F1
6. FlavorDB 已知属性覆盖条件下的 functional-group oracle
7. 不同候选来源的独立新增 gold 数量
8. 每个来源的 unique gain 与 overlap

需要比较：

- H1 only；
- H1 + cooccurrence expansion；
- H1 + improved retrieval；
- H1 + direct occurrence evidence；
- H1 + LLM proposal；
- 全部 recall channels。

判断规则：

- 若扩大候选池的 oracle 相对 H1 没有明显上升，停止优化 Reviewer，继续改 Candidate Scientist。
- 若 oracle 明显上升但最终没有上升，瓶颈才属于 Reviewer/utility gate。
- 只有 oracle 上限和 Reviewer realization rate 同时改善，才允许正式运行。

## 8. Retrieval Scientist 的通用改进方向

当前 retrieval residual 已有小幅正 OOF，因此它是优先级最高的候选生成改进。

建议从当前 `token overlap + raw Jaccard + max support` 改为：

1. 对 partial molecules 使用 IDF/BM25 权重，降低常见分子的支配。
2. food-name、partial-profile、cooccurrence 分别检索，保留各自 provenance。
3. 对邻居 support 使用加权投票或校准聚合，不只取单个 max。
4. 记录独立食品数量，避免同源近重复 profile 伪造多证据。
5. 以 candidate Recall@K 和 verifier utility 联合选择检索策略。
6. Reviewer 发现 evidence insufficient 时，只允许一次本地 query reformulation，不允许自由生成最终集合。

这类改动是通用的“多答案检索—验证”设计，不依赖 FoodPuzzle 测试样本特例。

## 9. UniMol 路线变化

### 9.1 MFP

继续保留：

- 类别条件 UniMol set representation；
- OOF 融合；
- 固定候选 Reviewer。

后续补充同版本 `no_unimol` 消融，确认独立贡献。

### 9.2 MPC

下一正式候选暂时移除：

- UniMol feature scoring；
- UniMol residual proposal；
- UniMol set compatibility；
- Reviewer prompt 中的 UniMol 权重；
- 任何以结构相似推断食品出现的门控。

但必须保持独立、固定的 molecule catalog，避免关闭 UniMol 时意外缩小候选宇宙。

### 9.3 多构象

多构象 UniMol 暂不进入主线。若未来重新研究，只能作为独立实验：

- 冻结 Candidate Scientist、候选池、OOF folds、Reviewer 和预算；
- 只替换单构象表示；
- 若仍无正 OOF 增益，停止 MPC UniMol 路线；
- 无论结果如何，不影响 MFP 继续使用 UniMol。

## 10. 下一候选版本的准入标准

### 10.1 Candidate Scientist

- grouped OOF Recall@K 高于当前候选池；
- oracle 上限相对 H1 有实质空间；
- 至少一个扩展来源提供独立 gold 候选；
- 候选全部可规范化和审计；
- 不依赖测试集身份规则。

### 10.2 Reviewer

- 逐 action，而不是逐完整集合；
- 必须引用可验证 evidence ID；
- `ABSTAIN` 可用且默认安全；
- OOF accepted actions 中 wins > losses；
- accepted-action 平均 hidden-set F1 增益为正；
- 多数 fold 同向；
- Reviewer confidence 只记录，不决定权限。

### 10.3 Final Reviser

- budget=0 时逐项等于 H1；
- exact-N 100%；
- 无重复；
- 不包含 partial；
- 不生成候选池外分子；
- 所有改变都有 action ID、证据和 provenance。

### 10.4 正式测试

- 版本和阈值在 train grouped OOF 后冻结；
- test 只运行一次；
- 同时报告 H1、Candidate Oracle、Reviewer accepted subset、Final；
- 报告 Reviewer coverage、wins/losses 和净贡献；
- 配对 bootstrap 用于不确定性，不凭单点差异宣称显著。

## 11. 路线取舍

本次路线变化不是“UniMol 在整个项目中失败”，而是按任务对模块进行正确授权：

- MFP：UniMol 与任务方向对齐，继续作为核心结构专家。
- MPC：当前 UniMol 接口没有正 OOF 证据，暂时退出主预测。
- MPC Scientist：从完整答案生成器转为高召回候选与证据档案生成器。
- MPC Reviewer：从完整集合选择器转为候选/交换验证器。
- Reviser：从 LLM 自由融合转为确定性受约束执行器。

最终方法主张应是：

> 面向开放科学集合补全的 Evidence-Grounded Recall–Verify–Revise Scientist–Reviewer：以食品条件候选召回为基础，将复杂集合选择分解为可核查的候选级科学主张，通过证据关系约束、选择性拒答、训练内效用校准和 exact-N 确定性修正，减少无外部信息的自我纠错对强基础答案的破坏。

这比继续堆叠结构相似度、多个完整假设和自由 Reviewer 更符合历史实证、第一性原理与现有验证文献。

## 12. Train-only 候选上限审查结果

本轮新增：

- `scripts/audit_mpc_candidate_upper_bound.py`

执行协议：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
  scripts/audit_mpc_candidate_upper_bound.py
```

约束：

- 只读取 MPC train；
- normalized target food 五折 grouped OOF；
- seed=42；
- 每折重新拟合 v11 H1；
- 不读取 test label；
- 不调用 API；
- 不读取官方 LLM 功能团评测缓存；
- 不写入 `results/`；
- candidate catalog 独立来自 FlavorDB 和训练 profile 名称，不依赖 UniMol embedding。

### 12.1 候选召回

568 条训练 query 的 Macro Recall：

| Candidate channel | @N | @N+10 | @N+30 | @2N |
|---|---:|---:|---:|---:|
| v11 H1 | **0.681696** | 0.700902 | 0.722712 | **0.741141** |
| 当前 retrieval | 0.451613 | 0.478817 | 0.494249 | 0.489700 |
| IDF/containment retrieval | 0.678938 | **0.701993** | **0.731135** | 0.731136 |
| Cooccurrence-only | 0.435080 | 0.457395 | 0.470893 | 0.480915 |
| Direct-occurrence evidence | 0.000783 | 0.000783 | 0.000783 | 0.000783 |
| H1 + retrieval + cooccurrence RRF | 0.651588 | 0.688939 | 0.725745 | 0.735349 |
| 上述 RRF + evidence | 0.651822 | 0.689243 | 0.726029 | 0.735701 |

结论：

1. H1 仍是最强的 exact-N 排序，必须保留为默认答案。
2. IDF/containment retrieval 在 @N 没有超过 H1，但在 @N+30 达到 0.731135，高于 H1 的 0.722712，说明它更适合作为召回专家，而不是直接替代 H1。
3. 简单 RRF 在 @N 将召回从 0.681696 降至约 0.6518，证明“多个合理排序直接融合”仍会破坏强主干。
4. 当前离线 evidence 中可严格识别为 direct occurrence 且精确链接到分子名称的候选极少，不能承担主要候选召回。

### 12.2 多来源候选池 oracle

每个来源取 top-2N 后求集合并集：

- 平均候选池大小：263.79；
- Macro exact-molecule recall：0.776296；
- 因为 oracle 可从池中选择 exact-N，该值也是 candidate-pool exact-N molecule oracle F1；
- 相对 H1@2N：+0.035155；
- 247 条改善、0 条下降、321 条持平；
- 5/5 folds 为正。

各来源在其他来源之外独立新增的 gold 命中：

| 来源 | Unique gold hits |
|---|---:|
| H1 | 450 |
| IDF retrieval | **1067** |
| Cooccurrence | 135 |
| Direct evidence | 4 |

这些数据证明：

- H1 和 IDF retrieval 确实互补；
- Candidate Scientist 存在可利用的召回上限；
- 但这个上限来自更大的候选池和 oracle 选择，不是现成的可部署增益；
- RRF 的负结果说明真正瓶颈已经转向“如何识别池中的正确边界候选”。

### 12.3 功能团 oracle

在 FlavorDB intrinsic functional groups 可映射的 99.6383% gold molecules 上，使用 gold group 做不可部署的 greedy exact-N oracle：

| Candidate pool | Greedy functional-group Macro-F1 |
|---|---:|
| H1 top-2N | 0.950807 |
| 多来源 union top-2N | 0.958845 |

必须谨慎解释：

- 这不是官方 LLM-cache 功能团评测；
- oracle 直接读取 gold molecule 的功能团，只表示候选池覆盖能力；
- H1 候选池本身已经含有很高的功能团覆盖上限；
- 多来源扩展带来的功能团 oracle 增量只有约 +0.0080；
- 当前正式输出与 oracle 之间的巨大差距主要是选择问题，而不只是候选缺失问题。

所以 MPC 的下一步不能只继续扩大候选池，还必须训练一个不读取测试指标的候选级 utility verifier。

## 13. 审查后的下一步决策

候选上限审查已经排除了两条路线：

1. 不应让 IDF retrieval 或其他单一通道直接替代 H1。
2. 不应对多个排序做无条件平均或 RRF 后直接输出。

已获得支持的路线：

1. H1 锁定高置信 core。
2. IDF retrieval 作为主要 Candidate Scientist 召回扩展。
3. Cooccurrence 只补充少量独立候选。
4. Direct evidence 不负责广泛召回，只作为少数候选的高质量验证证据。
5. Scientist 将多来源候选转换为局部 add/remove action dossier。
6. Reviewer 验证 action，而不是选择完整集合。
7. train grouped OOF utility gate 学习哪些 Reviewer verdict 可以执行。
8. exact-molecule hidden-set F1 是主要训练效用；FlavorDB intrinsic group coverage 只能作为辅助，不能覆盖明显有害的成员交换。

因此，下一候选版本的实现顺序应为：

1. 使 MPC candidate catalog 与 UniMol 完全解耦。
2. 将 IDF/containment retrieval 加入候选生成，但不加入最终全局融合。
3. 构造 action dossier 与可机械验证的 evidence relation。
4. 用 grouped OOF 训练/校准 local action utility。
5. 只有 utility gate 通过后才调用 Reviewer。
6. Reviewer 失败、拒答或证据引用无效时逐项回退 H1。

在完成以上离线准入前，不需要正式 API 运行。
