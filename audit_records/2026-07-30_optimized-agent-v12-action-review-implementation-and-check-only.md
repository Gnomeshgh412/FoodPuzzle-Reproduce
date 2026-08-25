# optimized-agent v12：MPC candidate-action Scientist–Reviewer 实现与离线准入

## 1. 本轮目标

本轮不以测试集指标为调参依据，也不启动正式 API。目标是把此前的 MPC 整集假设选择改成更保守、可审计的局部 action：

1. MFP 继续使用冻结的单构象 UniMol；
2. MPC 不加载、不使用 UniMol；
3. MPC 的基础输出始终是 occurrence H1 exact-N；
4. IDF profile retrieval 只负责提出边界 add/remove action；
5. action 必须先通过训练集 grouped OOF utility gate；
6. Scientist 审计每个已准入 action，Reviewer 最多接受一个 action，否则回退 H1；
7. 最终执行器是确定性的，并强制 exact-N。

## 2. 实现范围

修改正式文件：

- `code/Only-Deepseek/optimized_agent.py`
- `scripts/run_optimized_agent.sh`

没有创建调试脚本、测试结果目录或临时报告；没有调用模型 API，没有覆盖现有正式结果，没有进行 Git 更新或提交。

## 3. 关键实现

### 3.1 任务级 UniMol 隔离

- `task=mfp` 且非 `no_unimol` 时加载 UniMol；
- `task=mpc` 时即使 `ablation=full` 也不实例化 UniMol；
- MPC 候选目录由 FlavorDB molecule catalog、`molecules_all.common_name` 和训练 profile 构成，与 UniMol 覆盖率解耦。

### 3.2 IDF retrieval action

对 food token 与 partial molecule containment 进行 IDF 加权，聚合近邻 profile 的缺失分子支持。它不直接覆盖 H1，只构造边界交换：

`remove ∈ H1 boundary, add ∉ H1`

每个 action 使用低容量逻辑回归估计 utility，并记录 occurrence、legacy retrieval、IDF retrieval、近邻支持数量及 H1 margin cost。

### 3.3 grouped OOF utility gate

- 按规范化 target food 分组；
- 5-fold；
- 每个 fold 重新训练 occurrence 与 action 模型；
- 选择指标为 hidden-molecule exact-set F1 的 paired gain；
- 要求平均增益为正、bootstrap 下界为正、wins > losses、超过半数 fold 为正；
- 不读取测试标签、官方 functional-group evaluation cache 或 API。

为降低策略选择对均值噪声的敏感性，最终阈值选择规则冻结为：

`最大化正的 paired-bootstrap lower bound，再比较 mean gain 与 threshold`

### 3.4 action dossier 与证据语义

每个 action dossier 包含：

- add/remove molecule；
- occurrence 与 retrieval 统计；
- OOF utility probability；
- 合法 evidence IDs；
- direct occurrence evidence IDs；
- independent statistical support count。

只有 `occurrence_support` 可作为直接出现证据；sensory、odor 或 functional relation 不能证明分子实际存在于目标食物中。

### 3.5 Scientist–Reviewer 约束

- Scientist 必须逐一审计所有已准入 action；
- Reviewer 只能选择一个已准入 action 或 `ABSTAIN`；
- 引用的 evidence ID 必须属于该 action；
- action 至少需要 direct occurrence 引用，或两个独立统计支持；
- Scientist 明确 refute 的 action 不允许执行；
- Reviewer 输出非法或证据不足时确定性回退 H1；
- 不再使用 LLM fusion 重写整套分子。

## 4. check-only 结果

### 4.1 MFP

- status: PASS
- method: `optimized_agent_v12`
- train/test: 567/71
- UniMol mapped occurrences: 53617/53617
- candidate ledger: 30/30
- task adapter: enabled

MFP 逻辑没有在本轮被重新设计，只验证了 v12 的任务级 UniMol 隔离没有破坏原路径。

### 4.2 MPC

- status: PASS
- method: `optimized_agent_v12`
- train/test: 568/71
- UniMol mapped occurrences: 0
- exact-N: 71/71
- candidate ledger size: 56–292
- maximum Scientist prompt: 4165 characters
- evidence-linked candidate occurrences: 30

最终 action policy：

- threshold: 0.90
- budget: 1
- OOF queries: 568
- changed queries: 41
- wins/losses: 31/2
- positive folds: 5/5
- mean hidden-molecule F1 gain: 0.00049068
- paired bootstrap 95% lower bound: 0.00032735
- test-side gated samples: 10/71
- planned Scientist/Reviewer calls: 20

作为对照，0.75 阈值虽然均值略高，但只有 4/5 positive folds，bootstrap 下界为 0.00023053，且有 18 个 loss；因此没有采用。

## 5. 审计判断

### 5.1 已通过

- MPC 已在代码层面与 UniMol 解耦；
- action 候选和效用准入只使用训练数据；
- exact-N 与 H1 fallback 是硬约束；
- Reviewer 不能自由生成、融合或扩大候选集合；
- 证据关系类型得到显式区分；
- 0.90 action gate 在所有 fold 上为正，且置信下界为正。

### 5.2 尚未证明

- OOF gain 数值很小，只能证明候选 action 模块在训练分布上通过保守准入，不能宣称 MPC 已显著提升；
- check-only 不能证明 DeepSeek Scientist–Reviewer 会正确接受有益 action；
- 也不能证明官方 functional-group F1 一定提升，因为准入目标刻意没有读取该评测缓存；
- 完整 5-fold 训练成本较高，属于工程风险。

## 6. 下一决策点

v12 已具备正式运行前的离线资格，但是否覆盖当前 v10 正式结果必须由用户再次授权。若正式运行，首先应比较：

1. H1 fallback 样本与 action-gated 样本的分层收益；
2. Reviewer 对 10 个 gated 样本的 accept/abstain/error 分布；
3. action 的实际 win/loss，而不仅是总体均值；
4. MPC Precision、Recall、F1、IoU 相对 v10、v9 和 Only-Deepseek baseline 的变化。

只有正式评测确认 MPC 有正收益后，v12 才能进入后续消融。
