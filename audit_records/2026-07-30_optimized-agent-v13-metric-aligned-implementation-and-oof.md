# optimized-agent v13：无 UniMol 的 MPC 指标对齐实现与 OOF 准入

- 日期：2026-07-30
- 范围：Only-Deepseek optimized-agent
- 本轮权限：修改正式代码并执行 check-only / train-only grouped OOF
- 本轮未执行：模型 API、正式预测、正式评测、结果覆盖、结果目录创建、Git 更新或提交
- 上游审查：`2026-07-30_optimized-agent-v12-failure-and-mpc-direction-reset-audit.md`

## 1. 目标

v13 不再尝试通过 UniMol 相似度、同模型 Reviewer 或 exact-molecule action gate 提升 MPC。实现目标为：

1. MFP 路径保持原有单构象 UniMol；
2. MPC 完全不加载 UniMol；
3. 保留 occurrence H1 作为冻结控制；
4. 从 FlavorDB 的 molecule-local `functional_groups` 确定性读取官能团；
5. 使用 train-only retrieval 估计 missing-set functional-group demand；
6. 在 H1 边界执行至多一次 exact-N functional-group-aware swap；
7. 只按 grouped OOF macro functional-group F1 准入；
8. v12 exact-molecule action gate 和 Reviewer 修改权限关闭。

## 2. 代码变更

修改：

- `code/Only-Deepseek/optimized_agent.py`
- `scripts/run_optimized_agent.sh`

方法版本：

`optimized_agent_v13`

### 2.1 MPC 官能团来源

MPC 从本地 FlavorDB：

- `molecules.common_name`
- `molecules.functional_groups`
- `molecules_all.common_name`
- `molecules_all.functional_groups`

构造分子到官能团的确定性映射。

方法侧不读取：

- LLM functional-group evaluation cache；
- test gold；
- `entity_molecule_link` 中测试食品的完整 profile；
- UniMol embeddings。

### 2.2 Functional-group demand

对每个 query：

1. 使用 food token 与 partial molecule profile 检索 15 个训练邻居；
2. 读取邻居训练样本的 missing molecules；
3. 将其映射为官能团集合；
4. 使用 retrieval relevance 和 rank discount 聚合官能团支持；
5. 与 train-only 全局官能团先验做固定比例平滑。

该模块估计的是：

`P(functional group needed | target food, partial molecules)`

而不是：

`P(candidate occurs | molecular structure similarity)`

### 2.3 Exact-N metric-aligned decoder

decoder 从 occurrence H1 开始，只允许在边界候选中执行一次交换。

候选效用同时考虑：

- 当前集合对预测官能团需求的 soft F1；
- 分子新增官能团的边际覆盖；
- occurrence rank 的轻量保护。

策略预先冻结为单次边界交换。budget 2/3/5 只保留为诊断，不能因本次 OOF 数值较好而被选为正式策略。

### 2.4 Scientist–Reviewer

v12 action gate 以 exact missing-molecule F1 为目标，与正式官能团 F1 不一致，因此：

- action ranker 不再训练；
- action policy 固定 budget 0；
- check-only 计划 API 调用为 0；
- Scientist–Reviewer 保留 typed-evidence audit 接口；
- occurrence claim 与 functional-replication claim 必须匹配相应 evidence relation；
- 在没有 metric-aligned 增量准入前，Reviewer 没有修改预测的权限。

## 3. MPC grouped OOF 结果

协议：

- 568 个训练样本；
- 按规范化 target food 分组；
- 5 folds；
- 每折重新训练 occurrence H1；
- 不读取 test；
- 不读取 LLM evaluation cache；
- 不使用 UniMol；
- 指标为每样本 functional-group set F1 的宏平均；
- paired bootstrap 500 次；
- 还检查按训练 OOF 的中位 `n=83` 分成上下两半后的收益方向。

### 3.1 Functional-group decoder

| Budget | Changed | Wins | Losses | Mean FG-F1 gain | Bootstrap lower bound | Positive folds | 准入 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 191 | 103 | 86 | **+0.00530381** | **+0.00053941** | 3/5 | **是** |
| 2 | 236 | 108 | 100 | +0.00517618 | -0.00043439 | 3/5 | 否 |
| 3 | 293 | 117 | 108 | +0.00364596 | -0.00150683 | 3/5 | 否 |
| 5 | 345 | 123 | 111 | +0.00419468 | -0.00094463 | 3/5 | 否 |

budget 1 的基数稳定性：

- `n <= 83`：平均增益 +0.00882749
- `n > 83`：平均增益 +0.00103974

五折增益：

- fold 0: -0.00150907
- fold 1: +0.01640868
- fold 2: +0.00891465
- fold 3: -0.00873834
- fold 4: +0.01137319

判断：

- budget 1 通过当前预设准入条件；
- 收益较小且存在两个负 fold，只能称为“有限正证据”；
- budget 2/3/5 虽然均值为正，但置信下界为负，必须拒绝；
- 不能据此宣称正式 test 已提升。

### 3.2 Legacy retrieval/action

原 retrieval residual 仍可作为诊断，但它的准入指标是 exact hidden-molecule F1，不进入 v13 最终 metric-aligned proposal。

v12 Reviewer action：

- action ranker 未训练；
- changed queries: 0；
- action budget: 0；
- Reviewer planned calls: 0；
- disabled reason: `exact_molecule_objective_not_aligned_with_functional_group_f1`。

## 4. check-only 结果

### 4.1 MPC

- status: PASS
- method: `optimized_agent_v13`
- train/test: 568/71
- UniMol mapped occurrences: 0
- exact-N: 71/71
- candidate ledger: 56–292
- functional-group decoder budget: 1
- Reviewer samples: 0
- planned API calls: 0

### 4.2 MFP 回归

- status: PASS
- method: `optimized_agent_v13`
- train/test: 567/71
- UniMol mapped occurrences: 53617/53617
- UniMol molecules: 1777
- UniMol dimension: 512
- candidate ledger: 30/30

说明 MPC 修改没有破坏 MFP 的 UniMol 路径。

## 5. 工程审计

第一次 check-only 被主动终止。原因不是死锁，而是每个 OOF fold 仍重复训练已被 v13 禁用的 v12 action/boundary ranker。

随后删除了这条无预测权限的训练路径并重新运行。第一次运行：

- 未调用 API；
- 未写入结果；
- 未生成日志或中间文件；
- 未留下可恢复断点。

完整第二次 check-only 通过。

剩余主要运行成本是：

- 五折重新训练 H1；
- 568 个留出 query 遍历完整合法候选目录。

后续可以做等价缓存或向量化，但不能通过缩小候选目录改变评测定义。

## 6. 证据边界与下一决策

v13 当前证明的是：

> 在 train-only grouped OOF 中，完全不使用 UniMol、最多一次边界替换的 metric-aligned functional-group decoder，相对同折 occurrence H1 获得小幅但 bootstrap 下界为正的增益。

v13 当前没有证明：

- 当前正式 71 样本会提升；
- 会超过 v2 历史 0.6819；
- 会超过 v9 0.6737；
- 当前代码中的 occurrence H1 与历史 v7/v9 H1 完全相同；
- Scientist–Reviewer 对 MPC 有正增量。

在正式运行前仍需用户单独授权。正式运行后必须分别记录：

1. occurrence H1；
2. metric-aligned decoder；
3. typed-evidence Reviewer（当前应为 disabled）；
4. final exact-N；
5. 按 `n` 与官能团覆盖分层的增减；
6. 相对 v12、v10、v9、v2 和 BM25-ICL 的差异。

只有正式结果超过强基线后，v13 才进入消融；否则应回到 H1 重建或候选信息源，而不是增加更多 Agent 层。
