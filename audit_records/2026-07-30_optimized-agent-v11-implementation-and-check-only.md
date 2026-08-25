# Optimized Agent v11 实现与 Check-only 审查

- 日期：2026-07-30
- 路线：Only-Deepseek Scientist–Reviewer
- 方法版本：`optimized_agent_v11`
- 本轮范围：正式代码实现、训练内准入检查、MFP/MPC 本地回归
- API：未调用
- 正式结果：未运行、未覆盖
- Git：未执行更新、暂存或提交

## 1. 实际修改范围

仅修改：

- `code/Only-Deepseek/optimized_agent.py`
- `scripts/run_optimized_agent.sh`

仅新增本审查记录。没有创建新的代码目录、结果目录、缓存或测试结果文件。

## 2. v11 已实现的方法变化

### 2.1 训练观测机制

移除 v10 每条食品额外构造的 15%、35%、60% 低缺失率 query。

v11 直接使用 MPC train 中的 task-shaped partial/missing：

- Train 平均缺失率约 0.8687。
- Test 平均缺失率约 0.8593。
- 每个原始食品只贡献其真实任务形态的 query，不再被三个容易 surrogate query 稀释。

### 2.2 H1 食品相关性主干

H1 只使用：

- frequency；
- partial-conditioned cooccurrence mean/max；
- retrieved-profile residual support。

UniMol 和感知属性不再进入 H1 的主排序特征，避免弱结构信号污染食品相关性主干。

训练仍采用低容量、低权重的 positive–unlabeled pairwise ranking；代码和元数据没有将其表述为严格 nnPU。

### 2.3 移除 v10 全局 set energy

已停止训练和使用：

- hard exact-cardinality corrupted-set ranker；
- learned global set-energy decoder；
- structure-seeded global set search。

这些模块不再拥有重写完整 H1 集合的权限。

### 2.4 Grouped OOF 准入

新增 train-only 五折 grouped OOF：

- group 为规范化后的 `target_food`；
- 同一食品的派生记录不跨折；
- 选择指标为 hidden molecule set F1；
- 不读取 LLM 功能团评测缓存；
- 不使用 N 分桶；
- 不使用 test ID、食品名规则或 test 指标。

候选预算：

- Retrieval H2：1、2；
- Structural UniMol H3：1、2；
- Local attribute complementarity：1、2、3；
- 预算0始终存在，并严格返回 H1。

准入条件：

- 至少25条 OOF query；
- 至少5条发生改变；
- Macro hidden-molecule F1 平均增益为正；
- paired bootstrap 95% 下界为正；
- 改善次数大于损害次数；
- 多数 fold 的方向为正。

### 2.5 H2/H3 权限

- H2 是检索支持的截断边界残差，只允许1–2个局部名额。
- H3 是 task-adapted UniMol/分子属性边界残差。
- 局部互补只在 H1 边界候选中运行。
- 未通过 OOF 的通道预算自动为0，输出逐项退化为 H1。

### 2.6 Selective Reviewer

Reviewer 的调用条件改为：

- 至少一个残差专家通过 OOF；
- proposal 确实与 H1 不同；
- 分歧边界上存在明确候选级 evidence，或两个独立残差支持相同加入候选。

Reviewer 输出现在允许：

- 选择一个预先准入的 exact-N proposal；
- `ABSTAIN`，确定性回退 H1。

自然语言 confidence 只记录，不作为权限门控。

重复 proposal 只保留一个 eligible 代表，不再形成伪多数。

### 2.7 Deterministic Fusion

正式推理路径不再调用 Fusion LLM：

- Reviewer 只能选择固定 exact-N proposal 或弃权；
- Fusion 只确定性执行选择；
- 校验去重和 exact-N；
- 不生成新分子，不进行第二轮自由改写。

## 3. MPC Check-only 结果

运行环境：

- `PYTHONDONTWRITEBYTECODE=1`
- `PYTHONHASHSEED=0`
- DeepSeek API 未加载、未调用

总体检查：

| 检查项 | 结果 |
|---|---:|
| Train samples | 568 |
| Test samples | 71 |
| UniMol mapped occurrences | 5739/5739 |
| Candidate ledger min/max | 56/292 |
| Deterministic exact-N | 71/71 |
| Reviewer-gated samples | 1/71 |
| 预计 Scientist API calls | 1 |
| 预计 Reviewer API calls | 1 |
| 预计 Fusion API calls | 0 |

最终准入策略：

```json
{
  "retrieval": {"global": 1},
  "structural": {"global": 0},
  "complementarity": {"global": 0}
}
```

### 3.1 Retrieval H2

预算1：

- OOF queries：568；
- changed：166；
- Macro hidden-molecule F1 gain：+0.0019075；
- bootstrap 95% lower bound：+0.0001432；
- wins/losses：50/25；
- positive folds：3/5；
- 结论：通过。

预算2：

- 平均增益仍为正，约 +0.0012876；
- bootstrap 下界为负；
- 结论：拒绝。

因此 v11 只允许一次 retrieval boundary exchange。

### 3.2 Structural UniMol H3

预算1：

- changed：505；
- 平均增益：-0.0041166；
- wins/losses：56/156；
- positive folds：1/5；
- 结论：拒绝。

预算2：

- 平均增益：-0.0079957；
- positive folds：1/5；
- 结论：拒绝。

所以当前单构象 task-adapted UniMol 残差没有进入正式 v11。代码保留通用接口和严格零预算回退，后续可以在不改变其他模块的情况下继续研究结构适配目标。

### 3.3 Local complementarity

预算1：

- 平均增益：-0.0118861；
- bootstrap 下界：-0.0158336；
- positive folds：0/5；
- 结论：拒绝。

预算2、3同样为负，均拒绝。

说明 v5 的“集合互补”阶段收益不能直接迁移到当前更强 H1；现有 soft functional-demand 实现会替换正确成员。该思想只能保留为待研究模块，不能因历史上曾有效就强行组合。

## 4. MFP Check-only 回归

MFP 没有更改方法逻辑。

| 检查项 | 结果 |
|---|---:|
| Train/Test | 567/71 |
| UniMol mapped occurrences | 53617/53617 |
| Adapter enabled | True |
| Candidate ledger | 每条30 |
| Check status | PASS |

本轮没有调用 MFP API，因此不能声称正式准确率仍为35/71；只能确认结构、依赖和候选生成路径通过本地回归。

## 5. 方法判断

本次准入结果没有为了维持“三专家形式”而强行启用 UniMol：

- H2 有小而严格为正的训练内证据，因此进入。
- H3 和互补项虽然具有方法动机，但当前实现的实证方向为负，因此归零。
- Reviewer 仍存在于 Scientist–Reviewer 方法中，但覆盖率由可验证证据决定。
- v11 的正式 MPC 在绝大多数样本上等价于强 H1，只在训练内验证过的检索边界和极少数证据充分样本上介入。

这比“直接关闭 Reviewer”更容易在方法上解释：

> Reviewer 是一个选择性风险控制层；当候选专家没有通过准入或当前实例没有独立证据时，正确动作就是弃权，而不是假装审查一定有价值。

## 6. 尚未证明的内容

Check-only 不能证明：

- v11 的正式 MPC F1 一定高于 v9/v2；
- OOF 的小增益会转化成官方功能团 F1 增益；
- 单构象 UniMol 已经发挥有效作用；
- Reviewer 的唯一一次调用一定改善结果；
- MFP 正式结果一定复现35/71。

因此下一步只能是一次冻结后的正式运行，而不是继续查看 test 后调准入参数。

## 7. 正式运行建议

如果获准运行：

1. 使用现有 `scripts/run_optimized_agent.sh`。
2. 继续覆盖现有 `results/Only-Deepseek/optimized-agent/...`，不创建新结果层级。
3. 先完成 MFP 回归，再完成 MPC。
4. MPC 预计仅1条进入 Scientist–Reviewer，共2次生成调用；Fusion为0次调用。
5. 评测继续使用 Only-Deepseek 共享功能团缓存。
6. 完成后保存 v11 正式结果审查，并与 v2、v7、v9、v10 分阶段对比。

在用户明确允许正式运行前，不启动 API 任务。
