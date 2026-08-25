# FoodPuzzle 项目综合理解与复现审计

- 日期：2026-08-01
- 范围：论文、作者 GitHub 官方仓库、本地数据处理、三条近似复现、统一评测、UniMol / Multi-Agent / Optimized-Agent 探索
- 性质：只读审计与离线检查
- 未执行：模型 API、正式预测、正式结果覆盖、Git 暂存/提交
- 新增文件：仅本审计记录

## 1. 总体判断

本项目的主线可以清晰分成两层：

1. **近似复现层**：在官方公开资源不完整的条件下，重建 MFP / MPC 输入、80/10/10 split、zero-shot、BM25 ICL、Scientific Agent 和公开评测逻辑。
2. **优化探索层**：围绕 UniMol、任务隔离 Multi-Agent、Scientist–Reviewer 权限收缩、PU / retrieval、exact-N 和 functional-group metric alignment 进行迭代。

这条路线本身是合理的，而且本地仓库已经主动标注了 reconstructed split、FlavorDB-derived MPC 和非 Table-2 exact reproduction。当前最大的障碍不是“没有做实验”，而是：

- 官方发布本身不足以精确复现；
- 当前工作树、README、代码版本和结果版本不同步；
- reconstructed random split 中存在大量相同/近重复 molecular profile；
- 结果目录存在多套历史评测口径，README 数值与当前 summary 不一致；
- 优化版本长期覆盖同一路径，历史正式产物只剩审计记录；
- 缺少依赖锁定、自动测试、统一 manifest 和安全的密钥管理边界。

因此，现阶段最重要的不是继续增加方法层数，而是先冻结一套**可追溯、可重跑、无近重复污染歧义**的复现协议。

## 2. 论文理解

论文：`FOODPUZZLE: Developing Large Language Model Agents as Flavor Scientists`，arXiv:2409.12832v3。

### 2.1 任务

- **MFP**：输入分子集合 `M`，输出食品来源 `F`；评测时映射到 21 个宏观类别，报告 accuracy。
- **MPC**：输入食品 `F`、部分已知分子 `M_partial` 和缺失数量 `n`，输出 `M_missing`；评测预测/金标准分子集合对应的 functional-group 并集 F1。

论文报告数据规模为 978 foods / 1,766 flavor molecules，并称按 80/10/10 划分。

### 2.2 方法

- Foundation LLM zero-shot：LLaMA3-8B-Instruct、Gemini-1.5-Pro、GPT-3.5-Turbo。
- Domain LLM：MolT5、BioT5 微调。
- ICL：BM25 / Sentence Transformer / DPR，最终 BM25 最优；passage 形如 `Food: F'. Molecules: M'`。
- Scientific Agent：
  1. MFP 根据训练频率的信息熵选最多 10 个低熵分子；MPC 不做 starting-point selection。
  2. 离线 Google Custom Search 收集 food/molecule evidence。
  3. BM25 检索 3 条训练 demonstrations。
  4. Scientist 生成 3 个 hypotheses。
  5. Reviewer 选择或拒绝 hypothesis，得到最终预测。

### 2.3 论文 Table 2

| 方法 | 模型 | MFP Accuracy | MPC F1 |
|---|---|---:|---:|
| Zero-shot | LLaMA3-8B | 15.0% | 0.292 |
| Zero-shot | Gemini-1.5-Pro | 19.0% | 0.340 |
| Zero-shot | GPT-3.5 | 12.2% | 0.327 |
| ICL | LLaMA3-8B | 31.6% | 0.349 |
| ICL | Gemini-1.5-Pro | 34.6% | 0.373 |
| ICL | GPT-3.5 | 23.2% | 0.360 |
| Scientific Agent | LLaMA3-8B | 35.5% | 0.374 |
| Scientific Agent | Gemini-1.5-Pro | 34.2% | 0.333 |
| Scientific Agent | GPT-3.5 | 26.9% | 0.374 |

论文正文两次声称 Appendix B 提供 prompt，但当前 11 页 v3 PDF 不包含 Appendix B。Prompt、DSPy signature、split seed、完整输入构造均无法从论文恢复。

## 3. 官方 GitHub 审计

### 3.1 仓库身份

- URL：https://github.com/tenghaohuang/FoodPuzzle
- 所有者：论文第一作者 Tenghao Huang
- 当前 main commit：`febc2660b29ec9f9955d22d5e8261157e77a8997`
- 最新提交日期：2025-02-22
- 完整历史：6 commits；无其他 branch、tag、release。

### 3.2 实际公开内容

官方仓库仅包含：

- `code/evaluation.py`
- `data/raw/flavordb.db`
- `data/processed/MFP_tasks.jsonl`
- `data/processed/MPC_tasks.jsonl`
- 两个 collected-evidence pickle
- README / IDE 文件

README 所列 `code/scientific_agent/` 从未出现在任何历史 commit 中；README 的 MIT badge 指向不存在的 `LICENSE`。

### 3.3 官方评测脚本不可独立运行

公开 `evaluation.py` 依赖未发布的：

- `DSP_functions.py`
- `config.py`
- `utils.py`
- DSPy signature / prompt
- baseline result pickle
- pickle 版 FlavorDB

而仓库发布的是 SQLite `flavordb.db`。此外脚本 main 中无参调用需要必填 `results` 的 `eval_task2()`。因此官方仓库不是完整 reference implementation，而是**数据 + 不完整 evaluator fragment**。

### 3.4 官方数据规模与论文不一致

当前公开资源实际为：

- `food_entities`: 936
- `molecules`: 1,781
- `molecules_all`: 25,239
- MFP rows: 709
- MPC rows: 710

与论文/README 的 978 foods、1,766 molecules、25,595 total molecules 不完全一致。公开 task 是 benchmark 的一个子集或不同快照，但官方未说明过滤规则。

## 4. 本地官方资源与数据处理

### 4.1 官方文件逐字节一致

以下本地文件与作者仓库当前 main 完全一致：

| 文件 | SHA-256 |
|---|---|
| `collected_evidences_task1.pkl` | `42428e09c5c0ba3eda39b5e3efbe83aebe568f473d86b9ec5b3ba03ef577292b` |
| `collected_evidences_task2.pkl` | `230100bbe14f2c3c57c9eb2c03a44e9c85fc3be1d6f5d6f6a1ac82919e06dde1` |
| `MFP_tasks.jsonl` | `a2d54ed47308c3d94e5b4d0b4d65965905694fab6f6a82b7b558031896697c2a` |
| `MPC_tasks.jsonl` | `420aad296c54b46ba3fbeec59b2ba9ee646eceebcfe125e19476c3fe1f3f459a` |
| `flavordb.db` | `f377cac1186ae8572ffe7855dc93c84fbb538727b34caf6d1a81f0b3a199e259` |

### 4.2 MPC reconstruction

`reconstruct_mpc_data.py`：

- 分级匹配 FlavorDB food alias；
- 710/710 food 唯一匹配；
- 710/710 public missing list 是 FlavorDB full profile 子集；
- `partial = full - missing`；
- `n = len(missing)`；
- 重新生成与当前 `MPC_reconstructed_tasks.jsonl` 字节级一致。

709 个 MFP/MPC 共用 food 的 reconstructed full profile 与 MFP molecules 归一集合全部一致；MPC 额外一项为 Parsley。因此 reconstruction 很可能恢复了公开快照对应的 known set，但官方未发布原始 `partial/n`，仍不能称 exact reproduction。

### 4.3 Split

`split_data.py` 使用：

- seed 42；
- row-level shuffle；
- 80/10/10；
- MFP 567/71/71；
- MPC 568/71/71。

两套 split 均可确定性重生成并与当前 JSONL 字节级一致。

### 4.4 近重复与污染风险

完整 profile 审计：

| 任务 | rows | unique exact profiles | duplicate-profile rows | 最大 group | test exact profile 已在 train |
|---|---:|---:|---:|---:|---:|
| MFP | 709 | 467 | 257 | 72 | 23/71 |
| MPC | 710 | 468 | 257 | 72 | 23/71 |

MFP test 到 train 最近 Jaccard：

- median ≈ 0.9211；
- 37/71 ≥ 0.9；
- 23/71 exact profile overlap。

一些完全相同的 molecule profile 对应不同宏观类别，说明存在输入不可辨识或数据抽取默认 profile。随机 split 的绝对分数同时测量 profile 记忆和泛化。

MPC BM25 的 top-1 demo missing set 对 test gold 已有约：

- exact-molecule precision 0.665；
- recall 0.650；
- F1 0.643。

现有 ICL 输出 exact-molecule F1 约 0.706。这个高分主要来自相似/重复 profile 与 labeled train demonstrations，而不是与论文同条件的可比提升。

### 4.5 Evidence

- Task1：1,535 molecule keys，14,760 snippets。
- Task2：936 food keys，9,358 snippets。
- 当前 MFP/MPC split 均有 evidence coverage。
- pickle 只保存 snippet 字符串，缺少 URL、title、repository、query、timestamp 等结构化 provenance。
- 多条 snippet 明显不是 PubMed/arXiv，无法核验论文所称搜索域约束。

## 5. 三条近似复现

### 5.1 Zero-shot

优点：

- MFP prompt 只读 molecules；MPC 只读 target food / partial / n。
- MPC 输出做去重、过滤 partial、按 n 截断。
- 当前 prediction 文件不含 test `missing_molecules`。

差异/风险：

- 使用 DeepSeek / AIHubMix 模型，不是论文三模型配置。
- API 异常多数被折叠成 `parse_failed`；脚本只要生成 output 文件就可打印 PASS。
- 无统一 run manifest / prompt hash。

### 5.2 BM25 ICL

优点：

- MFP corpus/query 与论文 `Food + Molecules` 描述基本一致。
- 纯 Python BM25、固定 k1/b、稳定 tie-break。
- query 不读 test gold，demos 只含 train labels。
- retrieval metadata 完整落盘。

差异/风险：

- MPC passage 在公开字段不足下被重建为 food + partial + n；不是可验证的官方逐行实现。
- resume 可能追加同 ID 的失败重试行；evaluator last-row-wins。
- metadata 没有绑定 train/test/top-k/hash，改变数据后 resume 可能留下陈旧 metadata。
- near-duplicate profile 使 ICL 结果明显偏向记忆。

### 5.3 Scientific Agent

优点：

- Scientist 三 hypotheses + Reviewer 两阶段结构与论文相符。
- MFP/MPC 都使用 BM25 top-3 demos。
- MPC food-centered official evidence、Reviewer 默认不读 raw evidence，与论文描述大体一致。
- MPC 输出只按 partial / n 做 normalization，不读 gold。

差异/风险：

- MFP 正式 DeepSeek run 使用 5 个 starting molecules；论文为最多 10。
- 使用 train IDF 近似低 entropy，并优先 evidence availability；不等价于公开的 entropy 定义。
- MFP 使用 test-label-aware answer masking；只有 1/71 样本、5 个 occurrence 实际被遮蔽，但它仍是 gold-aware preprocessing，不是论文公开流程。
- prompt / parser / Reviewer schema 均为本地 reconstruction。
- 不保存 raw model response，难以独立重放 parser。
- status PASS 不代表零失败；metadata 在 resume 后可能与最新 prediction 不完全绑定。

## 6. Evaluation 审计

### 6.1 MFP

本地 evaluator：

- 从 SQLite 建 21 类 gold mapping；
- 用固定 judge 把 predicted free text 映射到 21 类；
- accuracy 分母包含缺失/失败 prediction。

这比官方 fragment 更可运行，也修复了官方 `Vegetable` 大小写重复。但 MFP mapping 没有共享 cache 或 prompt-version metadata，同一预测的重评仍可能受 judge 漂移影响。

### 6.2 MPC

本地 evaluator忠实重建官方公开逻辑：

- gold molecule groups 来自 FlavorDB；
- predicted molecule groups 由 LLM 限制到公开 53-item vocabulary；
- 对每样本求 set precision / recall / F1 / IoU，再宏平均；
- 共享 cache 绑定 provider/model/endpoint/prompt version/vocabulary hash。

已发现的固有问题：

- 官方空格切分 parser 产生 5 个词表外/异常 gold token：`compound`、`or`、`(sulfanyl`、`(dialkylamine)`、`(trialkylamine)`。
- test gold group 中平均约 88.2% 在 53-item predicted vocabulary 内，gold/pred mapping 不对称。
- empty cache entry 同时表示“合法空组”和“judge/parse failure”，且会永久共享；当前 DeepSeek cache 2,910 keys 中 194 个为空。
- MFP 与 MPC 的 judge 与论文 GPT-3.5 不同；Only-Deepseek 又使用 DeepSeek 自评，因此绝对分数不能直接对照 Table 2。

## 7. 当前可直接复核的结果

### 7.1 DeepSeek 基线

| 方法 | MFP Accuracy | MFP Correct | MPC 当前 shared-cache F1 | MPC exact molecule F1（诊断） |
|---|---:|---:|---:|---:|
| Zero-shot | 0.2113 | 15/71 | 0.5355 | 0.1241 |
| BM25 ICL | 0.2394 | 17/71 | 0.6758 | 0.7060 |
| Scientific Agent | 0.2958 | 21/71 | 0.6012 | 0.4901 |

MPC result README 仍写旧值 0.2366 / 0.2810 / 0.2660，与当前 evaluation summary 不一致；应以带 details 的当前 summary 为准，同时保留旧值只作历史记录。

### 7.2 探索结果

| 方法 | MFP | MPC F1 | 状态 |
|---|---:|---:|---|
| Heterogeneous Multi-Agent v1 | 30/71 = 0.4225 | 0.6565 | 当前 code/output/hash 一致，可直接复核 |
| Optimized Agent v12 | 33/71 = 0.4648 | 0.6539 | 当前结果为 v12；当前代码已是 v13 |
| Historical best MFP v9 | 35/71 | 0.6737 | 仅审计记录，正式文件已被覆盖 |
| Historical best MPC v2 | 19/71 | 0.6819 | 仅审计记录，正式文件已被覆盖 |
| Current code v13 | check-only PASS | check-only PASS | 无正式 v13 result |
| v14 | 不改 MFP | 仅设计 | 尚未实现 |

当前 v13 check-only 复核结果：

- MFP：567/71、53,617/53,617 UniMol occurrences mapped、candidate ledger 30、PASS。
- MPC：568/71、UniMol 不使用、71/71 exact-N、Reviewer API 0。
- metric-aligned budget 1：OOF mean group-F1 gain `+0.00526929`，bootstrap lower bound `+0.00049949`，102 wins / 86 losses，3/5 positive folds。

但 OOF 按 unique target food 分组，而 568 个 train food 基本唯一；相同/近重复 full profiles 可跨 fold。v14 记录已正确识别这个问题，下一步必须改为 exact/near-profile cluster OOF。

### 7.3 Multi-Models 当前未完成

目前只有：

- coding-glm MFP zero-shot：71 rows，其中 46 parse failures；
- DeepSeek MPC zero-shot：71 rows；
- GPT-4.1 shared cache：仅 2 keys；
- 其余 model/method/evaluation 尚未形成完整结果。

`run_multi_models.sh` 会无限重试不完整输出且吞掉子命令失败，缺少最大重试/失败退出策略。

## 8. 优化路线理解

### 8.1 Multi-Agent

该路线不是简单 persona voting，而是：

- MFP：训练 profile retrieval + occurrence + frozen UniMol + evidence critic + deterministic arbiter。
- MPC：train-only PU/occurrence/retrieval candidate model + optional evidence/swap arbiter + exact-N planner。
- test gold 在进入 prediction components 前显式删除。

优点是任务隔离、exact-N、run metadata/hash 完整；缺点是 MPC UniMol/NNPU 路线后来被多轮 OOF 否定，且尚无完整消融。

### 8.2 Optimized Agent v1-v13

纵向规律：

- MFP 的稳定正信号来自 category-conditioned UniMol set adapter、直接输出 macro category、受约束 Reviewer。
- MPC 的稳定正信号来自 occurrence H1、BM25/IDF retrieval、exact-N 和有限集合覆盖。
- MPC 的反复负信号来自 raw/global UniMol、profile 外伪负例、全局固定融合、自由 Reviewer/Fusion 和 exact-molecule objective 与 functional-group metric 错位。
- v12 Reviewer 10/10 abstain；v12 正式结果实际等于新 H1。
- v13 改为无 UniMol、FlavorDB intrinsic group demand、最多一次 metric-aligned boundary swap；只有有限 OOF 正证据。

### 8.3 v14 方向

v14 设计是当前最合理的 MPC 下一步：

1. 冻结强 H1；
2. high-recall counterfactual Action Bank；
3. group-cardinality Expected-F1 Scientist；
4. independent metric/evidence Reviewer；
5. risk-controlled exact-N executor；
6. profile-cluster nested OOF；
7. Scientist 未通过离线准入前不调用 Reviewer。

当前代码仍是 v13；v14 尚未实现。

## 9. UniMol 审计

- 本地 UniMol clone：`deepmodeling/Uni-Mol` commit `90f52c41299a1a582da0f9765e9f87aa21faa16a`。
- embedding：1,777 molecules × 512，全部 finite，name/CSV/metadata 一致。
- 模型权重在嵌套 UniMol 仓库内未跟踪。
- NPZ 只记录 names/smiles/embeddings，没有 UniMol commit、checkpoint hash、package version、RDKit version、seed/config。
- 日志显示 7 个 molecule 的 3D conformer 失败并使用非 3D fallback：`1-Popc`、`Calcium Oxide`、`hydrochloric acid`、`hydrogen cyanide`、`hydrogen sulfide`、`Silica`、`sulfur dioxide`。

结论：当前 embedding 足以重用，但尚不能从仓库独立再生同一表示。MFP 的 UniMol 收益需要 frozen full/no-Unimol 单变量消融；MPC 当前不应使用 UniMol。

## 10. 当前仓库状态与高优先级风险

### P0：密钥文件

- `API-KEY.txt` 当前 working copy 非空（354 bytes）。
- Git index 已跟踪/暂存同名空文件，状态为 `AM`。
- 即使 `.gitignore` 已写入 `API-KEY.txt`，tracked file 仍可能在后续 `git add` 中把密钥加入 commit。
- `.env.example` 同时处于 staged deletion。

必须在下一次提交前安全处理；本审计未读取或输出密钥内容。

### P0：README / 实体不一致

README 和 `code/README.md` 宣称存在：

- `code/agent_v2_best.py`
- `code/agent_v2_unimol_fusion.py`
- `code/prompts/`
- `results/agent_v2/`
- `results/agent_v2_unimol_fusion/`

当前均不存在。README 中 Agent V2 / fusion 数值没有可审计产物支持。

### P1：版本与结果错位

- 当前 `optimized_agent.py` / runner 是 v13；正式目录是 v12。
- 直接运行 `run_optimized_agent.sh` 会因版本/hash 不匹配删除同目录 v12 artifacts 后写 v13。
- 历史 v1-v11 已被覆盖，只剩文字审计记录。
- 当前工作树 105 个 non-ignored untracked files；核心新代码/结果尚未纳入 Git。

### P1：复现工程缺失

- 无 `pyproject.toml` / requirements / lock file；
- 无 tests / CI / lint 配置；
- 无顶层 LICENSE / CITATION；
- baseline Only/Multi 两套近重复代码易漂移；
- hard-coded `/Library/Frameworks/.../python3`，跨机器不可移植；
- `validate_data.py` 只做存在性/JSON/SQLite 结构检查，不验证 hash、语义、split 互斥、重建等式、gold isolation 或 near duplicates。

### P1：holdout 名称误导

`data/holdout` 的 100 MFP/MPC rows 全部仍位于现有 train split。它只是从 train 中复制出的候选集合；除非训练时从 corpus 移除这些 rows，否则不能作为独立 holdout 评测。

## 11. 建议执行顺序

### 阶段 A：先冻结可复现基线

1. 处理 `API-KEY.txt` tracking 风险并恢复无值 `.env.example`。
2. 冻结当前工作树到独立 branch/commit；不要覆盖已有结果。
3. 统一 README，使路径、方法版本、结果与实体一致。
4. 建立 `pyproject.toml` / lock，至少锁定 Python、NumPy、scikit-learn；UniMol 单列 optional environment。
5. 为每个 run 写 immutable manifest：code/data/split/prompt/model/judge/cache/output hash。
6. 把旧口径结果放入 `results/archive/<protocol-id>/`，不再同目录覆盖。

### 阶段 B：定义两套评测协议

1. **Paper-aligned reconstructed protocol**：保留 seed42 random split，用于和现有所有结果连续比较。
2. **Decontaminated protocol**：按 exact/near full-profile cluster split，报告类别分布、profile overlap 和外推性能。
3. 对 MFP 同时报 overall accuracy 与按类别 macro accuracy / coverage。
4. 对 MPC 同时报 official-compatible FG-F1、exact-molecule F1、candidate recall、exact-N 和 unmapped rate。
5. MFP judge 增加 cache/version；MPC cache 区分 valid-empty 与 parse/API failure。

### 阶段 C：完善自动审计

至少加入测试：

- official file SHA；
- MPC reconstruction 710/710 与字节稳定；
- split count/disjoint/coverage；
- profile duplicate/nearest-Jaccard report；
- prompt/query gold-field isolation；
- parser fixtures；
- functional-group vocabulary/parser snapshot；
- prediction unique ID / no duplicate / exact-N；
- run metadata hash integrity；
- no tracked secret files。

### 阶段 D：再推进优化

- MFP：冻结 v9/v12 可复核 control，运行 full vs no-UniMol 单变量消融，再决定是否继续 Reviewer。
- MPC：先实现 v14 Phase 1 的 profile-cluster OOF、Action Bank 和 Expected-F1 Scientist；未过准入不调用 LLM Reviewer。
- 不继续扩大候选池、增加 Agent 层或引入多构象，直到独立增益得到验证。

## 12. 本轮实际检查

- 论文 11 页逐页渲染与全文提取阅读：完成。
- 官方 GitHub 当前 main 与完整 6-commit history：完成。
- 官方/本地五个资源 SHA 与 byte compare：通过。
- MPC reconstruction 临时重建 byte compare：通过。
- MFP/MPC split 临时重建 byte compare：通过。
- `validate_data.py`：PASS（仅代表其现有结构性检查）。
- 当前 Python 文件 `py_compile`：PASS。
- Optimized Agent v13 MFP check-only：PASS。
- Optimized Agent v13 MPC check-only：PASS。
- holdout selector check-only：PASS。
- 当前 results JSONL parse、ID uniqueness、summary/detail 聚合：完成。
- run metadata/output hash：Multi-Agent 与 v12 artifacts 内部一致；v12 code hash与当前 v13 code不同，符合版本错位判断。
