# Looped LM 上的 RL：实验计划

- 状态：草案 v0，2026-11-20。还没有做任何实验，文中的预期都是假设。
- 修订：2026-11-20，RSI 的基线受到公开质疑，涉及 RSI 的条目已调整，记录在 3.6 节。
- 配套讲解页：LoopLM RL 设计笔记.md
- 记号：`P_t` 是第 t 个 loop 的 next-token 分布，`q_t` 是 exit 分布，`ω` 是 RLTT 的 loop 权重，`ρ_t ∝ ω_t·P_t(y)` 是后验，`A` 是 advantage，`gap = log Σ_t ω_t P_t(y) − Σ_t ω_t log P_t(y)`。

## 1. 目标

用尽量少的训练回答三个问题，再据此决定主线。

1. RLTT 的收益是否来自 `A > 0` 的 gap 项（跨 loop 蒸馏），`A < 0` 的 gap 项是否有害。
2. 把深度或 exit 当作策略的一部分并用 outcome reward 训练，能否在总计算量对准确率的前沿上胜过固定深度和现有 gate。
3. 在跨 loop 自蒸馏（RSI）上加 reward，是否有额外收益。前提是自蒸馏本身的增益能复现（H0）。

对应三个方案：方案 1（Sandwiched RLTT）、方案 2（Exit-marginal RL，分序列级 2a 和逐 token 2b）、方案 3（以更深的自己为参考的自蒸馏）。主线押方案 2，方案 1 作为诊断和消融。方案 3 原定保底，依据是 RSI 报告的增益；这个结果现在有争议（3.6 节），H0 通过之前不算保底。

## 2. 要检验的假设

| 编号 | 假设 | 什么结果算推翻 | 阶段 |
|---|---|---|---|
| H0 | D-LoopOPD（`ψ = 0`）在我们自己的评测下高于未训练的基座，差值超出重复评测的波动 | 差值落在波动之内 | 3 |
| H1 | RLTT 训练中，`A < 0` 样本上的平均 gap 上升，`A > 0` 样本上的下降 | 两者同向变化，或没有差别 | 1 |
| H2 | gap 系数从 `A` 改成 `max(A, 0)` 后，长预算准确率不低于 RLTT；系数取 0 时回答不再变短 | 系数 0 的臂同样变短，或 sandwich 更差 | 1 |
| H3 | loop 和 CoT token 互相替代：固定深度越浅，回答越长 | 回答长度与深度无关 | 0 |
| H4 | 每个要用的深度都需要带 reward 的 rollout；多深度 RL 后浅层准确率明显高于只在 4 个 loop 上做 RL | 单深度 RL 的浅层表现已经相当 | 2 |
| H5 | 深度由 reward 训练后，相同准确率下的总计算量低于固定深度和现有 gate | 前沿没有外移 | 2、4 |
| H6 | reward-tilted 蒸馏在 over-looping 的题上优于纯 D-LoopOPD，整体不差于它 | 所有 `ψ > 0` 都不优于 `ψ = 0` | 3 |

## 3. 实验设定

### 3.1 模型

| 模型 | 用途 | 备注 |
|---|---|---|
| Ouro-1.4B-Thinking | 跑全部消融臂 | 迭代更便宜。基线以 Ouro 论文 Table 9、Table 12 和我们的实测为准，不用 RSI Table 2 的 Vanilla 行（3.6 节） |
| Ouro-2.6B-Thinking | 复现关键对比 | RLTT 和 RSI 的主模型 |
| TML-d24-varloop-research | 可选的第二个 testbed | 26.4B 总参数、3.4B 激活的 MoE，只有权重没有 API，算力门槛高 |

### 3.2 数据与 reward

- 训练集：MATH train split（RLTT 用）或 DAPO-Math-17k（RSI 用）。先用 MATH，便于和 RLTT 的数字对照。
- reward：答案与标准答案匹配记 1，否则 0。advantage 用组内均值和标准差归一化。

### 3.3 默认超参

取自 RLTT 论文 Table 6，作为所有臂的起点。

| 项 | 值 |
|---|---|
| loop 数 | 4 |
| 训练步数 | 140 |
| 每步题数 × 每题 rollout 数 | 32 × 8 |
| 最大生成长度 | 2048 |
| 学习率 | 1e-6，constant with warmup，warmup 比例 0.1 |
| 优化器 | AdamW 8bit，β = (0.9, 0.99)，weight decay 0.1 |
| 梯度裁剪 | 最大范数 0.1 |
| KL 系数 | 1e-3（对冻结的初始策略） |
| 精度 | BF16 |

### 3.4 基线

- 未训练的基座。数字由我们的脚手架实测，与各训练臂同批评测，不引用别的论文里的 Vanilla 行。
- GRPO，4 个 loop。学习率单独扫（见第 10 节的风险 1）。
- RLTT，`ω` 取 exit 分布。
- D-LoopOPD（RSI）：`KL(P₂ ‖ sg P₄)`，rollout 来自 loop 2。它相对基座的增益有争议，先过 H0。

### 3.5 开工前要确认的工程前提

- [ ] Ouro 的 Hugging Face 实现能否在一次前向里取出每个 loop 的 logits 和 gate 值。
- [ ] RL 框架。RLTT 用的是 VERL 加 vLLM rollout；我们用什么待定。
- [ ] RLTT 的代码是否公开。不公开就按论文 Eq. 4–7 重写。
- [ ] RSI 的代码。arXiv v2 仍写录用后放出，没有检索到公开仓库；公开评论的作者说跑的是官方代码，来源待确认（可能是 OpenReview 的补充材料）。能拿到就用官方代码做 H0 的复现，拿不到再按论文 Eq. 11 重写。
- [ ] 推理时能否按请求指定固定 loop 数（阶段 0 和阶段 2 需要）。
- [ ] 是否有可用的逐 token 早退推理实现（Ragged KV 或 continuous depth batching）。没有的话阶段 4 的部署评测降级为估算。

### 3.6 RSI 的基线争议（2026-11-20 记录）

OpenReview 上有两条公开评论（2026-11-06 和 11-09，同一位评论者）质疑 RSI 的基线：

- RSI Table 2 的 Vanilla 行低于 Ouro 论文 Table 9。
- 评论者称跑了官方代码，重复十次，D-LoopOPD 与 Vanilla 的成绩和波动都相同。日志尚未公开。

评论内容来自转贴，OpenReview 页面和作者的回复还没有核对到。第一点可以直接对照两篇论文。下表中 Ouro 的数字读自 arXiv v1 的 Table 9（R4），RSI 的数字读自 arXiv v2 的 Table 2（d=4）。Ouro 论文没有定义 pass@1 的采样次数，这里把它当平均准确率，与 Avg@16 对照。

| 模型 | 项目 | Ouro 论文 | RSI Vanilla | 差 |
|---|---|---|---|---|
| 2.6B | AIME24 平均：pass@1 对 Avg@16 | 64.7 | 62.08 | −2.6 |
| 2.6B | AIME24：pass@10 对 Pass@16 | 90.0 | 83.33 | −6.7 |
| 2.6B | AIME25 平均 | 50.3 | 48.96 | −1.3 |
| 2.6B | AIME25：pass@10 对 Pass@16 | 76.7 | 73.33 | −3.4 |
| 2.6B | OlympiadBench | 76.4 | 64.30 | −12.1 |
| 2.6B | BeyondAIME | 39.0 | 35.00 | −4.0 |
| 1.4B | AIME24 平均 | 65.0 | 51.25 | −13.8 |
| 1.4B | AIME24：pass@10 对 Pass@16 | 83.3 | 73.33 | −10.0 |
| 1.4B | AIME25 平均 | 46.3 | 41.25 | −5.1 |
| 1.4B | AIME25：pass@10 对 Pass@16 | 73.3 | 70.00 | −3.3 |
| 1.4B | OlympiadBench | 71.6 | 59.40 | −12.2 |
| 1.4B | BeyondAIME | 34.0 | 24.00 | −10.0 |

怎么读这张表：

- 评论里的对照属实。采样从 10 次加到 16 次，Pass@16 反而低于 pass@10，方向不对。
- 幅度要分开看。2.6B 的 AIME 平均准确率两边差不到 3 个点；AIME 每套 30 题，pass@k 上一题是 3.33 个点，2.6B 的差距是 1 到 2 题。明显偏低的是两个模型的 OlympiadBench，以及 1.4B 的 AIME24 和 BeyondAIME。
- 两边写的解码设置相同（temperature 1.0，top-p 0.7），判分方式不同：Ouro 用内部脚手架加 LLM 判分，RSI 抽取 boxed 答案后做数值和符号等价判断。Ouro 没有写生成长度上限，RSI 是 31,744。
- 判分方式解释不了全部。同一张表里 Qwen3-4B 和 Qwen3-8B 的 AIME、BeyondAIME 与 Ouro Table 9 基本一致（Qwen3-4B 的 AIME24 是 61.25 对 61.3），OlympiadBench 低 5 到 6 个点，而 Ouro 的行低 12 个点。Qwen3-1.7B 的 AIME 在 RSI 里反而高出十几个点。原因未知。
- 第二点（复现不出增益）目前无法核实。RSI 没有报告种子数、方差或置信区间，也没有解释与 Ouro 论文的差别。它的主要结论是 2.6B 的 AIME24 Avg@16 从 62.08 升到 71.67，量自 30 道题。

对本计划的处理：

1. RSI 的数字不作为基线，也不作为脚手架的校验目标。校验对 Ouro 论文，见第 4 节。
2. D-LoopOPD 相对基座的增益当作待检验的假设（H0），不当作已知结果。方案 3 的定位随之调整，见第 1 节和第 7 节。
3. 我们自己的结果都报告波动，基座与训练臂同批评测，见第 9 节。
4. 第 10 节风险 1、2 里引用 RSI Table 2 的地方已加说明。方案 1、方案 2 和阶段 0 不依赖 RSI。

## 4. 阶段 0：只测不训

目的：在花训练算力之前估计三个方案各自的空间，并验证评测脚手架。

先做一项校验：基座的准确率要能对上论文，对不上就先修脚手架。RSI 的 Vanilla 行不作为校验目标（3.6 节）。

- 逐深度：RLTT Table 9（MATH-500），Ouro Table 12 和 Table 13（AIME、OlympiadBench）。
- 4 个 loop：Ouro Table 9。AIME 的答案是整数，受判分方式影响小，作为主要校验点；2.6B-Thinking 的 AIME24 是 pass@1 64.7、pass@10 90.0，AIME25 是 50.3、76.7。OlympiadBench 在 Ouro 论文里用 LLM 判分，规则判分会偏低，只作参考。
- 生成长度上限要放够。RSI Table 6 里，基座的 AIME24 Avg@16 在 8K、16K、31K 上限下是 32.29、55.00、62.08。这些数字本身待核实，趋势可以参考：上限不够会直接量低基线。
- 基座重复评测若干次并记录波动，它是后面所有对比的噪声底。

| 编号 | 测什么 | 怎么算 | 决定什么 |
|---|---|---|---|
| M1 | gap 的集中程度 | 4 个 loop 的 rollout 上逐 token 算 gap，报告 gap 总量里前 1% 和前 5% token 的占比 | 方案 2b 的空间 |
| M2 | 答对与答错样本的 gap 差异 | 按 rollout 的对错分组比较平均 gap | 方案 1 是否值得做 |
| M3 | 逐 token 一致率 | loop 2、3 与 loop 4 的 top-1 一致的 token 比例 | 早退能省多少计算 |
| M4 | over-looping 比例 | 同一批题分别在固定 2、3、4 个 loop 下生成，统计浅层对而 4 个 loop 错的题，以及取并集的 oracle 准确率 | 方案 2a 的准确率空间，方案 3 是否需要 reward |
| M5 | token 与 loop 的替代 | M4 的 rollout 上统计各深度的平均回答长度和总计算量 | H3；计价方式 |
| M6 | 现有 gate 的表现 | 用 Ouro 自带的 Q-exit 和一个置信度阈值做早退，画准确率对平均深度 | 方案 2 要打败的基线 |

决策规则（阈值是暂定的，看到数据后再定）：

- M4 的 oracle 比固定 4 个 loop 高不到约 2 个点，且 M3 的一致率低：放弃方案 2，主线转方案 3。
- M3 一致率高但 M6 里早退掉点多：方案 2 继续，训练里要加入 ragged 模式。
- M2 没有差异：方案 1 降级为只做一组对照。

## 5. 阶段 1：方案 1 与基线

目的：搭好训练管线，复现 GRPO 与 RLTT，检验 H1、H2。

目标函数统一写成 `A·log Σ_t ω_t P_t(y) − γ·gap`，各臂只差 `γ`。

| 臂 | `γ` | 含义 |
|---|---|---|
| GRPO | 不适用 | 只用 `log P₄(y)` |
| 精确项 | 0 | 只有混合似然，没有 gap 项 |
| RLTT | `A` | 论文原式 |
| Sandwich | `max(A, 0)` | 方案 1 |
| 常数 | 训练初期 `|A|` 的均值 | gap 项与 advantage 符号解耦 |

每步记录：reward、平均回答长度、最后一个 loop 的熵、`A > 0` 与 `A < 0` 样本上的平均 gap、各 loop 与 loop 4 的 top-1 一致率。

通过标准：见 H1、H2。评测按第 8 节。

## 6. 阶段 2：方案 2a，序列级的深度动作

目的：检验 H4、H5。这是方案 2 的第一版，rollout 是固定深度的静态图，节省的计算是真实的，不依赖早退推理。

做法：

- 每条 rollout 采一个深度 `k`，按固定深度生成。`k` 的候选集合由阶段 0 决定，默认 {2, 3, 4}。
- 策略梯度作用在 `log q(k|x) + log π^(k)(y|x)` 上，同一道题的一组 rollout 共用一个基线。
- return 里扣总计算量：`R − β·C`，`C = (Σ_j k_j) / (4 × 最大生成长度)`。

| 臂 | 深度怎么来 | `β` |
|---|---|---|
| GRPO@4 | 固定 4 | 0 |
| 多深度 | 在候选集合上均匀分层 | 0 |
| 多深度加计价 | 同上 | 扫 0.05、0.1、0.2 |
| 学习的深度先验 | `q(k|x)` 是 prompt 末位状态上的线性头，初始偏向 4 | 上一臂里最好的 |

通过标准：多深度臂在 2、3 个 loop 下的准确率明显高于 GRPO@4 在同深度下的准确率（H4）；最后一臂的准确率对总计算量曲线在固定深度的各点之外（H5）。

## 7. 阶段 3：方案 3

目的：先检验 H0，再检验 H6。可与阶段 2 并行。

损失：`KL(P₂ ‖ sg P₄) − ψ·A·log P₂(y)`，rollout 来自 loop 2。

先跑 `ψ = 0` 一臂，与基座比较（H0）。RSI 报告它在 2.6B 的 AIME24 Avg@16 上比基座高 9.6 个点，公开评论称复现不出差别（3.6 节）。这一臂跑两个训练种子，评测按第 9 节重复。H0 不成立时，H6 的对照改为基座和 GRPO@2，问题变成 reward 能否让跨 loop 蒸馏起作用，是否继续投入另行决定。

| 臂 | `ψ` | 备注 |
|---|---|---|
| D-LoopOPD | 0 | 复现 RSI，同时检验 H0 |
| reward-tilted | 两到三档 | 先量 KL 项与 reward 项的梯度范数，取使两者同量级的值及其上下各一档 |
| GRPO@2 | 只有 reward 项 | 去掉 KL |

通过标准：见 H6。另外报告各臂在 2、3、4 个 loop 下的准确率，看浅层的收益是否带到深层。

## 8. 阶段 4：方案 2b，逐 token 的 exit 求和

前提：阶段 0 的 M1、M3 显示有空间，且阶段 2 的结果为正。

做法：

- forward 返回 `log Σ_t q_t P_t` 作为 logits，rollout 仍算满 4 个 loop。
- 损失是对 `π_mix` 的 PPO 目标，`q` 保留梯度，return 里扣总计算量（`k_j` 用 `Σ_t t·q_t` 的期望）。
- `q` 初始偏向 loop 4，`β` 从 0 逐步调大。

评测分两层：先在算满 4 个 loop 的模式下看准确率和 `q` 是否随 token 变化；有早退推理实现时，再按 `q` 采样 exit 做真实部署评测。训练与部署的落差大时，在更新里交替加入 ragged 模式。

## 9. 评测协议（所有阶段通用）

- **按总计算量对齐。** 报告准确率对 loop-token 数（每个 token 用的 loop 数之和），不只报告 token 预算下的准确率。
- **多个解码预算。** 1024、2048、4096，再加一个不设限的档。主结论看长预算。AIME 类的题不设限档要到 31K 左右，上限不够会量低所有模型（见第 4 节的校验）。
- **采样评测。** 用 Avg@16 一类的多次采样加配对检验，温度沿用所复现论文的设置，在阶段 0 定下来。
- **报告波动。** 每个数字给出重复评测的均值和标准差。AIME 每套 30 题，一题是 3.33 个点，结论以 Avg@k 为准，Pass@k 只作参考。H0 的 D-LoopOPD 臂跑两个训练种子，其余对比的种子数看到阶段 0 的波动后再定。
- **基线同批评测。** 未训练的基座与每个训练臂用同一脚手架、同一设置、同一批次评测，保留逐样本日志。不拿别的论文里的基座数字做对比。
- **逐深度。** 每个模型都在 1 到 4 个 loop 下评测。
- **数据集。** MATH-500、AIME24、GSM8K、BeyondAIME；迁移看 GPQA 和 MBPP。
- **可监控性。** 至少报告 reasoning token 数。有余力时加一个 monitor 指标，因为方案 2 和方案 3 都会把推理从可见 token 挪进 loop。

## 10. 风险

1. **基线偏弱或被量低。** RLTT 论文里 GRPO 的 reward 曲线几乎不动。RSI Table 2 里 RLTT 低于未训练的模型，但那张表的 Vanilla 行有争议（3.6 节），只能当线索。GRPO 的学习率要单独扫，各臂要比较更新范数，否则结论可能只是有效步长的差别。未训练的基座同样要量准，RSI 的争议就出在这一步。
2. **reward 在 Ouro 上可能提不了多少准确率。** Ouro 论文 4.5 节说自己的 RL 尝试没有明显超过 SFT checkpoint。RSI Table 2 里 GRPO 只比 Vanilla 高 0.8 个点，这个差值依赖有争议的 Vanilla 行。方案 2 的评价因此以计算量前沿为主。
3. **训练与部署的落差（方案 2b）。** 见阶段 4。
4. **简单阈值可能已经够好。** M6 就是为了提前看到这一点。
5. **被抢先。** 方案 3 离 RSI 最近。阶段 3 与阶段 2 并行，不排在最后。
6. **方案 3 的前提可能不成立。** 方案 3 假设不用 reward 的跨 loop 蒸馏已经有增益，reward 是在它之上的修正。H0 不成立时，方案 3 没有可靠的 `ψ = 0` 起点。对策见阶段 3。

## 11. 算力估计

参考点：RLTT 论文 Table 1 报告，Ouro-2.6B 上一次 140 步训练在 4 张 H200 上约 49 到 54 小时，约合 200 GPU 小时。下表按这个参考点粗估，1.4B 会更便宜，具体倍数要实测。

| 阶段 | 训练次数 | 说明 |
|---|---|---|
| 0 | 0 | 只有推理 |
| 1 | 约 7 | 5 个臂，加 GRPO 学习率的两档 |
| 2 | 约 6 | 含 `β` 的三档 |
| 3 | 约 6 | rollout 只算 2 个 loop，单次更便宜。含 `ψ = 0` 的第二个种子 |
| 4 | 约 4 | |

全部在 2.6B 上跑约 23 次，上限约 4600 GPU 小时。评测的开销没有计入：AIME 类的题要 31K 左右的生成上限，再加重复评测，具体数字在阶段 0 实测。建议全部臂先在 1.4B 上跑，2.6B 只复现每个阶段的关键对比。

## 12. 待决定

- 可用的算力和硬件: L20X * 8
- 主模型用 2.6B-thinking。
- RL 框架: slime https://github.com/THUDM/slime
- 暂时不把 TML 的 varloop 模型作为第二个 testbed。
- 论文范围：恒等式加方案 1 和方案 2 一篇。方案 3 是否单独成文，等 H0 的结果。
- 保底方案：方案 3 在 H0 通过前不算保底，替代选项待定（候选：恒等式加方案 1 的诊断结果）。

## 13. 参考

- RLTT：https://arxiv.org/abs/2602.10520
- TraceRL：https://arxiv.org/abs/2509.06949
- Ouro：https://arxiv.org/abs/2510.25741
- RSI / D-LoopOPD：https://arxiv.org/abs/2610.10623 （预印本，基线有公开质疑，见 3.6 节）
- LoopRPT：https://arxiv.org/abs/2603.19714
- Enabling Dynamic Computation in Looped LMs（Ragged KV）：https://arxiv.org/abs/2610.09013
- Adaptive Depth in Looped Transformers（gate 诊断）：https://arxiv.org/abs/2607.20519
- Continuous Depth Batching：https://arxiv.org/abs/2608.09444
- SPG：https://arxiv.org/abs/2510.09541
- Inkling 2 Mini：https://thinkingmachines.ai/news/inkling-2-mini/
- TML-d24-varloop-research 模型卡：https://thinkingmachines.ai/model-card/tml-d24-varloop-research/
