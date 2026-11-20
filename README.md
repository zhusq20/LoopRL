# Looped LM 上的 RL：实验计划

- 状态：草案 v0，2026-11-20。还没有做任何实验，文中的预期都是假设。
- 配套讲解页：LoopLM RL 设计笔记.md
- 记号：`P_t` 是第 t 个 loop 的 next-token 分布，`q_t` 是 exit 分布，`ω` 是 RLTT 的 loop 权重，`ρ_t ∝ ω_t·P_t(y)` 是后验，`A` 是 advantage，`gap = log Σ_t ω_t P_t(y) − Σ_t ω_t log P_t(y)`。

## 1. 目标

用尽量少的训练回答三个问题，再据此决定主线。

1. RLTT 的收益是否来自 `A > 0` 的 gap 项（跨 loop 蒸馏），`A < 0` 的 gap 项是否有害。
2. 把深度或 exit 当作策略的一部分并用 outcome reward 训练，能否在总计算量对准确率的前沿上胜过固定深度和现有 gate。
3. 在跨 loop 自蒸馏（RSI）上加 reward，是否有额外收益。

对应三个方案：方案 1（Sandwiched RLTT）、方案 2（Exit-marginal RL，分序列级 2a 和逐 token 2b）、方案 3（以更深的自己为参考的自蒸馏）。主线押方案 2，方案 3 保底，方案 1 作为诊断和消融。

## 2. 要检验的假设

| 编号 | 假设 | 什么结果算推翻 | 阶段 |
|---|---|---|---|
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
| Ouro-1.4B-Thinking | 跑全部消融臂 | RSI 给了它的基线数字，迭代更便宜 |
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

- 未训练的基座。
- GRPO，4 个 loop。学习率单独扫（见第 10 节的风险 1）。
- RLTT，`ω` 取 exit 分布。
- D-LoopOPD（RSI）：`KL(P₂ ‖ sg P₄)`，rollout 来自 loop 2。

### 3.5 开工前要确认的工程前提

- [ ] Ouro 的 Hugging Face 实现能否在一次前向里取出每个 loop 的 logits 和 gate 值。
- [ ] RL 框架。RLTT 用的是 VERL 加 vLLM rollout；我们用什么待定。
- [ ] RLTT 的代码是否公开。不公开就按论文 Eq. 4–7 重写。
- [ ] RSI 的代码写的是录用后放出，目前按论文 Eq. 11 重写。
- [ ] 推理时能否按请求指定固定 loop 数（阶段 0 和阶段 2 需要）。
- [ ] 是否有可用的逐 token 早退推理实现（Ragged KV 或 continuous depth batching）。没有的话阶段 4 的部署评测降级为估算。

## 4. 阶段 0：只测不训

目的：在花训练算力之前估计三个方案各自的空间，并验证评测脚手架。

先做一项校验：基座在 1 到 4 个 loop 下的准确率要能对上论文（RLTT Table 9、Ouro 的逐深度表）。对不上就先修脚手架。

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

目的：检验 H6。可与阶段 2 并行。

损失：`KL(P₂ ‖ sg P₄) − ψ·A·log P₂(y)`，rollout 来自 loop 2。

| 臂 | `ψ` | 备注 |
|---|---|---|
| D-LoopOPD | 0 | 复现 RSI |
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
- **多个解码预算。** 1024、2048、4096，再加一个不设限的档。主结论看长预算。
- **采样评测。** 用 Avg@16 一类的多次采样加配对检验，温度沿用所复现论文的设置，在阶段 0 定下来。
- **逐深度。** 每个模型都在 1 到 4 个 loop 下评测。
- **数据集。** MATH-500、AIME24、GSM8K、BeyondAIME；迁移看 GPQA 和 MBPP。
- **可监控性。** 至少报告 reasoning token 数。有余力时加一个 monitor 指标，因为方案 2 和方案 3 都会把推理从可见 token 挪进 loop。

## 10. 风险

1. **基线偏弱。** RLTT 论文里 GRPO 的 reward 曲线几乎不动，RSI 的复现里 RLTT 又低于未训练的模型。GRPO 的学习率要单独扫，各臂要比较更新范数，否则结论可能只是有效步长的差别。
2. **reward 在 Ouro 上可能提不了多少准确率。** RSI 的复现里 GRPO 只比未训练高 0.8 个点。方案 2 的评价因此以计算量前沿为主。
3. **训练与部署的落差（方案 2b）。** 见阶段 4。
4. **简单阈值可能已经够好。** M6 就是为了提前看到这一点。
5. **被抢先。** 方案 3 离 RSI 最近。阶段 3 与阶段 2 并行，不排在最后。

## 11. 算力估计

参考点：RLTT 论文 Table 1 报告，Ouro-2.6B 上一次 140 步训练在 4 张 H200 上约 49 到 54 小时，约合 200 GPU 小时。下表按这个参考点粗估，1.4B 会更便宜，具体倍数要实测。

| 阶段 | 训练次数 | 说明 |
|---|---|---|
| 0 | 0 | 只有推理 |
| 1 | 约 7 | 5 个臂，加 GRPO 学习率的两档 |
| 2 | 约 6 | 含 `β` 的三档 |
| 3 | 约 5 | rollout 只算 2 个 loop，单次更便宜 |
| 4 | 约 4 | |

全部在 2.6B 上跑约 22 次，上限约 4400 GPU 小时。建议全部臂先在 1.4B 上跑，2.6B 只复现每个阶段的关键对比。

## 12. 待决定

- 可用的算力和硬件: L20X * 8
- 主模型用 2.6B-thinking。
- RL 框架: slime https://github.com/THUDM/slime
- 暂时不把 TML 的 varloop 模型作为第二个 testbed。
- 论文范围：恒等式加方案 1 和方案 2 一篇，方案 3 建议单独成文。

## 13. 参考

- RLTT：https://arxiv.org/abs/2602.10520
- TraceRL：https://arxiv.org/abs/2509.06949
- Ouro：https://arxiv.org/abs/2510.25741
- RSI / D-LoopOPD：https://arxiv.org/abs/2610.10623
- LoopRPT：https://arxiv.org/abs/2603.19714
- Enabling Dynamic Computation in Looped LMs（Ragged KV）：https://arxiv.org/abs/2610.09013
- Adaptive Depth in Looped Transformers（gate 诊断）：https://arxiv.org/abs/2607.20519
- Continuous Depth Batching：https://arxiv.org/abs/2608.09444
- SPG：https://arxiv.org/abs/2510.09541
- Inkling 2 Mini：https://thinkingmachines.ai/news/inkling-2-mini/
- TML-d24-varloop-research 模型卡：https://thinkingmachines.ai/model-card/tml-d24-varloop-research/
