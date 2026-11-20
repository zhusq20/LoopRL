研究笔记 · 2026-11-20 · Looped LM × 强化学习

# 从 RLTT 的恒等式到三个 RL 方案

Looped LM 对每个 token 把同一组 Transformer 权重执行几个 loop 再输出。RLTT 让每个 loop 的读出都拿到 reward，在 Ouro 上报告了明显高于 GRPO 的结果。这页说明 RLTT 的目标函数可以精确拆成两项，以及这个拆分指向的三个改法。三个方案都还没有做实验。

[恒等式](#identity) [证据](#evidence) [从 dLLM 借什么](#lessons) [三个方案](#designs) [建议顺序](#order) [参考文献](#refs)

∑t *ω*t log *P*t(*y*)

RLTT 的 surrogate

\=

log ∑t *ω*t *P*t(*y*)

exit 混合策略的精确 log 似然

−

KL(*ω* ‖ *ρ*(·|*y*))

loop 分歧 gap，恒 ≥ 0

*P*t 是第 *t* 个 loop 的 next-token 分布，*y* 是采到的 token，*ω* 是 RLTT 给各 loop 的权重。*ρ*t ∝ *ω*t *P*t(*y*) 是这个 token 归功于 loop *t* 的后验。等式对每个 token 逐点成立。

扩散语言模型（dLLM）的 RL 技巧大多在处理同一个困难：生成轨迹的似然算不出来，只能用 ELBO 或逐步估计代替。Looped LM 没有这个困难。它只有 4 个 exit，一次 teacher-forced 前向就能得到同一条样本在每个深度下的精确 log 概率。所以值得借的是 dLLM RL 找到的问题，估计器可以直接换成精确值。

## Looped LM 与 RLTT 做了什么

以 Ouro 为例，模型对每个 token 把整个 Transformer 堆栈执行 4 个 loop，权重共享。每个 loop 结束时，同一个 LM head 都能读出一个 next-token 分布 *P*t。另有一个 exit gate，给出在第 *t* 个 loop 退出的概率 *q*t。

一个 token 位置上的计算。四个方框用的是同一组权重。GRPO 和 RLTT 的 token 都只从 *P*4 采样，exit gate 给出的 *q* 在 RLTT 里只作为固定权重。

GRPO 只对最后一个 loop 求梯度：∇ log *P*4(*y*) · *A*。*A* 是 advantage，由同一道题的一组回答的对错归一化得到，答对的回答 *A* > 0，答错的 *A* \< 0。RLTT 的改动是把 log *P*4(*y*) 换成四个读出的加权和 ∑t *ω*t log *P*t(*y*)。权重 *ω* 取 exit 分布、均匀或递增，都不参与求导。

需要先澄清一点。loop 之间的转移是确定性的，GRPO 的梯度经反向传播已经流过全部四个 loop。RLTT 增加的是每个读出上的直接监督，也就是 deep supervision。它并没有把 latent 状态变成可以做策略梯度的随机动作。

## RLTT 实际在优化什么

把 *ω* 看成 exit 步的先验，把 *P*t(*y*) 看成 loop *t* 对这个 token 的似然，页首的等式就是 ELBO 的标准分解。右边第一项是 exit 混合策略 *π*mix = ∑t *ω*t *P*t 的精确 log 似然。第二项是先验 *ω* 到后验 *ρ* 的 KL，这里叫它 gap。四个读出给 *y* 的概率相同时 gap 为 0，分歧越大 gap 越大。

乘上 advantage 后，RLTT 的更新方向是

*A* · ∇ ∑t *ω*t log *P*t(*y*) = *A* · ∇ log *π*mix(*y*) − *A* · ∇ gap(*y*)

两种写法的差别只在权重。精确项的梯度是 ∑t *ρ*t ∇ log *P*t(*y*)，按后验分配。RLTT 是 ∑t *ω*t ∇ log *P*t(*y*)，按先验分配。由此有三个后果。

- ***A* > 0 时缩小 gap。**所有 loop 都被拉向好 token，本质是跨 loop 蒸馏。我认为 RLTT 报告的更早收敛和回答变短来自这一项。这是假设，需要消融验证。
- ***A* \< 0 时放大 gap。**目标函数在奖励 loop 之间产生分歧。压得最重的是本来就不支持这个 token 的浅层 loop，而它们并没有参与采样。
- **采样和目标不一致。**样本来自 *P*4，目标却是 *π*mix 的下界。*ω* 不参与求导，exit gate 从未见过 reward。

拖动四个读出的概率，看两种权重怎样分配压力

推力 = 权重 × (1 − Pt(y))，即 |A| = 1 时这个 token 的 logit 在 loop t 上受到的梯度大小

预设

这条样本

答错，A \< 0 答对，A > 0

loop 权重 ω

均匀 递增 集中在 loop 2–3（示例）

loop 1 给 y 的概率0.05

loop 2 给 y 的概率0.30

loop 3 给 y 的概率0.80

loop 4 给 y 的概率0.90

−1.132

RLTT surrogate

\=

−0.668

精确 log 似然

−

0.464

gap

surrogate 的大小里，gap 占 41%

每个 loop 分到的权重

ω 是固定的先验。ρ 是看到 y 之后的后验，即这个 token 归功于哪个 loop。

- ω，RLTT 使用
- ρ，精确似然使用

每个 loop 受到的推力

方向：压低这个 token（A \< 0）

- RLTT
- 方案 1

负样本：RLTT 把最大的压力放在 loop 1，而它对采到这个 token 的责任很小。

| loop | Pt(y) | ωt | ρt | ρt − ωt | RLTT 推力 | 方案 1 推力 |
| --- | --- | --- | --- | --- | --- | --- |

顶部三个数的单位是 nat。方案 1 在答对的样本上与 RLTT 相同，在答错的样本上改用 ρ。方案 2 里 exit 分布受到的 reward 梯度方向就是表里的 *ρ*t − *ω*t，四个 loop 一致时它恒为 0。“集中在 loop 2–3”的权重是示意值，不是 Ouro 的实测 exit 分布。

## 与这个解读一致的现象

下面的数字都来自已发表的论文，没有新实验。它们与上面的解读一致，但不能证明它。

RLTT 对 GRPO 的优势在紧预算下最大

MATH-500 准确率（%），按解码 token 预算，Ouro-2.6B-Thinking

- RLTT
- GRPO
- 未做 RL 的基座

1024 token 时两者差 36.0 个点，4096 token 时差 9.0。RLTT 训练后回答变短，这在紧预算下直接变成准确率。

数据：RLTT 论文 Table 3

2 个 loop 时差距已经拉满，SFT 模型在第 4 个 loop 反而下降

MATH-500 准确率（%），按推理时执行的 loop 数

- RLTT
- GRPO
- SFT

RLTT 在 2 个 loop 时比 GRPO 高 15.0 个点，与 4 个 loop 时的 14.4 相当，符合浅层 loop 被蒸馏的解读。SFT 模型从 3 个 loop 到 4 个 loop 下降 5.2 个点，更深不总是更好。

数据：RLTT 论文 Table 9

另一篇论文的复现里，RLTT 低于未训练的模型

AIME 2024 Avg@16（%），Ouro-2.6B-Thinking，执行 4 个 loop

D-LoopOPD 完全不用 reward，只把 loop 4 的分布蒸馏给 loop 2。它比未训练的模型高 9.6 个点，RLTT 低 5.4 个点。

数据：RSI 论文 Table 2

另外，RLTT 论文 Table 5 显示均匀、递增和 exit 分布三种 *ω* 的结果几乎相同。任何一组权重都带着一个 gap 项，如果起作用的是 gap，这个结果是预期的。

## 从 dLLM RL 借什么

左两列是 dLLM RL 这一年多逐步发现的问题和对应做法，右列是同一个问题在 Looped LM 里的样子。

| dLLM RL 遇到的问题 | dLLM 的做法 | Looped LM 里的对应 |
| --- | --- | --- |
| 训练用的似然和真实采样过程不一致 | d1 用单步估计。TraceRL、d2、AGRPO 改为沿真实轨迹逐步计算。 | 混合策略的边缘似然只有 4 项，直接精确计算（方案 2） |
| 下界适合往上推，不适合往下压 | SPG 对负样本改用证据上界 | 负样本直接用精确似然（方案 1） |
| 解码顺序也是策略的一部分 | DCoLT、Mask-aware PG 采样 unmask 位置并写进似然 | exit 步是 4 选 1，可以求和掉，不用采样（方案 2） |
| RL 可以写成向 reward-tilted teacher 蒸馏 | wd1、GDSD | teacher 取模型自己更深的读出（方案 3） |
| 训练步数可以少于推理步数 | Flow-GRPO 的 denoising reduction，TraceRL 的 block-size 适配 | 2 个 loop 做 rollout，4 个 loop 部署（方案 3） |
| 中间步的输出可能比最终步好 | MDPO 观察到的 over-denoising | over-looping，用 reward 修正 teacher（方案 3） |

## 三个方案

三个方案对应三个目标：保持 RLTT 的设定只修估计器，让深度自适应，把 RL 的成本降下来。

### 方案 1Sandwiched RLTT

答对的样本保留 RLTT 原式，答错的样本换成精确的混合似然。只改一行。

*S*(*y*) = ∑t *ω*t log *P*t(*y*)当 A > 0\
*S*(*y*) = log ∑t *ω*t *P*t(*y*)当 A ≤ 0

```
# lp[b, j, t] = log P_t(y_j)，RLTT 已经在算的张量；w[t] 是 loop 权重；A[b] 是 advantage
jensen = (w * lp).sum(-1)                        # RLTT 原式
exact  = torch.logsumexp(w.log() + lp, dim=-1)   # log π_mix(y_j)
S      = torch.where(A[:, None] > 0, jensen, exact)
loss   = -(A[:, None] * S).mean()
```

为什么

答对时，Jensen 形式把每个 loop 都拉向好 token，这是想要的蒸馏。答错时，精确形式按责任 *ρ*t 分配压力，没参与采样的 loop 几乎不受影响。等价的说法是把 gap 的系数从 *A* 改成 max(*A*, 0)。

dLLM 对应

SPG 对正样本用 ELBO，对负样本用证据上界，因为把下界往下压并不保证真实似然下降。这里负样本不需要上界，精确值只有 4 项。

预测

负样本上的 gap 不再随训练上升。长解码预算下不再低于 GRPO。

怎么验

四组消融，gap 的系数分别取 0、*A*（即 RLTT）、max(*A*, 0)、正常数。

### 方案 2Exit-marginal RL

把在第几个 loop 退出当作策略的一部分，从混合分布采样，对精确的混合似然求梯度。exit gate 第一次直接从答案对错里学习。

*π*mix(*y*j) = ∑t *q*t *P*t(*y*j)\
*r* = *π*mix,θ(*y*j) / *π*mix,old(*y*j)\
loss = − min( *r* · *A*, clip(*r*, 1 − *ε*, 1 + *ε*) · *A* ) + *c* · ∑t *t* · *q*t\
∇ log *π*mix(*y*j) = ∑t *ρ*t ( ∇ log *q*t + ∇ log *P*t(*y*j) )

*q*t 是 Ouro exit gate 给出的 exit 分布，这里保留梯度，此时 *ρ*t ∝ *q*t *P*t(*y*j)。*c* 是每多算一个 loop 的价格。

Rollout

forward 直接返回 log *π*mix 作为 logits，仍是固定 4 个 loop 的静态图。Ouro 论文把自适应 RL 的失败主要归因于动态计算图，这里不需要它。

梯度方差

exit 被精确求和，不需要采样 exit 步。LoopRPT 也把 exit 当动作，但它采样 exit 并加高斯噪声，reward 是逐 token 的 next-token 概率，不是答案对错。

自适应从哪来

exit 分布用 softmax 参数化时，reward 对 exit logit 的梯度是 *ρ*t − *q*t。四个 loop 一致的 token 上它恒为 0，只剩价格 *c* 在起作用，模型学会在这些 token 上早退。有分歧的 token 由 reward 决定信哪个 loop。Mishra 等人认为 Ouro 现有的 exit 分布在不同 token 上差别很小。

策略的来历

Ouro 预训练的任务项是各 loop 交叉熵按 *q* 的加权和。按 Jensen 不等式，它是 *π*mix 负 log 似然的上界。所以 *π*mix 就是预训练目标隐含的生成模型。

起步

把 *q* 初始化成偏向最后一个 loop，此时等同 GRPO。再把 *c* 从 0 逐步调大。

部署与风险

推理时按 *q* 采样 exit，配合 Ragged KV 或 continuous depth batching 真正省掉计算。训练时后面的 token 能看到前面每个 token 的全部 4 个 loop 状态，部署时看不到被跳过的 loop。这个落差是主要风险。

dLLM 对应

TraceRL 和 d2 沿真实生成轨迹计算似然。DCoLT 和 Mask-aware PG 把 unmask 位置的选择写进似然。它们必须采样这些选择再估计似然，这里 4 选 1 可以直接求和。

### 方案 3以更深的自己为参考的自蒸馏

用 2 个 loop 做 rollout，把 GRPO 里冻结的参考策略换成模型自己 loop 4 的读出。reward 负责在更深的读出出错时纠正它。

*y* \~ *π*(2)\
lossj = KL( *P*j(2) ‖ sg\[ *P*j(4) \] ) − *ψ* · *A* · log *P*j(2)(*y*j)

上标是 loop 编号，sg 表示不回传梯度，*ψ* 控制 reward 的权重。

含义

这个目标的最优解是 *π*(4) · exp(*ψA*)，也就是被 reward 倾斜过的更深读出。*ψ* = 0 退化为 RSI 的 D-LoopOPD，去掉 KL 退化为 loop 2 上的 GRPO。

为什么要 reward

RSI 的结果说明深度本身是策略改进算子：把 loop 4 蒸馏给 loop 2，所有深度都变好。但它默认更深等于更好。上面的图里 SFT 模型从 3 个 loop 到 4 个 loop 掉了 5.2 个点；Ouro 论文 Table 13 里，2.6B-Thinking 在 AIME24 上 3 个 loop 是 70.3，4 个 loop 是 64.7。更深的读出出错时，reward 项让学生不必跟着错。

梯度

rollout 来自被训练的那个 loop，reward 项是 *π*(2) 的 on-policy 精确梯度。RLTT 的中间 loop 项用的是 *P*4 的样本，又没有重要性权重。

成本

rollout 只算 2 个 loop。更新时前向 4 个 loop、反向 2 个 loop，只需要两个 loop 的 logits，显存低于 RLTT。

dLLM 对应

wd1 和 GDSD 把 RL 写成向 reward-tilted teacher 的蒸馏。训练 2 个 loop、推理 4 个 loop，对应 Flow-GRPO（图像 flow 模型）的 denoising reduction 和 TraceRL 的 block-size 适配。

预测

loop 2 上的 reward 增益经共享权重传到 loop 4。在 over-looping 的题上优于纯 RSI。

### 各方法把采样和 reward 接在哪里

| 方法 | token 采样自 | reward 梯度落在 | 各 loop 的权重 | exit gate | 备注 |
| --- | --- | --- | --- | --- | --- |
| GRPO | P₄ | P₄ | 无 | 不学 | 梯度经反向传播到达每个 loop |
| RLTT | P₄ | P₁ 到 P₄ | 先验 ω | 不学 | 中间 loop 项是 off-policy，无重要性权重 |
| RSI（D-LoopOPD） | P₂ | 不用 reward | 无 | 不用 | KL(P₂ ‖ sg P₄) |
| LoopRPT | 逐 token 采样 exit 步 | exit 步 | 采样得到 | 采样式策略梯度 | reward 是 next-token 概率 |
| 方案 1 | P₄ | P₁ 到 P₄ | A > 0 用 ω，A ≤ 0 用 ρ | 不学 | 只改 RLTT 的一行 |
| 方案 2 | ∑ qt Pt | P₁ 到 P₄，以及 q | 后验 ρ | 精确梯度 ρ − q | rollout 仍算满 4 个 loop |
| 方案 3 | P₂ | P₂ | 无 | 不涉及 | 另加 KL(P₂ ‖ sg P₄) |

## 次要的映射

| dLLM 技巧 | Looped LM 里的对应 | 判断 |
| --- | --- | --- |
| AGRPO 的步采样摊销，BGPO 的线性化 | 权重（ω 或 ρ）取定后，梯度对各 loop 是线性的，可以逐个 loop 反传 LM head | RLTT 的显存随 loop 数增长，这个改法顺手可做 |
| TraceRL 的 value model | 每个 loop 的 hidden state 上挂 value head，相邻两个 loop 的差就是再算一个 loop 的价值 | 可以替代 Ouro 第二阶段用 LM loss 改善量训练 gate 的做法。有方案 2 之后优先级低 |
| Flow-GRPO 的 ODE→SDE，Latent-GRPO 等给 latent 加噪声 | 给每个 loop 的 hidden state 加噪声 | 不建议。token 已经在采样，确定性的 loop 由反向传播给出精确梯度，加噪声只是换成方差更高的 score-function 梯度 |

## 建议顺序

1. **先测再训。**在 Ouro-Thinking 的 rollout 上做一次前向，统计三个数。gap 在 token 上的集中程度决定方案 2 的空间，答对与答错样本的 gap 差异对应方案 1，答案 token 上 loop 3 对而 loop 4 错的比例对应方案 3。
2. **实现顺序 1 → 3 → 2。**改动量依次增大。
3. **评测用长解码预算。**RLTT 的大部分优势出现在紧预算下，短预算会把回答变短和推理变好混在一起。
4. **如果只写一篇。**恒等式加方案 1 和方案 2 是一个完整的故事。

## 不确定的地方

- 三个方案都没有实验结果。恒等式和梯度是我推导的，可以用上面的计算器核对数值。
- *A* \< 0 时放大 gap 有害，以及 RLTT 的收益来自 *A* > 0 的 gap 项，这两点都是假设。
- 没有检索到相同做法，但检索范围只到 2026 年 10 月的 arXiv 论文。方案 3 离 RSI 最近，最容易被抢先。
- RLTT 的数字读自 PDF 原文。RSI、Ouro、TraceRL 和其余论文读的是 arXiv 的 HTML 版，部分只读了摘要和方法节。
- Ouro 现有的 gate 是否随 token 难度变化，两篇论文说法不同（Mishra 等与 Popescu 等）。

## 参考文献

- 主论文：[RLTT](https://arxiv.org/abs/2602.10520)，[TraceRL](https://arxiv.org/abs/2509.06949)，[Ouro](https://arxiv.org/abs/2510.25741)
- Looped LM 后续：[LoopRPT](https://arxiv.org/abs/2603.19714)，[RSI / D-LoopOPD](https://arxiv.org/abs/2610.10623)，[Mishra 等，Enabling Dynamic Computation in Looped LMs](https://arxiv.org/abs/2610.09013)，[Popescu 等，Adaptive Depth in Looped Transformers](https://arxiv.org/abs/2607.20519)，[Continuous Depth Batching](https://arxiv.org/abs/2608.09444)
- dLLM RL：[d1](https://arxiv.org/abs/2504.12216)，[SPG](https://arxiv.org/abs/2510.09541)，[d2](https://arxiv.org/abs/2509.21474)，[AGRPO](https://arxiv.org/abs/2510.04019)，[DCoLT](https://arxiv.org/abs/2505.10446)，[Mask-aware PG](https://arxiv.org/abs/2607.15200)，[wd1](https://arxiv.org/abs/2507.08838)，[GDSD](https://arxiv.org/abs/2605.29398)，[BGPO](https://arxiv.org/abs/2510.11683)，[MDPO](https://arxiv.org/abs/2508.13148)，[Flow-GRPO](https://arxiv.org/abs/2505.05470)
- Latent 推理的 RL：[Soft Tokens, Hard Truths](https://arxiv.org/abs/2509.19170)，[Latent-GRPO](https://arxiv.org/abs/2604.27998)，[LEPO](https://arxiv.org/abs/2604.17892)，[Dropout-GRPO](https://arxiv.org/abs/2606.10184)