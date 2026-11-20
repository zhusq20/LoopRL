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

顶部三个数的单位是 nat。方案 1 在答对的样本上与 RLTT 相同，在答错的样本上改用 ρ。方案 2 里 reward 把 exit 分布从先验推向后验，表里的 *ρ*t − *ω*t 就是这个方向，四个 loop 一致时它恒为 0；落到 Ouro 逐 loop 的 gate 上的写法见方案 2。“集中在 loop 2–3”的权重是示意值，不是 Ouro 的实测 exit 分布。

## 与这个解读一致的现象

下面的数字都来自论文或预印本，没有新实验。它们与上面的解读一致，但不能证明它。第三组来自 RSI，它的基线在 2026 年 11 月受到公开质疑，见该组下面的补记。

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

**这组数字有争议（2026-11-20 补记）。**OpenReview 上的公开评论指出，RSI Table 2 的未训练基线低于 Ouro 论文 Table 9；评论者称用官方代码重复十次，D-LoopOPD 与未训练的模型没有差别。我对照了两篇论文。2.6B 的 AIME24 平均准确率两边接近：RSI 的 Avg@16 是 62.08，Ouro 的 pass@1 是 64.7。明显偏低的是 OlympiadBench（64.30 对 76.4）和 1.4B 的各项。RSI 没有报告种子数和方差，AIME24 只有 30 题。评论者的复现我无法核实，作者的回复也没有看到。在我们自己复现之前，9.6 个点的增益不能当作已成立的结果。

RLTT 低于未训练的模型这一点也要打折扣。RSI Table 6 按解码上限给了更细的数字：上限在 8K 以内时 RLTT 高于基座，16K 和 31K 时低于基座。这与 RLTT 让回答变短的解读相容，但它出自同一篇论文，只能当线索。完整的对照表在 README 的 3.6 节。

另外，RLTT 论文 Table 5 显示均匀、递增和 exit 分布三种 *ω* 的结果几乎相同。任何一组权重都带着一个 gap 项，如果起作用的是 gap，这个结果是预期的。

## 从 dLLM RL 借什么

左两列是 dLLM RL 这一年多逐步发现的问题和对应做法，右列是同一个问题在 Looped LM 里的样子。

| dLLM RL 遇到的问题 | dLLM 的做法 | Looped LM 里的对应 |
| --- | --- | --- |
| 训练用的似然和真实采样过程不一致 | d1 用单步估计。TraceRL、d2、AGRPO 改为沿真实轨迹逐步计算。 | 沿真实的早退轨迹，每一步的联合 log 概率是精确的，不用估计；本位置的 4 个 exit 还可以求和（方案 2） |
| 下界适合往上推，不适合往下压 | SPG 对负样本改用证据上界 | 负样本直接用精确似然（方案 1） |
| 解码顺序也是策略的一部分 | DCoLT、Mask-aware PG 采样 unmask 位置并写进似然 | exit 步是 4 选 1，写进似然；它对读出的影响可以求和掉（方案 2） |
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

把在第几个 loop 退出当作策略的一部分。rollout、update 和部署用同一个早退过程，梯度是这个过程的精确策略梯度，拆成对混合似然的一项和一个校正项。exit gate 第一次直接从答案对错里学习。

*e*j \~ *q*(· | *s*j)， *y*j \~ *P*e(· | *s*j)，*e* = *e*j\
*π*mix(*y*j | *s*j) = ∑t *q*t *P*t(*y*j)， *ρ*t ∝ *q*t *P*t(*y*j)\
lossj = − *A*j · ( log *π*mix(*y*j | *s*j) + *κ* · log *ρ*(*e*j | *s*j, *y*j) ) + *c* · ∑t *t* · *q*t\
log *π*mix(*y*j) + log *ρ*(*e*j) = log *q*e *P*e(*y*j)

*e*j 是第 *j* 个 token 的退出深度，*s*j 是它看到的历史：前面的 token，以及每个 token 到各自退出深度为止的 KV。*q*t 是 Ouro exit gate 给出的 exit 分布，这里保留梯度。Ouro 的 gate 在每个 loop 给出一个退出概率 *λ*t，*q*t = *λ*t ∏s\<t (1 − *λ*s)。Ouro 论文没有写最后一个 loop 怎么归一，这里令 *q*4 取剩余的 ∏s\<4 (1 − *λ*s)。

*c* 是每多算一个 loop 的价格，loss 里的 *c* 项是本位置期望深度的精确梯度。*A*j 的 return 是答案对错减去 *c* 乘以其余位置的退出深度之和；本位置的深度已经由 *c* 项计入，不再放进 return。*κ* 取 1 或 0，见“梯度”。实现层面的细节在 README 第 8 节。

策略

逐 loop 执行：算完第 *t* 个 loop，以 *λ*t 的概率退出并从 *P*t 采样 token，否则继续，第 4 个 loop 必退。第 *t* 个 loop 的判定不需要更深的 loop。退出的 token 只留下 loop 1 到 *e*j 的 KV，后面的 token 在 loop *s* 读它在 loop min(*s*, *e*j) 的 KV，这是 Ragged KV 的规则。prompt 算满 4 个 loop。rollout、update 和部署都按这一套。

Rollout

按上面的策略生成，记录每个 token 的退出深度。用真实早退的推理实现时，rollout 的计算量是各 token 退出深度之和。没有实现时用静态图模拟，语义相同，只是不省计算。

Update

一次 ragged teacher forcing：仍是 4 个 loop 的静态图，注意力按记录下来的退出深度取 KV。每个位置都把 4 个 loop 算完，得到四个读出和四个 gate 值；超出实际退出深度的 loop 只用来算 *π*mix 和 *ρ*，后面的位置读不到它们。一个 token 继续算深层时读的是自己的深层 KV，别的 token 读它时按补位规则，两种读法要分开实现。update 不省 loop，成本与 RLTT 相同。Ouro 论文把自适应 RL 的失败主要归因于动态计算图；这里求梯度的图是静态的，动态的部分只在 rollout 的推理引擎里。

梯度

如果每个位置都算满 4 个 loop，exit 只通过采到的 token 影响后面的位置，逐 token 的四项求和就是整条轨迹的边缘似然，只用 loss 的第一项就精确。这页早先的版本就是这样训练的。但部署时要早退，训练和部署成了两个策略，这个版本已删除。真实早退时 exit 还决定后面的 token 能看到哪些 KV，轨迹的边缘似然是 4^n 条 exit 路径的和，不能逐 token 分解。

由 log *q*t *P*t(*y*) = log *π*mix(*y*) + log *ρ*t(*y*)，早退过程的精确策略梯度是

E\[ *A* · ∑j ∇ log *π*mix(*y*j | *s*j) \] + E\[ *A* · ∑j ∇ log *ρ*(*e*j | *s*j, *y*j) \]

第一项对本位置的退出深度求了和，方差低，管的是信哪个 loop 的读出。第二项是校正项，在后验 *ρ* 下均值为零，只有当回报在给定 token 之后还依赖退出深度时，它的期望才不为零；它管的是早退让后面的 token 少看了什么。*κ* = 1 时两项都在，梯度精确，合起来就是联合动作（退出深度和 token）的策略梯度。*κ* = 0 只留第一项，方差更低但有偏：gate 的退出决定拿不到早退伤到后文的信号，只有表示会经 KV 的梯度去适应。默认 *κ* = 1，*κ* = 0 作消融。这个分解用一个可以穷举的小过程核对过（2 个位置、4 个 exit、144 条路径，脚本是仓库里的 `scripts/exit_marginal_check.py`），校正项在 Ouro 上有多大未知。

自适应从哪来

reward 经第一项落到 gate 在第 *t* 个 loop 的 logit 上，梯度是 *ρ*≥t · ( *ρ*t / *ρ*≥t − *λ*t )，其中 *ρ*≥t = ∑s≥t *ρ*s。括号里是后验的退出概率减去 gate 给的退出概率。四个 loop 一致的 token 上 *ρ* = *q*，梯度恒为 0，只剩价格 *c* 在起作用，模型学会在这些 token 上早退。有分歧的 token 由 reward 决定信哪个 loop。这页早先写的是 softmax 参数化下的 *ρ*t − *q*t；softmax 要四个 logit 一起归一化，部署时不能逐 loop 判定，所以改用 Ouro 原生的形式。Mishra 等人认为 Ouro 现有的 exit 分布在不同 token 上差别很小。

策略的来历

Ouro 预训练的任务项是各 loop 交叉熵按 *q* 的加权和。按 Jensen 不等式，它是 *π*mix 负 log 似然的上界。所以 *π*mix 就是预训练目标隐含的生成模型。

起步

gate 的 logit 统一减一个常数，让初始的平均深度接近 4。这时几乎每个 token 都退在 loop 4，训练从接近 GRPO 的状态开始。不能直接按 Ouro 原生的 *q* 采样：它是给 Q-exit 阈值用的，没有被当作采样分布用过，Mishra 等说它在各 token 上熵都高，按它采样很可能有不少 token 退在很浅的 loop。常数也不能大到让 sigmoid 饱和，否则 gate 拿不到梯度。再把 *c* 从 0 逐步调大。gate 头的学习率要单独设，RLTT 的 1e-6 是给 LM 权重的。

部署

与 rollout 相同的策略，配合 Ragged KV 真正省掉计算。改用 Q-exit 阈值、换补位规则，或者引擎让退出延迟生效，都是另一个策略。Continuous depth batching 有两种缓存布局，last-exited 与 Ragged KV 语义相同；shared cache 每层只留一份，后面的 token 总是读到前面 token 最近一次写入的状态，语义不同。它的退出信号还要晚一步才生效。要用这样的引擎，训练就按它的规则来。一次训练对应一个 *c*，也就是计算量前沿上的一个点。

风险

Ragged KV 的现成证据不能直接用。Mishra 等报告 Ragged KV 不经训练就能用（macro pass@1，Table 1、2）：Ouro-2.6B-Thinking 固定 4 个 loop 是 62.1，阈值 .5 下 62.0 到 62.3，阈值 .3 下 58.5；Ouro-1.4B 固定 4 个 loop 是 64.2，阈值 .5 下 62.4 到 62.9（平均深度约 3.1 到 3.2），阈值 .3 下 58.0 到 59.5（平均深度约 2.4 到 2.6）。他们给的解释是，现有 gate 下相邻 token 的退出深度很少跳变。方案 2 的目的是让 *q* 随 token 变化，训练成功就会离开这个区间。另外 Ouro 论文 Table 14 里，只用 loop 1 的 KV 顶替全部深度时 GSM8K 从 78.92 掉到 18.73。价格 *c* 把 token 推得越浅，模型离预训练时的状态越远。

另外两条是实现上的。update 的 ragged 重算要和 rollout 引擎的实际计算对得上，否则训练的又不是 rollout 的那个策略。校正项靠采样估计，方差多大要实测。Mishra 等第 5 节也有 ragged-aware 的训练，那一节我没有读到原文。

与 LoopRPT 的关系

LoopRPT 也采样 exit 并做策略梯度。差别有三处。这里的 reward 是答案对错，LoopRPT 是逐 token 的 next-token 概率。这里不给 hidden state 加噪声。这里本位置的 4 个 exit 在 loss 的第一项和价格项里被求和掉，只有校正项靠采样。另外，早退的 token 在训练时留给后文哪些状态，LoopRPT 的论文里我没有读到。

dLLM 对应

TraceRL 和 d2 沿真实生成轨迹计算似然。DCoLT 和 Mask-aware PG 把 unmask 位置的选择写进似然。这里同样沿真实的早退轨迹，把 exit 写进似然；不同的是每一步的 log 概率是精确的，本位置的 4 选 1 还能求和掉一部分方差。

### 方案 3以更深的自己为参考的自蒸馏

用 2 个 loop 做 rollout，把 GRPO 里冻结的参考策略换成模型自己 loop 4 的读出。reward 负责在更深的读出出错时纠正它。

*y* \~ *π*(2)\
lossj = KL( *P*j(2) ‖ sg\[ *P*j(4) \] ) − *ψ* · *A* · log *P*j(2)(*y*j)

上标是 loop 编号，sg 表示不回传梯度，*ψ* 控制 reward 的权重。

含义

这个目标的最优解是 *π*(4) · exp(*ψA*)，也就是被 reward 倾斜过的更深读出。*ψ* = 0 退化为 RSI 的 D-LoopOPD，去掉 KL 退化为 loop 2 上的 GRPO。

为什么要 reward

RSI 报告把 loop 4 蒸馏给 loop 2 之后所有深度都变好，也就是深度本身可以当策略改进算子。这个结果目前有争议（见证据一节的补记），所以方案 3 的第一步是在自己的评测下复现 *ψ* = 0。RSI 还默认更深等于更好。上面的图里 SFT 模型从 3 个 loop 到 4 个 loop 掉了 5.2 个点；Ouro 论文 Table 13 里，2.6B-Thinking 在 AIME24 上 3 个 loop 是 70.3，4 个 loop 是 64.7。RSI Table 5 对同一个模型量到的是 61.46 和 62.08，没有这个下降。两篇论文在这一点上不一致，AIME24 又只有 30 题，over-looping 的幅度要等阶段 0 实测。更深的读出出错时，reward 项让学生不必跟着错。

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
| 方案 2 | gate 采样到的那个 loop 的读出 | P₁ 到 P₄，以及 gate | 后验 ρ，加退出深度上的校正项 | 精确策略梯度（κ = 1） | rollout、update、部署同一个早退过程；update 仍算满 4 个 loop |
| 方案 3 | P₂ | P₂ | 无 | 不涉及 | 另加 KL(P₂ ‖ sg P₄) |

## 次要的映射

| dLLM 技巧 | Looped LM 里的对应 | 判断 |
| --- | --- | --- |
| AGRPO 的步采样摊销，BGPO 的线性化 | 权重（ω 或 ρ）取定后，梯度对各 loop 是线性的，可以逐个 loop 反传 LM head | RLTT 的显存随 loop 数增长，这个改法顺手可做 |
| TraceRL 的 value model | 每个 loop 的 hidden state 上挂 value head，相邻两个 loop 的差就是再算一个 loop 的价值 | 可以替代 Ouro 第二阶段用 LM loss 改善量训练 gate 的做法。有方案 2 之后优先级低 |
| Flow-GRPO 的 ODE→SDE，Latent-GRPO 等给 latent 加噪声 | 给每个 loop 的 hidden state 加噪声 | 不建议。token 已经在采样，确定性的 loop 由反向传播给出精确梯度，加噪声只是换成方差更高的 score-function 梯度 |

## 建议顺序

1. **先测再训。**在 Ouro-Thinking 的 rollout 上做一次前向，统计四个数。gap 在 token 上的集中程度决定方案 2 的空间，答对与答错样本的 gap 差异对应方案 1，答案 token 上 loop 3 对而 loop 4 错的比例对应方案 3。同一条 rollout 在完整历史和 ragged 历史下 *π*mix 的差异，对应方案 2 的校正项有多重要。
2. **实现顺序 1 → 3 → 2。**改动量依次增大。方案 3 先只跑 *ψ* = 0 的复现，确认它高于未训练的模型，再加 reward 项。
3. **评测用长解码预算。**RLTT 的大部分优势出现在紧预算下，短预算会把回答变短和推理变好混在一起。
4. **如果只写一篇。**恒等式加方案 1 和方案 2 是一个完整的故事。

## 不确定的地方

- 三个方案都没有实验结果。恒等式和梯度是我推导的，可以用上面的计算器核对数值。方案 2 的 gate 梯度和“梯度”一段里的分解另用可穷举的小过程核对过（`scripts/exit_marginal_check.py`），没有在 Ouro 上验证。
- 方案 2 的校正项靠采样估计，方差多大未知；去掉它（*κ* = 0）的偏差多大，取决于后面的读出对 KV 历史有多敏感。两者要等阶段 0 的测量和阶段 4 的消融。Ragged KV 不经训练就能用的结果量自现有的 gate，不能当作方案 2 训练之后的估计。
- 方案 2 在 *κ* = 1 时就是对联合动作（退出深度和 token）的策略梯度，做法上离 LoopRPT 更近了，差别见方案 2 里“与 LoopRPT 的关系”一段。
- *A* \< 0 时放大 gap 有害，以及 RLTT 的收益来自 *A* > 0 的 gap 项，这两点都是假设。
- 没有检索到相同做法，但检索范围只到 2026 年 10 月的 arXiv 论文。方案 3 离 RSI 最近，最容易被抢先。
- RSI 的基线在 OpenReview 上受到公开质疑，D-LoopOPD 的增益能否复现未知。本页依赖 RSI 数字的地方有两处：证据一节的第三组，以及方案 3 的动机。恒等式、方案 1 和方案 2 不依赖 RSI。
- RLTT 的数字读自 PDF 原文。RSI、Ouro、TraceRL 和其余论文读的是 arXiv 的 HTML 版，部分只读了摘要和方法节。2026-11-20 重新逐字核对了 Ouro 的 Table 9、12、13（v1 的 HTML）和 RSI 的 Table 2、5、6（v2 的 HTML）。同一天为方案 2 补读了 Ouro 的 3.2 节、4.5 节和 Table 14，Mishra 等的第 3、4 节和 Table 1、2，Continuous Depth Batching 的第 4 节（都是 v1 的 HTML）；Mishra 等第 5 节和 Continuous Depth Batching 附录里 shared cache 的精度数字没有读到原文。
- Ouro 现有的 gate 是否随 token 难度变化，两篇论文说法不同（Mishra 等与 Popescu 等）。

## 参考文献

- 主论文：[RLTT](https://arxiv.org/abs/2602.10520)，[TraceRL](https://arxiv.org/abs/2509.06949)，[Ouro](https://arxiv.org/abs/2510.25741)
- Looped LM 后续：[LoopRPT](https://arxiv.org/abs/2603.19714)，[RSI / D-LoopOPD](https://arxiv.org/abs/2610.10623)，[Mishra 等，Enabling Dynamic Computation in Looped LMs](https://arxiv.org/abs/2610.09013)，[Popescu 等，Adaptive Depth in Looped Transformers](https://arxiv.org/abs/2607.20519)，[Continuous Depth Batching](https://arxiv.org/abs/2608.09444)
- dLLM RL：[d1](https://arxiv.org/abs/2504.12216)，[SPG](https://arxiv.org/abs/2510.09541)，[d2](https://arxiv.org/abs/2509.21474)，[AGRPO](https://arxiv.org/abs/2510.04019)，[DCoLT](https://arxiv.org/abs/2505.10446)，[Mask-aware PG](https://arxiv.org/abs/2607.15200)，[wd1](https://arxiv.org/abs/2507.08838)，[GDSD](https://arxiv.org/abs/2605.29398)，[BGPO](https://arxiv.org/abs/2510.11683)，[MDPO](https://arxiv.org/abs/2508.13148)，[Flow-GRPO](https://arxiv.org/abs/2505.05470)
- Latent 推理的 RL：[Soft Tokens, Hard Truths](https://arxiv.org/abs/2509.19170)，[Latent-GRPO](https://arxiv.org/abs/2604.27998)，[LEPO](https://arxiv.org/abs/2604.17892)，[Dropout-GRPO](https://arxiv.org/abs/2606.10184)