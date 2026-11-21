# 方案二实现与首轮实验

## 正式队列中断核查与修复（2026-11-22）

11 月 22 日核查确认旧队列均已停止：gate β=0 在 11 月 21 日 18:57 完成 9/140 次更新后失败，最近完整保存为 iteration 4（5 次更新）；固定四轮在 23:22 完成 54/140 次更新后失败，最近完整保存为 iteration 49（50 次更新）。两组各有最后 4 次更新未保存，续训须从保存边界重算。模型分片、rollout 数据游标、Adam 步数和 scheduler 样本数已核对，[边界记录](results/native2b/20261122/recovery-boundaries.json) 保留。下方 11 月 21 日的“运行中”描述属于当时的历史状态。

固定组失败题目为 `train/geometry/101`：应用聊天模板后输入为 2071 token，加 2048-token 回答预算超过入口写死的 4096。入口与评测现在按实际题目、同一聊天模板和 tokenizer 预检容量，不删题、不缩短回答预算。真实引擎测试还发现 pinned SGLang 的 worker 与 scheduler 共保留两个位置，设置 4119 虽能接收请求却只生成 2046 token；这一失败证据保留为 [fixed-long-prompt.json](results/native2b/20261122/fixed-long-prompt.json)。最终容量为 4121，最长题目已完整生成 2048 token，CUDA Graph 保持开启，见 [通过报告](results/native2b/20261122/fixed-long-prompt-v2.json)。超过模型或 KV 容量时预检报错，避免静默截断。

gate 组在 18:53:28 出现 `KV cache pool is full. Retract requests`。SGLang 的剩余生成预算估计衰减至 0.098，允许 KV 超额接纳；撤回后将已有回答重新作 dense prefill，既在 trace 中混入第二个嵌套预填充块，也改变 last-exited KV，不能靠裁剪/拼接 trace 修补。真实 32 请求压力测试已[复现](results/native2b/20261122/retraction-root-cause.json)原解析错误。gate 训练、评测和采样验收入口现均将预算比例固定为 1，预留完整剩余输出，并把估计上限至少设为回答预算；解析端显式要求 `num_retractions=0`、token 概率和 completion 数一致，失败时保存原请求及返回证据。同一低容量压力测试修复后撤回为 0，全部 32 条 trace 通过，见 [retraction-repair.json](results/native2b/20261122/retraction-repair.json)。

34 项针对性回归测试通过，见 [日志](results/native2b/20261122/regression-tests.log)。正式容量的 gate 压力测试也已通过：32 条回答各生成 2048 token，共 65,536 个 token，全部 trace 对齐、撤回为 0，见 [gate-capacity-stress-v2.json](results/native2b/20261122/gate-capacity-stress-v2.json)。服务容量和 gate 的预算预留配置写入 protocol，恢复必须提供匹配配置及源码哈希的显式服务验证报告。训练步数、数据、seed、β、2048-token 预算、精度、优化器和训练/采样卡配置保持原实验设置。两条正式队列已从原保存边界恢复到 `runs/native2b-fixed-repaired-20261122` 与 `runs/native2b-gate-repaired-20261122`；当前验收状态见 [recovery-verification.json](results/native2b/20261122/recovery-verification.json)。完整 140 步、held-out 效果和同硬件墙钟加速仍未完成。

### Reward 趋势与续训判断

本次两个错误都发生在未完成的采样/请求阶段，失败批次没有进入优化器更新。gate 的首次缓存撤回出现在第 10 批，早于它的 5 次更新检查点可保留；固定组第 55 批才因长题被拒绝，50 次更新检查点可保留。恢复实读 gate 模型、FP32 主权重及 Adam 分片完全匹配，三个 gate 采样进程的完整预算预留配置也已实读确认。不需要从原基座重新开始；应重算每组最后 4 个未保存的更新。

中断前固定组的训练日志呈上升趋势。下面选择第 1–10 批与第 41–50 批，各为 320 道题、2560 条回答，后一窗口全部处于已保存边界之内；不把续跑产生的重复批次混入统计。

| 固定四轮指标 | 第 1–10 批 | 第 41–50 批 |
|---|---:|---:|
| 原始 0/1 outcome reward | 46.17% | 67.03% |
| 回答截断率 | 53.36% | 31.41% |
| 平均回答 token 数 | 1702.87 | 1386.94 |
| 未截断回答中的正确比例 | 98.83% | 97.67% |

[曲线](results/native2b/20261122/reward-trends.png) 同时画出逐批值与五批移动平均；[窗口明细](results/native2b/20261122/reward-window-analysis.json) 保留分母和失败原因。reward 提升伴随更少截断和更短回答，支持模型更会在固定 2048-token 预算内结束思考、提交可验证答案的解释。未截断回答的正确比例没有相同上升，而且这部分样本的组成在变，不能据此推断纯数学能力提升或下降。不同窗口题目不同，训练曲线不能替代基座与检查点在同一 held-out 集上的配对评测。

固定与 gate 的共同前 9 批原始 reward 均值分别为 46.70%、46.18%；gate 平均退出深度在 3.9004–3.9057，仍处于 14 步学习率 warmup 内，尚无自适应深度优势证据。β=0 本来也没有显式节省计算的激励，正价格扫描尚未开始。所有已完成批次训练指标有限，独立 grad/no-grad 重放误差为 0，TIS ESS 最低固定 99.926%、gate 99.913%，见 [稳定性统计](results/native2b/20261122/training-stability.json)。48 条正 reward 回答重新评分全部一致，错配标准答案的负对照全部为 0，见 [评分抽查](results/native2b/20261122/reward-scorer-audit.json)。这些检查未发现明显数值发散或评分常量错误，但不是全量正确性证明。

看训练效果应使用 `rollout-metrics.jsonl` 的 `reward_mean` 或日志中的 `rollout/raw_reward`。`rollout/rewards` 已被自定义 post-process 替换为组内中心化的训练信号，均值接近零是预期行为。


## 用户确认 token gate 与原生 2b 验收（2026-11-21）

用户明确选择每个 token、每个 loop 决定退出。β=0 与四档正价格 gate 组直接从相同原始基座训练，取消的混合训练不再是前置条件。当前实现采用 κ=1 采样联合 exit/token score，实际轨迹成本进入 return 一次；与原先 mixture/posterior 加解析成本的方差降低实现有所区别，详见 README 8.0。

原生小模型独立串行 oracle 已核对并行重放的读出、gate 概率及全部参数梯度，也覆盖跨 loop KV 梯度、packed 隔离和激活检查点。完整 2.6B 的 FP32 真实退出采样/训练联合 log 概率最大差约 1.29e-5；BF16 两条 128-token 轨迹的 TIS ESS 均超过 99.9%，grad/no-grad 差为 0，gate 与主模型梯度非零。logit shift=3 的平均深度约 3.8，包含 1/2/3/4 退出。报告见 [native2b/20261121](results/native2b/20261121)。前两次服务测试分别因 hidden-state 返回模式和 ForwardBatch 接口错误失败，修正后通过，失败日志保留。

`runs/native2b-pilot-20261121` 两步真实 MATH、2048-token、DP2、每步 32 回答已通过 [验收](results/native2b/20261121/pilot-acceptance.json)。两步平均深度分别为 3.9119/3.8978；joint 与 gate 的 grad/no-grad 差均为 0。gate FP32 主权重最大变化 7.16e-6，Adam 矩非零，scheduler 样本数为 64。第一步学习率为 0，更新仅推进矩与 schedule；第二步才产生实际权重变化。主模型 BF16 final norm 尚未跨过量化间隔，FP32 主权重已有 1.19e-7 变化。

第一次恢复测试发现 CLI 相对路径在源码快照 `chdir` 后失效，slime 回退 HF 基座。该运行在采样阶段被主动停止，不计作恢复成功；[停止记录](results/native2b/20261121/rejected-resume.json) 保留。入口现提前绝对化路径、检查 model+rollout 边界完整性，并在参数解析后拒绝意外 finetune/no-load-optimizer。新 helper 在更新前逐元素核对 gate BF16 模型、FP32 主权重和 Adam 分片。

第二次恢复因短运行边界改变默认 WD 调度长度而被 Megatron 拒绝；已改为新 Ouro 任务按原始总步数构建调度，恢复显式采用保存的调度器，保持恒定 WD=0.1。第三次已通过 gate 模型/FP32 主权重/Adam 分片的精确恢复检查，但对采样引擎实读发现 gate 仍等于基座：非 offload 初始化在加载 ref 后未切回 actor，而 NCCL updater 直接遍历 live model、忽略传入的 backup getter。版本一致检查不足以发现此问题。第三次在采样阶段主动停止；[记录](results/native2b/20261121/rejected-resume-weight-sync.json)。更新入口现保证广播前切回 actor，并增加实际调用路径的回归测试；第四次恢复已通过 gate 模型、FP32 主权重与 Adam 分片精确比对，三个采样引擎实读权重与训练检查点完全一致；1182 个 BF16 gate 权重元素区别于基座。[实际同步报告](results/native2b/20261121/gate-weight-sync.json)。恢复后的第 3 步与保存已完成，[续训验收](results/native2b/20261121/resume-acceptance.json) 通过：Adam 步数 3、scheduler 样本 96、gate 主权重相对基座最大变化 2.14e-5，joint/gate 重放误差均为 0，TIS ESS 99.92%。这验证状态恢复与续训，不等于完整运行逐位一致。52 项定向回归测试通过；原生和 SGLang 补丁的正向/反向应用检查通过。

旧固定四轮速度升级运行同样在非 offload 恢复首批经过这条同步路径，不能继续作为严格配对的正式参照。保留第 15 次更新的 checkpoint 和日志，旧队列已停止，`runs/native2b-fixed-formal-20261121` 已从原始基座重新运行 BF16 固定四轮 β=0，与本轮 gate 组保持相同初始 checkpoint、140 步、32×8、2048-token 和 seed。旧结果仅保留为工程历史。

17:22 正式 gate 队列已启动，目录 `runs/native2b-formal-20261121`：先运行 β=0，随后依次运行 β=0.02、0.05、0.1、0.2，每组均从相同原始基座初始化。两卡训练、三个采样副本、微批次 4、每步 256 回答；17:34 首个完整批次已通过验收并开始下一批。启动前证据绑定见 [training-validation.json](results/native2b/20261121/training-validation.json)，它分别记录 32 回答 pilot 和 256 回答正式首批的证据。

首批完整运行见 [formal-verification.json](results/native2b/20261121/formal-verification.json)：256 回答、438,142 个 response token，四个退出深度均有覆盖，平均深度 3.90265，joint 与 gate 的独立重算误差均为 0，TIS ESS 99.918%。训练 107.16 秒，采样/等待 488.13 秒，合计 595.29 秒；不含一次性启动和保存。固定组首批 504.62 秒，但训练/采样卡分配和生成文本不同，不能由此隔离 CUDA Graph 或 gate 的速度贡献。首批仍处于学习率为零的 warmup 起点，实际参数更新已由 pilot 第二步和恢复续训验证；全部 140 步及 held-out 效果尚未完成。

**CUDA Graph 的当前限制：**固定四轮参照保持开启；2b 暂时关闭现有整模型 decode graph。当前实现每轮按 gate 结果压缩活跃请求，包含动态索引、CPU 元数据更新与重新构造 attention 元数据，尚未接入图重放。逐 token gate 并非原则上不能使用 CUDA Graph，后续可实现单轮图和活跃 batch 分桶；不能只打开旧开关就认为正确。实际执行轮数与端到端耗时分开报告，目前没有 2b 墙钟加速结论。固定组与 gate 组还使用不同训练/采样卡分配，队列总耗时不能直接充当模型推理速度对照，最终需在相同硬件预算下单独测量。

## 取消均匀混合深度，改做 gate 对照（2026-11-21 16:19）

用户明确取消均匀混合 2/3/4 轮及其计算价格扫描，要求直接比较带 gate、不惩罚计算与带 gate、不同强度计算惩罚。混合训练是 H4 的诊断对照，不是训练可学习深度策略的必要前置步骤；H4 对照本次不继续，实验主问题转为 H5。

已停止 `runs/native2a-speed-formal-20261121/stratified` 队列及其所属 driver/worker，后续四个正价格混合臂不会启动。保留 `iter_0000009`（10 次更新）的完整检查点，日志记录到已完成第 12 次更新；未将后两次无检查点更新计作可恢复状态。固定四轮 β=0 队列继续执行，提供原始参照。停止记录及旧计划快照位于 [gate-plan-20261121](results/native2a/gate-plan-20261121/decision.json)。

新的主对照为：固定四轮 β=0；gate β=0；gate β=0.02、0.05、0.1、0.2。gate 的所有组从相同原始基座、相同 gate 初始化开始，不使用被取消的混合训练权重。β=0 应保留：它给出仅由正确性奖励驱动的深度策略，帮助区分 gate 本身和计算惩罚的贡献；没有计算惩罚时不保证会省算，多个深度准确率相同时也没有偏好浅层的额外奖励。正价格组用于测量计算节省与准确率损失的取舍。选价只用验证集上的实际 gated policy，不再使用均匀固定深度的平均结果；测试集独立。后续保留选中价格下的固定四轮对照，以检查收益是否仅来自缩短答案。

正在澄清 gate 的粒度：现有原生 `sequence_depth_head` 在每条回答开始前选择 2/3/4 轮，预训练 `early_exit_gate` 仍被冻结；逐 token、逐 loop 的 hazard gate 属于方案 2b，需要相同的退出/KV 规则贯穿采样、训练与评测，不能以序列头替代。现有 BF16 入口明确拒绝未验收的 learned-depth 路径，原因是深度动作概率校正尚未完成。已将新 gate 组记为待实施，尚未启动或声称验收通过。当前活动 `arms` 仅含继续运行的固定基线，新组列于 `planned_gate_experiments`，防止旧混合扫描再次启动。

以下章节保留历史记录，当前执行状态以本节为准。



## 训练概率复用、微批次与数据并行（2026-11-21）

用户授权继续做速度优化。新增三个可显式启用的执行选项：`--reuse-old-log-probs`、`--micro-batch-size 4`、`--trainer-gpus 2`。只有每批一次 optimizer update、无 dropout 的固定/分层 GRPO/TIS 可以复用旧策略概率；此时带梯度 forward 的停止梯度结果就是当前更新前的训练策略，不再额外执行一遍全量旧策略 forward。每卡、每个实际深度仍独立重放一个最长完整微批次，要求带梯度/无梯度 logp 最大差不超过 0.001；更新前验证每条回答恰好覆盖一次、参考与采样概率有限，跨卡聚合完整 token 级 TIS 指标。按深度、长度组微批次，保留尾部样本，并保持全局损失归一化。

相同真实长回答的原生训练回放如下，含参考/旧策略或抽查 forward、真实自定义 loss、backward 及 FP32 梯度同步，不含采样、optimizer step、保存和完整编排：

| 回放 | 耗时 | 峰值 allocated 显存 |
|---|---:|---:|
| 12 条，单卡，微批次 1，原全量重算 | 15.37 秒 | 17.48 GiB |
| 相同 12 条，单卡，微批次 1，概率复用 | 13.45 秒 | 17.48 GiB |
| 相同 12 条，单卡，微批次 2，概率复用 | 9.45 秒 | 19.98 GiB |
| 相同 12 条，单卡，微批次 4，概率复用 | 9.74 秒 | 24.97 GiB |
| 24 条，单卡，微批次 4，概率复用 | 17.77 秒 | 24.97 GiB |
| 相同 24 条，两卡，微批次 4，概率复用 | 9.70 秒 | 24.83 GiB/卡 |

12 条的小回放中，微批次 4 会独立审计全部三个微批次；正式 256 条批次的抽查占比更低，最终选择以完整正式更新计时为准。两卡对同一 24 条回放加速 1.83 倍，loss 一致；归一化参数梯度的跨配置相对 L2 差为 1.655%，同配置重复波动为 1.387%–1.788%。只检查了归一化参数梯度，不能声称全部参数梯度或 BF16 运算逐位一致。详见 [replay-summary.json](results/native2a/performance-v3-20261121/replay-summary.json)。选择性重算单轨迹约快 11%，但显存显著增加；此次采用完整逐层重算，优先使用组批和数据并行。

两步原生试跑 `runs/native2a-speed-pilot-v2-20261121` 已通过 [检查点验收](results/native2a/performance-v3-20261121/pilot-acceptance.json)：每步 32 条真实 MATH 回答，2/3/4 轮、2048-token 上限、两卡、微批次 4、β=0.1。每步独立审计 24 条，grad/no-grad 最大差为 0，完整 token 级 TIS ESS 均超过 99.9%。实读 checkpoint 验证 BF16 模型、FP32 主权重/Adam 矩、Adam 步数 2、scheduler 样本 64，冻结 exit gate 未变。第一轮 pilot 因遗漏 SGLang 外部模型环境变量而在训练前失败，日志保留；入口现自动设置并校验原生模型包。29 项定向测试通过，原生补丁正向和反向应用检查通过。[training-validation.json](results/native2a/performance-v3-20261121/training-validation.json) 绑定准确执行配置、模型 revision 与源码哈希。

正式队列从原 BF16 运行的 `iter_0000004`（已完成 5 次更新）恢复到新根 `runs/native2a-speed-formal-20261121`。仅改变经过验证的执行配置；140 步、32×8、2048-token、精度、FP32 主权重/梯度/优化器、价格、数据和种子全部保持。原输出与失败记录保留；新 run 的 `training-upgrade.json` 记录源路径和边界。固定组仍用 GPU 4 训练、5/6 采样；分层组改为 GPU 0/1 训练，2/3/7 分别采样二/三/四轮，总计仍是 8 张 GPU。逐位恢复一致性、六组最终效果、验证集选价和学习头评测仍不属于本次工程验收。

15:53 的正式更新观测：分层恢复后的两步分别为 220.47、216.37 秒，均值 218.42 秒；原配置第 2–5 次更新均值 468.57 秒，约快 2.15 倍。训练部分由 356.03 秒降至 72.79 秒；采样/等待由 112.54 秒增至 145.63 秒，体现将一张采样卡转给训练的取舍。按实际生成 response loop-token 归一化后，吞吐约为原来的 2.00 倍。这是不同题目批次的前后观测，不是相同生成文本的严格 A/B；不含一次性启动及 checkpoint 保存开销。完整统计随 [formal-speed-verification.json](results/native2a/performance-v3-20261121/formal-speed-verification.json) 更新。

两个原检查点的 Adam 步数均为 5，scheduler 已消费 1,280 条回答，数据游标为第 160 题。分层恢复后的第一个 batch 确认为样本 1280–1535、prompt group 160–191；学习率继续为第 6 次更新对应值，未重复前五步题目。首次暂停辅助脚本遇到容器内 `pidfd_open` 不兼容；逐一核对所属进程已退出后完成记录，并改用 PID/create-time 轮询，固定队列随后由修复版正常停止。边界状态见 [resume-boundaries.json](results/native2a/performance-v3-20261121/resume-boundaries.json)，进程归属和停止记录见同目录 `fixed-stop.json`、`stratified-stop.json`。


正式切换验收截至 2026-11-21 15:59：

| 队列 | 旧配置第 2–5 次更新均值 | 加速配置观测 | 端到端倍数 |
|---|---:|---:|---:|
| 分层，两卡训练＋三个采样引擎 | 468.57 秒 | 3 步平均 223.39 秒 | 2.10 |
| 固定四轮，单卡训练＋两个采样引擎 | 672.57 秒 | 首步 398.05 秒 | 1.69 |

固定组首步训练为 184.91 秒，旧稳态 461.43 秒；其 256 条回答全部覆盖、4 条独立重放审计通过。两个队列均保持运行，已累计完成分层 8/140、固定 6/140 步。分层后续 β=0.02、0.05、0.1、0.2 按原队列继续。固定单步样本不足以确定长期稳态，完整六组训练尚未结束；表中是采样＋更新计时，未将启动、检查点保存及人工验证间隔算入。按生成 response loop-token 归一化的吞吐变化分别为分层 2.02 倍、固定 1.56 倍。


## BF16、训推校正与新正式队列（2026-11-21 14:56）

用户要求进一步提速，并指出训推差异应使用校正方法处理。此前把跨引擎 `max |Δlogp| < 0.001` 当作必须使用 FP32 的理由过于保守；现为固定/分层 2a 接入 **BF16 + tokenwise TIS**。训练端对采样文本重算旧策略概率，以停止梯度的 `min(2, exp(log p_old_train − log p_sampler))` 乘在逐 token PPO 策略损失上；PPO 本身仍使用 `p_current_train / p_old_train`。固定生成预算归一化、价格、冻结参考 KL 保持原定义。实现依据 [slime 的 TIS 文档](https://thudm.github.io/slime/advanced/policy-mismatch.html#tis)，明确属于有偏的 tokenwise surrogate，不能声称完全校正前缀分布。

BF16 用于模型训练与采样计算；主权重、梯度累积、Adam 一二阶矩保留 FP32。跨引擎检查改为显式 TIS 协议、有限值检查，并记录权重、截断率、ESS 与原始 logp 差；同一训练实现的带梯度/无梯度重算仍保留 0.001 门槛，深度、权重版本和共享策略同步检查继续执行。原始严格跨引擎审计仍如实记为 `passed=false`，没有将原始误差伪装为消失。学习深度头的 BF16 路径需要额外验证一次性深度动作校正，当前入口明确拒绝该组合。

服务配置采用 FlashInfer attention、融合 BF16 RMSNorm/RoPE/MLP、decode CUDA Graph、每引擎并发 32、KV 容量 65,536、显存比例 0.85。实测 overlap scheduler 略慢，未启用。性能与校正证据见 [完整验证报告](results/native2a/performance-v2-20261121/bf16-tis-validation.json)：

| 测量范围 | 优化后 FP32 | BF16 新配置 | 结果 |
|---|---:|---:|---|
| 相同 32 个数学采样请求/种子，四轮，最多 2048 token，单张同型号 GPU | 274.24 token/s | 1,034.21 token/s | 吞吐 3.77 倍；实际文本和 token 数随精度变化 |
| 同一长轨迹原生四轮 forward/backward，预热后 | 约 5.41 秒 | 约 1.01 秒 | 约 5.35 倍；不含参考重算、优化器及完整训练编排 |
| 上述单轨迹训练探针峰值 allocated 显存 | 23.03 GiB | 11.70 GiB | 不代表完整 trainer 显存 |

32 条 BF16 长轨迹共 63,536 token，原始最大 logp 差 0.39709，平均绝对差 0.01067；TIS 权重均值 0.999924，范围 0.6723–1.4005，截断率 0，ESS 比例 99.918%。这验证该批权重没有退化，不证明后续所有批次或最终任务收益。

`runs/native2a-bf16-tis-pilot-v2-20261121` 完成两步真实 MATH 训练，覆盖 2/3/4 轮、每步 12 条回答、2048-token 上限及非零价格。两步 ESS 比例均超过 99.9%，无截断，第二步梯度有限非零；保存后实读检查点确认 BF16 模型、FP32 主权重及优化器矩，主权重 final norm 最大变化 1.0133e-6，Adam 步数 2，scheduler 已消费 24 条样本。BF16 导出的 final norm 可能因量化保持相同，不能用它否定 FP32 主权重更新。验收见 [bf16-tis-pilot-acceptance.json](results/native2a/performance-v2-20261121/bf16-tis-pilot-acceptance.json)。第一次 pilot 因新增嵌套指标不被 slime logger 接受而在更新前失败，已改为标量张量，失败记录保留。35 项定向测试通过，包括实际自定义损失梯度、TIS 比值方向/停止梯度、非法值拒绝、CUDA Graph 重放、队列拓扑及拒绝跨精度续训；原生补丁正反应用检查通过。

新的持久运行根目录为 `runs/native2a-bf16-formal-20261121`，14:45 启动：

| 队列 | GPU | 正式训练组 |
|---|---|---|
| `fixed` | 4 训练，5/6 两个四轮采样副本 | `fixed4-beta0` |
| `stratified` | 0 训练，1/2 分别二/三轮，3/7 两个四轮副本 | β=0、0.02、0.05、0.1、0.2 顺序执行 |

所有组统一从同一基座重新开始，140 步、32 题 × 8 条回答、2048-token、seed=20261121、标准 AdamW、学习率 1e-6、temperature=top_p=1；不是从 FP32 检查点跨精度续训。两个首组的协议仅 `arm` 不同，源码随各 run 固定保存。14:55 分层首个正式更新完成：采样墙钟 125.19 秒，训练 350.08 秒（参考重算 64.60 秒、旧策略重算 64.66 秒、forward/backward 220.53 秒），端到端 484.86 秒；原 FP32 首步为 3835.90 秒，整体约快 7.91 倍。梯度范数 0.08171，四个服务引擎均已同步为权重版本 2，并开始第二批。首批 448,071 个 token 的 TIS ESS 比例 99.935%，仅 1 个 token 上截断，原始最大 logp 差 1.47903。固定组首批采样为 231.66 秒，校正检查通过，首轮训练仍在运行。这一整体变化包含精度、引擎优化、并发与新增 GPU，不能全部归因于 BF16；首步也不能替代稳态测量。详见 [首批计时与校正](results/native2a/performance-v2-20261121/formal-first-batch.json) 和 [六个服务实例启动核验](results/native2a/performance-v2-20261121/formal-launch-verification.json)。

旧 FP32 分层组在已完成两次更新、第三批采样期间停止，尚未达到五步检查点；原固定组首批数值失败记录保留。此前等待检查点的 FP32 优化切换队列已标为 `superseded`，不再自动启动。只停止本实验已核对的进程，其他 GPU 任务自行结束后才使用空闲设备。停止记录见 [fp32-run-retirement.json](results/native2a/performance-v2-20261121/fp32-run-retirement.json)。以下 FP32 章节为历史记录，当前状态以本节为准。

还测量了 selective checkpointing（约快 9%，单轨迹显存由 23 增至 71 GiB）、FP32x3 GEMM（单算子收益未转化为完整训练收益）以及 overlap（短基准慢约 3%），均未并入正式配置。完整六组训练、正式吞吐统计、验证集选价、学习深度头及配对效果评测仍未完成，不声称已达到硬件的绝对最优性能。


## 采样性能修改与切换安排（2026-11-21 14:07）

按用户授权修改原生服务实现：解码直接读取分页 KV，避免逐层复制完整历史；prefill 元数据每批仅处理一次，只计算新增 query；融合 FP32 RMSNorm 与 RoPE。保留 FP32、关闭 TF32、原训练参数、数据、种子与 0.001 一致性门槛。入口增加 `--attention-implementation paged-fused`，旧实现仍可显式选择。原生补丁包含新 Triton kernels，已通过固定上游版本的正向及当前树反向应用检查。

同一 GPU 0、相同 FP32 模型、depth=4、batch=8，每条生成 64 token，预热并复用 prompt cache 后测量：

| 上下文 | 原实现 token/s | 新实现 token/s | 倍数 |
|---|---:|---:|---:|
| 512 token | 59.23 | 110.57 | 1.87 |
| 2048 token | 54.82 | 121.64 | 2.22 |

真实数学题长回答批次共生成 14,038 token，耗时 132.43 秒，106.00 token/s。28 项定向测试通过；46 条轨迹覆盖 2/3/4 轮、长前缀、2048-token 回答与原失败题组的种子重跑，采样/原生重算最大 logp 差 0.000100613。最长 2458-token 样本带梯度/无梯度 logp 差为 0，梯度非零。此结果验证服务实现；未完成整个 256 条回答的优化版正式更新，不能将采样加速直接当作整步训练加速。完整数据及源码哈希见 [optimization-validation.json](results/native2a/performance-20261121/optimization-validation.json)。

原固定组在首个 optimizer update 前因 sampler/recompute 差 0.00446677 超过阈值而停止，旧代码独立重放也能复现误差。保留失败数据；新实现按原种子重跑该题组的 8 条、每条 2048-token 回答已通过。随机生成文本可能随浮点顺序改变；通过此回归不表示后续全部训练批次已通过，训练内检查继续执行。

新的持久调度根目录为 `runs/native2a-optimized-20261121`：

- `fixed/`：GPU 4、5，优化实现从基座重跑 `fixed4-beta0`。14:07 时 GPU 4 被其他任务占用，队列等待设备空闲，最长等待 24 小时，不终止其他任务。
- `stratified-upgrade/`：监视仍在运行的原分层组，等待 `iter_0000004`（第 5 次更新）的模型、优化器/RNG 和数据游标完整保存。确认文件稳定、恢复协议及源码验证通过后，仅停止本队列所属进程，再以新实现从同一边界续训，随后扫描其余 β。14:07 状态为 `waiting_for_checkpoint`，原分层组仍运行旧快照。

切换前的协议检查已通过；服务实现变化单独记录在 `serving-upgrade.json`，其余训练协议必须完全一致。切换保留训练状态，不声称与旧服务不中断运行逐位相同。原先尚未完成的严格同边界恢复、学习头验收及最终效果评测仍保持未完成状态。

## 正式 2a 对照（2026-11-21 11:27 起）

用户要求直接运行正式 2a 对照。已停止此前的验收队列和 8192-token 基线评测，保留其全部结果；小批量 v4 验收在第 6 次更新后的 `iter_0000005` 检查点处终止。六次 sampler/recompute 最大 logp 差均低于 0.001。严格同边界恢复和学习深度头端到端验收尚未完成，不记作通过；它们不再阻挡从基座开始的固定/分层正式组。

运行根目录：`runs/native2a-formal-20261121`。两条持久队列并行执行：

| 队列 | GPU | 正式训练组 |
|---|---|---|
| `fixed` | 4、5 | `fixed4-beta0` |
| `stratified-v2` | 0、1、2、3 | `stratified-beta0`，随后 β=0.02、0.05、0.1、0.2 |

每组均从相同 Ouro-2.6B-Thinking 基座开始，140 步、32 题 × 8 条回答/步、2048-token 上限、seed=20261121、FP32、标准 AdamW、temperature=top_p=1、学习率 1e-6；每 5 步保存检查点。提示词和判分沿用已锁定的原生协议，各组一致。这里的“正式”指完整训练规格，不表示与 RLTT 在精度、优化器和提示词上完全复现。共享权重版本、0.001 概率一致性阈值、有限值和结束后检查点检查保留，任何失败会停止所在队列。

入口为 `scripts/advance_native_2a.py --direct-formal --arm-names ... --gpus ...`；实际命令和进程记录在根目录 `launch.json`，两队列的 `status.json`、各组 `protocol.json`、源码快照和日志用于追踪。调度、原生协议和共享权重检查共 14 项通过。首批正式任务已启动，尚未得到正式训练增益结论。六组扫描后的验证集选价、学习深度组、同价固定深度组和配对评测仍是后续阶段，未宣称已经完成。

首次并行启动时，两任务在服务绑定前都分配了 15000 端口，分层组在初始化权重通信组时失败，未执行更新。其证据保留在 `stratified/`。现为新任务增加按 GPU 分配的独立端口区间；分层队列从基座重新启动于 `stratified-v2/`，三个引擎使用 25000、25132、25264，固定组继续使用 15000。两组训练协议仅 `arm` 不同，模型、奖励、损失和采样源码相同；源码差异仅为启动器和端口分配。核对记录见 `launch-verification.json`，原生补丁已更新并通过反向应用检查。

## 正式训练吞吐排查（2026-11-21 13:32–13:41）

按用户要求先诊断慢速原因。运行中的模型权重、训练协议与服务配置未修改；在采样引擎等待训练时发出少量固定种子的短诊断请求，并采集四次 forward 的 CPU/CUDA trace。诊断会填充 prefix cache、产生少量额外耗时，不能把该时段算作正常训练吞吐。完整证据见 [diagnosis.json](results/native2a/performance-20261121/diagnosis.json)。

- 固定组首批 256 条回答共 446,175 个 token，采样 7,550.81 秒（125.85 分钟），实际总吞吐 59.09 token/s。分层组前两批采样分别 43.48、43.65 分钟；第一次重算/训练约 20.4 分钟。任务规模与实测吞吐可以解释耗时，不是仅凭参数量推算。
- 正式任务沿用了数值验收配置：FP32、TF32 关闭、原生 attention、CUDA Graph/overlap 关闭、每引擎最多 8 个请求及 16,384 个 KV token。训练侧每组一个 rank，microbatch=1，逐层全量激活重算；采样与训练串行。
- 2.6B 是共享参数量；模型有 48 个物理 block，2/4 轮实际执行 96/192 个 block。4 轮的 FP32 MHA KV 每个缓存 token 需要 3 MiB，16K-token 池已占 48 GiB；32 条互不共享前缀的 2048-token 序列仅 KV 就需 192 GiB，不能只调大并发而不处理容量。
- 真实引擎 trace 中，每次 4 轮、batch=8 解码出现 11,362 个 GPU kernel、768 次 linear、192 次 SDPA。native cached prefill 对每个请求逐个执行并读取 GPU 标量：8 条 512-token 历史、各只续一个 token 的 prefill 出现 13,440 次 DtoH/同步；同时构造完整长度的 query 再丢弃前缀输出。decode 每层先 gather/materialize 全历史 K/V，再执行 FP32 SDPA。trace 时长含 profiler 开销，不作正常速度使用。
- 独立随机张量单 block 微基准（batch=8、context=2048，省略 RoPE/KV 写入）：FP32 0.642 ms，TF32 0.541 ms，BF16 0.335 ms；FP32 CUDA Graph 0.609 ms，仅约 1.05 倍。不能声称开启 CUDA Graph 就会带来数量级加速，也不能把 BF16 微基准当作完整模型一致性验收。
- 优先级：修复 attention 中多余同步/缓存搬运与 cached prefill；解决 BF16/TF32 的训练—采样数值一致性后再决定精度；联合调整并发和 KV 容量；然后融合算子、适配 graph 和改善训练/采样资源利用率。当前仍未验证端到端加速后的正式配置。

复测脚本：`scripts/benchmark_native_2a_live.py`（仅对空闲引擎发短请求）和 `scripts/benchmark_native_2a_ops.py`（单 GPU 小内存算子对照，使用 `integrations/slime/native/env.sh` 的兼容库环境）。

## 此前原生 2a 验收记录（2026-11-21，已停止）

入口 `scripts/run_native_2a.py` 使用原生 Megatron actor 和三个共享更新权重的 SGLang 深度引擎；支持真实 MATH 判分、绝对步数的价格升温、冻结初始策略 KL，以及运行源码快照。以下为正式启动前的验收记录，最新运行状态以上一节为准。

- `runs/native2a-acceptance-20261121` 在首批回答判分阶段失败，未执行模型更新：math-verify 的 signal 超时不支持 Ray 工作线程。已改为独立进程判分并通过工作线程回归测试，未移除超时保护。
- `runs/native2a-acceptance-v2-20261121` 首步完成：12 条回答中 2 条答对，截断率 83.3%，非零梯度范数 0.796，三个深度重算最大 logp 差 7.15e-5。第二步在带梯度 forward 时显存不足而失败，未达到验收标准。已保留失败日志，未缩短生成预算或放宽精度门槛。
- `runs/native2a-acceptance-v3-20261121` 被新加的权重版本检查拒绝，未进入更新：上游 slime 的多模型入口仅更新第一个可训练 server，深度 3/4 仍为 `default`，深度 2 为 `1`。因此 v2 也不能作为多深度权重同步验收。现已为 Ouro 显式共享策略增加全部深度组的同步/恢复视图，保留其他模型的原有行为。
- `runs/native2a-acceptance-v4-20261121` 原定 10 步、每步 2 题 × 6 条回答、2048-token、目标 β=0.1；完成 6 次更新并保存 `iter_0000005` 后按用户最新要求停止。前两步 sampler/recompute 最大 logp 差为 7.15e-5、1.11e-4；后四步也全部通过阈值。仅证明运行链路，不能据此声称任务收益。
- 第一份 checkpoint 已实读验收：final norm 最大变化 1.01e-6，Adam 一、二阶矩非零、步数为 2，scheduler 已消费 24 条样本，冻结 token gate 逐位不变。三个深度引擎已同步版本 `3`，报告 `results/native2a/first-checkpoint.json`。激活重算修复了此前第二步的显存失败；完整 10 步验收及严格恢复对照仍未完成。
- `runs/native2a-baseline-20261121` 的 8192-token 基座评测已停止，为正式训练释放 GPU；已完成的逐题结果保留。不同预算及不同 top-p 的结果不能直接作增益对照。
- 原生 prompt 深度头、一次/trajectory 的动作损失、独立学习率及保存/同步路径已实现，端到端训练仍待验收。完整 2.6B 上非零深度头的 SGLang/Megatron logq 最大差 5.96e-7，head 与共享 LM 梯度均非零，报告 `results/native2a/depth-head-consistency.json`。
- 60 项定向检查通过，记录 `results/native2a/tests.json`。原生 checkpoint 的 533 个模型张量已成功导出为评测用 HF 格式，重新导入后的全部 389 个原生参数与源 checkpoint 逐位一致；它仍来自原生训练。
- FP32 批量 decode 已通过 ragged mask 单测、128/2048-token 双请求与缓存重用对照；2048-token 的最大 logp 差 0.000246，低于既定 0.001。v2 显存失败批次里最长的四轮样本（总长 2458、回答 2048）单独使用原生重算完成 backward，带梯度/无梯度 logp 差为 0，峰值 allocated 23.03 GiB；该数值不含完整 trainer、冻结参考备份及 Adam 状态。
- `scripts/advance_native_2a.py` 的旧队列已停止，替换为上述直接正式训练的两条队列。原验收队列记录保留在 `runs/native2a-queue-v2-20261121`。正式组仍在任一命令或数值检查失败时停止，不静默重试或覆盖证据。训练扫描结束不等于 H4/H5 得到验证。
- 验证集选择入口 `scripts/select_native_2a_beta.py` 检查完整配对数据、正式训练 checkpoint 的臂/价格/步数和预先固定的训练集留出划分；拒绝错误数据、遗漏/重复样本及错误 checkpoint。
- `scripts/compare_native_2a_evaluations.py` 在题目层面重采样，报告配对准确率、回答长度、实际 loop-token 计算量和截断率差的区间；同题重复回答不会被误当成独立题目。不同生成预算的结果拒绝作配对比较。

待完成：正式六组训练与吞吐统计、验证集选价、学习深度头训练/恢复、同价固定深度对照和配对评测。均匀分层采样的平均深度天然接近 3，因此选 β 使用预先固定单位计算价格下的验证集效用；测试集不参与选择。

**2026-11-21 路线修正：用户要求正式实现落在 slime 代码上。现有自定义 HF trainer/sampler 保留为数值参考，后续先补 Megatron/SGLang 原生 Ouro 接入，再推进正式 2a 扫描。原生模型、采样、深度元数据与损失已接通：33 项测试通过，完整 2.6B 的 2/3/4 轮 logits/梯度对齐，SGLang 与 Megatron 重算最大 log 概率差 5.25e-6；固定深度原生两步训练、NCCL 同步与 checkpoint 参数/Adam 状态检查已通过（合成诊断 reward），恢复续训也已通过（尚未验收逐位一致），详见 [原生接入进度](integrations/slime/native/README.md)。**

本文件是 README 实验计划的执行记录。正式主模型沿用第 12 节已确定的 **Ouro-2.6B-Thinking**。2026-11-21 已完成参考路径的两步学习深度头和三步分层深度验收；进一步将参考后端接入 slime 训练编排（trainer/sampler 仍为自定义 HF 实现），在 FP32 路径通过固定文本的 2048-token 一致性检查，并完成两步长序列端到端训练、checkpoint 保存、权重同步和恢复续训。恢复与独立连续训练的模型/采样数值对照通过，优化器状态的严格跨运行对照未通过，详见下文。尚未得到 H4/H5 成立的证据，也没有启动 140 步的完整扫描或方案 2b 正式训练。

## 已实现的范围

| 文件 | 对应计划与功能 |
|---|---|
| `looprl/objectives.py` | 2a 序列深度策略损失；2b hazard、概率混合、后验校正项、分开 PPO 裁剪；β 升温、全 batch 原始 outcome 标准差、排除当前位置成本 |
| `looprl/ouro.py` | 真正只执行 k 个 loop 的固定深度；逐 loop 采样早退；可导 Ragged KV 重放；Q-exit/置信度基线；prompt 末位线性深度头 |
| `looprl/experiments.py` | 2a 固定 4、多深度分层、学习深度先验三种训练臂；冻结初始参考策略 KL；固定/动态/学习深度评测；M1–M5 汇总 |
| `looprl/history.py` | M7：同一文本上，完整历史与 ragged 历史的全词表 `KL(π_ragged || π_dense)`，含原生 gate 采样和 4/1 交替退出 |
| `looprl/data.py`、`rewards.py` | 固定版本的 MATH train / MATH-500；数据哈希；最后一个 boxed 答案的数学等价判分；未结束的思考不计正确 |
| `looprl/slime_hooks.py` | slime 的 reward 与计价后处理 hook，按完整 rollout batch 归一化 |
| `looprl/slime_train.py`、`slime_backend.py` | slime 原生训练循环 + 单训练 GPU / 单 HF 采样 GPU 的 Ouro 后端；权重同步、冻结参考、整段重算、checkpoint 恢复 |
| `looprl/consistency.py` | 固定文本、多个序列长度和深度的缓存/整段 log 概率与反向传播验收 |
| `integrations/slime/`、`scripts/prepare_slime.py` | 固定 slime revision、最小入口补丁、安装与复现说明 |
| `scripts/launch_stage0.py` | 先检查 GPU 空闲，再启动有限规模的配对先导评测；记录每个进程 PID、参数和日志 |

官方 Ouro `exit_at_step` 只在算满各 loop 后挑选读出；本项目不以它计固定深度节省。官方 `use_weighted_exit` 混合 logits，也不用于这里的 `Σ q_t P_t`。适配层直接使用每个 loop 的隐藏状态和原生 gate，最后一个 exit 接收剩余概率质量。

## 已完成的验收

- 24 项测试通过。除原有的 hazard 梯度、一致读出时 gate 梯度为零、极端 logits、β 不被归一化约掉、当前位置成本排除、深度动作只计一次、ragged 补位及其梯度、答案判分和配对评测汇总外，新增真实 slime `BatchBuilder`/训练入口集成、过期权重拒绝、参数探针、恢复配置和非有限数值拒绝检查。
- 两位置、四 exit 的穷举测试覆盖 **带价格** 的梯度：在独立 baseline、旧策略点、不发生 clipping 时，`κ=1` 分解与联合策略目标的梯度一致。原仓库 `scripts/exit_marginal_check.py` 也通过，残差在约 `1e-15`。
- 实际 2.6B 权重的 1/2/3/4 loop 调用计数正确；固定深度隐藏状态与全深度对应读出完全相同。
- 真实早退采样与 ragged replay 的逐 token 联合 log 概率最大误差为 `1.19e-7`（gate shift 0）和 `8.17e-4`（shift 2），此次各 8 token。
- 同一 Ouro 实现的小尺寸实例中，全部 exit=4 时 dense/ragged 读出差为 `2.38e-7`；早退后的最后位置 loss 对历史 prompt embedding 的梯度非零。
- 实际 2.6B 的学习深度头训练完成两步并保存 checkpoint。两步的 sampler/无梯度重放/带梯度重放 logp 最大误差均为 **0**；第二步产生非零梯度（裁剪前范数 `0.007394`），并确认初始为零的深度头权重发生更新。32-token 回答全部截断、outcome reward 全零，因此仅证明计价信号和训练链路有效，不证明数学能力提升。详见 [验收数据](results/2026-11-21/learned-depth-check.json)。

首轮 M7 只取两题各 32 token，结果是机制探针，不能外推为准确率或训练增益：

| 题目 | 原生 gate 采样，平均 KL | 4/1 交替，平均 KL |
|---|---:|---:|
| `test/precalculus/697.json` | 0.003047 | 0.016212 |
| `test/algebra/101.json` | 0.004325 | 0.031294 |

当前结果提示需要保留校正项及其消融，但还不足以选择最终 κ。

## 首轮任务记录

1. **阶段 0 配对先导评测（已完成 64/64）**：8 道随机固定的 MATH-500 题 × 2 次采样 × 固定深度 1/2/3/4，共 64 条回答；temperature=1，top-p=0.7，最大生成长度 4096。深度 4 额外记录 M1/M2/M3，合并四臂后统计 M4/M5。原 GPU 2/3/4/5 已释放。运行目录 `runs/stage0-pilot-20261121/`，完整结果为 `summary.json`。2/3/4 轮分别答对 12/16、13/16、12/16，截断率为 25%、18.75%、25%；只作基座先导，不构成训练收益。
2. **阶段 2a 工程冒烟验证（已完成）**：MATH train，3 步，每步 2 题 × 3 条分层深度 rollout，64-token 上限。目标 β=0.003125，使单价与正式 `β=0.1, Lmax=2048` 相同。有效尝试为 `runs/stage2a-smoke-20261121-v4/`，使用缓存重放和标准 AdamW；原 GPU 1 已释放。
3. **学习深度头验收（已完成）**：在 GPU 6 上执行 2 步、每步 1 题 × 3 条回答、32-token 上限、β=0.0015625 的 `learned` 臂，验证线性深度头能和 LM 一起优化。目录为 `runs/stage2a-learned-check-20261121/`，模型、深度头与优化器保存在 `checkpoint.pt`。

4096 仍可能截断 Thinking 的回答；必须报告截断比例。8 题、两次采样的先导结果不能替代 README 第 9 节的长预算、Avg@16 和重复评测。32/64-token 训练仅检查数据—采样—梯度—更新链路，尤其不能用其 reward 判断 H4/H5。

## 暴露的问题与处理

**BF16 训练/采样数值一致性。** 8-token 验收中，缓存采样与整段 teacher forcing 的最大 logp 差为 0.0984；扩展至真实 128-token 样本后达到 0.245591，超过 0.15 阈值，程序在优化器更新前拒绝继续。已保留失败日志 `runs/stage2a-smoke-20261121-v2.log`，没有放宽阈值。后续短程验收改为 `--replay-mode cached --logprob-tolerance 0.001`，并分别检查无梯度重算与带梯度重算。128-token 缓存重放在首步更新前显存达到 140451 MiB，已主动停止该次任务（v3），缩到 64 token；代码也限制此模式最多 64 token。新增 FP32 整段重算路径已通过下述 2048-token 检查；BF16 的数值问题本身仍未解决。

**优化器。** 系统 bitsandbytes CUDA 库要求 `GLIBC_2.34`，本环境加载失败。第一轮任务在采样期间被停止，尚未做更新。参考验证显式切换到标准 AdamW；正式计划的 AdamW 8bit 需要兼容的环境。两种优化器不可混作同一正式对照。

**slime。** 固定版本 `THUDM/slime@79aa41a884ab0bf65222c94418f6357d3c31192b` 没有现成的 Ouro Megatron 模型转换。现已实现保留共享循环权重的 HF Ouro trainer/sampler，运行实际 slime 训练循环、`RayTrainGroup` 和 `BatchBuilder`，通过文件校验、参数探针与版本号同步模型和深度头。新任务标记 `backend=slime_custom_ouro, slime_training=true`；先前参考任务仍标记 `hf_reference`。当前仅支持一个 trainer rank，Megatron/SGLang 与分布式吞吐优化尚未完成。详见 [接入说明](integrations/slime/README.md)。

**数学口径。** `κ=1` 的“精确梯度”指旧策略点的未裁剪 score 分解。分开 PPO clipping 在参数更新后并不等于一个精确的联合 PPO 比值。README 的组内均值含样本自身，沿用 GRPO 的有限组估计偏差；穷举恒等式使用动作无关 baseline，不声称消除了这个偏差。

**零方差 batch。** 若整个 batch 的原始 0/1 reward 都相同，`σ_R=0`，本实现改用单位尺度 1，保留价格梯度与 β 成比例，避免除极小 ε 放大价格信号。这是计划未指定边界条件的显式选择，会记录 `reward_scale`。

**自回归位置与计算账。** `e[j]` 属于预测 `response[j]` 的 query。j=0 的 query 是 prompt 末位，prompt 始终 dense，故它的 e 仅选择读出；j≥1 时 e 控制输入 token `response[j-1]` 留下的 KV。采样、重放、M7 均遵守这一索引。计划中的计价仍为 `Σ e[j]`；实际执行计算另外报告 `prefill_loop_tokens + Σ e[j] − e[0]`。学习序列深度的 prompt 四轮特征计算也计入 prefill，避免把额外开销隐去。Ragged KV 当前以深层缓存补位实现真实计算早退，尚未优化缓存内存占用。

## 2048-token 一致性与 slime 接入验收

在相同的 128-token 固定文本上，BF16 缓存/整段最大 logp 差为 0.096032；仅输出 head 使用 FP32 为 0.067005，SDPA math backend 为 0.097054；整个模型使用 FP32 并禁用 TF32 后降至 0.00001341。这是定位数值问题的对照，不是训练效果对照。

FP32 进一步使用同一条记录的前 2048 token，保持实际共享循环权重，整段训练启用非重入 gradient checkpointing：

| 实际深度 | 缓存/整段最大 logp 差 | 带梯度/无梯度重算差 | 反向传播峰值显存 |
|---|---:|---:|---:|
| 2 | 0.00003350 | 0 | 20.83 GiB |
| 3 | 0.00002563 | 0 | 21.64 GiB |
| 4 | 0.00021231 | 0 | 22.44 GiB |

均低于预设 `0.001` 阈值，梯度有限且非零。这里显存不含冻结参考与 AdamW 状态；完整训练的实测峰值约 60.4 GiB，学习深度臂约 60.5 GiB。深度 4 另有 512-token 检查通过。这组结果只验证 2a 固定序列深度在该文本上的一致性，不包含长序列 ragged 2b 验收。[原始数值与固定输入](results/2026-11-21/long-consistency/manifest.json) 已保留。

长序列端到端任务为 `runs/slime-ouro-acceptance-20261121/`：MATH train，两步，每步一题两条回答，深度 2/4，最大生成 2048，FP32。两步均完成，checkpoint 已保存，权重版本 0→1→2 全部通过同步校验。[完整验收摘要与源码快照](results/2026-11-21/slime-acceptance/summary.json) 已保留。

| 更新 | 长度（深度 4 / 2） | β | 原始 reward | 采样/训练最大误差 | 梯度范数（裁剪前） |
|---|---|---:|---|---:|---:|
| 第 1 步 | 1833 / 1645 | 0 | 1 / 1 | 0.00003338 | 0 |
| 第 2 步 | 2048 / 2048 | 0.1 | 0 / 0 | 0.00011611 | 0.109918 |

两步带梯度/无梯度重算差均为 0。第一步只有 AdamW weight decay 引起参数变化；第二步两条回答均截断，原始 reward 相同，计价差异产生非零策略梯度，LM 参数探针更新 L2 为 0.002375。旧策略点的组均值会使标量 loss 接近零，不代表梯度为零。这些结果证明真实长序列训练链路成立，不能说明准确率提升。

该次长序列任务在严格恢复数据哈希校验加入前启动，其历史 checkpoint 未含 `data_sha256`；随 run 保存了当时的源码。恢复测试使用随后生成的当前格式 checkpoint。后续运行会直接保存当前格式，不能把不同代码版本的 checkpoint 混用。

恢复验收另用 `learned` 臂、32-token 上限、两步、每题三条回答：`slime-resume-part1-20261121` 完成第一步后保存，`slime-resume-part2-20261121` 从第一步恢复并完成第二步；`slime-resume-control-20261121` 连续完成两步。保持总步数、价格调度、数据选择和随机种子相同。恢复前后的整份模型/深度头权重文件 SHA-256 完全相同，见 [恢复校验](results/2026-11-21/slime-resume/restore-proof.json)。恢复后的三条回答与连续训练的 token IDs 一致，采样 logp 最大差约 0.00000906。

独立连续运行与中断运行在**恢复前的第一步**已有最大 2.98e-8 的权重差，深度头完全相同；因此不声称跨运行逐位可复现。最终状态对照同时报告逐位差异和绝对误差，预先使用模型 1e-7、优化器状态 1e-10、采样 logp 1e-5 的诊断容差；训练自身的 0.001 阈值没有改变。

最终共比较 2136 个 tensor、8,003,943,919 个元素，见 [完整对照结果](results/2026-11-21/slime-resume-comparison.json)：模型最大差 5.96e-8，深度头 3.85e-8，采样 logp 9.06e-6，均在对应容差内；AdamW 状态最大差 6.53e-9，201 个状态 tensor 超过 1e-10 的严格阈值，因此整体 `passed=false`、`bitwise_equal=false`。未放宽阈值。恢复前已存在的跨运行浮点差异尚未与恢复后的差异单独隔离；后续需使用同一 checkpoint 边界的确定性对照。当前结论是能够恢复并继续训练、模型和采样结果数值接近，不能声称优化器状态的严格跨运行复现已经通过。

运行中另发现两个吞吐问题：多个 worker 被绑定在 CPU 0/96 上，现按 GPU 分配不同 CPU 核；共享盘 mmap 加载约 30 GiB checkpoint 产生大量小块读，现先顺序复制到本地后加载。对活动任务的 CPU 调整已记录在各 run 的 `resource-adjustments.json`。独立实测顺序复制/校验用时约 129 秒；这些检查也说明当前后端的 I/O 与串行采样成本仍需优化，不能直接外推正式训练吞吐。

## 复现命令

隔离环境安装，不改已有服务的 Python 环境：

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[ouro,test]'
# 本机 Jinja2 元数据与实际版本不一致，需要在隔离环境重新安装：
.venv/bin/pip install --ignore-installed --no-deps jinja2==3.1.6
.venv/bin/python -m pytest -q
.venv/bin/python scripts/exit_marginal_check.py
.venv/bin/python -m looprl prepare-data --output data
```

本轮使用 `/tmp/looprl-venv/bin/python` 和 `/tmp/looprl-models/Ouro-2.6B-Thinking`。远端加载默认锁定模型 revision `f1edd81e7ac41355db670500ceaf204e0f73af68`。每个 run 的 manifest 保存数据、源码哈希和版本；新 run 同时保存源码副本，已有目录拒绝覆盖。

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m looprl preflight --output runs/preflight-new
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m looprl history-probe \
  --data data/math500.jsonl --limit 2 --max-new-tokens 32 --output runs/m7-new
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m looprl train \
  --data data/math_train.jsonl --arm stratified --steps 3 \
  --prompts-per-step 2 --group-size 3 --limit 12 --max-new-tokens 64 \
  --beta 0.003125 --optimizer adamw --replay-mode cached \
  --logprob-tolerance 0.001 --output runs/stage2a-smoke-new
```

每个任务实际状态以 run 内 `status.json` 及阶段 0 的 `progress.json` 为准。空闲后可用 launcher 重做配对先导评测；launcher 会拒绝占用超过 1 GiB 的 GPU。评测完成后合并：

```bash
.venv/bin/python -m looprl summarize \
  runs/stage0-pilot-20261121/depth{1,2,3,4}/samples.jsonl \
  --output runs/stage0-pilot-20261121/summary.json
```

本轮还启动了只读监控 `python -m looprl.monitor --launch runs/stage0-pilot-20261121/launch.json --watch`，每 30 秒更新 `progress.json`，所有 worker 退出后写 `summary.json` 并自行结束。它不停止任何 GPU 进程；其 `summary.json` 的 `metrics` 字段是合并指标。手动 `summarize` 输出直接是指标字典。

训练数据共 7498 条：源 MATH train 中两条缺少可抽取 boxed 答案，暂时过滤；MATH-500 保留 500 条。准备时核对 train/test 的题目文本没有完全重合。原始下载、过滤后的文件均记录哈希，不能把 MATH-500 输入用于训练。正式 β 扫描配置在 `configs/stage2a.json`，这是实验规格，不是已完成的实验记录。

## 下一步的完成门槛

先完成 Megatron/SGLang 原生 Ouro 模型、采样、权重同步与损失适配；现有 HF 参考用于对照其数值。随后隔离严格恢复复现中的优化器差异，统一原生训练/采样的精度、解码和优化器协议，并完成正式 batch 规模和吞吐验收，再开始 2a 的完整多深度/β 扫描，同时训练同 β 的 GRPO@4 长度计价对照；学习先验的工作点不能从测试集挑选。阶段 0 的 64 条先导已完成，仍需扩大配对评测、补长预算基线和 M6 同设置对照。更长序列需单独验收，不能从两步试运行外推。只有阶段 2 为正，才启动 2b 的 κ=1/0 正式对照。当前 2b 已有损失、采样和 KV 重放实现，但没有正式训练驱动或结果。

实现依据：[Ouro 固定版本源码](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking/blob/f1edd81e7ac41355db670500ceaf204e0f73af68/modeling_ouro.py)、[slime 固定版本 batch builder](https://github.com/THUDM/slime/blob/79aa41a884ab0bf65222c94418f6357d3c31192b/slime/data/batch_builder.py)。
