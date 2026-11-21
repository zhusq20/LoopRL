# Ouro 原生 slime 接入

2026-11-21，正式方案 2a 在 slime 的原生 Megatron trainer、rollout manager 和 SGLang 引擎上实现。`looprl/slime_backend.py` 保留作 HF 参考；本目录结果不使用该 HF actor。14:45 起正式固定/分层扫描统一使用 BF16 + TIS，最新状态见 [EXPERIMENTS.md](../../../EXPERIMENTS.md)。

## 代码与版本

- slime：Git 子模块 `third_party/slime`，上游 [THUDM/slime](https://github.com/THUDM/slime)，固定 `79aa41a884ab0bf65222c94418f6357d3c31192b`。GitHub 中点击子模块目录会打开这个上游版本。`train.py` 保持上游原样；[ouro-native.patch](../ouro-native.patch) 保存全部 slime 修改，已在新 checkout 验证可应用。
- Megatron：`ae76c60e6c35d913867d6dd7bad0f30b128b54e7`；SGLang：`e00930c5489053f26d86b179cee0d087f846acbb`（0.5.21）。两者必须应用 slime `docker/patch/v0.5.21` 的对应补丁；SGLang 再应用 [sglang-ouro.patch](sglang-ouro.patch)。Megatron 再应用 [megatron-adam-resume.patch](megatron-adam-resume.patch)，修复 PyTorch Adam 空状态模板、恢复后参数间步数张量别名与分片恢复的标量 step 处理。缺少上游兼容补丁会导致暂停生成期间的权重同步被拒绝。
- Ouro checkpoint：`f1edd81e7ac41355db670500ceaf204e0f73af68`，本机 `/tmp/looprl-models/Ouro-2.6B-Thinking`。
- 运行环境：Python 3.12.16、PyTorch 2.13.0+cu130、Transformers 5.12.1。HF oracle 单独保留 Transformers 4.57.1。精确依赖见 [runtime-lock.txt](runtime-lock.txt)、[oracle-lock.txt](oracle-lock.txt)。
- 本机使用隔离的 CUDA 13 compatibility library（580.178.04）和 GCC 12.4；系统 driver 未修改。`nvcc` 为系统 CUDA 12.8。编译器环境见 [compiler-lock.txt](compiler-lock.txt)，路径与变量见 [env.sh](env.sh)。

## 已实现并验证

1. `slime_plugins/models/ouro.py`：原生 Megatron 模型，48 个物理层共享，按序列执行 k 次；每层四个 RMSNorm，每轮结束复用 final norm。packed attention 隔离不同序列。exit gate 随 checkpoint 保存和同步，2a 固定深度下冻结。
2. HF ↔ Megatron 映射覆盖真实 checkpoint 的全部 **533** 个张量，包括 sandwich norms；没有展开成 192 份层权重。
3. `slime_plugins/sglang_models/ouro.py`：SGLang 外部模型，投影权重共享，每轮使用独立 KV 槽；一个引擎固定一个深度。两处 SGLang 核心缓存计数通过单独补丁适配。
4. `slime_plugins/ouro_2a.py`：样本执行深度贯穿 batch 分发、旧策略重算与带梯度 forward；混合深度 microbatch 拒绝执行，当前使用 micro-batch-size=1。`--ouro-beta` 指定计算价格（默认 0）；计价后按 prompt group 中心化，除以整个 batch 原始 outcome 标准差，恒定 outcome 使用单位尺度。PPO token 项按固定生成预算归一化。
5. 原生训练支持 FP32 与 BF16、forward kwargs hook。FP32 无校正模式的采样/重算误差超过 0.001 时拒绝更新；BF16 要求显式 `--rollout-correction tis`，自定义 PPO loss 乘以停止梯度的旧训练策略/采样策略重要性权重，上截断为 2。同一训练实现的带梯度/无梯度重放保留 0.001 门槛。记录原始误差、TIS 权重、截断率和 ESS，不声称 TIS 使两个引擎数值相等。

数值验收：

| 检查 | 结果 |
|---|---|
| 实际 MCore vs 缩小的真实 HF Ouro，MHA/GQA、1/2/3/4 轮 logits/全部梯度、packed 隔离、转换 | 15 项通过 |
| 2a 计算价格、batch reward 尺度、元数据分发、错误深度拒绝、packed loss 梯度 | 12 项通过 |
| PyTorch Adam 空状态、步数一致性与恢复后独立计数、分片矩恢复及缺失矩拒绝 | 6 项通过 |
| 完整 2.6B checkpoint，128/2048 token，2/3/4 轮 | log 概率最大差 1.53e-5；2048 token 各深度差为零；argmax 全相同 |
| 完整 checkpoint 的全部参数梯度，16 token，2/3/4 轮 | 相对 L2 误差 6.28e-6～7.42e-6 |
| SGLang 双请求与单请求重复，2/3/4 轮，各生成 8 token | 与 Megatron 重算最大 log 概率差 5.25e-6；重复请求实际复用了 27 token 的 prefix cache |

原始数据：[完整模型检查](../../../results/native/megatron-full-checkpoint.json)、[采样一致性](../../../results/native/serving-consistency.json)、`results/native/sglang-depth{2,3,4}.json`。上述早期检查使用 FP32/SDPA，是数值验收。后续 BF16/FlashInfer/CUDA Graph 和两步 TIS 训练验证见 [性能与校正报告](../../../results/native2a/performance-v2-20261121/bf16-tis-validation.json)；TP/PP/CP>1 仍未验收。

## 原生生命周期检查

固定深度 2 的两步原生 smoke **已通过**（64-token 上限，每步 2 条回答）：梯度范数 16.83/30.99；更新后采样与重算最大差 2.53e-5；两次 checkpoint 与三次原生 NCCL 权重同步完成。读取最终 checkpoint 验证 final norm 的 2048 个元素全部变化（最大 2.03e-6），Adam 一、二阶矩非零，冻结 exit gate 逐元素未变。报告：[lifecycle.json](../../../results/native/lifecycle.json)。

第一步 checkpoint 的恢复续训也已通过：恢复 iteration 0 后完成 rollout 1，梯度范数 16.29，采样与重算最大差 1.62e-5；Adam 步数从 1 增到 2，scheduler 的样本计数从 2 增到 4，final norm 在恢复后继续变化（最大 1.01e-6），冻结 exit gate 未变，并完成保存及权重同步。报告：[resume.json](../../../results/native/resume.json)。这证明恢复状态并继续更新可用，尚未验收与独立连续运行逐位一致。

入口 [run_ouro_native_smoke.py](../../../scripts/run_ouro_native_smoke.py) 调用未修改的 slime `train.main`，使用真正的 MegatronTrainRayActor 与 SGLangEngine、原生 optimizer、NCCL 权重同步和 checkpoint。事件摘录见 [生命周期日志](../../../results/native/lifecycle-events.log)、[恢复日志](../../../results/native/resume-events.log)。

该检查明确使用 `slime_plugins.ouro_smoke.reward` 的 **合成 0/1 reward**，构造短序列更新诊断；验收另外要求每步梯度非零、checkpoint 参数确实变化。它仅验证更新链路，不能报告为数学任务收益。真实数学训练的新入口为 `scripts/run_native_2a.py`，当前执行状态见 [EXPERIMENTS.md](../../../EXPERIMENTS.md)。

本机 glibc 不兼容 `nvidia-resiliency-ext==0.7.0` 的预编译包，因此 smoke 显式传入 `--disable-checkpoint-metadata-cache`，使用 Megatron 标准 checkpoint reader，保存内容与恢复的 optimizer 状态不变。该开关默认关闭，常规 slime 环境仍沿用原有缓存设置。

## 本机复现

新 clone 可以使用 `git clone --recurse-submodules` 获取锁定的 slime 源码；普通 clone 运行下方准备脚本时也会自动初始化子模块，再应用原生补丁。已有 checkout 只核对版本与补丁，不覆盖本地修改。应用补丁后子模块显示为 modified 是预期状态，修改由主仓库中的补丁文件保存。

```bash
python scripts/prepare_slime.py --backend native
source integrations/slime/native/env.sh
# 在全新机器先按 lock 文件建立两个 Python 3.12 环境；env.sh 的 /tmp 路径需对应。
# 源码准备器下载固定 revision，应用补丁并记录 SHA-256；已有目录只验证，不覆盖。
"$LOOPRL_NATIVE_PYTHON" scripts/prepare_native_sources.py

CUDA_VISIBLE_DEVICES=0 OURO_HF_CHECKPOINT=/tmp/looprl-models/Ouro-2.6B-Thinking \
  "$LOOPRL_ORACLE_PYTHON" -m pytest -q \
  third_party/slime/tests/test_ouro_model.py third_party/slime/tests/test_ouro_weight_conversion.py
CUDA_VISIBLE_DEVICES=0 "$LOOPRL_NATIVE_PYTHON" -m pytest -q third_party/slime/tests/test_ouro_2a.py
"$LOOPRL_NATIVE_PYTHON" -m pytest -q tests/native/test_megatron_adam_checkpoint.py

CUDA_VISIBLE_DEVICES=1 "$LOOPRL_NATIVE_PYTHON" scripts/check_ouro_sglang.py \
  --checkpoint /tmp/looprl-models/Ouro-2.6B-Thinking --depth 2 \
  --output /tmp/sglang-depth2-new.json
CUDA_VISIBLE_DEVICES=0 "$LOOPRL_ORACLE_PYTHON" scripts/check_ouro_native.py \
  --checkpoint /tmp/looprl-models/Ouro-2.6B-Thinking \
  --sglang-output /tmp/sglang-depth2-new.json --output /tmp/native-check-new.json

# 选择两张空闲 GPU；输出目录必须不存在。只清理本次创建的 Ray cluster。
CUDA_VISIBLE_DEVICES=2,3 "$LOOPRL_NATIVE_PYTHON" scripts/run_ouro_native_smoke.py \
  --checkpoint /tmp/looprl-models/Ouro-2.6B-Thinking \
  --depth 2 --steps 2 --max-new-tokens 64 --output /tmp/native-smoke-new \
  > /tmp/native-smoke-new.log 2>&1

# 从第一次更新后的边界恢复；steps 仍是总 rollout 数，复用原始数据路径。
CUDA_VISIBLE_DEVICES=2,3 "$LOOPRL_NATIVE_PYTHON" scripts/run_ouro_native_smoke.py \
  --checkpoint /tmp/looprl-models/Ouro-2.6B-Thinking \
  --resume /tmp/native-smoke-new/checkpoints --resume-step 0 \
  --prompt-data /tmp/native-smoke-new/diagnostic-prompts.jsonl \
  --depth 2 --steps 2 --output /tmp/native-resume-new \
  > /tmp/native-resume-new.log 2>&1

"$LOOPRL_NATIVE_PYTHON" scripts/summarize_ouro_native_smoke.py \
  --run /tmp/native-resume-new --log /tmp/native-resume-new.log \
  --base-checkpoint /tmp/looprl-models/Ouro-2.6B-Thinking \
  --output /tmp/native-resume-report.json
```

## 仍待完成

- 长序列真实奖励两步 BF16/TIS 验收及首个正式 32×8 更新已完成；更长稳定性、同一 checkpoint 边界的严格恢复对照和正式稳态吞吐仍待验证。
- 学习深度头的原生端到端训练/保存/恢复验收。模型与动作 loss 已实现，完整模型的采样/重算深度概率及共享 LM 梯度检查已通过。
- 140 步 β 扫描、验证集选价、同价固定深度对照及配对重复评测。数学 reward、价格升温、初始参考 KL、分层引擎路由和权重版本检查均已接入新入口。

用户已授权先运行固定/分层 140 步扫描；未完成的严格恢复与学习深度头验收继续单独记录，不声称 2a 已有算法收益。

当前正式训练使用 BF16、标准 AdamW、逐层激活重算，主权重、梯度累积与优化器状态保留 FP32；学习率、采样参数、价格与数据划分固定在每次运行的 `protocol.json`。运行保存独立源码快照，开发中的后续修改不影响已启动的 actor。`scripts/advance_native_2a.py --direct-formal` 可直接排队正式组，任一阶段失败就停止后续阶段；`status.json` 与各阶段日志保存于队列目录。具体计划见 [native2a-plan.json](../../../configs/native2a-plan.json)。

BF16 服务配置参数为 `--precision bf16 --rollout-correction tis --attention-implementation bf16-fast --cuda-graph --max-running-requests 32 --kv-token-capacity 65536 --serving-memory-fraction .85 --depth4-replicas 2`。两副本配置需要固定组 3 张、分层组 5 张 GPU；显存容量按本机 L20X 验证，换硬件需重新确定 KV 容量。默认入口仍保留 FP32 参考配置，跨精度恢复会被协议检查拒绝。
