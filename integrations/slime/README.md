# Ouro / slime 方案 2a 后端

**2026-11-21 路线修正：正式实验改在 slime 的 Megatron/SGLang 后端上实现。下文已验收的是自定义 HF 参考后端，不能作为原生后端就绪的证据。** 原生代码位于 `third_party/slime`；可复现补丁和当前缺口见 [原生接入进度](native/README.md)。原有 `/tmp/looprl-slime` 与历史结果保留作数值对照。

当前实现使用 slime 原生训练循环、`RayTrainGroup`、`BatchBuilder` 和 checkpoint 调度，接入自定义 HF Ouro trainer 与 sampler。它保留 Ouro 的 48 个共享 decoder 层和真实循环执行，不展开成 192 个独立层。支持固定 4 loop、分层深度和学习序列深度三种 2a 训练臂。

这是单机、一个训练 GPU 加一个采样 GPU 的正确性后端。本段只描述历史 HF 参考路径；当前已完成的原生 Megatron/SGLang 工作见上方链接。多卡并行训练与方案 2b 正式训练尚未验收。HF sampler 串行生成，不能把此处吞吐量当作 SGLang 的吞吐量。

## 安装和锁定版本

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install -e '.[ouro,slime,test]'
.venv/bin/python scripts/prepare_slime.py --backend hf-reference
LOOPRL_SLIME_PATH=third_party/slime-hf-reference .venv/bin/python -m pytest -q
```

上述命令显式准备 HF 参考补丁。`prepare_slime.py` 现在默认准备原生 checkout `third_party/slime`；两套补丁必须使用不同目录。

slime 锁定到 `79aa41a884ab0bf65222c94418f6357d3c31192b`。`prepare_slime.py` 应用本目录的 `custom-actor.patch`：给 `train()` 开放已有的 `actor_cls` 参数，并把默认 CLI 的 SGLang 依赖延迟到 CLI 使用时导入；训练循环的执行顺序保持原样。已有 checkout 的版本不符时脚本拒绝修改。Ouro 远端权重锁定到 `f1edd81e7ac41355db670500ceaf204e0f73af68`。

本环境使用 `/tmp/looprl-venv`、`/tmp/looprl-slime` 和 `/tmp/looprl-models/Ouro-2.6B-Thinking`。系统 Jinja2 元数据与实际代码不一致，已仅在隔离环境重装 `jinja2==3.1.6`。bitsandbytes 的系统库不兼容，因此此后端使用标准 AdamW。

## 长序列一致性

默认 `--precision fp32`，关闭 TF32，整段 teacher forcing 配合非重入 gradient checkpointing。训练参数、采样参数和 AdamW 状态使用 FP32；并非只有输出 head 转 FP32。这样既降低了缓存采样与整段重算的误差，也避免 BF16 参数更新小于可表示步长的问题。

每个 batch 在更新前验证：

1. sampler 权重版本与 trainer 相同，收到完整 rollout batch。
2. 每条回答的采样 log 概率与整段旧策略重算误差不超过 `0.001`。
3. 带梯度与不带梯度重算误差不超过 `0.001`；学习深度臂同时检查旧深度动作概率。
4. loss 与梯度范数有限；记录参数探针的实际更新量。

超过阈值直接拒绝更新。BF16 的整段重算目前未通过；只转换输出 head 或切换 SDPA math backend 也未解决。`--precision bf16` 保留为诊断入口，不属于已验收配置。

独立固定文本检查命令如下。它验证数值与梯度，不能当作准确率实验：

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m looprl.consistency \
  --model /tmp/looprl-models/Ouro-2.6B-Thinking \
  --rollouts results/2026-11-21/long-consistency/long-input.jsonl \
  --output runs/consistency-new --precision fp32 \
  --depths 2 3 4 --lengths 512 2048 --backward
```

## 运行和恢复

入口要求显式指定两个不同且空闲的物理 GPU。每次建立自己的本地 Ray cluster，只清理自己创建的 actor，不调用全局 `ray stop`。worker 在 driver 可用 CPU 集合内按物理 GPU 分配四个逻辑核，避免继承到相同的单核亲和性；本机已做真实 Ray/CUDA 检查。CPU 调整不改变采样策略或数值阈值。

```bash
CUDA_VISIBLE_DEVICES=0,6 .venv/bin/python -m looprl.slime_train \
  --slime-path /tmp/looprl-slime \
  --model /tmp/looprl-models/Ouro-2.6B-Thinking \
  --data data/math_train.jsonl --output runs/slime-acceptance-new \
  --steps 2 --prompts-per-step 1 --group-size 2 --depths 2 4 \
  --max-new-tokens 2048 --limit 4 --precision fp32
```

默认温度为 1、top-p 为 1；旧参考评测的 top-p=0.7、BF16 配置不能直接充当此 FP32 训练的正式对照。全 batch 原始正确性 reward 的标准差用于归一化，先减计算价格再求组均值；全 batch reward 恒定时用单位尺度。KL 参考始终是初始基座。学习深度特征由完整 prompt 的四轮读出产生，其额外执行计算在该组第一条 rollout 的 prefill 账中记录一次。

每轮更新后，trainer 原子写出模型和深度头权重，校验整个文件的 SHA-256；sampler 严格加载，再比对每个参数的固定探针和版本。日志 `weight-sync.jsonl` 同时保留文件 SHA 与探针指纹，后者不是整个 GPU 参数的逐元素哈希。

checkpoint 包含模型、深度头、AdamW 状态、随机状态、数据哈希和更新位置。FP32 checkpoint 约 30 GiB；共享盘直接 mmap 会产生大量小块读取，因此恢复前先顺序复制到本地 `/tmp`，记录 SHA-256，再加载。需要相应本地磁盘空间。恢复到**新的输出目录**，保留原总步数和实验配置；`--stop-after` 只设置本次停止边界，不改变价格或学习率调度。

```bash
# 原计划 140 步，本次在完成 10 步并保存后停止。
CUDA_VISIBLE_DEVICES=0,6 .venv/bin/python -m looprl.slime_train \
  --data data/math_train.jsonl --output runs/slime-part1 \
  --steps 140 --stop-after 10 --precision fp32

# 使用相同模型、数据选择、臂、batch、价格和学习率配置继续。
CUDA_VISIBLE_DEVICES=0,6 .venv/bin/python -m looprl.slime_train \
  --data data/math_train.jsonl --output runs/slime-part2 \
  --steps 140 --precision fp32 \
  --resume runs/slime-part1/checkpoints/step-000010.pt
```

`manifest.json` 保存源码副本的哈希、slime revision、补丁、数据哈希和配置；`rollouts.jsonl`、`metrics.jsonl`、`weight-sync.jsonl` 与 `checkpoint.json` 对应采样、更新、同步和保存证据。完整 140 步扫描与 H4/H5 的配对效果验证仍应按主 README 执行；本后端的短程验收不替代它们。

恢复对照可用 `scripts/check_slime_resume.py` 逐元素比较模型、深度头、AdamW、随机状态和采样记录。默认要求完全相同；若用于独立 GPU 运行的数值诊断，必须显式给出容差。它会分别报告 `passed` 和 `bitwise_equal`，不会把数值通过写成逐位相同。本次独立运行在恢复前已出现约 3e-8 的 FP32 权重差，因此同时保存这一参照。

```bash
PYTHONPATH=. .venv/bin/python scripts/check_slime_resume.py \
  --control runs/slime-resume-control-20261121 \
  --resumed runs/slime-resume-part2-20261121 \
  --output runs/resume-comparison-new.json \
  --model-atol 1e-7 --optimizer-atol 1e-10 --logprob-atol 1e-5 \
  --local-cache /tmp/looprl-resume-comparison-new
```

比较两份完整 checkpoint 需要约 60 GiB 本地缓存空间；复制结果保留在指定目录，脚本拒绝覆盖已有缓存文件。

本次命令的实际结果是模型/采样在容差内，但优化器最大差 6.53e-9 超过 1e-10，整体返回失败；原始结果保留在 [恢复对照报告](../../results/2026-11-21/slime-resume-comparison.json)。恢复起点的权重文件校验完全相同，独立两次第一步训练已出现浮点差异，严格跨运行复现仍需继续隔离。不要把“能恢复续训”写成“逐位可复现”。
