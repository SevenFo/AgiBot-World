# GO-1 分布式训练与 DeepSpeed 配置指南

> 本文档详细解析 GO-1 项目中分布式训练的核心配置，包括 torchrun 环境变量、DeepSpeed ZeRO 优化策略以及实际部署建议。

---

## 1. Torchrun 分布式环境变量详解

### 1.1 核心环境变量

从 [`go1/shell/train.sh`](../go1/shell/train.sh) 脚本中提取的关键环境变量：

| 环境变量         | 默认值    | 含义                   | 示例                     |
| ---------------- | --------- | ---------------------- | ------------------------ |
| `WORLD_SIZE`     | 1         | 总节点数量（机器数量） | 单机=1，双机=2           |
| `MASTER_ADDR`    | 127.0.0.1 | 主节点IP地址           | 192.168.1.100            |
| `RANK`           | 0         | 当前节点在集群中的排名 | 主节点=0，从节点=1,2,... |
| `NPROC_PER_NODE` | 8         | 每个节点的GPU/进程数   | 8卡机器=8，4卡=4         |
| `MASTER_PORT`    | 12345     | 主节点通信端口         | 任意空闲端口             |

### 1.2 NCCL 通信优化

```bash
export NCCL_P2P_LEVEL=NVL
```

- **NVL (NVLink)**：启用 NVIDIA NVLink 点对点高速通信
- **适用场景**：多 GPU 间有 NVLink 连接的高端服务器（如 DGX、RTX 4090 多卡）
- **性能收益**：相比 PCIe，NVLink 提供更高带宽和更低延迟

### 1.3 分布式启动命令解析

```bash
torchrun \
  --nnodes=${WORLD_SIZE} \          # 节点数量
  --node-rank=${RANK} \             # 当前节点排名
  --master-addr=${MASTER_ADDR} \    # 主节点地址
  --nproc-per-node=${NPROC_PER_NODE} \ # 每节点进程数
  --master-port=${MASTER_PORT} \    # 通信端口
  go1/internvl/train/go1_train.py \ # 训练脚本
  --cfg ${CFG}                      # 配置文件
```

### 1.4 常见部署场景

#### 单机多卡训练
```bash
# 8卡 RTX 4090
export WORLD_SIZE=1
export RANK=0
export NPROC_PER_NODE=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12345
```

#### 双机16卡训练
```bash
# 主节点 (192.168.1.100)
export WORLD_SIZE=2
export RANK=0
export NPROC_PER_NODE=8
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=12345

# 从节点 (192.168.1.101)
export WORLD_SIZE=2
export RANK=1
export NPROC_PER_NODE=8
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=12345
```

---

## 2. DeepSpeed ZeRO 优化策略详解

### 2.1 ZeRO 概述

**ZeRO (Zero Redundancy Optimizer)** 是 DeepSpeed 的核心内存优化技术，通过分片存储模型状态来大幅降低显存占用。

### 2.2 三个 Stage 对比

| Stage       | 分片内容         | 内存节省 | 通信开销 | 适用场景        |
| ----------- | ---------------- | -------- | -------- | --------------- |
| **Stage 1** | 优化器状态       | ~4x      | 低       | 大模型+充足显存 |
| **Stage 2** | 优化器+梯度      | ~8x      | 中等     | 中等显存约束    |
| **Stage 3** | 参数+优化器+梯度 | ~64x     | 高       | 严重显存不足    |

### 2.3 Stage 1 配置详解

**文件**: [`go1/zero_stage1_config.json`](../go1/zero_stage1_config.json)

```json
{
  "zero_optimization": {
    "stage": 1,
    "allgather_partitions": true,      // 收集分片时的优化
    "allgather_bucket_size": 1e9,      // 收集桶大小 (1GB)
    "overlap_comm": true,              // 通信与计算重叠
    "reduce_scatter": true,            // 分布式规约优化
    "reduce_bucket_size": 1e9,         // 规约桶大小 (1GB)
    "contiguous_gradients": true       // 梯度内存连续化
  }
}
```

**特点**：
- 仅分片优化器状态 (Adam 的 momentum 和 variance)
- 内存节省有限但通信开销最小
- 适合显存充足的场景

### 2.4 Stage 2 配置详解

**文件**: [`go1/zero_stage2_config.json`](../go1/zero_stage2_config.json)

```json
{
  "zero_optimization": {
    "stage": 2,
    "allgather_bucket_size": 1e8,      // 更小的桶 (100MB)
    "reduce_bucket_size": 1e8,         // 更频繁的通信
    // ... 其他配置同 Stage 1
  }
}
```

**特点**：
- 分片优化器状态 + 梯度
- 平衡内存节省与通信效率
- **推荐用于大多数场景**

### 2.5 Stage 3 配置详解

**文件**: [`go1/zero_stage3_config.json`](../go1/zero_stage3_config.json)

```json
{
  "zero_optimization": {
    "stage": 3,
    "stage3_prefetch_bucket_size": 1e9,        // 预取桶大小
    "stage3_param_persistence_threshold": 1e7, // 参数持久化阈值
    "stage3_max_live_parameters": 1e9,         // 最大活跃参数
    "stage3_max_reuse_distance": 1e9,          // 参数重用距离
    "stage3_gather_16bit_weights_on_model_save": true // 保存时收集16位权重
  }
}
```

**特点**：
- 分片所有模型状态（参数、梯度、优化器）
- 最大内存节省，但通信开销高
- 需要精细调优通信参数

### 2.6 混合精度配置

所有 Stage 都支持 FP16/BF16 混合精度：

```json
{
  "fp16": {
    "enabled": "auto",           // 自动检测GPU支持
    "auto_cast": true,          // 自动类型转换
    "loss_scale": 0,            // 动态损失缩放
    "initial_scale_power": 32,  // 初始缩放指数
    "loss_scale_window": 1000,  // 缩放窗口
    "hysteresis": 2,            // 缩放阈值
    "min_loss_scale": 1         // 最小缩放值
  },
  "bf16": {
    "enabled": "auto"           // BF16支持（需要较新GPU）
  }
}
```

### 2.7 优化器配置

```json
{
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",              // 从训练配置读取
      "betas": [0.9, 0.999],     // Adam beta参数
      "eps": 1e-8,               // 数值稳定性
      "weight_decay": "auto"     // 权重衰减
    }
  }
}
```

---

## 3. GO-1 训练脚本中的 DeepSpeed 集成

### 3.1 配置文件选择

GO-1 训练脚本会根据模型大小和显存情况自动选择 ZeRO 策略：

```python
# 在训练配置中指定
deepspeed_config = "go1/zero_stage2_config.json"  # 默认推荐
```

### 3.2 训练参数自动映射

配置文件中的 `"auto"` 参数会从训练脚本中自动读取：

```python
training_args = TrainingArguments(
    learning_rate=1e-5,                    # 映射到 optimizer.params.lr
    per_device_train_batch_size=2,         # 映射到 train_micro_batch_size_per_gpu
    gradient_accumulation_steps=4,         # 映射到 gradient_accumulation_steps
    weight_decay=0.01,                     # 映射到 optimizer.params.weight_decay
    max_grad_norm=1.0,                     # 映射到 gradient_clipping
)
```

### 3.3 显存优化建议

根据显存大小选择配置：

| 显存大小         | 推荐配置 | 备注                |
| ---------------- | -------- | ------------------- |
| 24GB+ (RTX 4090) | Stage 1  | 充足显存，追求速度  |
| 16GB (RTX 4080)  | Stage 2  | 平衡性能与内存      |
| 12GB (RTX 4070)  | Stage 3  | 显存紧张            |
| 8GB 以下         | 不推荐   | 建议使用更大显存GPU |

---

## 4. 实际部署与调优

### 4.1 启动脚本示例

**单机8卡训练**：
```bash
# 设置环境变量
export RUNNAME="go1_custom_$(date +%Y%m%d_%H%M)"
export NPROC_PER_NODE=8
export NCCL_P2P_LEVEL=NVL

# 启动训练
bash go1/shell/train.sh go1/configs/go1_sft_custom.py
```

**多机训练**：
```bash
# 主节点启动
export RUNNAME="go1_multi_node"
export WORLD_SIZE=2
export RANK=0
export MASTER_ADDR=192.168.1.100
bash go1/shell/train.sh go1/configs/go1_sft_custom.py

# 从节点启动
export RUNNAME="go1_multi_node"
export WORLD_SIZE=2
export RANK=1
export MASTER_ADDR=192.168.1.100
bash go1/shell/train.sh go1/configs/go1_sft_custom.py
```

### 4.2 性能监控

**日志监控**：
```bash
# 实时查看训练日志
tail -f experiment/${RUNNAME}/log/training_log_*.txt

# 监控GPU使用情况
watch -n 1 nvidia-smi
```

**关键指标**：
- GPU 利用率：应保持 90%+ 
- 内存使用率：Stage 2 下应 < 90%
- 通信效率：看 `samples/sec` 指标

### 4.3 常见问题与解决

#### 显存不足 (OOM)
```bash
# 解决方案：
# 1. 降低 batch size
per_device_train_batch_size: 1  # 从 2 降到 1

# 2. 增加梯度累积
gradient_accumulation_steps: 8  # 从 4 增到 8

# 3. 使用更激进的 ZeRO stage
deepspeed_config: "go1/zero_stage3_config.json"
```

#### 通信超时
```bash
# 增加超时时间
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=INFO  # 开启调试信息
```

#### 节点同步失败
```bash
# 检查网络连通性
ping ${MASTER_ADDR}

# 确保端口未被占用
netstat -ln | grep ${MASTER_PORT}
```

---

## 5. 最佳实践总结

### 5.1 配置选择指南

1. **新手推荐**：使用 Stage 2 + FP16，兼顾内存效率与训练稳定性
2. **高级用户**：根据模型大小和硬件条件精细调优桶大小和通信参数
3. **生产环境**：使用 Stage 1 确保最高性能，配合充足的GPU内存

### 5.2 调试技巧

```bash
# 开启调试模式
export DEBUG_MODE=true

# 使用小数据集快速验证
export RUNNAME="debug_run"
bash go1/shell/train_dev.sh go1/configs/go1_sft_debug.py
```

### 5.3 扩展建议

- **更大模型**：考虑使用 ZeRO-Infinity (offload 到 CPU/NVMe)
- **更多节点**：配置 InfiniBand 网络以降低通信延迟
- **混合训练**：结合数据并行和模型并行策略

---

## 6. 训练中断恢复机制详解

### 快速参考：训练恢复配置速查表

| 场景                       | `overwrite_output_dir` | `resume_from_checkpoint` | 行为                               |
| -------------------------- | ---------------------- | ------------------------ | ---------------------------------- |
| **自动恢复（推荐）**       | `False`                | `None`                   | 自动从最新 checkpoint 恢复         |
| **从指定 checkpoint 恢复** | `False`                | 指定路径                 | 从指定的 checkpoint 恢复           |
| **强制重新训练**           | `True`                 | 任意                     | 忽略所有 checkpoint，从头开始      |
| **跨实验恢复**             | `False`                | 其他实验的 checkpoint    | 将 A 实验的 checkpoint 用于 B 实验 |

### 6.1 自动 Checkpoint 检测

GO-1 训练脚本内置了智能的训练恢复机制，可以从意外中断（断电、OOM、网络故障等）自动恢复。

#### 核心代码逻辑

```python
# 源码位置: go1/internvl/train/go1_train.py:435-453
last_checkpoint = get_last_checkpoint(training_args.output_dir)

if (
    os.path.isdir(training_args.output_dir)
    and training_args.do_train
    and not training_args.overwrite_output_dir
):
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        # 输出目录非空但没有检测到 checkpoint，抛出错误
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )
    elif (
        last_checkpoint is not None and training_args.resume_from_checkpoint is None
    ):
        # 自动检测到最新 checkpoint，准备恢复训练
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. "
            "To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )
```

#### Checkpoint 目录结构

```
experiment/lxy_test_001/
├── checkpoint-1000/
│   ├── global_step1000/        # DeepSpeed 特有，保存完整训练状态
│   ├── model-00001-of-00002.safetensors  # 模型权重分片
│   ├── model-00002-of-00002.safetensors
│   ├── model.safetensors.index.json      # 权重索引
│   ├── trainer_state.json                # 训练器状态（epoch、step、lr等）
│   ├── training_args.bin                 # 训练参数
│   ├── rng_state_*.pth                   # 随机数种子（确保可复现）
│   ├── config.json                       # 模型配置
│   ├── tokenizer*                        # tokenizer 文件
│   └── zero_to_fp32.py                   # DeepSpeed 权重合并脚本
├── checkpoint-2000/
├── checkpoint-3000/
└── ...
```

---

### 6.2 三种恢复训练方式

#### 方式 1：自动恢复（推荐）

**配置**：
```python
@dataclass
class GOTrainingArguments(TrainingArguments):
    output_dir: str = field(default=f"experiment/{RUNNAME}")
    overwrite_output_dir: bool = field(default=False)  # ← 关键：设为 False
```

**启动命令**：
```bash
# 保持 RUNNAME 不变，直接重新运行训练脚本
export RUNNAME="lxy_test_001"
bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py
```

**行为**：
- 脚本自动调用 `get_last_checkpoint()` 查找最新 checkpoint
- 检测到 `experiment/lxy_test_001/checkpoint-23000/` 
- 自动从 step 23000 恢复训练
- 保留优化器状态、学习率调度器、随机数种子

**日志输出**：
```
Checkpoint detected, resuming training at experiment/lxy_test_001/checkpoint-23000.
Loading model from experiment/lxy_test_001/checkpoint-23000.
Continuing training from epoch 8.26, global_step 23000
```

---

#### 方式 2：手动指定 Checkpoint

**配置**：
```python
@dataclass
class GOTrainingArguments(TrainingArguments):
    output_dir: str = field(default=f"experiment/{RUNNAME}")
    resume_from_checkpoint: str = field(
        default="experiment/lxy_test_001/checkpoint-20000"  # ← 手动指定
    )
```

**或通过环境变量**：
```bash
export RUNNAME="lxy_test_001_continued"
export RESUME_CKPT="experiment/lxy_test_001/checkpoint-20000"

# 修改配置文件读取环境变量
# resume_from_checkpoint: str = field(default=os.environ.get("RESUME_CKPT"))

bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py
```

**适用场景**：
- 从非最新的 checkpoint 恢复（如发现最新 checkpoint 有 NaN）
- 跨实验目录恢复（将 A 实验的 checkpoint 用于 B 实验）
- 精确控制恢复点

**注意事项**：
- 确保指定的 checkpoint 目录完整（包含 `trainer_state.json`）
- 若跨数据集恢复，可能导致数据采样顺序不一致

---

#### 方式 3：从零开始（覆盖现有输出）

**配置**：
```python
@dataclass
class GOTrainingArguments(TrainingArguments):
    output_dir: str = field(default=f"experiment/{RUNNAME}")
    overwrite_output_dir: bool = field(default=True)  # ← 强制覆盖
```

**启动命令**：
```bash
export RUNNAME="lxy_test_001"  # 即使目录存在也会清空
bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py
```

**行为**：
- 忽略现有 checkpoint
- 从 step 0 开始训练
- **危险**：会覆盖所有已有训练结果

---

### 6.3 恢复训练状态完整性

#### 完整保存的状态

| 状态类型                | 保存位置                         | 恢复后效果                     |
| ----------------------- | -------------------------------- | ------------------------------ |
| **模型权重**            | `model-*.safetensors`            | 完整恢复所有参数               |
| **优化器状态**          | `global_step*/`（ZeRO 分片）     | 恢复 Adam momentum 和 variance |
| **学习率调度器**        | `trainer_state.json`             | 恢复当前 epoch 和学习率        |
| **随机数种子**          | `rng_state_*.pth`                | 保证数据采样顺序一致           |
| **梯度累积状态**        | `trainer_state.json`             | 恢复累积步数                   |
| **训练步数/Epoch**      | `trainer_state.json`             | 从正确的 global_step 继续      |
| **日志历史**            | `trainer_state.json.log_history` | 继续记录 loss/lr 曲线          |
| **DeepSpeed ZeRO 状态** | `global_step*/zero_pp_rank_*`    | 恢复分片策略和通信状态         |

#### 不完整的 Checkpoint 检测

如果 checkpoint 目录损坏或不完整，脚本会报错：

```bash
RuntimeError: Error(s) in loading state_dict for GO1Model:
    Missing key(s) in state_dict: "action_model.layers.0.self_attn.wqkv.weight", ...
    
# 解决方案：删除损坏的 checkpoint，从上一个完整的恢复
rm -rf experiment/lxy_test_001/checkpoint-23000
# 系统会自动回退到 checkpoint-22000
```

---

### 6.4 跨机器/跨环境恢复

#### 场景 1：单机 → 多机迁移

**原训练**：
```bash
# 单机 8 卡
export WORLD_SIZE=1
export NPROC_PER_NODE=8
```

**迁移到双机**：
```bash
# 主节点
export WORLD_SIZE=2
export RANK=0
export NPROC_PER_NODE=8
export MASTER_ADDR=192.168.1.100

# 从节点
export WORLD_SIZE=2
export RANK=1
export NPROC_PER_NODE=8
export MASTER_ADDR=192.168.1.100

# 都执行
bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py
```

**注意事项**：
- 需要调整 `per_device_train_batch_size`（如从 32 降到 16）
- 全局 batch size = `per_device_train_batch_size` × `gradient_accumulation_steps` × `WORLD_SIZE` × `NPROC_PER_NODE`
- DeepSpeed 会自动重新分片模型状态

#### 场景 2：ZeRO Stage 切换

**从 Stage 1 切换到 Stage 2**：

```python
# 修改配置
deepspeed: str = field(default="go1/zero_stage2_config.json")  # 原为 stage1
```

```bash
# 启动时 DeepSpeed 会自动转换分片格式
bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py
```

**日志输出**：
```
[DeepSpeed] Detected checkpoint from ZeRO stage 1, converting to stage 2...
[DeepSpeed] Conversion completed, resuming training.
```

---

### 6.5 常见中断场景处理

#### 场景 1：显存不足导致 OOM

**问题定位**：
```bash
# 查看日志
tail -f experiment/lxy_test_001/log/training_log_*.txt

# 发现
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 23.99 GiB total capacity; ...)
```

**解决方案**：
```python
# 修改配置
@dataclass
class GOTrainingArguments(TrainingArguments):
    per_device_train_batch_size: int = field(default=16)  # 从 32 降低
    gradient_accumulation_steps: int = field(default=2)   # 从 1 增加
    # 全局 batch size 保持不变: 32×1 = 16×2
```

**恢复训练**：
```bash
# 自动从最后一个成功的 checkpoint 恢复
bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py
```

---

#### 场景 2：网络中断（多机训练）

**症状**：
```
[E ProcessGroupNCCL.cpp:828] Watchdog caught collective operation timeout: WorkNCCL(...)
RuntimeError: NCCL communicator was aborted on rank 1.
```

**恢复步骤**：

1. **检查网络连通性**：
```bash
# 在所有节点上执行
ping ${MASTER_ADDR}
telnet ${MASTER_ADDR} ${MASTER_PORT}
```

2. **清理僵尸进程**：
```bash
# 在所有节点上执行
pkill -9 python
pkill -9 torchrun
```

3. **增加 NCCL 超时**：
```bash
# 在启动脚本中添加
export NCCL_TIMEOUT=3600  # 1小时
export NCCL_DEBUG=INFO    # 开启调试日志
```

4. **重新启动训练**（自动从 checkpoint 恢复）：
```bash
bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py
```

---

#### 场景 3：手动中断（Ctrl+C）

**安全中断流程**：

1. 按 `Ctrl+C` 一次（发送 SIGINT 信号）
2. 等待训练脚本保存当前 checkpoint（可能需要 1-2 分钟）
3. 看到 `Saving model checkpoint to experiment/.../checkpoint-XXXXX` 后再退出

**强制中断风险**：
```bash
# 不推荐：直接 kill -9
pkill -9 python  # ← 可能导致 checkpoint 损坏

# 推荐：优雅关闭
pkill -15 python  # 发送 SIGTERM，等待保存
```

**恢复训练**：
```bash
# 如果最后一个 checkpoint 完整，直接恢复
bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py

# 如果损坏，删除损坏的 checkpoint
rm -rf experiment/lxy_test_001/checkpoint-23000
# 自动回退到 checkpoint-22000
```

---

### 6.6 Checkpoint 管理最佳实践

#### 自动清理策略

```python
@dataclass
class GOTrainingArguments(TrainingArguments):
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=10)  # ← 仅保留最近 10 个 checkpoint
```

**行为**：
- 每 1000 steps 保存一次
- 当 checkpoint 数量超过 10 时，自动删除最旧的
- 节省磁盘空间（每个 checkpoint 约 20-30GB）

#### 关键 Checkpoint 手动备份

```bash
# 备份重要的 checkpoint
cp -r experiment/lxy_test_001/checkpoint-20000 \
      backup/lxy_test_001_checkpoint-20000_best

# 恢复时指定备份路径
resume_from_checkpoint: str = field(
    default="backup/lxy_test_001_checkpoint-20000_best"
)
```

#### 合并 DeepSpeed 分片权重

```bash
# 使用 DeepSpeed 提供的工具合并 ZeRO 分片
cd experiment/lxy_test_001/checkpoint-23000
python zero_to_fp32.py . pytorch_model.bin

# 或者使用官方脚本
python go1/zero_to_fp32.py \
  --checkpoint-dir experiment/lxy_test_001/checkpoint-23000 \
  --output-file experiment/lxy_test_001/merged_model.bin
```

---

### 6.7 验证恢复训练正确性

#### 检查 1：训练步数连续性

```bash
# 查看日志，确认步数从正确位置继续
grep "global_step" experiment/lxy_test_001/log/training_log_*.txt

# 预期输出（中断前）：
# global_step 22990: loss=0.234
# global_step 23000: loss=0.231

# 预期输出（恢复后）：
# Continuing training from global_step 23000
# global_step 23010: loss=0.229
```

#### 检查 2：学习率调度一致性

```python
# 查看 trainer_state.json 中的学习率
import json
with open("experiment/lxy_test_001/checkpoint-23000/trainer_state.json") as f:
    state = json.load(f)
    print(f"Last LR: {state['log_history'][-1]['learning_rate']}")
```

#### 检查 3：Loss 曲线平滑性

```bash
# 使用 TensorBoard 查看
tensorboard --logdir experiment/lxy_test_001/runs

# 在浏览器中打开 http://localhost:6006
# 检查 loss 曲线在恢复点是否有突变
```

---

### 6.8 常见问题排查

| 问题                     | 症状                                  | 解决方案                            |
| ------------------------ | ------------------------------------- | ----------------------------------- |
| **找不到 checkpoint**    | `last_checkpoint is None`             | 检查 `output_dir` 路径是否正确      |
| **权重加载失败**         | `Missing key(s) in state_dict`        | 删除损坏的 checkpoint，从上一个恢复 |
| **学习率突变**           | 恢复后 lr 从 warmup 重新开始          | 检查 `trainer_state.json` 是否存在  |
| **数据采样顺序不一致**   | 恢复后看到重复的样本                  | 确保 `rng_state_*.pth` 正确加载     |
| **DeepSpeed 分片不匹配** | `ZeRO stage mismatch`                 | 保持 `deepspeed_config` 文件一致    |
| **跨机器路径不一致**     | `FileNotFoundError: checkpoint-23000` | 使用共享存储（NFS/CEPH）或手动同步  |

---

### 6.9 高级恢复技巧

#### 技巧 1：从 Checkpoint 提取特定模块

```python
# 仅加载 Action Expert，忽略 ViT 和 LLM
from transformers import AutoModel

checkpoint_path = "experiment/lxy_test_001/checkpoint-23000"
model = GO1Model.from_pretrained(
    checkpoint_path,
    ignore_mismatched_sizes=True  # 允许部分加载
)

# 仅提取 Action Expert 权重
action_expert_state = {
    k: v for k, v in model.state_dict().items() 
    if k.startswith("action_model.")
}
torch.save(action_expert_state, "action_expert_only.pth")
```

#### 技巧 2：跨配置迁移（维度变化）

```bash
# 场景：从 7DOF 单臂模型恢复训练 14DOF 双臂模型
export RUNNAME="dual_arm_from_single"

# 配置中设置
resume_from_checkpoint: str = field(default="experiment/single_arm/checkpoint-20000")
ignore_mismatched_sizes: bool = field(default=True)  # 允许维度不匹配

# 启动训练（适配器层会重新初始化）
bash go1/shell/train.sh go1/configs/go1_dual_arm.py
```

---

### 6.10 自动化恢复脚本示例

```bash
#!/bin/bash
# auto_resume_train.sh - 自动恢复训练脚本

set -e

RUNNAME=$1
CONFIG=$2

if [ -z "$RUNNAME" ] || [ -z "$CONFIG" ]; then
    echo "Usage: $0 <RUNNAME> <CONFIG>"
    exit 1
fi

OUTPUT_DIR="experiment/${RUNNAME}"
LATEST_CKPT="${OUTPUT_DIR}/$(ls -t ${OUTPUT_DIR} | grep checkpoint | head -1)"

if [ -d "$LATEST_CKPT" ]; then
    echo "Found checkpoint: ${LATEST_CKPT}"
    echo "Resuming training from step $(cat ${LATEST_CKPT}/trainer_state.json | jq .global_step)"
    
    # 验证 checkpoint 完整性
    if [ ! -f "${LATEST_CKPT}/trainer_state.json" ]; then
        echo "Warning: Incomplete checkpoint detected, falling back to previous one"
        rm -rf ${LATEST_CKPT}
        LATEST_CKPT="${OUTPUT_DIR}/$(ls -t ${OUTPUT_DIR} | grep checkpoint | head -1)"
    fi
else
    echo "No checkpoint found, starting from scratch"
fi

export RUNNAME=${RUNNAME}
bash go1/shell/train.sh ${CONFIG}
```

**使用方法**：
```bash
chmod +x auto_resume_train.sh
./auto_resume_train.sh lxy_test_001 go1/configs/go1_air_sft_lxy.py
```

---

## 7. 配置参数覆盖方法详解

### 7.1 当前系统限制说明

⚠️ **重要提示**：GO-1 训练脚本目前**不支持标准的命令行参数覆盖**（与标准 HuggingFace Trainer 不同）。

#### 当前实现方式

```python
# 源码位置: go1/internvl/train/go1_train.py:537-549
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", help="config path", required=True)  # ← 仅支持配置文件路径
    args = parser.parse_args()

    cfg, dataset_args, model_args, training_args, space_args = get_config_args(args.cfg_path)
    # ↑ 直接实例化 dataclass，无法从命令行覆盖
```

**对比标准 HuggingFace 训练脚本**：

| 特性                    | GO-1 当前实现  | 标准 HuggingFace Trainer                        |
| ----------------------- | -------------- | ----------------------------------------------- |
| 使用 `HfArgumentParser` | ❌ 否           | ✅ 是                                            |
| 命令行覆盖配置          | ❌ 不支持       | ✅ 支持                                          |
| 示例命令                | `--cfg xxx.py` | `--output_dir xxx --resume_from_checkpoint yyy` |

---

### 7.2 方法 1：环境变量覆盖（推荐）

GO-1 配置文件支持从环境变量读取部分参数。这是目前**最灵活的覆盖方法**。

#### 已支持的环境变量

```python
# 源码位置: go1/configs/go1_air_sft_lxy.py:13-14
RUNNAME = os.environ.get("RUNNAME")
DEBUG_MODE = get_bool_env("DEBUG_MODE")
```

#### 使用示例

```bash
# 场景 1：指定实验名称
export RUNNAME="my_experiment_001"
bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py

# 场景 2：启用调试模式
export DEBUG_MODE=true
export RUNNAME="debug_run"
bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py

# 场景 3：一行命令组合
RUNNAME="quick_test" DEBUG_MODE=false bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py
```

#### 扩展配置文件以支持更多环境变量

如果需要覆盖其他参数（如 `resume_from_checkpoint`），可以修改配置文件：

```python
# go1/configs/go1_air_sft_lxy.py
import os
from dataclasses import dataclass, field

RUNNAME = os.environ.get("RUNNAME")
DEBUG_MODE = get_bool_env("DEBUG_MODE")
RESUME_CKPT = os.environ.get("RESUME_CKPT")  # ← 新增
OVERWRITE_DIR = os.environ.get("OVERWRITE_DIR", "true") == "true"  # ← 新增

@dataclass
class GOTrainingArguments(TrainingArguments):
    output_dir: str = field(default=f"experiment/{RUNNAME}")
    resume_from_checkpoint: str = field(default=RESUME_CKPT)  # ← 使用环境变量
    overwrite_output_dir: bool = field(default=OVERWRITE_DIR)  # ← 使用环境变量
    # ... 其他配置
```

**使用扩展后的配置**：

```bash
# 从指定 checkpoint 恢复
export RUNNAME="lxy_test_002"
export RESUME_CKPT="experiment/lxy_test_001/checkpoint-23000"
export OVERWRITE_DIR="false"
bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py

# 强制覆盖现有输出
export RUNNAME="lxy_test_001"
export OVERWRITE_DIR="true"
bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py
```

---

### 7.3 方法 2：动态生成临时配置文件（高级）

创建一个 Shell 脚本，根据命令行参数动态生成配置文件。

#### 完整脚本：`go1/shell/train_with_args.sh`

```bash
#!/bin/bash
# train_with_args.sh - 支持命令行参数覆盖的训练启动脚本

set -e

# 解析命令行参数
BASE_CONFIG=""
RUNNAME=""
RESUME_CKPT=""
OVERWRITE_DIR="true"
BATCH_SIZE=""
LEARNING_RATE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            BASE_CONFIG="$2"
            shift 2
            ;;
        --runname)
            RUNNAME="$2"
            shift 2
            ;;
        --resume)
            RESUME_CKPT="$2"
            OVERWRITE_DIR="false"
            shift 2
            ;;
        --overwrite)
            OVERWRITE_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 验证必需参数
if [ -z "$BASE_CONFIG" ] || [ -z "$RUNNAME" ]; then
    echo "Usage: $0 --config <path> --runname <name> [options]"
    echo "Options:"
    echo "  --resume <checkpoint_path>     从指定 checkpoint 恢复"
    echo "  --overwrite <true|false>       是否覆盖现有输出"
    echo "  --batch-size <int>             每卡 batch size"
    echo "  --lr <float>                   学习率"
    exit 1
fi

# 创建临时配置文件
TEMP_CONFIG="experiment/${RUNNAME}_temp_config.py"
mkdir -p "experiment"

# 复制基础配置
cp "$BASE_CONFIG" "$TEMP_CONFIG"

# 使用 sed 覆盖配置值
if [ -n "$RESUME_CKPT" ]; then
    # 添加或修改 resume_from_checkpoint
    if grep -q "resume_from_checkpoint" "$TEMP_CONFIG"; then
        sed -i "s|resume_from_checkpoint:.*|resume_from_checkpoint: str = field(default=\"$RESUME_CKPT\")|" "$TEMP_CONFIG"
    else
        sed -i "/class GOTrainingArguments/a\\    resume_from_checkpoint: str = field(default=\"$RESUME_CKPT\")" "$TEMP_CONFIG"
    fi
fi

if [ -n "$BATCH_SIZE" ]; then
    sed -i "s/per_device_train_batch_size: int = field(default=[0-9]*)/per_device_train_batch_size: int = field(default=$BATCH_SIZE)/" "$TEMP_CONFIG"
fi

if [ -n "$LEARNING_RATE" ]; then
    sed -i "s/learning_rate: float = field(default=[0-9.e-]*)/learning_rate: float = field(default=$LEARNING_RATE)/" "$TEMP_CONFIG"
fi

# 覆盖 overwrite_output_dir
sed -i "s/overwrite_output_dir: bool = field(default=True)/overwrite_output_dir: bool = field(default=$OVERWRITE_DIR)/" "$TEMP_CONFIG"
sed -i "s/overwrite_output_dir: bool = field(default=False)/overwrite_output_dir: bool = field(default=$OVERWRITE_DIR)/" "$TEMP_CONFIG"

# 导出环境变量
export RUNNAME="$RUNNAME"

# 启动训练
bash go1/shell/train.sh "$TEMP_CONFIG"

# 清理临时文件（可选）
# rm "$TEMP_CONFIG"
```

#### 使用示例

```bash
# 1. 赋予执行权限
chmod +x go1/shell/train_with_args.sh

# 2. 从指定 checkpoint 恢复
./go1/shell/train_with_args.sh \
    --config go1/configs/go1_air_sft_lxy.py \
    --runname lxy_test_002 \
    --resume experiment/lxy_test_001/checkpoint-23000

# 3. 覆盖多个参数
./go1/shell/train_with_args.sh \
    --config go1/configs/go1_air_sft_lxy.py \
    --runname lxy_test_003 \
    --batch-size 16 \
    --lr 1e-5 \
    --overwrite true

# 4. 快速调试
./go1/shell/train_with_args.sh \
    --config go1/configs/go1_air_sft_lxy.py \
    --runname debug_001 \
    --batch-size 2
```

---

### 7.4 方法 3：直接修改配置文件（传统方法）

如果不需要频繁更改参数，直接修改 `.py` 配置文件是最简单的方法。

#### 修改步骤

```bash
# 1. 复制基础配置
cp go1/configs/go1_air_sft_lxy.py go1/configs/my_custom_config.py

# 2. 编辑配置文件
vim go1/configs/my_custom_config.py

# 修改示例：
# @dataclass
# class GOTrainingArguments(TrainingArguments):
#     output_dir: str = field(default=f"experiment/{RUNNAME}")
#     resume_from_checkpoint: str = field(
#         default="experiment/lxy_test_001/checkpoint-23000"  # ← 修改这里
#     )
#     overwrite_output_dir: bool = field(default=False)      # ← 修改这里

# 3. 启动训练
export RUNNAME="my_custom_run"
bash go1/shell/train.sh go1/configs/my_custom_config.py
```

#### 版本控制建议

```bash
# 将自定义配置放在 experiment 目录下（不纳入 git）
cp go1/configs/go1_air_sft_lxy.py experiment/my_custom_config.py

# 或使用 git 忽略
echo "go1/configs/*_custom.py" >> .gitignore
```

---

### 7.5 方法 4：使用配置模板（推荐用于团队协作）

创建可复用的配置模板，通过环境变量填充。

#### 配置模板示例：`go1/configs/go1_template.py`

```python
import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments
from go1.configs.go1_base_cfg import *
from go1.tools.env_parse import get_bool_env

# 从环境变量读取所有可配置参数
RUNNAME = os.environ.get("RUNNAME", "default_run")
DEBUG_MODE = get_bool_env("DEBUG_MODE", default=False)
RESUME_CKPT = os.environ.get("RESUME_CKPT", None)
OVERWRITE_DIR = os.environ.get("OVERWRITE_DIR", "true") == "true"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-5"))
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", "100.0"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "1000"))
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/dataset")

@dataclass
class DatasetArguments(BaseDatasetArguments):
    data_root_dir: list = field(default_factory=lambda: [DATA_ROOT])
    transforms: list = field(default_factory=lambda: [dict(type="Normalize")])

@dataclass
class GOModelArguments(BaseModelArguments):
    model_name_or_path: str = field(default="agibot-world/GO-1-Air")
    freeze_llm: bool = field(default=True)
    freeze_backbone: bool = field(default=True)
    freeze_mlp: bool = field(default=True)

@dataclass
class GOTrainingArguments(TrainingArguments):
    output_dir: str = field(default=f"experiment/{RUNNAME}")
    resume_from_checkpoint: Optional[str] = field(default=RESUME_CKPT)
    overwrite_output_dir: bool = field(default=OVERWRITE_DIR)
    per_device_train_batch_size: int = field(default=BATCH_SIZE)
    learning_rate: float = field(default=LEARNING_RATE)
    num_train_epochs: float = field(default=NUM_EPOCHS)
    save_steps: int = field(default=SAVE_STEPS)
    # ... 其他固定配置

@dataclass
class SpaceArguments(BaseSpaceArguments):
    state_dim: int = field(default=7)
    action_dim: int = field(default=7)
    # ... 其他配置
```

#### 使用模板

```bash
# 场景 1：快速恢复训练
export RUNNAME="lxy_test_002"
export RESUME_CKPT="experiment/lxy_test_001/checkpoint-23000"
export OVERWRITE_DIR="false"
bash go1/shell/train.sh go1/configs/go1_template.py

# 场景 2：调整超参数
export RUNNAME="hyperp_tuning_001"
export BATCH_SIZE="8"
export LEARNING_RATE="1e-5"
export NUM_EPOCHS="200"
bash go1/shell/train.sh go1/configs/go1_template.py

# 场景 3：使用不同数据集
export RUNNAME="new_dataset_001"
export DATA_ROOT="/path/to/new/dataset"
export OVERWRITE_DIR="true"
bash go1/shell/train.sh go1/configs/go1_template.py
```

---

### 7.6 各方法对比与选择建议

| 方法                 | 灵活性 | 易用性 | 适用场景             | 推荐度 |
| -------------------- | ------ | ------ | -------------------- | ------ |
| **环境变量覆盖**     | ⭐⭐⭐    | ⭐⭐⭐⭐   | 快速更改少量参数     | ⭐⭐⭐⭐⭐  |
| **动态生成配置**     | ⭐⭐⭐⭐   | ⭐⭐⭐    | 需要频繁更改多个参数 | ⭐⭐⭐⭐   |
| **直接修改配置文件** | ⭐⭐     | ⭐⭐⭐⭐⭐  | 一次性配置，长期使用 | ⭐⭐⭐    |
| **配置模板**         | ⭐⭐⭐⭐⭐  | ⭐⭐⭐⭐   | 团队协作，标准化配置 | ⭐⭐⭐⭐⭐  |

#### 推荐实践

1. **日常训练**：使用**配置模板**（方法 4）+ 环境变量
   ```bash
   export RUNNAME="experiment_$(date +%Y%m%d_%H%M)"
   export RESUME_CKPT="experiment/prev/checkpoint-20000"
   bash go1/shell/train.sh go1/configs/go1_template.py
   ```

2. **快速实验**：使用**环境变量覆盖**（方法 1）
   ```bash
   RUNNAME="quick_test" DEBUG_MODE=true bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py
   ```

3. **生产环境**：使用**专用配置文件**（方法 3）+ 版本控制
   ```bash
   git add go1/configs/production_config.py
   git commit -m "Add production training config"
   ```

4. **自动化脚本**：使用**动态生成配置**（方法 2）+ CI/CD
   ```bash
   ./train_with_args.sh --config base.py --runname ci_build_$BUILD_ID --batch-size 32
   ```

---

### 7.7 未来改进建议

如果需要完全兼容 HuggingFace 标准，可以修改训练脚本集成 `HfArgumentParser`：

```python
# 改进后的 go1_train.py (示例)
from transformers import HfArgumentParser

if __name__ == "__main__":
    parser = HfArgumentParser((DatasetArguments, GOModelArguments, 
                                GOTrainingArguments, SpaceArguments))
    
    # 支持三种输入方式：配置文件、JSON、命令行
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 从 JSON 文件读取
        dataset_args, model_args, training_args, space_args = parser.parse_json_file(sys.argv[1])
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".py"):
        # 保持兼容：从 .py 配置文件读取
        cfg, dataset_args, model_args, training_args, space_args = get_config_args(sys.argv[1])
    else:
        # 从命令行参数读取（支持覆盖）
        dataset_args, model_args, training_args, space_args = parser.parse_args_into_dataclasses()
```

**使用改进后的版本**：
```bash
# 命令行覆盖
python go1/internvl/train/go1_train.py \
    --output_dir experiment/test \
    --resume_from_checkpoint experiment/prev/checkpoint-20000 \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-5
```

---

### 7.8 常见配置覆盖场景速查

#### 场景 1：从 checkpoint 恢复（不覆盖配置文件）

```bash
# 方法 A：环境变量（需先扩展配置文件）
export RESUME_CKPT="experiment/lxy_test_001/checkpoint-23000"
export OVERWRITE_DIR="false"
bash go1/shell/train.sh go1/configs/go1_template.py

# 方法 B：动态脚本
./go1/shell/train_with_args.sh \
    --config go1/configs/go1_air_sft_lxy.py \
    --runname lxy_test_002 \
    --resume experiment/lxy_test_001/checkpoint-23000
```

#### 场景 2：调整 batch size 应对 OOM

```bash
# 方法 A：环境变量
export BATCH_SIZE="8"  # 从 16 降低
bash go1/shell/train.sh go1/configs/go1_template.py

# 方法 B：动态脚本
./go1/shell/train_with_args.sh \
    --config go1/configs/go1_air_sft_lxy.py \
    --runname lxy_test_001 \
    --batch-size 8 \
    --resume experiment/lxy_test_001/checkpoint-22000
```

#### 场景 3：使用不同数据集

```bash
# 环境变量
export DATA_ROOT="/path/to/new/dataset"
export RUNNAME="new_dataset_$(date +%Y%m%d)"
bash go1/shell/train.sh go1/configs/go1_template.py
```

#### 场景 4：多次实验快速迭代

```bash
# 使用循环
for lr in 1e-5 2e-5 5e-5; do
    export LEARNING_RATE="$lr"
    export RUNNAME="lr_sweep_${lr}"
    bash go1/shell/train.sh go1/configs/go1_template.py
done
```

---

## 参考资料

- [DeepSpeed ZeRO 官方文档](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch DistributedDataParallel](https://pytorch.org/docs/stable/distributed.html)
- [NCCL 性能调优指南](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)
- [HuggingFace Trainer Checkpointing](https://huggingface.co/docs/transformers/main_classes/trainer#checkpointing)