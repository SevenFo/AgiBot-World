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

## 参考资料

- [DeepSpeed ZeRO 官方文档](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch DistributedDataParallel](https://pytorch.org/docs/stable/distributed.html)
- [NCCL 性能调优指南](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)