# 第 8 章：Action Expert 模型架构深度解析

## 8.1 概述与设计理念

### 8.1.1 背景与动机

Action Expert 是 GO-1 模型中的**动作预测专家模块**，负责根据视觉-语言模型（VLM）提取的高层语义特征，生成机器人的未来动作序列。它是整个 GO-1 架构中**决策生成**的核心组件。

**设计理念**：
- **条件生成**：以 VLM 的 KV cache 作为条件，生成符合视觉语义的动作轨迹
- **并行处理**：采用 Transformer Decoder 架构，并行处理 33 tokens（3 state + 30 action）
- **Diffusion 去噪网络**：作为 Diffusion Model 的核心，通过迭代去噪生成动作序列
- **高效推理**：支持 Flash Attention 2 加速，适配实时机器人控制场景
- **轻量化设计**：相比 InternLM2-2B，隐藏层维度减半（1024 vs 2048），降低计算开销

### 8.1.2 核心创新点

与标准 Transformer Decoder 的主要区别：

| 特性           | 标准 Transformer Decoder              | GO-1 Action Expert                                    |
| -------------- | ------------------------------------- | ----------------------------------------------------- |
| **输入形式**   | Token embeddings                      | State-action trajectory embeddings                    |
| **条件信息**   | Encoder outputs (固定长度)            | VLM KV cache (动态长度)                               |
| **注意力机制** | Cross-attention + Self-attention      | 统一的 **条件自注意力**（Q: action, K/V: VLM+action） |
| **因果掩码**   | 标准下三角掩码                        | **混合掩码**（state 可互看，action 因果）             |
| **位置编码**   | 统一位置编码                          | **分离位置编码**（action 和 VLM 独立编码）            |
| **模型规模**   | 通常较大（如 InternLM2-2B: 2048 dim） | 轻量化（1024 dim，参数量减半）                        |

**关键设计思想**：
```
VLM KV Cache (语义条件) + State-Action Trajectory (当前状态+历史动作)
    ↓
Conditional Self-Attention (统一注意力层)
    ↓
Action Prediction (未来动作序列)
```

### 8.1.3 模型架构概览

```
ActionExpertModel
├── Embedding Layer (替换为外部 projection)
│   └── state_action_traj: [B, horizon+state_tokens, hidden_size]
│
├── Decoder Layers × 24
│   ├── ActioinExpertAttention (标准 / Flash Attention 2)
│   │   ├── Query: 从 state_action_traj 计算
│   │   ├── Key/Value: 拼接 VLM KV cache + action KV
│   │   └── Rotary Position Embedding (RoPE)
│   ├── Feed-Forward Network (InternLM2 MLP)
│   └── RMSNorm × 2
│
└── Output Normalization (RMSNorm)
```

**文件组织**：
- `configuration_action_expert.py`: 配置类定义（ActionExpertConfig）
- `modeling_action_expert.py`: 模型实现
  - `ActioinExpertAttention`: 标准注意力机制
  - `ActioinExpertAttentionFlashAttention2`: Flash Attention 2 优化版本
  - `ActionExertDecoderLayer`: 解码器层
  - `ActionExpertModel`: 主模型类

---

## 8.2 配置详解 (ActionExpertConfig)

### 8.2.1 配置类定义

`ActionExpertConfig` 继承自 HuggingFace 的 `PretrainedConfig`，定义了模型的所有超参数。

**核心参数分类**：

#### 1. **模型结构参数**

```python
class ActionExpertConfig(PretrainedConfig):
    def __init__(
        self,
        input_hidden_size=2048,        # VLM 输出的隐藏层维度
        hidden_size=1024,              # Action Expert 隐藏层维度（比 InternLM2 减半）
        intermediate_size=2048,        # FFN 中间层维度
        num_hidden_layers=24,          # Decoder 层数
        num_attention_heads=16,        # 注意力头数
        num_key_value_heads=8,         # KV 头数（Grouped-Query Attention）
        hidden_act="silu",             # 激活函数（SwiGLU）
        ...
    ):
```

**参数说明**：

| 参数                  | 默认值 | 作用                   | 设计考量                             |
| --------------------- | ------ | ---------------------- | ------------------------------------ |
| `input_hidden_size`   | 2048   | VLM KV cache 的维度    | 需与 InternLM2-2B 对齐               |
| `hidden_size`         | 1024   | Action Expert 内部维度 | **减半设计**，降低计算量             |
| `intermediate_size`   | 2048   | FFN 中间层维度         | 通常为 `hidden_size` 的 2 倍         |
| `num_hidden_layers`   | 24     | Decoder 层数           | 与 InternLM2-2B 保持一致             |
| `num_attention_heads` | 16     | 注意力头数             | `hidden_size / head_dim = 1024 / 64` |
| `num_key_value_heads` | 8      | KV 头数                | **GQA 优化**，减少 KV cache 存储     |

**Grouped-Query Attention (GQA)**：
- 传统 MHA：Q、K、V 都有 16 个头
- GQA：Q 有 16 个头，K/V 只有 8 个头
- 优势：KV cache 减半，推理速度提升，精度损失极小

#### 2. **动作相关参数**

```python
state_dim=8,                 # 机器人状态维度（如关节角度数）
action_dim=7,                # 动作维度（如关节速度/位置）
action_chunk_size=30,        # 动作序列长度（预测未来 30 步）
state_token_num=3,           # 状态 token 数量（时间戳 + 控制频率 + 状态）
```

**场景示例**：
```
输入：
  - 机器人当前状态：[8 维关节角度]
  - 控制频率信息：[1 维]
  - 时间戳：[1 维]
  → 总共 3 个 state tokens

输出：
  - 未来 30 步动作：[30, 7] = 30 个时间步 × 7 维动作
```

#### 3. **注意力机制参数**

```python
max_position_embeddings=2048,   # 最大位置编码长度
rope_theta=10000,               # RoPE 基础频率
rope_scaling=None,              # RoPE 缩放策略（线性/动态）
attn_implementation="eager",    # 注意力实现方式（eager / flash_attention_2）
bias=True,                      # 线性层是否使用偏置
```

**RoPE Scaling 选项**：
```python
# 示例配置
rope_scaling = {
    "type": "linear",    # 或 "dynamic"
    "factor": 2.0        # 缩放因子（≥1.0）
}
```

#### 4. **训练相关参数**

```python
initializer_range=0.02,    # 权重初始化标准差
rms_norm_eps=1e-6,         # RMSNorm 的 epsilon（防止除零）
use_cache=False,           # 是否使用 KV cache（推理时启用）
```

### 8.2.2 配置验证

**RoPE Scaling 验证逻辑**：

```python
def _rope_scaling_validation(self):
    if self.rope_scaling is None:
        return
    
    # 必须包含 type 和 factor 两个字段
    if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
        raise ValueError(
            "`rope_scaling` must be a dictionary with two fields, `type` and `factor`"
        )
    
    # type 必须是 linear 或 dynamic
    rope_scaling_type = self.rope_scaling.get("type", None)
    if rope_scaling_type not in ["linear", "dynamic"]:
        raise ValueError(
            f"`rope_scaling`'s type field must be one of ['linear', 'dynamic']"
        )
    
    # factor 必须 >= 1.0
    rope_scaling_factor = self.rope_scaling.get("factor", None)
    if not isinstance(rope_scaling_factor, float) or rope_scaling_factor < 1.0:
        raise ValueError(
            f"`rope_scaling`'s factor field must be a float >= 1"
        )
```

**设计目的**：
- 支持更长的上下文（超过 2048 tokens）
- `linear`: 线性插值位置编码
- `dynamic`: 动态调整频率（NTK-aware）

### 8.2.3 派生属性

```python
self.head_dim = self.hidden_size // self.num_attention_heads  # 64
```

**计算示例**：
```
hidden_size = 1024
num_attention_heads = 16
head_dim = 1024 / 16 = 64

每个注意力头处理 64 维特征
```

---

## 8.3 位置编码与旋转嵌入 (RoPE)

### 8.3.1 RoPE 基础原理

**Rotary Position Embedding (RoPE)** 是一种相对位置编码方法，通过旋转向量来注入位置信息。

**核心思想**：
```
传统位置编码：x + PE(position)
RoPE：       rotate(x, position)
```

**数学形式**：
$$
\text{RoPE}(x, m) = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

其中：
- $m$: 位置索引
- $\theta$: 频率参数（与维度相关）
- $x_1, x_2$: 特征向量的相邻维度

**优势**：
1. **相对位置感知**：内积 $q^\top k$ 自动包含相对位置信息
2. **外推能力**：推理时可处理更长序列
3. **计算高效**：仅需旋转操作，无额外参数

### 8.3.2 RoPE 初始化

```python
def _init_rope(self):
    if self.config.rope_scaling is None:
        # 标准 RoPE
        self.rotary_emb = InternLM2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.config.rope_theta,
        )
    else:
        # 缩放 RoPE
        scaling_type = self.config.rope_scaling["type"]
        scaling_factor = self.config.rope_scaling["factor"]
        
        if scaling_type == "linear":
            self.rotary_emb = InternLM2LinearScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.config.rope_theta,
            )
        elif scaling_type == "dynamic":
            self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.config.rope_theta,
            )
```

**三种 RoPE 变体**：

| 类型          | 类名                                        | 适用场景         | 外推能力 |
| ------------- | ------------------------------------------- | ---------------- | -------- |
| **标准 RoPE** | `InternLM2RotaryEmbedding`                  | 固定长度序列     | 弱       |
| **线性缩放**  | `InternLM2LinearScalingRotaryEmbedding`     | 长序列（简单）   | 中       |
| **动态 NTK**  | `InternLM2DynamicNTKScalingRotaryEmbedding` | 长序列（高质量） | 强       |

### 8.3.3 GO-1 特有的位置编码策略

**关键函数**：`apply_rotary_pos_emb_go1`

```python
def apply_rotary_pos_emb_go1(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    GO-1 特有的 RoPE 应用方式，支持不同长度的 Q 和 K
    
    Args:
        q: (B, H, horizon, head_dim)        - Action tokens 的 Query
        k: (B, H, vlm_seq_len, head_dim)    - VLM + Action 的 Key
        cos/sin: 预计算的旋转矩阵
        position_ids: Action tokens 的位置 ID
    """
    
    # 关键创新：Q 和 K 的位置编码分离
    
    # 1. Action Query 的位置编码
    #    调整 position_ids 使其对齐到完整序列
    position_ids = position_ids + k.shape[2] - q.shape[2]
    cos_q = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin_q = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    
    # 2. VLM+Action Key 的位置编码
    #    从 0 开始的完整序列位置
    position_ids_k = torch.arange(k.shape[2]).repeat(k.shape[0], 1)
    cos_k = cos[position_ids_k].unsqueeze(unsqueeze_dim)
    sin_k = sin[position_ids_k].unsqueeze(unsqueeze_dim)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    
    return q_embed, k_embed
```

**设计动机**：

标准 Transformer 的 Q 和 K 长度相同，可以使用统一的位置编码。但在 GO-1 中：
- **Q（Query）**：仅包含 action tokens（长度 = horizon，如 30）
- **K（Key）**：包含 VLM tokens + action tokens（长度 = vlm_seq_len + horizon，如 1024 + 30）

**问题**：如何让短序列的 Q 和长序列的 K 正确交互？

**解决方案**：
1. **K 的位置编码**：从 0 开始，表示完整序列的绝对位置
2. **Q 的位置编码**：调整起始位置，使其对齐到 K 的后半部分

**图示**：
```
完整序列：[VLM tokens: 0-1023] [Action tokens: 1024-1053]
            ↓                        ↓
K:          [0, 1, ..., 1023,       1024, 1025, ..., 1053]
Q:                                  [1024, 1025, ..., 1053]
            
Q 的 position_ids 计算：
  原始: [0, 1, ..., 29]
  调整: [0, 1, ..., 29] + (1054 - 30) = [1024, 1025, ..., 1053]
```

**优势**：
- 保持相对位置关系正确
- Action tokens 可以正确关注到 VLM tokens
- 支持动态的 VLM 序列长度

---

## 8.4 注意力机制 (ActioinExpertAttention)

### 8.4.1 模块概览

`ActioinExpertAttention` 是 Action Expert 的核心计算单元，实现了**条件自注意力**机制。

**类定义**：
```python
class ActioinExpertAttention(nn.Module):
    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.hidden_size = config.hidden_size              # 1024
        self.num_heads = config.num_attention_heads        # 16
        self.head_dim = config.head_dim                    # 64
        self.num_key_value_heads = config.num_key_value_heads  # 8 (GQA)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 2
        
        # 核心线性层
        self.wqkv = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.bias,
        )
        self.wo = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.bias)
        
        self._init_rope()  # 初始化位置编码
```

**参数量计算**：
```
wqkv: 1024 → (16 + 2×8) × 64 = 1024 → 2048
      参数量 = 1024 × 2048 + 2048(bias) = 2,099,200

wo:   16 × 64 → 1024 = 1024 → 1024
      参数量 = 1024 × 1024 + 1024(bias) = 1,049,600

总计每层注意力参数：~3.15M
```

### 8.4.2 Grouped-Query Attention (GQA)

**标准 MHA vs GQA**：

```
标准 Multi-Head Attention (MHA):
  Q: [B, 16 heads, seq_len, 64] = [B, 16, L, 64]
  K: [B, 16 heads, seq_len, 64] = [B, 16, L, 64]
  V: [B, 16 heads, seq_len, 64] = [B, 16, L, 64]
  
Grouped-Query Attention (GQA):
  Q: [B, 16 heads, seq_len, 64] = [B, 16, L, 64]
  K: [B,  8 heads, seq_len, 64] = [B,  8, L, 64]  ← 减半
  V: [B,  8 heads, seq_len, 64] = [B,  8, L, 64]  ← 减半
```

**实现细节**：

```python
# 1. 统一计算 QKV（但 K/V 头数更少）
qkv_states = self.wqkv(hidden_states)  # [B, L, 2048]

# 2. 重塑为分组格式
qkv_states = rearrange(
    qkv_states,
    "b q (h gs d) -> b q h gs d",
    gs=2 + self.num_key_value_groups,  # 2 + 16/8 = 4
    d=self.head_dim,                   # 64
)
# 形状：[B, L, 8, 4, 64]
#        8 个 KV 头，每个头对应 4 个元素（2 个 Q + 1 个 K + 1 个 V）

# 3. 分离 Q, K, V
query_states = qkv_states[..., :self.num_key_value_groups, :]  # [..., :2, :]
query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
# [B, L, 8, 2, 64] → [B, L, 16, 64]

key_states = qkv_states[..., -2, :]    # [B, L, 8, 64]
value_states = qkv_states[..., -1, :]  # [B, L, 8, 64]
```

**分组关系**：
```
Q heads:  [0, 1] [2, 3] [4, 5] ... [14, 15]  (16 个头，分为 8 组)
          ↓      ↓      ↓           ↓
K/V head:  [0]    [1]    [2]   ...   [7]     (8 个头)

每 2 个 Q 头共享 1 个 K/V 头
```

**优势**：
- **推理加速**：KV cache 减半，内存带宽降低 50%
- **精度保持**：实验表明 GQA 几乎无精度损失
- **训练效率**：计算量略降，显存占用减少

### 8.4.3 前向传播流程

```python
def forward(
    self,
    hidden_states: torch.Tensor,           # [B, horizon, hidden_size]
    vlm_key_values: Tuple[torch.Tensor],   # VLM 的 KV cache
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, ...]:
```

**完整计算流程**：

#### Step 1: 计算 QKV

```python
bsz, q_len, _ = hidden_states.size()  # [B, 30, 1024]

# 计算 QKV（融合线性层）
qkv_states = self.wqkv(hidden_states)  # [B, 30, 2048]

# 分组并提取
qkv_states = rearrange(qkv_states, "b q (h gs d) -> b q h gs d", 
                       gs=4, d=64)  # [B, 30, 8, 4, 64]

query_states = rearrange(qkv_states[..., :2, :], "b q h gs d -> b q (h gs) d")
# [B, 30, 16, 64]

key_states = qkv_states[..., -2, :]    # [B, 30, 8, 64]
value_states = qkv_states[..., -1, :]  # [B, 30, 8, 64]

# 转置为 [B, H, L, D]
query_states = query_states.transpose(1, 2)  # [B, 16, 30, 64]
key_states = key_states.transpose(1, 2)      # [B, 8, 30, 64]
value_states = value_states.transpose(1, 2)  # [B, 8, 30, 64]
```

#### Step 2: 拼接 VLM KV Cache

```python
# VLM 的 KV cache
vlm_k, vlm_v = vlm_key_values  # 每个 [B, 8, vlm_len, 64]

# 拼接到 action 的 KV 前面
key_states = torch.cat([vlm_k, key_states], dim=2)
# [B, 8, vlm_len+30, 64]

value_states = torch.cat([vlm_v, value_states], dim=2)
# [B, 8, vlm_len+30, 64]

kv_seq_len = key_states.shape[-2]  # vlm_len + 30
```

#### Step 3: 应用 RoPE

```python
# 获取旋转矩阵
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

# GO-1 特有的位置编码
query_states, key_states = apply_rotary_pos_emb_go1(
    query_states, key_states, cos, sin, position_ids
)
```

#### Step 4: 重复 KV（GQA → MHA）

```python
# 将 8 个 KV 头扩展为 16 个（每个重复 2 次）
key_states = repeat_kv(key_states, self.num_key_value_groups)
# [B, 8, L, 64] → [B, 16, L, 64]

value_states = repeat_kv(value_states, self.num_key_value_groups)
# [B, 8, L, 64] → [B, 16, L, 64]
```

**repeat_kv 实现**：
```python
def repeat_kv(hidden_states, n_rep):
    """
    将 [B, num_kv_heads, L, head_dim] 扩展为 [B, num_heads, L, head_dim]
    """
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
```

#### Step 5: 计算注意力分数

```python
# Q × K^T
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
# [B, 16, 30, vlm_len+30]

# 缩放
attn_weights = attn_weights / math.sqrt(self.head_dim)  # / √64 = / 8

# 应用因果掩码
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask
    # attention_mask 中 -inf 表示不可见位置

# Softmax（上转型到 FP32 提高精度）
attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
attn_weights = attn_weights.to(query_states.dtype)
# [B, 16, 30, vlm_len+30]
```

#### Step 6: 加权求和并输出

```python
# Attention × V
attn_output = torch.matmul(attn_weights, value_states)
# [B, 16, 30, 64]

# 转置并合并头
attn_output = attn_output.transpose(1, 2).contiguous()
# [B, 30, 16, 64]

attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
# [B, 30, 1024]

# 输出投影
attn_output = self.wo(attn_output)
# [B, 30, 1024]

return attn_output, None, past_key_value
```

### 8.4.4 注意力权重可视化

**注意力矩阵形状**：
```
attn_weights: [B, 16, 33, vlm_len+33]
               ↓   ↓   ↓        ↓
            batch heads Q-len  K-len
```

**语义解释**：
- **行（Q 维度）**：每个 state/action token 关注什么
- **列（K 维度）**：VLM tokens + state tokens + action tokens

**典型模式**（假设 vlm_len=256）：
```
         VLM tokens      State    Action tokens
         [0 ... 255]    [256-258] [259 ... 288]
       ┌────────────────┬─────────┬────────────────┐
t      │ ▓▓▓░░░░░░░░░░░ │  ███    │ ░░░░░░░░░░░░░░ │  ← timestep 关注 VLM + states
f      │ ▓▓▓░░░░░░░░░░░ │  ███    │ ░░░░░░░░░░░░░░ │  ← freq 关注 VLM + states
s      │ ▓▓▓░░░░░░░░░░░ │  ███    │ ░░░░░░░░░░░░░░ │  ← state 关注 VLM + states
a_259  │ ▓▓▓░░░░░░░░░░░ │  ███    │ █░░░░░░░░░░░░░ │  ← action 1 关注 VLM+states+自己
a_260  │ ▓▓▓░░░░░░░░░░░ │  ███    │ ██░░░░░░░░░░░░ │  ← action 2 关注前面
  ...  │      ...       │   ...   │      ...       │
a_288  │ ▓▓▓░░░░░░░░░░░ │  ███    │ ██████████████ │  ← action 30 关注全部
       └────────────────┴─────────┴────────────────┘
         ↑                 ↑            ↑
      始终可见        互相可见    因果掩码（下三角）
```

**关键观察**：
1. **VLM tokens**：所有 state/action tokens 都可以关注（提供全局语义条件）
2. **State tokens**（前 3 个: timestep, freq, state）：可以互相关注（完整的条件信息）
3. **Action tokens**（30 个）：遵循因果性（后面的 action 不能看前面的，保持时间依赖建模）

**重要理解**：
- 虽然 action tokens 使用因果掩码，但这**不是**为了自回归生成
- 而是为了让模型学习 action 序列的**时间依赖关系**
- 实际生成时，30 个 action tokens 是**并行处理**的，通过 Diffusion 迭代优化整体

---

## 8.5 Flash Attention 2 优化

### 8.5.1 为什么需要 Flash Attention？

**标准注意力的性能瓶颈**：

```python
# 标准实现
Q = [B, H, L, D]
K = [B, H, L, D]
V = [B, H, L, D]

S = Q @ K^T          # [B, H, L, L] - 显存占用 O(L²)
P = softmax(S)       # [B, H, L, L] - 需要存储完整矩阵
O = P @ V            # [B, H, L, D]
```

**问题**：
1. **显存瓶颈**：注意力矩阵 $S$ 的大小为 $O(L^2)$，长序列时显存爆炸
2. **IO 效率低**：频繁在 HBM（高带宽内存）和 SRAM 间传输数据
3. **计算浪费**：掩码区域也参与计算，但结果会被丢弃

**Flash Attention 2 优化**：
- **Tiling 分块计算**：将 Q、K、V 分块加载到 SRAM，避免存储完整 attention matrix
- **Fused Kernel**：Softmax + MatMul 融合在同一个 CUDA kernel
- **IO 优化**：减少 70%+ 的显存访问
- **速度提升**：2-4× 训练速度提升，支持更长序列

### 8.5.2 ActioinExpertAttentionFlashAttention2

```python
class ActioinExpertAttentionFlashAttention2(ActioinExpertAttention):
    """
    继承标准注意力，仅重写 forward 方法以使用 Flash Attention
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        vlm_key_values: Tuple[torch.Tensor],
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,  # Flash Attention 不支持输出 attn weights
        use_cache: bool = False,
        **kwargs,
    ):
```

**关键区别**：

| 特性           | 标准注意力               | Flash Attention 2  |
| -------------- | ------------------------ | ------------------ |
| **注意力矩阵** | 显式存储 `[B,H,L,L]`     | 不存储（分块计算） |
| **掩码处理**   | 加法掩码 + Softmax       | 内置 causal mask   |
| **输出权重**   | 支持 `output_attentions` | 不支持（节省显存） |
| **内存占用**   | O(L²)                    | O(L)               |
| **速度**       | 基准                     | 2-4× 加速          |

### 8.5.3 Flash Attention 核心实现

```python
def _flash_attention_forward(
    self, 
    query_states,    # [B, L_q, H, D]
    key_states,      # [B, L_k, H, D]
    value_states,    # [B, L_k, H, D]
    attention_mask,  # [B, L_k] - padding mask
    query_length,
    dropout=0.0,
    softmax_scale=None
):
    """
    使用 Flash Attention 计算注意力
    """
    causal = self.is_causal and query_length != 1  # 训练时启用因果掩码
    
    if attention_mask is not None:
        # 有 padding tokens，需要 unpad 优化
        batch_size = query_states.shape[0]
        
        # 1. Unpad：移除 padding tokens
        query_states, key_states, value_states, indices_q, cu_seqlens, max_seqlens = \
            self._unpad_input(query_states, key_states, value_states, attention_mask, query_length)
        
        cu_seqlens_q, cu_seqlens_k = cu_seqlens
        max_seqlen_q, max_seqlen_k = max_seqlens
        
        # 2. 调用 Flash Attention（变长版本）
        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,      # Q 的累积序列长度
            cu_seqlens_k=cu_seqlens_k,      # K 的累积序列长度
            max_seqlen_q=max_seqlen_q,      # Q 的最大长度
            max_seqlen_k=max_seqlen_k,      # K 的最大长度
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )
        
        # 3. Pad：恢复原始形状
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        
    else:
        # 无 padding，直接调用
        attn_output = flash_attn_func(
            query_states,      # [B, L_q, H, D]
            key_states,        # [B, L_k, H, D]
            value_states,      # [B, L_k, H, D]
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    
    return attn_output  # [B, L_q, H, D]
```

### 8.5.4 Unpad 优化

**动机**：对于变长序列（有 padding），Flash Attention 支持跳过 padding tokens 的计算。

**Unpad 流程**：
```python
def _unpad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
    """
    移除 padding tokens，并记录每个样本的实际长度
    """
    # 获取非 padding 位置的索引
    indices_k, cu_seqlens_k, max_seqlen_k = _get_unpad_data(attention_mask)
    # indices_k: 所有非 padding 位置的全局索引
    # cu_seqlens_k: 累积序列长度 [0, len1, len1+len2, ...]
    # max_seqlen_k: 最长序列的长度
    
    batch_size, kv_seq_len, num_kv_heads, head_dim = key_layer.shape
    
    # 按索引提取非 padding tokens
    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_kv_heads, head_dim),
        indices_k
    )  # [total_valid_tokens, H, D]
    
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_kv_heads, head_dim),
        indices_k
    )
    
    # Query 同样处理（如果长度匹配）
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * query_length, num_heads, head_dim),
            indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_q = max_seqlen_k
    elif query_length == 1:
        # 解码阶段（单 token 推理）
        max_seqlen_q = 1
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
        query_layer = query_layer.squeeze(1)
    else:
        # ... 处理其他情况
    
    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_q, max_seqlen_k),
    )
```

**cumulative sequence lengths (cu_seqlens) 示例**：
```
Batch 包含 3 个样本：
  样本 1: 实际长度 5
  样本 2: 实际长度 3
  样本 3: 实际长度 7

cu_seqlens = [0, 5, 8, 15]
             ↑  ↑  ↑   ↑
             |  |  |   样本 3 结束位置
             |  |  样本 2 结束位置
             |  样本 1 结束位置
             起始位置

Flash Attention 使用 cu_seqlens 快速定位每个样本的起止位置
```

### 8.5.5 使用建议

**何时启用 Flash Attention 2**：

```python
# 配置文件中设置
config = ActionExpertConfig(
    attn_implementation="flash_attention_2",  # 或 "eager"
    ...
)
```

**性能对比**（经验值）：

| 序列长度 | 标准注意力（显存） | Flash Attention 2（显存） | 加速比 |
| -------- | ------------------ | ------------------------- | ------ |
| 512      | 2.1 GB             | 1.4 GB                    | 1.8×   |
| 1024     | 5.3 GB             | 2.1 GB                    | 2.5×   |
| 2048     | 16.2 GB            | 3.8 GB                    | 3.2×   |
| 4096     | OOM (爆显存)       | 7.1 GB                    | N/A    |

**注意事项**：
- 需要安装 `flash-attn` 库（CUDA 11.6+）
- 不支持 `output_attentions=True`
- BF16/FP16 效果最佳（FP32 加速不明显）

---

## 8.6 解码器层 (ActionExertDecoderLayer)

### 8.6.1 层结构

`ActionExertDecoderLayer` 是 Action Expert 的基本构建块，结合了注意力机制和前馈网络。

```python
class ActionExertDecoderLayer(nn.Module):
    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # 1. 注意力层（标准 or Flash Attention 2）
        self.attention = ACTIONEXPERT_ATTENTION_CLASSES[config.attn_implementation](config)
        
        # 2. 前馈网络（InternLM2 MLP）
        self.feed_forward = InternLM2MLP(config)
        
        # 3. 归一化层（RMSNorm）
        self.attention_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
```

**层拓扑**：
```
Input: [B, L, D]
    ↓
┌─────────────────────────────────────┐
│  RMSNorm                            │
│    ↓                                │
│  Conditional Self-Attention         │
│    ↓                                │
│  Residual Connection                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  RMSNorm                            │
│    ↓                                │
│  Feed-Forward Network (SwiGLU)      │
│    ↓                                │
│  Residual Connection                │
└─────────────────────────────────────┘
    ↓
Output: [B, L, D]
```

**Pre-Norm vs Post-Norm**：
- Action Expert 使用 **Pre-Norm**（RMSNorm 在子层之前）
- 优势：训练更稳定，梯度流更顺畅，支持更深网络

### 8.6.2 前馈网络 (InternLM2MLP)

虽然 FFN 复用了 InternLM2 的实现，但其配置有所调整：

```python
class InternLM2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size              # 1024
        self.intermediate_size = config.intermediate_size  # 2048
        
        # SwiGLU 架构
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()  # Swish 激活函数
    
    def forward(self, x):
        # SwiGLU: (gate(x) ⊙ up(x)) @ down
        gate = self.act_fn(self.gate_proj(x))  # [B, L, 2048]
        up = self.up_proj(x)                   # [B, L, 2048]
        return self.down_proj(gate * up)       # [B, L, 1024]
```

**SwiGLU 详解**：
$$
\text{SwiGLU}(x) = (\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}) W_{\text{down}}
$$

其中：
- $\text{SiLU}(x) = x \cdot \sigma(x)$（Swish 激活函数）
- $\odot$: 逐元素乘法（门控机制）

**优势**：
- 相比 ReLU FFN，SwiGLU 提升 1-2% 准确率
- 门控机制动态选择信息流
- InternLM2 / LLaMA 系列标配

**参数量**：
```
gate_proj: 1024 → 2048 = 2,097,152
up_proj:   1024 → 2048 = 2,097,152
down_proj: 2048 → 1024 = 2,097,152
总计：~6.29M 参数/层
```

### 8.6.3 RMSNorm

**Root Mean Square Normalization** 是 LayerNorm 的简化版本，省略了均值中心化。

```python
class InternLM2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        # 计算 RMS
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        # 应用可学习缩放
        return self.weight * hidden_states
```

**LayerNorm vs RMSNorm**：

| 操作       | LayerNorm                                    | RMSNorm                                        |
| ---------- | -------------------------------------------- | ---------------------------------------------- |
| 中心化     | $x - \mu$                                    | 无                                             |
| 归一化     | $\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$ | $\frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}}$ |
| 可学习参数 | $\gamma, \beta$                              | 仅 $\gamma$                                    |
| 计算复杂度 | 高                                           | 低 10-20%                                      |

**数学形式**：
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma
$$

**优势**：
- 计算更快（省略均值计算）
- 参数更少（无偏置 $\beta$）
- 效果相当（LLaMA、InternLM 等大模型验证）

### 8.6.4 前向传播

```python
def forward(
    self,
    hidden_states: torch.Tensor,           # [B, L, 1024]
    vlm_key_values: Tuple[torch.Tensor],   # VLM KV cache（单层）
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, ...]:
    
    # === 注意力子层 ===
    residual = hidden_states
    hidden_states = self.attention_norm(hidden_states)  # Pre-Norm
    
    hidden_states, self_attn_weights, present_key_value = self.attention(
        hidden_states=hidden_states,
        vlm_key_values=vlm_key_values,  # 关键：VLM 条件
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    
    hidden_states = residual + hidden_states  # Residual
    
    # === FFN 子层 ===
    residual = hidden_states
    hidden_states = self.ffn_norm(hidden_states)  # Pre-Norm
    hidden_states = self.feed_forward(hidden_states)
    hidden_states = residual + hidden_states  # Residual
    
    # 输出
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    
    return outputs
```

**关键设计**：
1. **VLM 条件注入**：通过 `vlm_key_values` 传递每层的 VLM KV cache
2. **Pre-Norm 架构**：归一化在子层之前，提升稳定性
3. **残差连接**：缓解梯度消失，支持 24 层深度

---

## 8.7 因果掩码机制

### 8.7.1 GO-1 的特殊掩码需求

在标准 Transformer Decoder 中，因果掩码是简单的下三角矩阵。但在 GO-1 中，需要处理三种 token：

1. **VLM tokens** ($I$)：视觉语言特征（如 1024 个 tokens）
2. **State tokens** ($q$)：机器人状态（3 个 tokens：时间戳 + 控制频率 + 状态）
3. **Action tokens** ($a$)：动作序列（30 个 tokens）

**可见性规则**：
- **VLM tokens**：所有 action/state tokens 都能看到（提供全局语义）
- **State tokens**：可以互相看到（当前状态的完整信息）
- **Action tokens**：只能看历史 action（因果性保证）

### 8.7.2 混合掩码实现

#### _make_causal_mask_go1

```python
def _make_causal_mask_go1(
    input_ids_shape: torch.Size,        # (batch_size, tgt_len)
    dtype: torch.dtype,
    device: torch.device,
    state_length: int = 3,              # State tokens 数量
    past_key_values_length: int = 4096, # VLM tokens 数量
):
    """
    创建 GO-1 特有的因果掩码
    """
    bsz, tgt_len = input_ids_shape
    
    # 1. 创建全掩码矩阵（初始化为 -inf）
    mask = torch.full(
        (tgt_len, tgt_len),
        torch.finfo(dtype).min,  # -inf
        device=device
    )
    
    # 2. State tokens 可以互相看到
    mask[:state_length, :state_length] = 0
    
    # 3. Action tokens 使用标准因果掩码
    mask[state_length:, :] = 0  # 先全部开放
    # （注：这里代码似乎有简化，实际应该是下三角掩码）
    
    # 4. 拼接 VLM 掩码（全部可见）
    if past_key_values_length > 0:
        vlm_mask = torch.zeros(
            tgt_len, past_key_values_length,
            dtype=dtype, device=device
        )
        mask = torch.cat([vlm_mask, mask], dim=-1)
    
    # 5. 扩展维度: [L, L] → [B, 1, L, L]
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )
```

**掩码矩阵示例**（tgt_len=5, state_length=3, vlm_len=1024）：

```
        VLM tokens  State    Action
        [0...1023]  [0,1,2]  [3,4]
        ─────────── ─────── ─────
q_0  [  全部可见   |  ✓✓✓ | ✓✗✗ ]  ← State token 0
q_1  [  全部可见   |  ✓✓✓ | ✗✓✗ ]  ← State token 1
q_2  [  全部可见   |  ✓✓✓ | ✗✗✓ ]  ← State token 2
a_3  [  全部可见   |  ✓✓✓ | ✓✗✗ ]  ← Action token 3
a_4  [  全部可见   |  ✓✓✓ | ✓✓✗ ]  ← Action token 4

✓ = 0（可见）
✗ = -inf（不可见）
```

**关键观察**：
- **第 1-2 列（VLM）**：全部为 0（全局可见）
- **State 区域**：3×3 全零矩阵（互相可见）
- **Action 区域**：下三角矩阵（因果掩码）

#### _concat_mask_go1

```python
def _concat_mask_go1(
    mask: torch.Tensor,  # [B, vlm_len] - VLM 的 padding mask
    dtype: torch.dtype,
    tgt_len: int         # Action tokens 长度
):
    """
    将 VLM 的 padding mask 与 action mask 合并
    """
    bsz, vlm_len = mask.size()
    src_len = vlm_len + tgt_len
    
    # 1. Action tokens 没有 padding（全部有效）
    action_padding_mask = torch.ones((bsz, tgt_len), device=mask.device)
    
    # 2. 拼接 VLM 和 Action 的 padding mask
    concated_mask = torch.cat((mask, action_padding_mask), dim=-1)
    # [B, vlm_len+tgt_len]
    
    # 3. 扩展为注意力掩码: [B, L] → [B, 1, tgt_len, src_len]
    expanded_mask = concated_mask[:, None, None, :].expand(
        bsz, 1, tgt_len, src_len
    ).to(dtype)
    
    # 4. 反转掩码（0 → -inf, 1 → 0）
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool),
        torch.finfo(dtype).min
    )
```

**Padding Mask 作用**：
- VLM 序列长度可能不一致（padding 对齐）
- Padding positions 设为 -inf，避免影响注意力计算

### 8.7.3 掩码组合

在 `ActionExpertModel._prepare_decoder_attention_mask` 中：

```python
def _prepare_decoder_attention_mask(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    combined_attention_mask = None
    
    # 1. 创建因果掩码
    if input_shape[-1] > 1:  # 训练时（序列长度 > 1）
        combined_attention_mask = _make_causal_mask_go1(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            state_length=self.state_token_num,
            past_key_values_length=past_key_values_length,
        )
    
    # 2. 合并 padding mask
    if attention_mask is not None:
        expanded_attn_mask = _concat_mask_go1(
            attention_mask,
            inputs_embeds.dtype,
            tgt_len=input_shape[-1]
        ).to(inputs_embeds.device)
        
        # 加法合并（-inf + -inf = -inf，0 + 0 = 0）
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )
    
    return combined_attention_mask
```

**合并逻辑**：
```
因果掩码（结构性约束）+ Padding 掩码（数据相关）
                ↓
         最终注意力掩码
```

---

## 8.8 主模型架构 (ActionExpertModel)

### 8.8.1 模型初始化

```python
class ActionExpertModel(ActionExpertPretrainedModel):
    def __init__(self, config: ActionExpertConfig):
        super().__init__(config)
        self.config = config
        self.rng_seeded = np.random.default_rng(seed=42)
        self.state_token_num = config.state_token_num
        
        # 检查 Flash Attention 可用性
        if not has_flash_attn:
            self.config.attn_implementation = "eager"
            print("Warning: Flash attention is not available")
        
        # 创建 24 层 Decoder
        self.layers = nn.ModuleList([
            ActionExertDecoderLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # 最终输出归一化
        self.norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        self.post_init()  # 初始化权重
```

**模型规模**：
```
每层参数量：
  Attention: ~3.15M
  FFN:       ~6.29M
  RMSNorm:   ~0.002M
  小计：     ~9.44M

总参数量：
  9.44M × 24 层 ≈ 226M
  加上最终 norm: ~226.5M 参数
```

**与 InternLM2-2B 对比**：
- InternLM2-2B: ~2B 参数
- Action Expert: ~227M 参数
- **减少约 90%**，推理速度大幅提升

### 8.8.2 输入输出接口

```python
def forward(
    self,
    # === 核心输入 ===
    state_action_traj: torch.FloatTensor,       # [B, horizon+state_tokens, hidden_size]
    vlm_key_values: Tuple[Tuple[torch.FloatTensor]],  # 24 层的 KV cache
    
    # === 可选输入 ===
    attention_mask: torch.Tensor = None,        # [B, vlm_seq_len] - VLM padding mask
    position_ids: Optional[torch.LongTensor] = None,  # [B, horizon+state_tokens]
    
    # === 控制参数 ===
    past_key_values: Optional[List[torch.FloatTensor]] = None,  # Action KV cache
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> BaseModelOutputWithPast:
```

**输入详解**：

| 参数                | 形状                                 | 说明                                                         |
| ------------------- | ------------------------------------ | ------------------------------------------------------------ |
| `state_action_traj` | `[B, L, D]`                          | 状态-动作轨迹嵌入<br>L = state_tokens(3) + action_tokens(30) |
| `vlm_key_values`    | 24 × 2 × `[B, H, vlm_len, head_dim]` | VLM 每层的 K, V cache<br>H = 8 (KV heads)                    |
| `attention_mask`    | `[B, vlm_len]`                       | VLM tokens 的 padding mask<br>1 = 有效, 0 = padding          |
| `position_ids`      | `[B, L]`                             | Action tokens 的位置 ID                                      |

**输出类型**：
```python
BaseModelOutputWithPast(
    last_hidden_state=[B, L, D],      # 最终隐藏状态
    past_key_values=Tuple,            # Action KV cache（如果 use_cache=True）
    hidden_states=Tuple,              # 所有层的隐藏状态（如果 output_hidden_states=True）
    attentions=Tuple,                 # 所有层的注意力权重（如果 output_attentions=True）
)
```

### 8.8.3 位置 ID 生成

```python
batch_size, seq_length = state_action_traj.shape[:2]
seq_length_with_past = seq_length
past_key_values_length = 0

if past_key_values is not None:
    # 增量解码时累积长度
    past_key_values_length = past_key_values[0][0].shape[2]
    seq_length_with_past += past_key_values_length

if position_ids is None:
    # 自动生成位置 ID
    device = state_action_traj.device
    position_ids = torch.arange(
        past_key_values_length,
        seq_length + past_key_values_length,
        dtype=torch.long,
        device=device
    )
    position_ids = position_ids.unsqueeze(0)  # [1, L] → broadcast
```

**示例**：
```
训练阶段（完整序列）：
  past_key_values_length = 0
  seq_length = 33（3 state + 30 action）
  position_ids = [0, 1, 2, ..., 32]

推理阶段（增量解码第 5 步）：
  past_key_values_length = 4（已生成 4 个 action）
  seq_length = 1（当前生成 1 个 action）
  position_ids = [4]  # 继续累积
```

### 8.8.4 注意力掩码处理

```python
dynamic_vlm_token_length = attention_mask.shape[-1]

if self.config.attn_implementation == "flash_attention_2":
    # Flash Attention 内置 causal mask
    attention_mask = None
else:
    # 标准注意力需要显式掩码
    if attention_mask is None:
        # 默认全部可见
        attention_mask = torch.ones(
            (batch_size, dynamic_vlm_token_length),
            dtype=torch.bool,
            device=state_action_traj.device
        )
    
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask=attention_mask,
        input_shape=(batch_size, seq_length),
        inputs_embeds=state_action_traj,
        past_key_values_length=dynamic_vlm_token_length,
    )
    # [B, 1, tgt_len, vlm_len+tgt_len]
```

---

## 8.9 前向传播流程

### 8.9.1 完整流程图

```
输入：state_action_traj [B, 33, 1024]
      vlm_key_values (24 layers × K/V)
      ↓
┌────────────────────────────────────────────────┐
│ 1. 位置 ID 生成                                │
│    position_ids = [0, 1, ..., 32]             │
└────────────────────────────────────────────────┘
      ↓
┌────────────────────────────────────────────────┐
│ 2. 注意力掩码准备                              │
│    混合因果掩码 + Padding 掩码                 │
└────────────────────────────────────────────────┘
      ↓
┌────────────────────────────────────────────────┐
│ 3. 逐层解码（24 层）                           │
│    for layer in self.layers:                  │
│      ├─ RMSNorm                               │
│      ├─ Conditional Self-Attention            │
│      │   ├─ Q: from state_action_traj        │
│      │   └─ K/V: VLM KV + action KV          │
│      ├─ Residual                              │
│      ├─ RMSNorm                               │
│      ├─ FFN (SwiGLU)                          │
│      └─ Residual                              │
└────────────────────────────────────────────────┘
      ↓
┌────────────────────────────────────────────────┐
│ 4. 最终归一化                                  │
│    hidden_states = self.norm(hidden_states)   │
└────────────────────────────────────────────────┘
      ↓
输出：last_hidden_state [B, 33, 1024]
```

### 8.9.2 核心循环

```python
hidden_states = state_action_traj  # [B, 33, 1024]

all_hidden_states = () if output_hidden_states else None
all_self_attns = () if output_attentions else None
next_decoder_cache = () if use_cache else None

for idx, decoder_layer in enumerate(self.layers):
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    
    past_key_value = past_key_values[idx] if past_key_values is not None else None
    
    if self.gradient_checkpointing and self.training:
        # 梯度检查点（节省显存）
        layer_outputs = torch.utils.checkpoint.checkpoint(
            create_custom_forward(decoder_layer),
            hidden_states,
            vlm_key_values[idx],  # 该层的 VLM KV
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
        )
    else:
        # 正常前向传播
        layer_outputs = decoder_layer(
            hidden_states=hidden_states,
            vlm_key_values=vlm_key_values[idx],  # 关键：每层对应的 VLM KV
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
    
    hidden_states = layer_outputs[0]  # [B, 33, 1024]
    
    if use_cache:
        next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
    
    if output_attentions:
        all_self_attns += (layer_outputs[1],)

# 最终归一化
hidden_states = self.norm(hidden_states)

if output_hidden_states:
    all_hidden_states += (hidden_states,)
```

### 8.9.3 梯度检查点（Gradient Checkpointing）

**动机**：深度网络训练时，需要存储所有中间激活值用于反向传播，显存占用巨大。

**原理**：
- 前向传播时**不存储**中间激活值
- 反向传播时**重新计算**需要的激活值
- 以 **计算换显存**

**使用方法**：
```python
model.gradient_checkpointing_enable()  # 启用

# 在 forward 中自动应用
if self.gradient_checkpointing and self.training:
    layer_outputs = torch.utils.checkpoint.checkpoint(...)
```

**效果**：
- 显存减少 30-50%
- 训练速度降低 10-20%（需要重算）
- 适合显存受限场景

### 8.9.4 输出封装

```python
next_cache = next_decoder_cache if use_cache else None

if not return_dict:
    # Tuple 输出（兼容模式）
    return tuple(v for v in [hidden_states, next_cache] if v is not None)

# Dict 输出（推荐）
return BaseModelOutputWithPast(
    last_hidden_state=hidden_states,     # [B, 33, 1024]
    past_key_values=next_cache,          # 24 layers × (K, V)
    hidden_states=all_hidden_states,     # 25 个张量（输入 + 24 层）
    attentions=all_self_attns,           # 24 个注意力矩阵
)
```

---

## 8.10 总结与关键创新点

### 8.10.1 核心创新

1. **条件生成架构**
   - 传统 Decoder：独立生成
   - GO-1 Action Expert：以 VLM KV cache 为条件，语义引导动作生成

2. **统一注意力机制**
   - 传统：Cross-attention（VLM → Action）+ Self-attention（Action ↔ Action）
   - GO-1：单一注意力层，Q 来自 action，K/V 拼接 VLM + action

3. **分离位置编码**
   - VLM tokens：绝对位置 `[0, vlm_len]`
   - Action tokens：相对位置，动态对齐

4. **混合因果掩码**
   - State tokens：全连接（捕获当前状态）
   - Action tokens：因果掩码（保证自回归性）
   - VLM tokens：全局可见（提供语义）

5. **轻量化设计**
   - 隐藏维度减半（1024 vs 2048）
   - Grouped-Query Attention（KV cache 减半）
   - 参数量降低 90%（227M vs 2B）

### 8.10.2 技术亮点

| 技术                       | 作用              | 效果                         |
| -------------------------- | ----------------- | ---------------------------- |
| **GQA**                    | KV 头数减半       | 推理速度 ↑30%，显存 ↓50%     |
| **Flash Attention 2**      | 分块计算，IO 优化 | 训练速度 ↑2-4×，支持更长序列 |
| **RoPE**                   | 相对位置编码      | 外推能力强，长序列性能好     |
| **RMSNorm**                | 简化归一化        | 计算 ↓15%，效果相当          |
| **SwiGLU**                 | 门控 FFN          | 准确率 ↑1-2%                 |
| **Pre-Norm**               | 归一化位置        | 训练稳定，支持深层网络       |
| **Gradient Checkpointing** | 重算换显存        | 显存 ↓40%，速度 ↓15%         |

### 8.10.3 与标准 Transformer 的对比

| 维度         | 标准 Transformer Decoder | GO-1 Action Expert            |
| ------------ | ------------------------ | ----------------------------- |
| **输入**     | Token embeddings         | State-action trajectory       |
| **条件**     | Encoder outputs（固定）  | VLM KV cache（动态）          |
| **注意力**   | Cross-attn + Self-attn   | 统一条件自注意力              |
| **掩码**     | 标准下三角               | 混合掩码（state/action 分离） |
| **位置编码** | 统一编码                 | 分离编码（VLM/action 独立）   |
| **规模**     | 通常较大                 | 轻量化（参数减少 90%）        |
| **应用场景** | 文本生成、翻译           | 机器人动作预测                |

### 8.10.4 适用场景

**优势场景**：
- 需要视觉语言引导的动作生成
- 并行生成多个 action chunk（配合 Diffusion）
- 长序列动作规划（Flash Attention 2 + RoPE）
- 多模态条件生成

**局限性**：
- 依赖 VLM 特征质量（条件生成）
- 需要与 Diffusion Model 配合使用（不能单独生成 action）
- 需要配对的 state-action 数据训练

### 8.10.5 与 Diffusion 的协同

**Action Expert 在完整 GO-1 模型中的角色**：

```
训练阶段:
  Noisy Actions [B, 30, 7] + State [B, 8] + Timestep
    ↓ Adaptors
  State-Action Trajectory [B, 33, 1024]
    ↓ Action Expert (本章内容)
  Denoised Features [B, 30, 1024]
    ↓ Final Layer
  Predicted Actions [B, 30, 7]
    ↓ MSE Loss
  与 GT Actions 对比

推理阶段 (迭代去噪):
  for t in [T, T-1, ..., 0]:
      Noisy Actions → Action Expert → Predicted Actions
      Noisy Actions = Scheduler.step(Predicted Actions, t)
  最终得到 Clean Actions
```

**关键理解**：
- Action Expert 是 **Diffusion 模型的核心网络**（去噪网络）
- 每次前向传播处理整个 30-chunk 的 action 序列
- 通过多次调用（如 10 次）逐步从噪声恢复到干净 action
- 详细的 Diffusion 机制请参见 **第 9 章：GO-1 完整模型架构**

### 8.10.6 扩展方向

1. **更高效的 Attention**：
   - 当前：Flash Attention 2
   - 改进：Flash Attention 3（支持更长序列）

2. **多尺度建模**：
   - 粗粒度：策略级动作序列
   - 细粒度：关节级轨迹

3. **强化学习集成**：
   - 当前：监督学习（模仿学习）
   - 扩展：RL fine-tuning（在线优化）

4. **跨具身迁移**：
   - 不同机器人共享 Action Expert
   - 通过 adapter 适配不同动作空间

5. **Diffusion 优化**：
   - 减少推理步数（10 → 5 steps）
   - 更快的 scheduler（DPM-Solver++）
   - 知识蒸馏（Diffusion → 直接回归）

---

## 参考资料

- **RoPE**: Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- **Flash Attention**: Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
- **GQA**: Ainslie et al. "GQA: Training Generalized Multi-Query Transformer" (2023)
- **InternLM2**: InternLM Team "InternLM2 Technical Report" (2024)
- **SwiGLU**: Shazeer "GLU Variants Improve Transformer" (2020)

---

**文档完成**！如有疑问或需要更多细节，请随时提问。🚀
