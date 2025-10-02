# GO-1 配置参数快速覆盖指南

## 🚨 重要说明

GO-1 训练脚本**当前不支持标准的命令行参数覆盖**（如 `--resume_from_checkpoint xxx`）。

本文档提供 4 种替代方案，推荐使用**方法 1（配置模板）**或**方法 2（动态脚本）**。

---

## 方法 1：环境变量 + 配置模板（⭐ 推荐）

### 使用配置模板

```bash
# 基础用法
export RUNNAME="my_experiment"
bash go1/shell/train.sh go1/configs/go1_template.py

# 从 checkpoint 恢复
export RUNNAME="exp_continued"
export RESUME_CKPT="experiment/exp/checkpoint-20000"
export OVERWRITE_DIR="false"
bash go1/shell/train.sh go1/configs/go1_template.py

# 调整超参数
export RUNNAME="hyperp_test"
export BATCH_SIZE="8"
export LEARNING_RATE="1e-5"
bash go1/shell/train.sh go1/configs/go1_template.py

# 一行命令
RUNNAME="quick_test" BATCH_SIZE="4" DEBUG_MODE="true" \
  bash go1/shell/train.sh go1/configs/go1_template.py
```

### 支持的环境变量

| 变量名          | 默认值        | 说明            |
| --------------- | ------------- | --------------- |
| `RUNNAME`       | `default_run` | 实验名称        |
| `RESUME_CKPT`   | `None`        | checkpoint 路径 |
| `OVERWRITE_DIR` | `true`        | 是否覆盖输出    |
| `BATCH_SIZE`    | `16`          | 每卡 batch size |
| `LEARNING_RATE` | `2e-5`        | 学习率          |
| `NUM_EPOCHS`    | `100.0`       | 训练轮数        |
| `SAVE_STEPS`    | `1000`        | checkpoint 间隔 |
| `DATA_ROOT`     | 见配置文件    | 数据集路径      |
| `DEBUG_MODE`    | `false`       | 调试模式        |

---

## 方法 2：动态生成配置脚本（⭐ 推荐）

### 使用示例

```bash
# 从 checkpoint 恢复
./go1/shell/train_with_args.sh \
  --config go1/configs/go1_air_sft_lxy.py \
  --runname lxy_test_002 \
  --resume experiment/lxy_test_001/checkpoint-23000

# 调整多个参数
./go1/shell/train_with_args.sh \
  --config go1/configs/go1_air_sft_lxy.py \
  --runname hyperp_001 \
  --batch-size 8 \
  --lr 1e-5

# 查看帮助
./go1/shell/train_with_args.sh --help
```

### 支持的命令行参数

```
必需参数:
  --config <path>         基础配置文件路径
  --runname <name>        实验名称

可选参数:
  --resume <ckpt_path>    从 checkpoint 恢复
  --overwrite <true|false> 是否覆盖输出
  --batch-size <int>      每卡 batch size
  --lr <float>            学习率
```

---

## 方法 3：直接修改配置文件

```bash
# 1. 复制配置
cp go1/configs/go1_air_sft_lxy.py go1/configs/my_config.py

# 2. 编辑配置
vim go1/configs/my_config.py
# 修改:
#   resume_from_checkpoint: str = field(default="experiment/xxx/checkpoint-20000")
#   overwrite_output_dir: bool = field(default=False)

# 3. 启动训练
export RUNNAME="my_experiment"
bash go1/shell/train.sh go1/configs/my_config.py
```

---

## 方法 4：扩展现有配置文件支持环境变量

在你的配置文件顶部添加：

```python
import os
from go1.tools.env_parse import get_bool_env

RESUME_CKPT = os.environ.get("RESUME_CKPT", None)
OVERWRITE_DIR = os.environ.get("OVERWRITE_DIR", "true") == "true"

@dataclass
class GOTrainingArguments(TrainingArguments):
    resume_from_checkpoint: Optional[str] = field(default=RESUME_CKPT)
    overwrite_output_dir: bool = field(default=OVERWRITE_DIR)
    # ... 其他配置
```

然后使用：

```bash
export RESUME_CKPT="experiment/xxx/checkpoint-20000"
export OVERWRITE_DIR="false"
bash go1/shell/train.sh go1/configs/your_config.py
```

---

## 常见场景速查

### 场景 1：训练中断，从最新 checkpoint 恢复

```bash
# 方法 A：自动恢复（保持 RUNNAME 不变 + overwrite_output_dir=False）
export RUNNAME="lxy_test_001"  # 保持不变
# 确保配置文件中 overwrite_output_dir=False
bash go1/shell/train.sh go1/configs/go1_air_sft_lxy.py

# 方法 B：使用模板
export RUNNAME="lxy_test_001"
export OVERWRITE_DIR="false"  # 不覆盖，自动恢复
bash go1/shell/train.sh go1/configs/go1_template.py
```

### 场景 2：从指定 checkpoint 恢复

```bash
# 方法 A：环境变量
export RUNNAME="lxy_test_002"
export RESUME_CKPT="experiment/lxy_test_001/checkpoint-23000"
export OVERWRITE_DIR="false"
bash go1/shell/train.sh go1/configs/go1_template.py

# 方法 B：动态脚本
./go1/shell/train_with_args.sh \
  --config go1/configs/go1_air_sft_lxy.py \
  --runname lxy_test_002 \
  --resume experiment/lxy_test_001/checkpoint-23000
```

### 场景 3：显存不足，降低 batch size 继续训练

```bash
# 删除损坏的 checkpoint（如果有）
rm -rf experiment/lxy_test_001/checkpoint-23000

# 使用环境变量
export RUNNAME="lxy_test_001"
export BATCH_SIZE="8"  # 从 16 降低
export OVERWRITE_DIR="false"
bash go1/shell/train.sh go1/configs/go1_template.py
```

### 场景 4：多次实验快速迭代

```bash
# 学习率扫描
for lr in 1e-5 2e-5 5e-5; do
  export RUNNAME="lr_${lr}"
  export LEARNING_RATE="$lr"
  bash go1/shell/train.sh go1/configs/go1_template.py
done

# 或使用动态脚本
for lr in 1e-5 2e-5 5e-5; do
  ./go1/shell/train_with_args.sh \
    --config go1/configs/go1_air_sft_lxy.py \
    --runname lr_${lr} \
    --lr $lr
done
```

---

## 对比总结

| 方法             | 优点             | 缺点           | 推荐场景          |
| ---------------- | ---------------- | -------------- | ----------------- |
| **配置模板**     | 灵活、简洁       | 需创建模板文件 | 日常训练 ⭐⭐⭐⭐⭐    |
| **动态脚本**     | 最接近标准命令行 | 脚本复杂       | 频繁更改参数 ⭐⭐⭐⭐ |
| **修改配置文件** | 最简单           | 每次都要改文件 | 一次性配置 ⭐⭐⭐    |
| **扩展配置**     | 灵活             | 需修改每个配置 | 团队标准化 ⭐⭐⭐⭐   |

---

## 未来改进方向

如果 GO-1 集成 HuggingFace 的 `HfArgumentParser`，将支持标准命令行覆盖：

```bash
# 期望的未来用法
python go1/internvl/train/go1_train.py \
  --config go1/configs/go1_air_sft_lxy.py \
  --output_dir experiment/test \
  --resume_from_checkpoint experiment/prev/checkpoint-20000 \
  --per_device_train_batch_size 16
```

---

## 相关文档

- 完整指南：[distributed_training_guide.md - 第 7 章](../docs/distributed_training_guide.md#7-配置参数覆盖方法详解)
- 训练恢复：[distributed_training_guide.md - 第 6 章](../docs/distributed_training_guide.md#6-训练中断恢复机制详解)
