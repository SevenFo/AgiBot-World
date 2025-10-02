import os
from dataclasses import dataclass, field
from typing import List, Optional

from transformers import TrainingArguments

from go1.configs.go1_base_cfg import (
    BaseDatasetArguments,
    BaseModelArguments,
    BaseSpaceArguments,
)
from go1.tools.env_parse import get_bool_env

# ============================================================================
# 环境变量配置（支持命令行覆盖）
# ============================================================================
RUNNAME = os.environ.get("RUNNAME", "default_run")
DEBUG_MODE = get_bool_env("DEBUG_MODE", default=False)

# 训练恢复相关
RESUME_CKPT = os.environ.get("RESUME_CKPT", None)
OVERWRITE_DIR = os.environ.get("OVERWRITE_DIR", "true").lower() == "true"

# 超参数
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-5"))
NUM_EPOCHS = float(os.environ.get("NUM_EPOCHS", "100.0"))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", "1000"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "1000"))

# 数据集路径
DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/your/dataset")

# DeepSpeed 配置
DEEPSPEED_CONFIG = os.environ.get("DEEPSPEED_CONFIG", "go1/zero_stage2_config.json")

# 模型路径
MODEL_PATH = os.environ.get("MODEL_PATH", "agibot-world/GO-1-Air")


# ============================================================================
# 数据集配置
# ============================================================================
@dataclass
class DatasetArguments(BaseDatasetArguments):
    dataset_type: Optional[str] = field(default="lerobot")
    data_root_dir: Optional[List[str]] = field(
        default_factory=lambda: [DATA_ROOT],
    )
    transforms: Optional[List[str]] = field(
        default_factory=lambda: [dict(type="Normalize")]
    )


# ============================================================================
# 模型配置
# ============================================================================
@dataclass
class GOModelArguments(BaseModelArguments):
    model_name_or_path: str = field(default=MODEL_PATH)
    freeze_llm: bool = field(default=True if not DEBUG_MODE else True)
    freeze_backbone: bool = field(default=True if not DEBUG_MODE else True)
    freeze_mlp: bool = field(default=True if not DEBUG_MODE else True)
    action_chunk_size: int = field(default=30)
    latent_planning: bool = field(default=False)


# ============================================================================
# 训练配置
# ============================================================================
@dataclass
class GOTrainingArguments(TrainingArguments):
    output_dir: str = field(default=f"experiment/{RUNNAME}")

    # 训练恢复（关键配置！）
    resume_from_checkpoint: Optional[str] = field(default=RESUME_CKPT)
    overwrite_output_dir: bool = field(default=OVERWRITE_DIR)

    # 数据加载
    dataloader_num_workers: int = field(default=8 if not DEBUG_MODE else 0)

    # 训练超参数
    bf16: bool = field(default=True)
    num_train_epochs: float = field(default=NUM_EPOCHS)
    per_device_train_batch_size: int = field(default=BATCH_SIZE)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=LEARNING_RATE)
    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=WARMUP_STEPS)

    # 训练控制
    do_train: bool = field(default=True)
    deepspeed: str = field(default=DEEPSPEED_CONFIG)

    # Checkpoint 管理
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=SAVE_STEPS)
    save_total_limit: int = field(default=10)

    # 日志
    logging_steps: int = field(default=10)
    report_to: str = field(default="tensorboard")


# ============================================================================
# 空间配置（状态/动作维度）
# ============================================================================
@dataclass
class SpaceArguments(BaseSpaceArguments):
    state_dim: int = field(default=7)
    action_dim: int = field(default=7)
    space_repack: dict = field(
        default_factory=lambda: {
            "state": "observation.state",
            "action": "action",
            "cam_head_color": "observation.images.camera_side",
            "cam_hand_left_color": "observation.images.camera_wrist",
            "final_prompt": "task",
        }
    )
    ctrl_freq: int = field(default=30)


# ============================================================================
# 使用说明
# ============================================================================
"""
这是一个完全由环境变量驱动的配置模板。

基础用法：
    export RUNNAME="my_experiment"
    bash go1/shell/train.sh go1/configs/go1_template.py

从 checkpoint 恢复：
    export RUNNAME="my_experiment_continued"
    export RESUME_CKPT="experiment/my_experiment/checkpoint-20000"
    export OVERWRITE_DIR="false"
    bash go1/shell/train.sh go1/configs/go1_template.py

调整超参数：
    export RUNNAME="hyperp_tuning"
    export BATCH_SIZE="8"
    export LEARNING_RATE="1e-5"
    export NUM_EPOCHS="200"
    bash go1/shell/train.sh go1/configs/go1_template.py

使用不同数据集：
    export RUNNAME="new_dataset"
    export DATA_ROOT="/path/to/new/dataset"
    bash go1/shell/train.sh go1/configs/go1_template.py

完整示例（一行命令）：
    RUNNAME="quick_test" \\
    BATCH_SIZE="4" \\
    LEARNING_RATE="1e-5" \\
    DEBUG_MODE="true" \\
    bash go1/shell/train.sh go1/configs/go1_template.py

支持的环境变量完整列表：
    - RUNNAME: 实验名称
    - DEBUG_MODE: 调试模式（true/false）
    - RESUME_CKPT: checkpoint 路径
    - OVERWRITE_DIR: 是否覆盖输出目录（true/false）
    - BATCH_SIZE: 每卡 batch size
    - LEARNING_RATE: 学习率
    - NUM_EPOCHS: 训练轮数
    - SAVE_STEPS: checkpoint 保存间隔
    - WARMUP_STEPS: warmup 步数
    - DATA_ROOT: 数据集根目录
    - DEEPSPEED_CONFIG: DeepSpeed 配置文件
    - MODEL_PATH: 预训练模型路径
"""
