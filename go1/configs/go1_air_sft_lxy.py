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

RUNNAME = os.environ.get("RUNNAME")
DEBUG_MODE = get_bool_env("DEBUG_MODE")


@dataclass
class DatasetArguments(BaseDatasetArguments):
    dataset_type: Optional[str] = field(default="lerobot")
    data_root_dir: Optional[List[str]] = field(
        default_factory=lambda: [
            "/quick_data/lxy_dataset/lerobot_datasets/09291324",
        ],
    )
    transforms: Optional[List[str]] = field(
        default_factory=lambda: [
            dict(type="Normalize"),
            dict(
                type="SelectDim",
                key="observation.state",
                index=([0, 1, 2, 3, 4, 5, -1], ...),
            ),
            # # 将 state 从 7 维 padding 到 14 维以匹配预训练模型
            # dict(
            #     type="Padding",
            #     key="observation.state",
            #     target_shape=14,
            #     target_dim=-1,
            #     pad_value=0.0,
            # ),
            # # 将 action 从 7 维 padding 到 14 维以匹配预训练模型
            # dict(
            #     type="Padding",
            #     key="action",
            #     target_shape=14,
            #     target_dim=-1,
            #     pad_value=0.0,
            # ),
        ],
    )


@dataclass
class GOModelArguments(BaseModelArguments):
    model_name_or_path: str = field(default="agibot-world/GO-1-Air")
    freeze_llm: bool = field(default=True if not DEBUG_MODE else True)
    freeze_backbone: bool = field(default=True if not DEBUG_MODE else True)
    freeze_mlp: bool = field(default=True if not DEBUG_MODE else True)
    action_chunk_size: int = field(default=30)
    latent_planning: bool = field(default=False)


@dataclass
class GOTrainingArguments(TrainingArguments):
    output_dir: str = field(default=f"experiment/{RUNNAME}")
    overwrite_output_dir: bool = field(default=True)
    dataloader_num_workers: int = field(default=2 if not DEBUG_MODE else 0)  # 20
    bf16: bool = field(default=True)
    num_train_epochs: float = field(default=100.0)
    per_device_train_batch_size: int = field(default=2 if not DEBUG_MODE else 2)  # 16
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=1000)
    do_train: bool = field(default=True)
    deepspeed: str = field(
        default="go1/zero_stage2_config.json"
    )  # 推荐使用 stage2 平衡内存和性能

    save_strategy: str = field(default="steps")
    save_steps: int = field(default=10000)
    save_total_limit: int = field(default=100)
    logging_steps: int = field(default=10)
    report_to: str = field(default="tensorboard")


@dataclass
class SpaceArguments(BaseSpaceArguments):
    state_dim: int = field(default=7)  # 恢复为14，匹配预训练模型 + padding后的维度
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
    ctrl_freq: int = field(default=20)
    default_prompt: str = field(default="take action to complete the task")
