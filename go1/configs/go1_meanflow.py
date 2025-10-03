"""
Flow Matching / MeanFlow 训练配置
基于 go1_air_sft_lxy.py 修改

使用方法:
1. Flow Matching (基础):
   RUNNAME=flow_matching_test bash go1/shell/train.sh go1/configs/go1_flow_matching.py

2. MeanFlow (进阶):
   修改 GOModelArguments 中的 action_generation_mode="meanflow"
"""

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
RESUME = get_bool_env("RESUME", default=False)


@dataclass
class DatasetArguments(BaseDatasetArguments):
    dataset_type: Optional[str] = field(default="lerobot")
    data_root_dir: Optional[List[str]] = field(
        default_factory=lambda: [
            "/quick_data/lxy_dataset/lerobot_datasets/100",
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
        ],
    )


@dataclass
class GOModelArguments(BaseModelArguments):
    """模型配置参数"""

    # 基础模型路径
    model_name_or_path: str = field(
        default="agibot-world/GO-1-Air",
        metadata={"help": "预训练模型路径（可以是 Diffusion 模型，会自动转换）"},
    )

    # 冻结策略
    freeze_llm: bool = field(default=True if not DEBUG_MODE else True)
    freeze_backbone: bool = field(default=True if not DEBUG_MODE else True)
    freeze_mlp: bool = field(default=True if not DEBUG_MODE else True)

    # Action 相关
    action_chunk_size: int = field(default=30)
    latent_planning: bool = field(default=False)

    # ===================== OpenLoop Evaluation 配置 =====================
    openloop_eval_steps: int = field(
        default=1000 if not DEBUG_MODE else 10,
        metadata={"help": "每隔多少步进行一次 OpenLoop Evaluation（真实推理测试）"},
    )
    openloop_eval_samples: int = field(
        default=100 if not DEBUG_MODE else 10,
        metadata={"help": "每次 OpenLoop Evaluation 使用多少个样本"},
    )

    # ===================== Action Generation Mode =====================
    # 选项: "diffusion", "flow_matching", "meanflow"
    action_generation_mode: str = field(
        default="flow_matching",  # 🔥 修改这里切换模式
        metadata={
            "help": "Action generation mode: diffusion, flow_matching, or meanflow"
        },
    )

    # ===================== Diffusion 配置 (仅当 mode=diffusion 时使用) =====================
    num_train_timesteps: int = field(
        default=100, metadata={"help": "Diffusion 训练时的总步数"}
    )
    num_inference_timesteps: int = field(
        default=10, metadata={"help": "Diffusion 推理时的采样步数"}
    )
    prediction_type: str = field(
        default="sample",
        metadata={"help": "Diffusion prediction type: epsilon or sample"},
    )
    beta_schedule: str = field(
        default="squaredcos_cap_v2", metadata={"help": "Diffusion noise schedule"}
    )
    clip_sample: bool = field(default=False)

    # ===================== Flow Matching 配置 (mode=flow_matching/meanflow) =====================
    fm_num_train_timesteps: int = field(
        default=1000,
        metadata={"help": "Flow Matching 训练时的时间分辨率（用于归一化）"},
    )
    fm_num_inference_steps: int = field(
        default=2,  # flow matching: 5, mean flow: 1
        metadata={"help": "Flow Matching 推理步数 (1=单步, 2-5=多步)"},
    )
    fm_time_sampling: str = field(
        default="logit_normal",  # 可选: "uniform" 或 "logit_normal"
        metadata={"help": "Flow Matching 时间采样策略"},
    )
    fm_sigma_min: float = field(
        default=0.0, metadata={"help": "Flow Matching 最小噪声水平"}
    )

    # ===================== MeanFlow 特有配置 (mode=meanflow) =====================
    mf_enable: bool = field(
        default=True,  # 设置为 True 启用 MeanFlow
        metadata={"help": "是否启用 MeanFlow (平均速度学习)"},
    )
    mf_data_proportion: float = field(
        default=0.25,
        metadata={
            "help": "MeanFlow 中 r=t 的样本比例 (学习瞬时速度)"
        },  # 0.25 is best in experiments
    )
    mf_guidance_omega: float = field(
        default=1.0, metadata={"help": "MeanFlow CFG 引导强度 omega"}
    )
    mf_guidance_kappa: float = field(
        default=0.5, metadata={"help": "MeanFlow CFG 混合系数 kappa"}
    )
    mf_norm_p: float = field(default=1.0, metadata={"help": "MeanFlow 自适应权重指数"})
    mf_norm_eps: float = field(
        default=0.01, metadata={"help": "MeanFlow 数值稳定性常数"}
    )
    mf_P_mean: float = field(
        default=-0.4, metadata={"help": "MeanFlow logit_normal 时间分布均值"}
    )
    mf_P_std: float = field(
        default=1.0, metadata={"help": "MeanFlow logit_normal 时间分布标准差"}
    )

    # ===================== 自动构造 flow_matching_config =====================
    def __post_init__(self):
        """自动构造 flow_matching_config 字典供模型使用"""
        # 根据 action_generation_mode 决定是否需要 flow_matching_config
        if self.action_generation_mode in ["flow_matching", "meanflow"]:
            self.flow_matching_config = {
                "num_train_timesteps": self.fm_num_train_timesteps,
                "num_inference_steps": self.fm_num_inference_steps,
                "time_sampling": self.fm_time_sampling,
                "sigma_min": self.fm_sigma_min,
                "enable_meanflow": self.mf_enable,
                "data_proportion": self.mf_data_proportion,
                "norm_p": self.mf_norm_p,
                "norm_eps": self.mf_norm_eps,
                "P_mean": self.mf_P_mean,
                "P_std": self.mf_P_std,
            }
        else:
            # Diffusion 模式不需要 flow_matching_config
            self.flow_matching_config = None


@dataclass
class GOTrainingArguments(TrainingArguments):
    output_dir: str = field(default=f"experiment/{RUNNAME}")
    overwrite_output_dir: bool = field(default=RESUME is False)
    dataloader_num_workers: int = field(default=4 if not DEBUG_MODE else 0)
    bf16: bool = field(default=True)
    num_train_epochs: float = field(default=400.0)
    per_device_train_batch_size: int = field(default=16 if not DEBUG_MODE else 2)
    gradient_accumulation_steps: int = field(default=1)

    learning_rate: float = field(default=1e-5)  # 原始 Diffusion 用 2e-5

    weight_decay: float = field(default=0.01)
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=1000)
    do_train: bool = field(default=True)
    deepspeed: str = field(default="go1/zero_stage2_config.json")

    save_strategy: str = field(default="steps")
    save_steps: int = field(default=10000)
    save_total_limit: int = field(default=20)
    logging_steps: int = field(default=10)
    report_to: str = field(default="tensorboard")


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


# ===================== 打印配置信息 =====================
def print_config_info():
    """打印配置摘要"""
    model_args = GOModelArguments()

    print("=" * 80)
    print("GO-1 Training Configuration")
    print("=" * 80)
    print(f"Model Path: {model_args.model_name_or_path}")
    print(f"Output Dir: experiment/{RUNNAME}")
    print("-" * 80)
    print(f"🔥 Action Generation Mode: {model_args.action_generation_mode.upper()}")
    print("-" * 80)

    if model_args.action_generation_mode == "diffusion":
        print("Diffusion Config:")
        print(f"  - Train Timesteps: {model_args.num_train_timesteps}")
        print(f"  - Inference Steps: {model_args.num_inference_timesteps}")
        print(f"  - Prediction Type: {model_args.prediction_type}")

    elif model_args.action_generation_mode == "flow_matching":
        print("Flow Matching Config:")
        print(f"  - Train Timesteps: {model_args.fm_num_train_timesteps}")
        print(f"  - Inference Steps: {model_args.fm_num_inference_steps} (目标单步)")
        print(f"  - Time Sampling: {model_args.fm_time_sampling}")

    elif model_args.action_generation_mode == "meanflow":
        print("MeanFlow Config:")
        print(f"  - Enable MeanFlow: {model_args.mf_enable}")
        print(f"  - Inference Steps: {model_args.fm_num_inference_steps}")
        print(f"  - Data Proportion (r=t): {model_args.mf_data_proportion}")
        print(f"  - Guidance Omega: {model_args.mf_guidance_omega}")
        print(f"  - Guidance Kappa: {model_args.mf_guidance_kappa}")

    print("=" * 80)


# 自动打印配置
if __name__ != "__main__":
    try:
        print_config_info()
    except:
        pass
