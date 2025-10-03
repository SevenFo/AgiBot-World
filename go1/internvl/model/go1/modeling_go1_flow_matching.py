"""
Flow Matching 版本的 GO1 模型实现
基于原始 modeling_go1.py，替换 Diffusion 为 Flow Matching

主要改动:
1. 移除 DDPMScheduler 和 DPMSolverMultistepScheduler
2. 使用线性插值 z_t = (1-t)x_0 + t*x_1
3. 训练目标改为速度预测 v = x_1 - x_0
4. 推理使用 Euler 积分，可配置为 1-10 步
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass

# 导入原始 GO1 模型的其他组件
from .modeling_go1 import (
    GO1Model,
    ActionModelOutputWithPast,
)
from .configuration_go1 import GO1ModelConfig


@dataclass
class FlowMatchingConfig:
    """Flow Matching 配置"""

    num_train_timesteps: int = 1000  # 训练时的时间分辨率（用于归一化）
    num_inference_steps: int = 5  # 推理步数（1 表示单步生成）
    time_sampling: str = "uniform"  # 时间采样策略: uniform, logit_normal
    sigma_min: float = 0.0  # 最小噪声水平

    # MeanFlow 相关（可选）
    enable_meanflow: bool = False  # 是否启用 MeanFlow
    data_proportion: float = 0.25  # MeanFlow 中 r=t 的比例

    # MeanFlow 自适应权重
    norm_p: float = 1.0  # 自适应权重的指数
    norm_eps: float = 0.01  # 自适应权重的 epsilon

    # Logit-normal 时间采样参数
    P_mean: float = -0.4  # logit_normal 的均值
    P_std: float = 1.0  # logit_normal 的标准差

    def to_dict(self):
        return {
            "num_train_timesteps": self.num_train_timesteps,
            "num_inference_steps": self.num_inference_steps,
            "time_sampling": self.time_sampling,
            "sigma_min": self.sigma_min,
            "enable_meanflow": self.enable_meanflow,
            "data_proportion": self.data_proportion,
            "norm_p": self.norm_p,
            "norm_eps": self.norm_eps,
            "P_mean": self.P_mean,
            "P_std": self.P_std,
        }


class GO1ModelFlowMatching(GO1Model):
    """
    使用 Flow Matching 的 GO1 模型

    核心区别:
    - 训练: z_t = (1-t)x_0 + t*x_1, 预测速度 v = x_1 - x_0
    - 推理: Euler 积分 z_{t-dt} = z_t - dt * v_θ(z_t, t)
    """

    def __init__(self, config: GO1ModelConfig, **kwargs):
        super().__init__(config, **kwargs)

        # 移除原始的 noise_scheduler
        if hasattr(self, "noise_scheduler"):
            delattr(self, "noise_scheduler")
        if hasattr(self, "noise_scheduler_sample"):
            delattr(self, "noise_scheduler_sample")

        # ✅ 添加 h_embedder (时间差 embedder)，用于 MeanFlow
        # 导入 TimestepEmbedder
        from .modeling_go1 import TimestepEmbedder

        self.h_embedder = TimestepEmbedder(
            config.action_config.hidden_size, dtype=self.torch_dtype
        )

        # 总是初始化新添加的 h_embedder (因为它不在父类中)
        torch.nn.init.normal_(self.h_embedder.mlp[0].weight, std=0.02)
        torch.nn.init.normal_(self.h_embedder.mlp[2].weight, std=0.02)

        # 初始化 Flow Matching 配置
        if hasattr(config, "flow_matching_config") and config.flow_matching_config:
            self.fm_config = FlowMatchingConfig(**config.flow_matching_config)
        else:
            self.fm_config = FlowMatchingConfig()

        print(
            f"[FlowMatching] Initialized with inference_steps={self.fm_config.num_inference_steps}"
        )

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        采样时间 t

        Returns:
            t: [B, 1, 1], 范围 [0, 1]
        """
        if self.fm_config.time_sampling == "uniform":
            t = torch.rand(batch_size, 1, 1, device=device)
        elif self.fm_config.time_sampling == "logit_normal":
            # Logit-normal 分布 (类似 MeanFlow)
            # t = sigmoid(N(μ, σ²))
            z = torch.randn(batch_size, 1, 1, device=device)
            # 使用配置中的参数
            mu = getattr(self.fm_config, "P_mean", -0.4)
            sigma = getattr(self.fm_config, "P_std", 1.0)
            t = torch.sigmoid(z * sigma + mu)
        else:
            raise ValueError(f"Unknown time_sampling: {self.fm_config.time_sampling}")

        # 添加小的 epsilon 避免 t=0 或 t=1 的数值问题
        eps = 1e-5
        t = torch.clamp(t, eps, 1.0 - eps)
        return t

    def sample_time_pairs(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样时间对 (t, r) 用于 MeanFlow

        Returns:
            t, r: [B, 1, 1], 满足 t >= r
        """
        t = self.sample_time(batch_size, device)
        r = self.sample_time(batch_size, device)

        # 确保 t >= r
        t, r = torch.maximum(t, r), torch.minimum(t, r)

        # 数据项: data_proportion 的样本设置 r = t (学习瞬时速度)
        if self.fm_config.enable_meanflow:
            data_size = int(batch_size * self.fm_config.data_proportion)
            zero_mask = torch.arange(batch_size, device=device) < data_size
            r = torch.where(zero_mask.view(batch_size, 1, 1), t, r)

        return t, r

    def linear_interpolation(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        线性插值: z_t = (1-t)x_0 + t*x_1

        Args:
            x_0: [B, H, D] 数据 (干净的 action)
            x_1: [B, H, D] 噪声
            t: [B, 1, 1] 时间

        Returns:
            z_t: [B, H, D] 插值结果
        """
        return (1 - t) * x_0 + t * x_1

    def compute_velocity_target(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算目标速度: v = x_1 - x_0

        这是从 x_0 (数据) 到 x_1 (噪声) 的方向

        Args:
            x_0: [B, H, D] 数据
            x_1: [B, H, D] 噪声

        Returns:
            v: [B, H, D] 速度场
        """
        return x_1 - x_0

    def compute_meanflow_loss_with_jvp(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        v_target: torch.Tensor,
        state: torch.Tensor,
        ctrl_freqs: int,
        vlm_key_values_downsample: list,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用 JVP 计算 MeanFlow 损失

        核心思想：
        u(z_t,r,t) = v(z_t,t) - (t-r) * du/dt
        其中 du/dt 通过 JVP 计算

        Args:
            z_t: [B, H, D] 插值样本
            t: [B, 1, 1] 当前时间
            r: [B, 1, 1] 目标时间
            v_target: [B, H, D] 目标速度
            state: [B, state_dim] 状态
            ctrl_freqs: 控制频率
            vlm_key_values_downsample: VLM KV cache
            attention_mask: attention mask

        Returns:
            u_pred: [B, H, D] 预测的平均速度
            action_loss: 标量损失 (加权后)
            action_loss_unweighted: 标量损失 (原始未加权)
        """
        import torch.func as func

        # 准备状态 embedding
        state_traj = self.state_adaptor(state)  # [B, 1, C]
        freq_tokens = self.freq_embedder(ctrl_freqs)  # [B, 1, C]

        # 定义模型函数：u_θ(z_t, t, r)
        # 注意：h = t - r 在内部计算，与 DiT 模型的设计一致
        def velocity_model_fn(z_input, t_input, r_input):
            """
            给定 (z_t, t, r)，返回模型预测的速度 u
            内部计算 h = t - r

            Args:
                z_input: [B, H, D] action 序列
                t_input: [B, 1, 1] 当前时间
                r_input: [B, 1, 1] 目标时间

            Returns:
                u: [B, H, D] 预测的速度
            """
            # 计算时间差 h = t - r
            h_input = t_input - r_input  # [B, 1, 1]

            # 时间 embedding (t)
            t_scaled = t_input.squeeze(-1) * self.fm_config.num_train_timesteps
            timestep_tokens = self.time_embedder(t_scaled)  # [B, 1, C]

            # 时间差 embedding (h)
            h_scaled = h_input.squeeze(-1) * self.fm_config.num_train_timesteps
            h_tokens = self.h_embedder(h_scaled)  # [B, 1, C]

            # 组合时间信息: c = t + h (类似 DiT 的做法)
            # 注意：DiT 中是 c = t + h + y，但我们没有 class label
            combined_time_tokens = timestep_tokens + h_tokens  # [B, 1, C]

            # Action embedding
            action_traj = self.action_adaptor(z_input)  # [B, H, C]

            # 拼接所有 tokens
            state_action_trajs_w_tfps = torch.cat(
                [combined_time_tokens, freq_tokens, state_traj, action_traj], dim=1
            )

            # Action expert 前向
            model_output = self.action_model(
                state_action_trajs_w_tfps,
                attention_mask,
                vlm_key_values_downsample,
            )

            # 提取 action 输出
            action_output_tokens = model_output[0][:, -self.action_chunk_size :, ...]
            u_pred = self.final_layer(action_output_tokens)  # [B, H, D]

            return u_pred

        # 准备 JVP 的切向量 (tangents)
        # tangent for z_t: v_target (速度方向)
        # tangent for t: 1.0 (时间导数 dt/dt = 1)
        # tangent for r: 0.0 (r 是常数，dr/dt = 0)
        tangent_z = v_target
        tangent_t = torch.ones_like(t, dtype=z_t.dtype)  # 保持与模型相同的dtype
        tangent_r = torch.zeros_like(r, dtype=z_t.dtype)  # 保持与模型相同的dtype

        # 计算 JVP: (u, du/dt)
        # JVP 计算: d/dλ u(z_t + λ*v_target, t + λ*1, r + λ*0) |_{λ=0}
        #         = ∂u/∂z * v_target + ∂u/∂t * 1 + ∂u/∂r * 0
        #         = ∂u/∂z * v_target + ∂u/∂t
        # 注意：∂u/∂t 包含了对内部 h = t - r 的链式求导
        u_pred, du_dt = func.jvp(
            velocity_model_fn,
            (z_t, t, r),  # primals: (z_t, t, r)
            (tangent_z, tangent_t, tangent_r),  # tangents: (v, 1, 0)
        )

        # 构造 MeanFlow 目标
        # u_target = v - (t-r) * du/dt
        time_diff = torch.clamp(t - r, min=0.0, max=1.0).to(
            dtype=z_t.dtype
        )  # [B, 1, 1] 保持dtype一致
        u_target = v_target - time_diff * du_dt

        # 停止梯度（target 不需要梯度）
        u_target = u_target.detach()

        # 计算逐样本损失（不立即 reduce）
        # [B, H, D] -> [B]
        loss_per_sample = F.mse_loss(u_pred, u_target, reduction="none")
        loss_per_sample = loss_per_sample.sum(dim=(1, 2))  # sum over H, D

        # 保持与输入相同的dtype
        loss_per_sample = loss_per_sample.to(dtype=z_t.dtype)

        # 📊 保存原始未加权的 loss (用于监控)
        action_loss_unweighted = loss_per_sample.mean()

        # 应用自适应权重 (Adaptive Weighting)
        # MeanFlow paper: loss = loss / (loss + eps)^p
        norm_p = getattr(self.fm_config, "norm_p", 1.0)
        norm_eps = getattr(self.fm_config, "norm_eps", 0.01)

        if norm_p > 0:
            # 确保 norm_eps 是正确的 dtype
            norm_eps_tensor = torch.tensor(
                norm_eps, dtype=loss_per_sample.dtype, device=loss_per_sample.device
            )
            adaptive_weight = torch.pow(loss_per_sample + norm_eps_tensor, norm_p)
            loss_per_sample = loss_per_sample / adaptive_weight.detach()

        # 最终损失：batch 平均 (加权后)
        action_loss = loss_per_sample.mean()

        return u_pred, action_loss, action_loss_unweighted

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        state: torch.Tensor = None,
        ctrl_freqs: int = None,
        action_gts: torch.Tensor = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, ActionModelOutputWithPast]:
        """
        前向传播 (训练模式使用 Flow Matching)
        """
        # 处理 VLM 部分 (与原始实现相同)
        vlm_outputs = self.common_process(
            pixel_values,
            input_ids,
            attention_mask,
            position_ids,
            image_flags,
            return_dict,
            labels=labels,
        )

        # 投影 VLM KV cache
        vlm_key_values = vlm_outputs.past_key_values
        vlm_key_values_downsample = []
        for vlm_key_value, k_proj, v_proj in zip(
            vlm_key_values, self.k_proj_layers, self.v_proj_layers
        ):
            vlm_key_values_downsample.append(
                (k_proj(vlm_key_value[0]), v_proj(vlm_key_value[1]))
            )

        B = input_ids.shape[0]

        # 处理 LAM (如果启用)
        if self.enable_lam:
            (
                latent_vlm_key_values_downsample,
                outputs_latent,
            ) = self.latent_planner(
                vlm_key_values_downsample,
                attention_mask,
            )

        action_loss = torch.tensor(0.0, dtype=state.dtype, device=state.device)
        action_logits = None

        if self.training:
            # ==================== Flow Matching 训练流程 ====================

            device = action_gts.device

            # 1. 采样时间
            r = None  # 初始化 r
            if self.fm_config.enable_meanflow:
                t, r = self.sample_time_pairs(B, device)
            else:
                t = self.sample_time(B, device)

            # 2. 采样噪声 x_1 ~ N(0, I)
            x_1 = torch.randn_like(action_gts, dtype=self.torch_dtype)

            # 3. 线性插值 z_t = (1-t)x_0 + t*x_1
            x_0 = action_gts.to(self.torch_dtype)
            t = t.to(self.torch_dtype)

            z_t = self.linear_interpolation(x_0, x_1, t)

            # 4. 计算目标速度 v_target = x_1 - x_0
            v_target = self.compute_velocity_target(x_0, x_1)

            # 5. 准备 LAM attention mask (如果启用)
            if self.enable_lam:
                vlm_key_values_downsample = latent_vlm_key_values_downsample
                attention_mask_extended = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            B,
                            self.latent_planner.latent_token_nums,
                            dtype=torch.bool,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
            else:
                attention_mask_extended = attention_mask

            # 6. 根据模式选择不同的计算路径
            if self.fm_config.enable_meanflow and r is not None:
                # ==================== MeanFlow 模式 ====================
                # 直接交给 JVP 函数处理所有 embedding 和前向传播
                (
                    action_logits,
                    action_loss,
                    action_loss_unweighted,
                ) = self.compute_meanflow_loss_with_jvp(
                    z_t=z_t,
                    t=t,
                    r=r,
                    v_target=v_target,
                    state=state,
                    ctrl_freqs=ctrl_freqs,
                    vlm_key_values_downsample=vlm_key_values_downsample,
                    attention_mask=attention_mask_extended,
                )
            else:
                # ==================== 基础 Flow Matching 模式 ====================
                # 准备模型输入 (只需要 t embedding，不需要 h)
                t_scaled = t.squeeze(-1) * self.fm_config.num_train_timesteps
                timestep_tokens = self.time_embedder(t_scaled)  # [B, 1, C]

                freq_tokens = self.freq_embedder(ctrl_freqs)  # [B, 1, C]

                state_trajs = self.state_adaptor(state)  # [B, 1, C]
                action_trajs = self.action_adaptor(z_t)  # [B, H, C]

                state_action_trajs_w_tfps = torch.cat(
                    [timestep_tokens, freq_tokens, state_trajs, action_trajs], dim=1
                )  # [B, H+3, C]

                # 前向传播
                model_output = self.action_model(
                    state_action_trajs_w_tfps,
                    attention_mask_extended,
                    vlm_key_values_downsample,
                )

                # 提取 action 输出
                state_action_output_tokens = model_output[0]
                action_output_tokens = state_action_output_tokens[
                    :, -self.action_chunk_size :, ...
                ]
                v_pred = self.final_layer(action_output_tokens)  # [B, H, action_dim]

                # 计算损失
                action_loss = F.mse_loss(v_pred, v_target)
                action_loss_unweighted = action_loss  # 基础 FM 没有加权
                action_logits = v_pred

        else:
            # ==================== 推理模式 ====================
            # 如果启用 LAM，需要扩展 attention_mask
            if self.enable_lam:
                vlm_key_values_downsample = latent_vlm_key_values_downsample
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            B,
                            self.latent_planner.latent_token_nums,
                            dtype=torch.bool,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )

            action_logits = self.condition_sample_flow_matching(
                state,
                vlm_key_values_downsample,
                attention_mask,
                ctrl_freqs,
            )
            action_loss_unweighted = None  # 推理时不计算 loss

        loss = action_loss

        # 构造输出
        if not return_dict:
            return (loss, action_logits)

        output = ActionModelOutputWithPast(
            loss=loss,
            action_logits=action_logits,
            action_loss=action_loss,
            action_gts=action_gts,
            action_loss_unweighted=action_loss_unweighted,  # ✅ 添加未加权 loss
        )
        return output

    def condition_sample_flow_matching(
        self,
        state: torch.Tensor,
        vlm_key_values_downsample: list,
        attention_mask: torch.Tensor,
        ctrl_freqs: int = 30,
    ) -> torch.Tensor:
        """
        Flow Matching / MeanFlow 推理采样

        - Flow Matching: Euler 积分 (多步)
        - MeanFlow: 直接跳转 (单步或多步)

        Args:
            state: [B, state_dim] 当前状态
            vlm_key_values_downsample: VLM 的 KV cache
            attention_mask: [B, seq_len] attention mask
            ctrl_freqs: 控制频率

        Returns:
            action: [B, H, action_dim] 预测的 action 序列
        """
        state_traj = self.state_adaptor(state)  # [B, 1, C]
        device, dtype = state_traj.device, state_traj.dtype
        B = state_traj.shape[0]

        # 1. 初始化为纯噪声 z_1 ~ N(0, I)
        z_t = torch.randn(
            size=(B, self.action_chunk_size, self.action_dim),
            device=device,
            dtype=dtype,
        )

        # 2. 设置采样步数
        num_steps = self.fm_config.num_inference_steps

        if self.fm_config.enable_meanflow:
            # ==================== MeanFlow 采样 ====================
            # 使用平均速度 u_θ(z_t, t, h) 进行直接跳转
            # 默认: t_steps = [1.0, 0.0] (单步)
            # 或多步: t_steps = [1.0, 0.5, 0.0] (2步)

            dt = 1.0 / num_steps
            for i in range(num_steps):
                t_current = 1.0 - i * dt  # t: 1.0 → 0.0
                r_current = 1.0 - (i + 1) * dt  # r: 0.0
                h_current = t_current - r_current  # h = dt

                # 准备时间 embedding (t)
                t_tensor = torch.full(
                    (B, 1),
                    t_current * self.fm_config.num_train_timesteps,
                    device=device,
                    dtype=dtype,
                )
                timestep_tokens = self.time_embedder(t_tensor)  # [B, 1, C]

                # 准备时间差 embedding (h)
                h_tensor = torch.full(
                    (B, 1),
                    h_current * self.fm_config.num_train_timesteps,
                    device=device,
                    dtype=dtype,
                )
                h_tokens = self.h_embedder(h_tensor)  # [B, 1, C]

                # 组合: c = t + h (MeanFlow 的关键)
                combined_time_tokens = timestep_tokens + h_tokens  # [B, 1, C]

                freq_tokens = self.freq_embedder(ctrl_freqs)  # [B, 1, C]
                action_traj = self.action_adaptor(z_t)  # [B, H, C]

                state_action_trajs_w_tfps = torch.cat(
                    [combined_time_tokens, freq_tokens, state_traj, action_traj], dim=1
                )

                # 模型预测平均速度 u
                model_output = self.action_model(
                    state_action_trajs_w_tfps,
                    attention_mask,
                    vlm_key_values_downsample,
                )

                action_output_tokens = model_output[0][
                    :, -self.action_chunk_size :, ...
                ]
                u_pred = self.final_layer(action_output_tokens)  # [B, H, action_dim]

                # MeanFlow 步进: z_r = z_t - (t-r) * u
                z_t = z_t - h_current * u_pred
                z_t = z_t.to(dtype)

        else:
            # ==================== 基础 Flow Matching 采样 ====================
            # 使用瞬时速度 v_θ(z_t, t) 进行 Euler 积分

            dt = 1.0 / num_steps
            for i in range(num_steps):
                t_current = 1.0 - i * dt  # 1.0, 0.9, 0.8, ..., 0.1

                # 准备时间 embedding (只有 t，没有 h)
                t_tensor = torch.full(
                    (B, 1),
                    t_current * self.fm_config.num_train_timesteps,
                    device=device,
                    dtype=dtype,
                )
                timestep_tokens = self.time_embedder(t_tensor)  # [B, 1, C]
                freq_tokens = self.freq_embedder(ctrl_freqs)  # [B, 1, C]

                # 准备 state & action embedding
                action_traj = self.action_adaptor(z_t)  # [B, H, C]
                state_action_trajs_w_tfps = torch.cat(
                    [timestep_tokens, freq_tokens, state_traj, action_traj], dim=1
                )

                # 模型预测瞬时速度 v
                model_output = self.action_model(
                    state_action_trajs_w_tfps,
                    attention_mask,
                    vlm_key_values_downsample,
                )

                action_output_tokens = model_output[0][
                    :, -self.action_chunk_size :, ...
                ]
                v_pred = self.final_layer(action_output_tokens)  # [B, H, action_dim]

                # Euler 步进: z_{t-dt} = z_t - dt * v_pred
                z_t = z_t - dt * v_pred
                z_t = z_t.to(dtype)

        return z_t  # 返回 z_0 (干净的 action)

    def set_inference_steps(self, num_steps: int):
        """
        设置推理步数

        Args:
            num_steps: 推理步数 (1 表示单步生成)
        """
        self.fm_config.num_inference_steps = num_steps
        print(f"[FlowMatching] Set inference steps to {num_steps}")


# ==================== 辅助函数 ====================


def convert_diffusion_to_flow_matching(
    diffusion_model_path: str,
    output_path: str,
    flow_matching_config: dict = None,
):
    """
    将训练好的 Diffusion 模型转换为 Flow Matching 模型

    Args:
        diffusion_model_path: 原始 Diffusion 模型路径
        output_path: 输出 Flow Matching 模型路径
        flow_matching_config: Flow Matching 配置
    """
    from transformers import AutoConfig, AutoModel

    # 加载原始模型
    print(f"Loading diffusion model from {diffusion_model_path}")
    config = AutoConfig.from_pretrained(diffusion_model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(diffusion_model_path, trust_remote_code=True)

    # 修改配置
    if flow_matching_config is None:
        flow_matching_config = FlowMatchingConfig().to_dict()

    config.flow_matching_config = flow_matching_config

    # 保存新模型
    print(f"Saving Flow Matching model to {output_path}")
    model.save_pretrained(output_path)
    config.save_pretrained(output_path)

    print("Conversion completed!")


if __name__ == "__main__":
    # 测试代码
    print("Flow Matching GO1 Model Implementation")
    print("=" * 50)

    # 示例配置
    fm_config = FlowMatchingConfig(
        num_inference_steps=1,
        enable_meanflow=False,
    )

    print("Flow Matching Config:")
    for key, value in fm_config.to_dict().items():
        print(f"  {key}: {value}")
