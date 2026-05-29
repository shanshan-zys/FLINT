"""
物理约束 Loss

在标准 cross-entropy loss 之上增加物理约束, 让LLM在训练时感知:
1. 碰撞惩罚: 行人间距离过近
2. 平滑性约束: 加速度不应剧烈变化
3. 可行走区域约束: 不应走到不可通行区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PhysicsAwareLoss(nn.Module):
    """
    物理约束loss, 作用于模型输出的coordinate token logits

    工作原理:
    1. 从 logits 通过 soft-argmax 得到可微的坐标预测
    2. 在坐标空间计算物理约束
    3. 加权求和到总loss
    """

    def __init__(
        self,
        collision_weight: float = 0.1,
        smoothness_weight: float = 0.05,
        walkability_weight: float = 0.2,
        collision_threshold: float = 10.0,  # 像素距离
        bin_size: int = 5,
        x_bins: int = 96,
        y_bins: int = 72,
    ):
        super().__init__()
        self.collision_weight = collision_weight
        self.smoothness_weight = smoothness_weight
        self.walkability_weight = walkability_weight
        self.collision_threshold = collision_threshold
        self.bin_size = bin_size
        self.x_bins = x_bins
        self.y_bins = y_bins

        # x/y bin 中心坐标 (用于soft-argmax)
        x_centers = torch.tensor(
            [(i - 0.5) * bin_size for i in range(1, x_bins + 1)],
            dtype=torch.float32,
        )
        y_centers = torch.tensor(
            [(j - 0.5) * bin_size for j in range(1, y_bins + 1)],
            dtype=torch.float32,
        )
        self.register_buffer("x_centers", x_centers)
        self.register_buffer("y_centers", y_centers)

    def soft_argmax_coords(
        self,
        x_logits: torch.Tensor,
        y_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 logits 得到可微的坐标 (soft-argmax)

        Args:
            x_logits: (B, x_bins) 或 (B, T*N, x_bins)
            y_logits: (B, y_bins) 或 (B, T*N, y_bins)

        Returns:
            x_coords, y_coords: 连续坐标值
        """
        x_probs = F.softmax(x_logits / temperature, dim=-1)
        y_probs = F.softmax(y_logits / temperature, dim=-1)
        x_coords = (x_probs * self.x_centers.unsqueeze(0)).sum(dim=-1)
        y_coords = (y_probs * self.y_centers.unsqueeze(0)).sum(dim=-1)
        return x_coords, y_coords

    def collision_loss(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        num_agents: int,
        num_steps: int,
    ) -> torch.Tensor:
        """
        碰撞惩罚: 惩罚同一时刻行人间距离过近

        Args:
            x_coords: (B, T*N) 所有坐标展平
            y_coords: (B, T*N)
        """
        B = x_coords.shape[0]
        # reshape to (B, T, N)
        x = x_coords.view(B, num_steps, num_agents)
        y = y_coords.view(B, num_steps, num_agents)

        total_loss = torch.tensor(0.0, device=x.device)
        count = 0

        for t in range(num_steps):
            xt = x[:, t, :]  # (B, N)
            yt = y[:, t, :]  # (B, N)
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = torch.sqrt(
                        (xt[:, i] - xt[:, j]) ** 2 +
                        (yt[:, i] - yt[:, j]) ** 2 +
                        1e-6
                    )
                    # 当距离 < threshold 时产生惩罚
                    penalty = F.relu(self.collision_threshold - dist)
                    total_loss = total_loss + penalty.mean()
                    count += 1

        return total_loss / max(count, 1)

    def smoothness_loss(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        num_agents: int,
        num_steps: int,
    ) -> torch.Tensor:
        """
        平滑性约束: 惩罚加速度的剧烈变化 (jerk)

        加速度 = v(t) - v(t-1)
        jerk = a(t) - a(t-1) → 应该尽可能小
        """
        B = x_coords.shape[0]
        x = x_coords.view(B, num_steps, num_agents)
        y = y_coords.view(B, num_steps, num_agents)

        if num_steps < 3:
            return torch.tensor(0.0, device=x.device)

        # 速度
        vx = x[:, 1:, :] - x[:, :-1, :]  # (B, T-1, N)
        vy = y[:, 1:, :] - y[:, :-1, :]

        # 加速度
        ax = vx[:, 1:, :] - vx[:, :-1, :]  # (B, T-2, N)
        ay = vy[:, 1:, :] - vy[:, :-1, :]

        # jerk (加速度变化)
        jerk = torch.sqrt(ax ** 2 + ay ** 2 + 1e-6)
        return jerk.mean()

    def walkability_loss(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        walkable_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        可行走区域约束: 惩罚走到不可通行区域的行为

        Args:
            walkable_mask: (B, grid_h, grid_w) 二值mask, 1=可行走
        """
        grid_h = walkable_mask.shape[1]
        grid_w = walkable_mask.shape[2]

        # 坐标转网格索引 (可微近似)
        gx = (x_coords / self.bin_size).clamp(0, grid_w - 1)
        gy = (y_coords / self.bin_size).clamp(0, grid_h - 1)

        # 双线性插值查询walkability
        gx_floor = gx.long().clamp(0, grid_w - 2)
        gy_floor = gy.long().clamp(0, grid_h - 2)
        gx_frac = gx - gx_floor.float()
        gy_frac = gy - gy_floor.float()

        B = walkable_mask.shape[0]
        walkability = torch.zeros_like(x_coords)
        for b in range(B):
            w00 = walkable_mask[b, gy_floor[b], gx_floor[b]]
            w01 = walkable_mask[b, gy_floor[b], (gx_floor[b] + 1).clamp(max=grid_w - 1)]
            w10 = walkable_mask[b, (gy_floor[b] + 1).clamp(max=grid_h - 1), gx_floor[b]]
            w11 = walkable_mask[b, (gy_floor[b] + 1).clamp(max=grid_h - 1),
                                   (gx_floor[b] + 1).clamp(max=grid_w - 1)]
            interp = (
                w00 * (1 - gx_frac[b]) * (1 - gy_frac[b]) +
                w01 * gx_frac[b] * (1 - gy_frac[b]) +
                w10 * (1 - gx_frac[b]) * gy_frac[b] +
                w11 * gx_frac[b] * gy_frac[b]
            )
            walkability[b] = interp

        # 惩罚: 1 - walkability (走到obstacle区域时惩罚大)
        return (1.0 - walkability).mean()

    def forward(
        self,
        x_logits: torch.Tensor,
        y_logits: torch.Tensor,
        num_agents: int,
        num_steps: int,
        walkable_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        计算所有物理约束loss

        Args:
            x_logits: (B, T*N, x_bins) x坐标的logits
            y_logits: (B, T*N, y_bins) y坐标的logits
            num_agents: 行人数
            num_steps: 时间步数
            walkable_mask: (B, grid_h, grid_w)

        Returns:
            {total, collision, smoothness, walkability} loss dict
        """
        x_coords, y_coords = self.soft_argmax_coords(x_logits, y_logits)

        losses = {}
        total = torch.tensor(0.0, device=x_logits.device)

        if self.collision_weight > 0 and num_agents > 1:
            col_loss = self.collision_loss(x_coords, y_coords, num_agents, num_steps)
            losses["collision"] = col_loss
            total = total + self.collision_weight * col_loss

        if self.smoothness_weight > 0 and num_steps >= 3:
            smooth_loss = self.smoothness_loss(x_coords, y_coords, num_agents, num_steps)
            losses["smoothness"] = smooth_loss
            total = total + self.smoothness_weight * smooth_loss

        if self.walkability_weight > 0 and walkable_mask is not None:
            walk_loss = self.walkability_loss(x_coords, y_coords, walkable_mask)
            losses["walkability"] = walk_loss
            total = total + self.walkability_weight * walk_loss

        losses["total_physics"] = total
        return losses
