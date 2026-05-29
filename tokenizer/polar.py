"""
极坐标 Tokenizer

编码: 首帧用绝对坐标, 后续帧用 (距离d, 角度θ) 表示相对上一帧的位移
      d ∈ [0, max_dist], θ ∈ [0°, 360°)

优势:
- 方向/速度自然解耦: 距离=速度, 角度=方向, LLM可以分别推理
- 旋转等变性: "向左转30度"在任何朝向下编码一致
- 适合方向性运动: 疏散、巡逻、编队等有明确方向的场景
- 与自然语言描述对齐更好: "向北走"/"转向东" 直接映射到角度token

劣势:
- 原点奇异性: 静止时 d≈0, 角度无意义 → 需特殊处理
- 离散化不均匀: 远距离处角度bin对应的实际面积更大
- 误差累积: 同relative, 但角度误差会导致方向性偏移
"""

import numpy as np
from typing import List, Tuple, Optional
from .base import BaseCoordinateTokenizer

ABSENT_MARKER = -1.0


class PolarTokenizer(BaseCoordinateTokenizer):

    def __init__(
        self,
        resolution: Tuple[int, int] = (360, 480),
        max_distance: float = 60.0,
        dist_bins: int = 30,
        angle_bins: int = 36,
        abs_bin_size: int = 5,
    ):
        self.max_distance = max_distance
        self.dist_bins = dist_bins
        self.angle_bins = angle_bins
        self.abs_bin_size = abs_bin_size
        super().__init__(resolution)

    def _build_vocab(self):
        # 绝对坐标token (首帧)
        self.x_bins_abs = self.W // self.abs_bin_size
        self.y_bins_abs = self.H // self.abs_bin_size
        self._abs_x_tokens = [f"<x_{i}>" for i in range(self.x_bins_abs + 1)]
        self._abs_y_tokens = [f"<y_{j}>" for j in range(self.y_bins_abs + 1)]

        # 距离token: <d_0> (静止) 到 <d_{dist_bins}>
        self._dist_tokens = [f"<d_{i}>" for i in range(self.dist_bins + 1)]
        # 角度token: <a_0> (0°) 到 <a_{angle_bins-1}> (350°)
        self._angle_tokens = [f"<a_{i}>" for i in range(self.angle_bins)]

        # absence tokens
        self._dist_tokens.append("<d_absent>")
        self._angle_tokens.append("<a_absent>")
        # 静止token (d≈0时角度无意义)
        self._angle_tokens.append("<a_still>")

        # 距离bin中心
        self._dist_centers = np.linspace(
            0, self.max_distance, self.dist_bins + 1
        )
        # 角度bin中心 (度)
        self._angle_centers = np.linspace(
            0, 360, self.angle_bins, endpoint=False
        ) + (180.0 / self.angle_bins)

        abs_x_centers = np.array(
            [0.0] + [(i - 0.5) * self.abs_bin_size for i in range(1, self.x_bins_abs + 1)]
        )
        abs_y_centers = np.array(
            [0.0] + [(j - 0.5) * self.abs_bin_size for j in range(1, self.y_bins_abs + 1)]
        )
        self._abs_x_centers = abs_x_centers
        self._abs_y_centers = abs_y_centers

        self._vocab = (
            self._abs_x_tokens + self._abs_y_tokens +
            self._dist_tokens + self._angle_tokens
        )

        self.still_threshold = self.max_distance / self.dist_bins * 0.5

    def tokenize(
        self,
        trajectories: np.ndarray,
        initial_positions: Optional[np.ndarray] = None,
    ) -> List[str]:
        N, T, _ = trajectories.shape
        tokens = []

        # 首帧: 绝对坐标
        for n in range(N):
            x, y = trajectories[n, 0]
            if x == ABSENT_MARKER:
                tokens.extend(["<x_0>", "<y_0>"])
            else:
                xi = int(np.clip(np.round(x / self.abs_bin_size), 1, self.x_bins_abs))
                yj = int(np.clip(np.round(y / self.abs_bin_size), 1, self.y_bins_abs))
                tokens.extend([f"<x_{xi}>", f"<y_{yj}>"])

        # 后续帧: 极坐标位移
        for t in range(1, T):
            for n in range(N):
                x_cur, y_cur = trajectories[n, t]
                x_prev, y_prev = trajectories[n, t - 1]

                if x_cur == ABSENT_MARKER or x_prev == ABSENT_MARKER:
                    tokens.extend(["<d_absent>", "<a_absent>"])
                    continue

                dx = x_cur - x_prev
                dy = y_cur - y_prev
                dist = np.sqrt(dx ** 2 + dy ** 2)

                # 距离bin
                di = int(np.clip(
                    np.round(dist / self.max_distance * self.dist_bins),
                    0,
                    self.dist_bins,
                ))

                # 静止处理
                if dist < self.still_threshold:
                    tokens.extend([f"<d_{di}>", "<a_still>"])
                else:
                    angle = np.degrees(np.arctan2(dy, dx)) % 360  # [0, 360)
                    ai = int(np.round(angle / 360 * self.angle_bins)) % self.angle_bins
                    tokens.extend([f"<d_{di}>", f"<a_{ai}>"])

        return tokens

    def detokenize(
        self,
        tokens: List[str],
        num_agents: int,
        num_steps: int,
        initial_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        traj = np.full((num_agents, num_steps, 2), ABSENT_MARKER)
        idx = 0

        # 解码首帧
        for n in range(num_agents):
            if idx + 1 >= len(tokens):
                break
            x_tok, y_tok = tokens[idx], tokens[idx + 1]
            idx += 2
            try:
                xi = int(x_tok.strip("<>").split("_")[1])
                yj = int(y_tok.strip("<>").split("_")[1])
            except (ValueError, IndexError):
                continue
            if xi > 0 and yj > 0:
                traj[n, 0, 0] = self._abs_x_centers[xi]
                traj[n, 0, 1] = self._abs_y_centers[yj]

        # 解码极坐标位移
        for t in range(1, num_steps):
            for n in range(num_agents):
                if idx + 1 >= len(tokens):
                    break
                d_tok, a_tok = tokens[idx], tokens[idx + 1]
                idx += 2

                if "absent" in d_tok or "absent" in a_tok:
                    continue
                if traj[n, t - 1, 0] == ABSENT_MARKER:
                    continue

                try:
                    di = int(d_tok.strip("<>").split("_")[1])
                except (ValueError, IndexError):
                    continue

                dist = self._dist_centers[di]

                if "still" in a_tok:
                    traj[n, t] = traj[n, t - 1]
                else:
                    try:
                        ai = int(a_tok.strip("<>").split("_")[1])
                    except (ValueError, IndexError):
                        continue
                    angle_rad = np.radians(self._angle_centers[ai])
                    traj[n, t, 0] = traj[n, t - 1, 0] + dist * np.cos(angle_rad)
                    traj[n, t, 1] = traj[n, t - 1, 1] + dist * np.sin(angle_rad)

        return self.validate_trajectory(traj)

    @property
    def vocab(self) -> List[str]:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def mode(self) -> str:
        return "polar"
