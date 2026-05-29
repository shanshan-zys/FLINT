"""
相对坐标 (位移) Tokenizer

编码: 首帧用绝对坐标 <x_i><y_j>, 后续帧用位移 <dx_i><dy_j>
      Δx, Δy 被裁剪到 [-max_disp, max_disp] 后离散化

优势:
- 平移不变性: 学到的是"怎么走"而非"在哪走", 跨场景泛化更好
- Token分布更均匀: 位移集中在小范围, 不会像绝对坐标那样稀疏分布
- 运动模式捕捉: 直接编码速度信息, 更容易学到加速/减速/转弯模式

劣势:
- 误差累积: detokenize时每步误差叠加, 长轨迹末端偏差大
- 需要首帧绝对坐标: 作为锚点
- 全局位置感知弱: 模型难以判断是否越出walkable area边界
"""

import numpy as np
from typing import List, Tuple, Optional
from .base import BaseCoordinateTokenizer

ABSENT_MARKER = -1.0


class RelativeTokenizer(BaseCoordinateTokenizer):

    def __init__(
        self,
        resolution: Tuple[int, int] = (360, 480),
        max_displacement: int = 50,
        bin_size: int = 2,
        abs_bin_size: int = 5,
    ):
        self.max_displacement = max_displacement
        self.disp_bin_size = bin_size
        self.abs_bin_size = abs_bin_size
        super().__init__(resolution)

    def _build_vocab(self):
        # 绝对坐标token (仅首帧使用)
        self.x_bins = self.W // self.abs_bin_size
        self.y_bins = self.H // self.abs_bin_size
        self._abs_x_tokens = [f"<x_{i}>" for i in range(self.x_bins + 1)]
        self._abs_y_tokens = [f"<y_{j}>" for j in range(self.y_bins + 1)]

        # 位移token
        self.num_disp_bins = 2 * (self.max_displacement // self.disp_bin_size) + 1
        half = self.max_displacement // self.disp_bin_size
        self._dx_tokens = [f"<dx_{i}>" for i in range(self.num_disp_bins)]
        self._dy_tokens = [f"<dy_{i}>" for i in range(self.num_disp_bins)]
        # dx_0 → -max_disp, dx_{half} → 0, dx_{2*half} → +max_disp
        self._disp_centers = np.array(
            [(i - half) * self.disp_bin_size for i in range(self.num_disp_bins)]
        )

        # absence tokens for displacement
        self._dx_tokens.append("<dx_absent>")
        self._dy_tokens.append("<dy_absent>")

        abs_x_centers = np.array(
            [0.0] + [(i - 0.5) * self.abs_bin_size for i in range(1, self.x_bins + 1)]
        )
        abs_y_centers = np.array(
            [0.0] + [(j - 0.5) * self.abs_bin_size for j in range(1, self.y_bins + 1)]
        )
        self._abs_x_centers = abs_x_centers
        self._abs_y_centers = abs_y_centers

        self._vocab = (
            self._abs_x_tokens + self._abs_y_tokens +
            self._dx_tokens + self._dy_tokens
        )

    def _coord_to_disp_bin(self, disp: float) -> int:
        half = self.max_displacement // self.disp_bin_size
        bin_idx = int(np.clip(
            np.round(disp / self.disp_bin_size) + half,
            0,
            self.num_disp_bins - 1,
        ))
        return bin_idx

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
                xi = int(np.clip(np.round(x / self.abs_bin_size), 1, self.x_bins))
                yj = int(np.clip(np.round(y / self.abs_bin_size), 1, self.y_bins))
                tokens.extend([f"<x_{xi}>", f"<y_{yj}>"])

        # 后续帧: 相对位移
        for t in range(1, T):
            for n in range(N):
                x_cur, y_cur = trajectories[n, t]
                x_prev, y_prev = trajectories[n, t - 1]
                if x_cur == ABSENT_MARKER or x_prev == ABSENT_MARKER:
                    tokens.extend(["<dx_absent>", "<dy_absent>"])
                else:
                    dx = x_cur - x_prev
                    dy = y_cur - y_prev
                    dxi = self._coord_to_disp_bin(dx)
                    dyj = self._coord_to_disp_bin(dy)
                    tokens.extend([f"<dx_{dxi}>", f"<dy_{dyj}>"])

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

        # 解码首帧绝对坐标
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

        # 解码后续帧位移
        for t in range(1, num_steps):
            for n in range(num_agents):
                if idx + 1 >= len(tokens):
                    break
                dx_tok, dy_tok = tokens[idx], tokens[idx + 1]
                idx += 2
                if "absent" in dx_tok or "absent" in dy_tok:
                    continue
                try:
                    dxi = int(dx_tok.strip("<>").split("_")[1])
                    dyj = int(dy_tok.strip("<>").split("_")[1])
                except (ValueError, IndexError):
                    continue
                if traj[n, t - 1, 0] != ABSENT_MARKER:
                    traj[n, t, 0] = traj[n, t - 1, 0] + self._disp_centers[dxi]
                    traj[n, t, 1] = traj[n, t - 1, 1] + self._disp_centers[dyj]

        return self.validate_trajectory(traj)

    @property
    def vocab(self) -> List[str]:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def mode(self) -> str:
        return "relative"
