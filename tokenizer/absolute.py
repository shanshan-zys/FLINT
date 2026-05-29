"""
绝对坐标 Tokenizer (论文原始方案)

编码: (x, y) → <x_i><y_j>  其中 i = round(x / bin_size), j = round(y / bin_size)
解码: <x_i><y_j> → ((i-1)*bin_size + bin_size/2, (j-1)*bin_size + bin_size/2)

特点:
- 直接映射像素坐标到离散token
- <x_0><y_0> 表示"不存在"(未出现或已离开)
- bin_size=5 时, x方向96个bin, y方向72个bin, 共170个token
"""

import numpy as np
from typing import List, Tuple, Optional
from .base import BaseCoordinateTokenizer

ABSENT_MARKER = -1.0


class AbsoluteTokenizer(BaseCoordinateTokenizer):

    def __init__(
        self,
        resolution: Tuple[int, int] = (360, 480),
        bin_size: int = 5,
    ):
        self.bin_size = bin_size
        super().__init__(resolution)

    def _build_vocab(self):
        self.x_bins = self.W // self.bin_size  # 96
        self.y_bins = self.H // self.bin_size  # 72
        # <x_0>, <y_0> 是 absence token
        self._x_tokens = [f"<x_{i}>" for i in range(self.x_bins + 1)]
        self._y_tokens = [f"<y_{j}>" for j in range(self.y_bins + 1)]
        self._vocab = self._x_tokens + self._y_tokens

        # bin中心坐标: bin 1 → bin_size/2, bin 2 → bin_size*1.5, ...
        self._x_centers = np.array(
            [0.0] + [(i - 0.5) * self.bin_size for i in range(1, self.x_bins + 1)]
        )
        self._y_centers = np.array(
            [0.0] + [(j - 0.5) * self.bin_size for j in range(1, self.y_bins + 1)]
        )

    def tokenize(
        self,
        trajectories: np.ndarray,
        initial_positions: Optional[np.ndarray] = None,
    ) -> List[str]:
        N, T, _ = trajectories.shape
        tokens = []
        for t in range(T):
            for n in range(N):
                x, y = trajectories[n, t]
                if x == ABSENT_MARKER or y == ABSENT_MARKER:
                    tokens.extend(["<x_0>", "<y_0>"])
                else:
                    xi = int(np.clip(np.round(x / self.bin_size), 1, self.x_bins))
                    yj = int(np.clip(np.round(y / self.bin_size), 1, self.y_bins))
                    tokens.extend([f"<x_{xi}>", f"<y_{yj}>"])
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
        for t in range(num_steps):
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
                if xi == 0 or yj == 0:
                    continue
                traj[n, t, 0] = self._x_centers[xi]
                traj[n, t, 1] = self._y_centers[yj]
        return traj

    @property
    def vocab(self) -> List[str]:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def mode(self) -> str:
        return "absolute"
