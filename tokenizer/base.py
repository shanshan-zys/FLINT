"""
Coordinate Tokenizer 基类

三种编码方式的对比:

┌──────────────┬──────────────────────────────┬──────────────────────────────┬────────────────────────────┐
│              │  绝对坐标 (Absolute)          │  相对坐标 (Relative)          │  极坐标 (Polar)             │
├──────────────┼──────────────────────────────┼──────────────────────────────┼────────────────────────────┤
│  编码内容     │  (x, y) 全局像素坐标          │  (Δx, Δy) 相对上一帧位移      │  (d, θ) 距离+角度           │
│  优势        │  简单直观; 保留全局位置信息;    │  平移不变性; 泛化性强;         │  方向性建模天然;             │
│              │  与walkable area直接对应       │  token分布更均匀              │  速度/转向解耦               │
│  劣势        │  位置依赖, 跨场景泛化差;       │  误差累积; 首帧需绝对坐标;     │  原点奇异性; 离散化不均匀;    │
│              │  token分布不均匀              │  丢失全局位置信息              │  实现复杂                   │
│  适合场景     │  单场景、需要精确位置          │  跨场景迁移、运动模式学习       │  方向性强的运动(如疏散)       │
│  vocab size  │  170 (97x + 73y)             │  102 (51dx + 51dy) + 170首帧  │  66 (30d + 36θ) + 170首帧   │
└──────────────┴──────────────────────────────┴──────────────────────────────┴────────────────────────────┘
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np


class BaseCoordinateTokenizer(ABC):
    """坐标tokenizer基类"""

    def __init__(self, resolution: Tuple[int, int] = (360, 480)):
        self.H, self.W = resolution
        self._build_vocab()

    @abstractmethod
    def _build_vocab(self):
        """构建token词汇表"""
        pass

    @abstractmethod
    def tokenize(
        self,
        trajectories: np.ndarray,
        initial_positions: Optional[np.ndarray] = None,
    ) -> List[str]:
        """
        将轨迹编码为token序列
        Args:
            trajectories: (N, T, 2) 轨迹数组, N=人数, T=时间步, 2=(x,y)
            initial_positions: (N, 2) 初始位置 (仅relative/polar需要)
        Returns:
            token列表
        """
        pass

    @abstractmethod
    def detokenize(
        self,
        tokens: List[str],
        num_agents: int,
        num_steps: int,
        initial_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        将token序列解码为轨迹
        Args:
            tokens: token列表
            num_agents: 人数
            num_steps: 时间步数
            initial_positions: (N, 2) 初始位置
        Returns:
            (N, T, 2) 轨迹数组
        """
        pass

    @property
    @abstractmethod
    def vocab(self) -> List[str]:
        """返回所有特殊token"""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    @property
    @abstractmethod
    def mode(self) -> str:
        pass

    def validate_trajectory(self, traj: np.ndarray) -> np.ndarray:
        """将轨迹裁剪到合法范围"""
        traj = traj.copy()
        traj[..., 0] = np.clip(traj[..., 0], 0, self.W - 1)
        traj[..., 1] = np.clip(traj[..., 1], 0, self.H - 1)
        return traj
