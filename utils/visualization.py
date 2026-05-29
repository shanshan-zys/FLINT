"""轨迹可视化工具"""

import numpy as np
from typing import List, Dict, Optional, Tuple

ABSENT = -1.0


def plot_trajectories(
    trajectories: np.ndarray,
    title: str = "",
    background: Optional[np.ndarray] = None,
    walkable_mask: Optional[np.ndarray] = None,
    bounds: Tuple[int, int] = (480, 360),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    可视化轨迹

    Args:
        trajectories: (N, T, 2) 轨迹数组
        title: 图标题
        background: 背景图像
        walkable_mask: 可行走区域mask
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(10, 7))

    if background is not None:
        ax.imshow(background, alpha=0.5)
    elif walkable_mask is not None:
        ax.imshow(walkable_mask, cmap='Greys_r', alpha=0.3,
                  extent=[0, bounds[0], bounds[1], 0])

    N, T, _ = trajectories.shape
    colors = cm.tab20(np.linspace(0, 1, max(N, 1)))

    for n in range(N):
        valid = trajectories[n, :, 0] != ABSENT
        if not valid.any():
            continue

        xs = trajectories[n, valid, 0]
        ys = trajectories[n, valid, 1]

        ax.plot(xs, ys, '-', color=colors[n % len(colors)],
                linewidth=1.5, alpha=0.8)
        ax.plot(xs[0], ys[0], 'o', color=colors[n % len(colors)],
                markersize=6)  # start
        ax.plot(xs[-1], ys[-1], 's', color=colors[n % len(colors)],
                markersize=6)  # end

    ax.set_xlim(0, bounds[0])
    ax.set_ylim(bounds[1], 0)  # y轴翻转
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_diversity_comparison(
    samples: List[np.ndarray],
    gt: Optional[np.ndarray] = None,
    title: str = "Diversity Visualization",
    save_path: Optional[str] = None,
):
    """
    可视化同一prompt生成的K条轨迹 (展示多样性)

    所有采样叠加在一张图上, GT用粗线标注
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(10, 7))

    # 绘制K次采样 (透明)
    for si, traj in enumerate(samples):
        N, T, _ = traj.shape
        for n in range(N):
            valid = traj[n, :, 0] != ABSENT
            if not valid.any():
                continue
            xs = traj[n, valid, 0]
            ys = traj[n, valid, 1]
            ax.plot(xs, ys, '-', color='steelblue', alpha=0.15, linewidth=1)

    # 绘制GT (粗线)
    if gt is not None:
        N, T, _ = gt.shape
        for n in range(N):
            valid = gt[n, :, 0] != ABSENT
            if not valid.any():
                continue
            xs = gt[n, valid, 0]
            ys = gt[n, valid, 1]
            ax.plot(xs, ys, '-', color='red', linewidth=2.5, alpha=0.9, label='GT' if n == 0 else '')

    ax.set_ylim(ax.get_ylim()[0], 0) if ax.get_ylim()[1] > 0 else None
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_metrics_radar(
    model_scores: Dict[str, Dict[str, float]],
    dimensions: List[str] = None,
    save_path: Optional[str] = None,
):
    """雷达图: 多模型多维度评分对比"""
    import matplotlib.pyplot as plt
    from math import pi

    if dimensions is None:
        dimensions = ["SA", "EA", "MP", "IR", "TI"]

    N = len(dimensions)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    for idx, (model_name, scores) in enumerate(model_scores.items()):
        values = [scores.get(d, 0) for d in dimensions]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2,
                color=colors[idx % len(colors)], label=model_name)
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=11)
    ax.set_ylim(0, 10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    plt.title('Model Evaluation Radar', fontsize=14, y=1.08)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
