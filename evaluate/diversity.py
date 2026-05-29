"""
多样性评估指标

核心思路: 同一prompt生成K次, 度量K条轨迹之间的差异

指标体系:
┌──────────────────────┬────────────────────────────────────────────────┐
│  指标                 │  含义                                          │
├──────────────────────┼────────────────────────────────────────────────┤
│  ATD                 │  K条轨迹两两平均L2距离 → 整体多样性             │
│  FDD                 │  终点位置方差 → 目的地多样性                     │
│  Coverage            │  空间网格覆盖率 → 空间探索多样性                 │
│  Endpoint Entropy    │  终点分布信息熵 → 分布均匀性                    │
│  Mode Count          │  DBSCAN聚类后的模式数 → 行为模式多样性          │
│  Self-Distance       │  最近邻距离均值 → 避免模式坍塌                  │
└──────────────────────┴────────────────────────────────────────────────┘
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from collections import defaultdict

ABSENT = -1.0


def _extract_valid_positions(traj: np.ndarray) -> np.ndarray:
    """提取非absent的有效位置"""
    mask = (traj[:, 0] != ABSENT) & (traj[:, 1] != ABSENT)
    return traj[mask]


def average_trajectory_diversity(trajectories: List[np.ndarray]) -> float:
    """
    ATD: Average Trajectory Diversity

    对K条轨迹 (每条 shape=(T, 2)), 计算两两平均L2距离

    ATD = (2 / K(K-1)) * Σ_{i<j} mean_t ||traj_i(t) - traj_j(t)||_2

    值越大 → 多样性越高
    """
    K = len(trajectories)
    if K < 2:
        return 0.0

    total_dist = 0.0
    count = 0
    for i in range(K):
        for j in range(i + 1, K):
            ti, tj = trajectories[i], trajectories[j]
            min_len = min(len(ti), len(tj))
            if min_len == 0:
                continue
            diff = ti[:min_len] - tj[:min_len]
            dist = np.sqrt((diff ** 2).sum(axis=-1) + 1e-8).mean()
            total_dist += dist
            count += 1

    return total_dist / max(count, 1)


def final_displacement_diversity(trajectories: List[np.ndarray]) -> float:
    """
    FDD: Final Displacement Diversity

    K条轨迹终点的标准差

    FDD = std(endpoints)

    终点分散 → 模型不会坍塌到同一个目的地
    """
    endpoints = []
    for traj in trajectories:
        valid = _extract_valid_positions(traj)
        if len(valid) > 0:
            endpoints.append(valid[-1])

    if len(endpoints) < 2:
        return 0.0

    endpoints = np.array(endpoints)  # (K, 2)
    return float(np.std(endpoints, axis=0).mean())


def spatial_coverage(
    trajectories: List[np.ndarray],
    grid_size: int = 20,
    bounds: Tuple[float, float, float, float] = (0, 0, 480, 360),
) -> float:
    """
    Coverage: 空间网格覆盖率

    将空间划分为 grid_size x grid_size 网格, 统计K条轨迹覆盖了多少格子

    Coverage = 被覆盖的网格数 / 总网格数

    值越大 → 轨迹空间探索越充分
    """
    x_min, y_min, x_max, y_max = bounds
    visited = set()

    for traj in trajectories:
        valid = _extract_valid_positions(traj)
        for x, y in valid:
            gi = int(np.clip((x - x_min) / (x_max - x_min) * grid_size, 0, grid_size - 1))
            gj = int(np.clip((y - y_min) / (y_max - y_min) * grid_size, 0, grid_size - 1))
            visited.add((gi, gj))

    total_cells = grid_size * grid_size
    return len(visited) / total_cells


def endpoint_entropy(
    trajectories: List[np.ndarray],
    num_bins: int = 10,
    bounds: Tuple[float, float, float, float] = (0, 0, 480, 360),
) -> float:
    """
    Endpoint Entropy: 终点分布信息熵

    将终点位置离散化到网格, 计算分布的Shannon熵

    H = -Σ p(i) log p(i)

    熵越高 → 终点分布越均匀, 不会集中在少数位置
    """
    x_min, y_min, x_max, y_max = bounds

    endpoints = []
    for traj in trajectories:
        valid = _extract_valid_positions(traj)
        if len(valid) > 0:
            endpoints.append(valid[-1])

    if len(endpoints) < 2:
        return 0.0

    endpoints = np.array(endpoints)
    # 离散化到网格
    gi = np.clip(
        ((endpoints[:, 0] - x_min) / (x_max - x_min) * num_bins).astype(int),
        0, num_bins - 1,
    )
    gj = np.clip(
        ((endpoints[:, 1] - y_min) / (y_max - y_min) * num_bins).astype(int),
        0, num_bins - 1,
    )
    cell_ids = gi * num_bins + gj

    # 计算分布
    counts = np.bincount(cell_ids, minlength=num_bins * num_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    return float(entropy(probs))


def mode_count(
    trajectories: List[np.ndarray],
    eps: float = 30.0,
    min_samples: int = 2,
) -> int:
    """
    Mode Count: 行为模式数量

    对K条轨迹的"特征向量"做DBSCAN聚类, 统计聚类数

    特征 = [起点x, 起点y, 终点x, 终点y, 平均速度, 主方向角]

    模式数越多 → 产生了越多种不同行为
    """
    from sklearn.cluster import DBSCAN

    features = []
    for traj in trajectories:
        valid = _extract_valid_positions(traj)
        if len(valid) < 2:
            continue
        start = valid[0]
        end = valid[-1]
        disps = np.diff(valid, axis=0)
        speed = np.sqrt((disps ** 2).sum(axis=-1)).mean()
        direction = np.arctan2(end[1] - start[1], end[0] - start[0])
        features.append([start[0], start[1], end[0], end[1], speed, direction * 50])

    if len(features) < 3:
        return 1

    features = np.array(features)
    # 标准化
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

    clustering = DBSCAN(eps=eps / 50, min_samples=min_samples).fit(features)
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    return max(n_clusters, 1)


def self_nearest_distance(trajectories: List[np.ndarray]) -> float:
    """
    Self-Distance: 最近邻距离均值

    每条轨迹到其最近邻轨迹的距离, 取均值

    值越大 → 轨迹之间差异越明显, 避免生成几乎相同的轨迹
    """
    K = len(trajectories)
    if K < 2:
        return 0.0

    # 计算两两距离矩阵
    dist_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            min_len = min(len(trajectories[i]), len(trajectories[j]))
            if min_len == 0:
                dist_matrix[i, j] = dist_matrix[j, i] = 0
                continue
            diff = trajectories[i][:min_len] - trajectories[j][:min_len]
            dist = np.sqrt((diff ** 2).sum(axis=-1) + 1e-8).mean()
            dist_matrix[i, j] = dist_matrix[j, i] = dist

    # 每行最小非零值
    nn_dists = []
    for i in range(K):
        row = dist_matrix[i]
        row[i] = float("inf")
        nn_dists.append(row.min())

    return float(np.mean(nn_dists))


# ====================================================================
# 聚合: 计算所有多样性指标
# ====================================================================
def compute_all_diversity_metrics(
    trajectories_per_agent: Dict[int, List[np.ndarray]],
    bounds: Tuple[float, float, float, float] = (0, 0, 480, 360),
) -> Dict[str, float]:
    """
    计算所有多样性指标

    Args:
        trajectories_per_agent: {agent_id: [K条轨迹, 每条(T,2)]}
            K条轨迹来自同一prompt的K次采样
        bounds: 坐标范围

    Returns:
        指标字典
    """
    all_atd, all_fdd, all_coverage, all_entropy = [], [], [], []
    all_modes, all_snd = [], []

    for agent_id, trajs in trajectories_per_agent.items():
        if len(trajs) < 2:
            continue
        all_atd.append(average_trajectory_diversity(trajs))
        all_fdd.append(final_displacement_diversity(trajs))
        all_coverage.append(spatial_coverage(trajs, bounds=bounds))
        all_entropy.append(endpoint_entropy(trajs, bounds=bounds))
        all_modes.append(mode_count(trajs))
        all_snd.append(self_nearest_distance(trajs))

    return {
        "ATD": float(np.mean(all_atd)) if all_atd else 0.0,
        "FDD": float(np.mean(all_fdd)) if all_fdd else 0.0,
        "Coverage": float(np.mean(all_coverage)) if all_coverage else 0.0,
        "Endpoint_Entropy": float(np.mean(all_entropy)) if all_entropy else 0.0,
        "Mode_Count": float(np.mean(all_modes)) if all_modes else 0.0,
        "Self_Nearest_Distance": float(np.mean(all_snd)) if all_snd else 0.0,
    }


def compute_diversity_from_raw_generations(
    generated_token_lists: List[str],
    tokenizer,
    num_agents: int,
    num_steps: int,
) -> Dict[str, float]:
    """
    从原始生成的token字符串计算多样性

    Args:
        generated_token_lists: K次生成的token字符串列表
        tokenizer: coordinate tokenizer
        num_agents: 行人数
        num_steps: 时间步数

    Returns:
        多样性指标
    """
    import re

    trajectories_per_agent = defaultdict(list)

    for gen_str in generated_token_lists:
        tokens = re.findall(r'<[^>]+>', gen_str)
        traj = tokenizer.detokenize(tokens, num_agents, num_steps)
        # traj: (N, T, 2)
        for n in range(num_agents):
            trajectories_per_agent[n].append(traj[n])

    return compute_all_diversity_metrics(trajectories_per_agent)
