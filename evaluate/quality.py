"""
轨迹质量评估指标

包括论文中已有的指标和补充指标
"""

import numpy as np
from typing import Dict, List, Tuple

ABSENT = -1.0


def joint_average_displacement_error(
    pred: np.ndarray,
    gt: np.ndarray,
) -> float:
    """
    JADE: Joint Average Displacement Error

    所有行人在所有时间步的平均位移误差

    JADE = (1/N*T) Σ_n Σ_t ||pred_{n,t} - gt_{n,t}||_2
    """
    N, T, _ = pred.shape
    total = 0.0
    count = 0
    for n in range(N):
        for t in range(T):
            if gt[n, t, 0] != ABSENT and pred[n, t, 0] != ABSENT:
                dist = np.sqrt(
                    (pred[n, t, 0] - gt[n, t, 0]) ** 2 +
                    (pred[n, t, 1] - gt[n, t, 1]) ** 2
                )
                total += dist
                count += 1
    return total / max(count, 1)


def joint_final_displacement_error(
    pred: np.ndarray,
    gt: np.ndarray,
) -> float:
    """
    JFDE: Joint Final Displacement Error

    所有行人最后一个有效时间步的位移误差
    """
    N, T, _ = pred.shape
    total = 0.0
    count = 0
    for n in range(N):
        last_t = -1
        for t in range(T - 1, -1, -1):
            if gt[n, t, 0] != ABSENT:
                last_t = t
                break
        if last_t >= 0 and pred[n, last_t, 0] != ABSENT:
            dist = np.sqrt(
                (pred[n, last_t, 0] - gt[n, last_t, 0]) ** 2 +
                (pred[n, last_t, 1] - gt[n, last_t, 1]) ** 2
            )
            total += dist
            count += 1
    return total / max(count, 1)


def collision_rate(
    traj: np.ndarray,
    threshold: float = 10.0,
) -> float:
    """
    碰撞率: 同一时刻行人间距离 < threshold 的比例

    Col = #(dist < threshold) / #(total_pairs)
    """
    N, T, _ = traj.shape
    total_pairs = 0
    collisions = 0
    for t in range(T):
        for i in range(N):
            if traj[i, t, 0] == ABSENT:
                continue
            for j in range(i + 1, N):
                if traj[j, t, 0] == ABSENT:
                    continue
                dist = np.sqrt(
                    (traj[i, t, 0] - traj[j, t, 0]) ** 2 +
                    (traj[i, t, 1] - traj[j, t, 1]) ** 2
                )
                total_pairs += 1
                if dist < threshold:
                    collisions += 1
    return collisions / max(total_pairs, 1)


def trajectory_smoothness(traj: np.ndarray) -> Dict[str, float]:
    """
    轨迹平滑性指标:
    - perpendicular_deviation: 轨迹偏离直线的程度
    - speed_change: 速度变化的标准差
    - angle_change: 方向变化的标准差
    """
    N, T, _ = traj.shape
    all_speed_change = []
    all_angle_change = []
    all_deviation = []

    for n in range(N):
        valid_positions = []
        for t in range(T):
            if traj[n, t, 0] != ABSENT:
                valid_positions.append(traj[n, t])
        if len(valid_positions) < 3:
            continue

        positions = np.array(valid_positions)
        disps = np.diff(positions, axis=0)
        speeds = np.sqrt((disps ** 2).sum(axis=1))
        angles = np.arctan2(disps[:, 1], disps[:, 0])

        # speed change
        if len(speeds) >= 2:
            all_speed_change.append(np.std(np.diff(speeds)))

        # angle change
        if len(angles) >= 2:
            angle_diff = np.diff(angles)
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            all_angle_change.append(np.std(angle_diff))

        # perpendicular deviation from straight line
        start, end = positions[0], positions[-1]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len > 1e-6:
            line_dir = line_vec / line_len
            deviations = []
            for p in positions[1:-1]:
                v = p - start
                proj = np.dot(v, line_dir) * line_dir
                perp = v - proj
                deviations.append(np.linalg.norm(perp))
            if deviations:
                all_deviation.append(np.mean(deviations))

    return {
        "perpendicular_deviation": float(np.mean(all_deviation)) if all_deviation else 0.0,
        "speed_change_std": float(np.mean(all_speed_change)) if all_speed_change else 0.0,
        "angle_change_std": float(np.mean(all_angle_change)) if all_angle_change else 0.0,
    }


def travel_distance(traj: np.ndarray) -> float:
    """平均行人行走距离"""
    N, T, _ = traj.shape
    distances = []
    for n in range(N):
        total = 0.0
        for t in range(1, T):
            if traj[n, t, 0] != ABSENT and traj[n, t - 1, 0] != ABSENT:
                d = np.sqrt(
                    (traj[n, t, 0] - traj[n, t - 1, 0]) ** 2 +
                    (traj[n, t, 1] - traj[n, t - 1, 1]) ** 2
                )
                total += d
        distances.append(total)
    return float(np.mean(distances))


def walkability_violation_rate(
    traj: np.ndarray,
    walkable_mask: np.ndarray,
    bin_size: int = 5,
) -> float:
    """越界率: 走到不可通行区域的位置占比"""
    N, T, _ = traj.shape
    total = 0
    violations = 0
    grid_h, grid_w = walkable_mask.shape
    for n in range(N):
        for t in range(T):
            if traj[n, t, 0] == ABSENT:
                continue
            gi = int(np.clip(traj[n, t, 0] / bin_size, 0, grid_w - 1))
            gj = int(np.clip(traj[n, t, 1] / bin_size, 0, grid_h - 1))
            total += 1
            if walkable_mask[gj, gi] == 0:
                violations += 1
    return violations / max(total, 1)


def compute_all_quality_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    walkable_mask: np.ndarray = None,
) -> Dict[str, float]:
    """计算所有质量指标"""
    metrics = {
        "JADE": joint_average_displacement_error(pred, gt),
        "JFDE": joint_final_displacement_error(pred, gt),
        "Collision_Rate": collision_rate(pred),
        "Travel_Distance": travel_distance(pred),
    }

    smoothness = trajectory_smoothness(pred)
    metrics.update(smoothness)

    if walkable_mask is not None:
        metrics["Walkability_Violation"] = walkability_violation_rate(pred, walkable_mask)

    return metrics
