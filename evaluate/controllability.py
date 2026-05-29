"""
可控性评估指标

验证"text改变 → 轨迹相应改变", 证明LLM真正理解了text指令
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import pearsonr


def speed_controllability(
    text_speed_labels: List[str],
    trajectories: List[np.ndarray],
) -> Dict[str, float]:
    """
    速度可控性: 修改text中的速度描述, 度量实际速度是否相应变化

    text_speed_labels: ["slow", "normal", "fast", ...]
    trajectories: 对应的生成轨迹

    返回: 速度标签与实际速度的相关系数
    """
    speed_map = {"very_slow": 1, "slow": 2, "normal": 3, "fast": 4, "very_fast": 5}
    label_scores = [speed_map.get(l, 3) for l in text_speed_labels]

    actual_speeds = []
    for traj in trajectories:
        N, T, _ = traj.shape
        speeds = []
        for n in range(N):
            for t in range(1, T):
                if traj[n, t, 0] >= 0 and traj[n, t - 1, 0] >= 0:
                    d = np.sqrt(
                        (traj[n, t, 0] - traj[n, t - 1, 0]) ** 2 +
                        (traj[n, t, 1] - traj[n, t - 1, 1]) ** 2
                    )
                    speeds.append(d)
        actual_speeds.append(np.mean(speeds) if speeds else 0)

    if len(label_scores) < 3:
        return {"speed_correlation": 0.0, "speed_pvalue": 1.0}

    corr, pval = pearsonr(label_scores, actual_speeds)
    return {
        "speed_correlation": float(corr),
        "speed_pvalue": float(pval),
        "label_scores": label_scores,
        "actual_speeds": actual_speeds,
    }


def density_controllability(
    text_density_labels: List[str],
    trajectories: List[np.ndarray],
) -> Dict[str, float]:
    """
    密度可控性: "sparse" vs "dense" 描述是否影响生成的人群密度

    密度 = 平均每帧活跃行人数 / 场景面积
    """
    density_map = {"very_sparse": 1, "sparse": 2, "moderate": 3, "dense": 4, "very_dense": 5}
    label_scores = [density_map.get(l, 3) for l in text_density_labels]

    actual_densities = []
    for traj in trajectories:
        N, T, _ = traj.shape
        active_per_frame = []
        for t in range(T):
            count = sum(1 for n in range(N) if traj[n, t, 0] >= 0)
            active_per_frame.append(count)
        actual_densities.append(np.mean(active_per_frame))

    if len(label_scores) < 3:
        return {"density_correlation": 0.0, "density_pvalue": 1.0}

    corr, pval = pearsonr(label_scores, actual_densities)
    return {
        "density_correlation": float(corr),
        "density_pvalue": float(pval),
    }


def direction_controllability(
    text_directions: List[str],
    trajectories: List[np.ndarray],
) -> Dict[str, float]:
    """
    方向可控性: "left-to-right" vs "top-to-bottom" 是否影响主运动方向

    计算主方向角与描述方向的一致性
    """
    direction_angles = {
        "left_to_right": 0, "right_to_left": 180,
        "top_to_bottom": 90, "bottom_to_top": 270,
        "diagonal_ne": 315, "diagonal_nw": 225,
        "diagonal_se": 45, "diagonal_sw": 135,
    }

    errors = []
    for direction, traj in zip(text_directions, trajectories):
        if direction not in direction_angles:
            continue
        target_angle = direction_angles[direction]

        N, T, _ = traj.shape
        angles = []
        for n in range(N):
            valid = [(traj[n, t, 0], traj[n, t, 1])
                     for t in range(T) if traj[n, t, 0] >= 0]
            if len(valid) >= 2:
                dx = valid[-1][0] - valid[0][0]
                dy = valid[-1][1] - valid[0][1]
                angle = np.degrees(np.arctan2(dy, dx)) % 360
                angles.append(angle)

        if angles:
            mean_angle = np.mean(angles) % 360
            error = min(
                abs(mean_angle - target_angle),
                360 - abs(mean_angle - target_angle),
            )
            errors.append(error)

    return {
        "direction_mean_error": float(np.mean(errors)) if errors else 180.0,
        "direction_accuracy_30deg": float(
            np.mean([1 if e < 30 else 0 for e in errors])
        ) if errors else 0.0,
    }


def compute_all_controllability(
    control_experiments: List[Dict],
) -> Dict[str, float]:
    """
    综合可控性评估

    control_experiments: [
        {
            "type": "speed",  # speed / density / direction
            "labels": ["slow", "fast", ...],
            "trajectories": [np.ndarray, ...],
        },
        ...
    ]
    """
    results = {}
    for exp in control_experiments:
        if exp["type"] == "speed":
            r = speed_controllability(exp["labels"], exp["trajectories"])
            results.update(r)
        elif exp["type"] == "density":
            r = density_controllability(exp["labels"], exp["trajectories"])
            results.update(r)
        elif exp["type"] == "direction":
            r = direction_controllability(exp["labels"], exp["trajectories"])
            results.update(r)
    return results
