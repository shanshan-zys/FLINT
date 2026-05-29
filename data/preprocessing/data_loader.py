"""ETH-UCY 和 SDD 数据加载与预处理"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def load_eth_ucy_trajectories(
    data_dir: str,
    subset: str,
    clip_length: int = 25,
) -> List[Dict]:
    """
    加载ETH-UCY数据并切分为clip

    ETH-UCY 标注格式: frame_id  ped_id  x  y
    (tab-separated, 有些数据集是space-separated)

    Args:
        data_dir: 包含 .txt 轨迹文件的目录
        subset: eth/hotel/univ/zara1/zara2
        clip_length: clip的帧数

    Returns:
        clip列表, 每个clip包含轨迹和元信息
    """
    txt_files = sorted(Path(data_dir).glob(f"{subset}*.txt"))
    if not txt_files:
        txt_files = sorted(Path(data_dir).glob(f"**/{subset}*.txt"))

    all_data = []
    for f in txt_files:
        data = np.loadtxt(f)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        all_data.append(data)

    if not all_data:
        print(f"Warning: no data found for subset {subset} in {data_dir}")
        return []

    data = np.concatenate(all_data, axis=0)
    # 列: frame_id, ped_id, x, y (有些数据集多列, 只取前4列)
    frames = data[:, 0].astype(int)
    ped_ids = data[:, 1].astype(int)
    positions = data[:, 2:4]

    unique_frames = sorted(set(frames))
    clips = []

    # 非重叠切分
    for start_idx in range(0, len(unique_frames), clip_length):
        clip_frames = unique_frames[start_idx:start_idx + clip_length]
        if len(clip_frames) < 2:
            continue

        # 提取该clip中的所有轨迹
        mask = np.isin(frames, clip_frames)
        clip_frames_data = frames[mask]
        clip_peds = ped_ids[mask]
        clip_pos = positions[mask]

        # 按行人组织
        unique_peds = sorted(set(clip_peds))
        trajectories = {}
        for pid in unique_peds:
            pid_mask = clip_peds == pid
            pid_frames = clip_frames_data[pid_mask]
            pid_pos = clip_pos[pid_mask]
            traj = []
            for cf in clip_frames:
                idx = np.where(pid_frames == cf)[0]
                if len(idx) > 0:
                    traj.append(pid_pos[idx[0]].tolist())
                else:
                    traj.append([-1.0, -1.0])  # absent
            trajectories[int(pid)] = traj

        # 计算初始状态 (位置+速度)
        initial_states = {}
        for pid, traj in trajectories.items():
            pos = traj[0]
            if pos[0] >= 0:  # 首帧存在
                if len(traj) > 1 and traj[1][0] >= 0:
                    vx = traj[1][0] - traj[0][0]
                    vy = traj[1][1] - traj[0][1]
                else:
                    vx, vy = 0.0, 0.0
                initial_states[pid] = {
                    "position": pos,
                    "velocity": [vx, vy],
                }

        clips.append({
            "clip_id": f"{subset}_clip{start_idx // clip_length:04d}",
            "dataset": "eth" if subset in ["eth", "hotel"] else "ucy",
            "subset": subset,
            "frame_start": clip_frames[0],
            "frame_end": clip_frames[-1],
            "num_frames": len(clip_frames),
            "num_pedestrians": len(unique_peds),
            "trajectories": trajectories,
            "initial_states": initial_states,
        })

    print(f"  {subset}: {len(clips)} clips, "
          f"{sum(c['num_pedestrians'] for c in clips)} total trajectories")
    return clips


def load_sdd_trajectories(
    data_dir: str,
    clip_length: int = 25,
    target_fps: float = 2.5,
) -> List[Dict]:
    """
    加载SDD (Stanford Drone Dataset)

    SDD 标注格式: (每个场景/视频一个annotations.txt)
    track_id, xmin, ymin, xmax, ymax, frame, lost, occluded, generated, label

    我们取 bounding box 中心作为位置, 只保留 "Pedestrian" 类别
    """
    sdd_root = Path(data_dir) / "annotations"
    if not sdd_root.exists():
        sdd_root = Path(data_dir)

    all_clips = []

    for scene_dir in sorted(sdd_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for video_dir in sorted(scene_dir.iterdir()):
            ann_file = video_dir / "annotations.txt"
            if not ann_file.exists():
                continue

            data = np.loadtxt(ann_file, delimiter=" ")
            if data.ndim == 1:
                data = data.reshape(1, -1)

            # 只保留行人 (label列通常需要从文本中解析)
            # SDD的label在最后一列, 但由于是txt, 需要特殊处理
            # 简化: 使用所有track
            track_ids = data[:, 0].astype(int)
            xmin, ymin = data[:, 1], data[:, 2]
            xmax, ymax = data[:, 3], data[:, 4]
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            frame_ids = data[:, 5].astype(int)

            # 下采样到目标fps (SDD原始30fps → 2.5fps, 每12帧取1帧)
            original_fps = 30.0
            sample_interval = int(original_fps / target_fps)
            unique_frames = sorted(set(frame_ids))
            sampled_frames = unique_frames[::sample_interval]

            # 切分clip
            for start_idx in range(0, len(sampled_frames), clip_length):
                clip_frames = sampled_frames[start_idx:start_idx + clip_length]
                if len(clip_frames) < 2:
                    continue

                mask = np.isin(frame_ids, clip_frames)
                clip_track_ids = track_ids[mask]
                clip_cx = cx[mask]
                clip_cy = cy[mask]
                clip_fids = frame_ids[mask]

                unique_tracks = sorted(set(clip_track_ids))
                if len(unique_tracks) == 0:
                    continue

                trajectories = {}
                for tid in unique_tracks:
                    tid_mask = clip_track_ids == tid
                    tid_frames = clip_fids[tid_mask]
                    tid_cx = clip_cx[tid_mask]
                    tid_cy = clip_cy[tid_mask]
                    traj = []
                    for cf in clip_frames:
                        idx = np.where(tid_frames == cf)[0]
                        if len(idx) > 0:
                            traj.append([float(tid_cx[idx[0]]), float(tid_cy[idx[0]])])
                        else:
                            traj.append([-1.0, -1.0])
                    trajectories[int(tid)] = traj

                scene_name = f"{scene_dir.name}_{video_dir.name}"
                clip_id = f"sdd_{scene_name}_clip{start_idx // clip_length:04d}"

                initial_states = {}
                for tid, traj in trajectories.items():
                    if traj[0][0] >= 0:
                        vx = traj[1][0] - traj[0][0] if len(traj) > 1 and traj[1][0] >= 0 else 0.0
                        vy = traj[1][1] - traj[0][1] if len(traj) > 1 and traj[1][0] >= 0 else 0.0
                        initial_states[tid] = {
                            "position": traj[0],
                            "velocity": [vx, vy],
                        }

                all_clips.append({
                    "clip_id": clip_id,
                    "dataset": "sdd",
                    "subset": scene_name,
                    "frame_start": clip_frames[0],
                    "frame_end": clip_frames[-1],
                    "num_frames": len(clip_frames),
                    "num_pedestrians": len(unique_tracks),
                    "trajectories": trajectories,
                    "initial_states": initial_states,
                })

    print(f"  SDD: {len(all_clips)} clips total")
    return all_clips


def generate_walkable_area_string(
    mask_path: str,
    grid_h: int = 72,
    grid_w: int = 96,
) -> str:
    """将walkable area mask下采样为0/1字符串"""
    from PIL import Image
    mask = np.array(Image.open(mask_path).convert("L"))
    # 下采样
    h, w = mask.shape
    block_h, block_w = h // grid_h, w // grid_w
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for i in range(grid_h):
        for j in range(grid_w):
            block = mask[
                i * block_h:(i + 1) * block_h,
                j * block_w:(j + 1) * block_w,
            ]
            grid[i, j] = 1 if block.mean() > 128 else 0
    return "".join(str(v) for row in grid for v in row)


def normalize_coordinates(
    trajectories: Dict,
    resolution: Tuple[int, int] = (360, 480),
    source_bounds: Tuple[float, float, float, float] = None,
) -> Dict:
    """
    将轨迹坐标归一化到目标分辨率

    ETH-UCY 使用世界坐标(米), 需要转换到像素坐标
    SDD 使用像素坐标, 可能需要缩放到统一分辨率
    """
    H, W = resolution
    all_coords = []
    for traj in trajectories.values():
        for x, y in traj:
            if x >= 0 and y >= 0:
                all_coords.append([x, y])

    if not all_coords:
        return trajectories

    coords = np.array(all_coords)

    if source_bounds:
        x_min, y_min, x_max, y_max = source_bounds
    else:
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0

    normalized = {}
    for pid, traj in trajectories.items():
        new_traj = []
        for x, y in traj:
            if x < 0 or y < 0:
                new_traj.append([-1.0, -1.0])
            else:
                nx = (x - x_min) / x_range * (W - 1)
                ny = (y - y_min) / y_range * (H - 1)
                new_traj.append([float(nx), float(ny)])
        normalized[pid] = new_traj

    return normalized
