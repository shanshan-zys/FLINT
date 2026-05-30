"""
ETH-UCY 预处理：原始数据 → 统一 640×480 像素坐标 clip

功能：
  默认模式: 坐标转换 + 切clip + 视频抽帧 + 可视化
  --generate_backgrounds: 生成干净背景图
  --generate_maps: 交互式标注可行走区域
"""

import os
import re
import argparse
import numpy as np
import cv2
from pathlib import Path

TARGET_W, TARGET_H = 640, 480
CLIP_LEN = 25
FPS_OUT = 2.5

SCENES = {
    'eth': {
        'obsmat': 'ETH/seq_eth/obsmat.txt',
        'H': 'ETH/seq_eth/H.txt',
        'video': 'ETH/seq_eth/video.avi',
        'native_res': (640, 480),
        'step': 6,
    },
    'hotel': {
        'obsmat': 'ETH/seq_hotel/obsmat.txt',
        'H': 'ETH/seq_hotel/H.txt',
        'video': 'ETH/seq_hotel/video.avi',
        'native_res': (720, 576),
        'step': 10,
    },
    'zara1': {
        'obsmat': 'UCY/zara01/obsmat.txt',
        'H': 'UCY/zara01/H.txt',
        'video': 'UCY/zara01/video.avi',
        'native_res': (720, 576),
        'step': 10,
    },
    'zara2': {
        'obsmat': 'UCY/zara02/obsmat.txt',
        'H': 'UCY/zara02/H-old.txt',
        'video': 'UCY/zara02/video.avi',
        'native_res': (720, 576),
        'step': 10,
    },
    'univ': {
        'obsmat': 'UCY/students03/obsmat_px.txt',
        'H': None,
        'video': 'UCY/students03/video.avi',
        'native_res': (720, 576),
        'step': 10,
    },
}


def load_obsmat(path):
    data = np.loadtxt(path)
    frame_ids = data[:, 0].astype(int)
    ped_ids = data[:, 1].astype(int)
    pos_x = data[:, 2]
    pos_y = data[:, 4]
    return frame_ids, ped_ids, pos_x, pos_y


def world_to_pixel(H_path, pos_x, pos_y):
    H = np.loadtxt(H_path)
    H_inv = np.linalg.inv(H)
    n = len(pos_x)
    world = np.vstack([pos_x, pos_y, np.ones(n)])
    pixel = H_inv @ world
    pixel /= pixel[2:3, :]
    return pixel[0], pixel[1]


def convert_coordinates(scene_name, cfg, opentraj_dir):
    obsmat_path = os.path.join(opentraj_dir, cfg['obsmat'])
    frame_ids, ped_ids, pos_x, pos_y = load_obsmat(obsmat_path)

    if cfg['H'] is not None:
        H_path = os.path.join(opentraj_dir, cfg['H'])
        px, py = world_to_pixel(H_path, pos_x, pos_y)
    else:
        px, py = pos_x.copy(), pos_y.copy()

    native_w, native_h = cfg['native_res']
    if native_w != TARGET_W or native_h != TARGET_H:
        px = px * TARGET_W / native_w
        py = py * TARGET_H / native_h

    return frame_ids, ped_ids, px, py


def remove_out_of_bounds(frame_ids, ped_ids, px, py):
    px = np.round(px, 1)
    py = np.round(py, 1)
    mask = (px >= 0) & (px < TARGET_W) & (py >= 0) & (py < TARGET_H)
    return frame_ids[mask], ped_ids[mask], px[mask], py[mask]


def find_continuous_segments(unique_frames, step):
    if len(unique_frames) == 0:
        return []
    segments = []
    seg_start = 0
    for i in range(1, len(unique_frames)):
        gap = unique_frames[i] - unique_frames[i - 1]
        if gap > step * 2:
            segments.append(unique_frames[seg_start:i])
            seg_start = i
    segments.append(unique_frames[seg_start:])
    return segments


def check_clip_quality(clip_data, clip_frames):
    if len(clip_frames) < CLIP_LEN:
        return False

    all_peds = set(clip_data[:, 1].astype(int))
    if len(all_peds) < 2:
        return False

    for t in range(min(3, len(clip_frames))):
        frame = clip_frames[t]
        rows = clip_data[clip_data[:, 0] == frame]
        peds_in_frame = set(rows[:, 1].astype(int))
        if len(peds_in_frame) < 2:
            return False
        if t > 0:
            prev_frame = clip_frames[t - 1]
            prev_rows = clip_data[clip_data[:, 0] == prev_frame]
            prev_peds = {int(r[1]): r[2:4] for r in prev_rows}
            has_movement = False
            for r in rows:
                pid = int(r[1])
                if pid in prev_peds:
                    disp = np.sqrt((r[2] - prev_peds[pid][0])**2 + (r[3] - prev_peds[pid][1])**2)
                    if disp > 0.5:
                        has_movement = True
                        break
            if not has_movement:
                return False

    return True


def process_scene(scene_name, cfg, opentraj_dir, output_dir):
    traj_dir = os.path.join(output_dir, 'trajectories')
    clip_dir = os.path.join(output_dir, 'clips')
    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    frame_ids, ped_ids, px, py = convert_coordinates(scene_name, cfg, opentraj_dir)
    frame_ids, ped_ids, px, py = remove_out_of_bounds(frame_ids, ped_ids, px, py)

    data = np.column_stack([frame_ids, ped_ids, px, py])
    unique_frames = np.sort(np.unique(frame_ids))
    step = cfg['step']
    segments = find_continuous_segments(unique_frames, step)

    video_path = os.path.join(opentraj_dir, cfg['video'])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Warning: cannot open video {video_path}")
        cap = None

    clip_num = 0
    total_clips = 0
    discarded = 0

    for seg_frames in segments:
        for start in range(0, len(seg_frames) - CLIP_LEN + 1, CLIP_LEN):
            clip_frames = seg_frames[start:start + CLIP_LEN]
            if len(clip_frames) < CLIP_LEN:
                discarded += 1
                continue

            mask = np.isin(data[:, 0].astype(int), clip_frames)
            clip_data = data[mask]

            if not check_clip_quality(clip_data, clip_frames):
                discarded += 1
                continue

            clip_num += 1
            clip_id = f"{scene_name}{clip_num:02d}"

            unique_peds = sorted(set(clip_data[:, 1].astype(int)))
            id_map = {old: new for new, old in enumerate(unique_peds, start=1)}
            frame_map = {old: new for new, old in enumerate(clip_frames, start=1)}

            lines = []
            for row in clip_data:
                fid = frame_map[int(row[0])]
                pid = id_map[int(row[1])]
                x, y = row[2], row[3]
                lines.append(f"{fid} {pid} {x:.1f} {y:.1f}")

            lines.sort(key=lambda l: (int(l.split()[0]), int(l.split()[1])))
            txt_path = os.path.join(traj_dir, f"{clip_id}.txt")
            with open(txt_path, 'w') as f:
                f.write('\n'.join(lines) + '\n')

            if cap is not None:
                write_clip_video(cap, clip_frames, clip_id, clip_dir, cfg)
                write_visualization(cap, clip_frames, clip_data, frame_map, id_map, clip_id, vis_dir, cfg)

            total_clips += 1

    if cap is not None:
        cap.release()

    remaining_frames = sum(len(seg) % CLIP_LEN for seg in segments if len(seg) % CLIP_LEN > 0)
    print(f"  {scene_name}: {len(segments)} segments, {total_clips} clips, "
          f"{discarded} discarded, {len(np.unique(ped_ids))} total peds")
    return total_clips


def write_clip_video(cap, clip_frames, clip_id, clip_dir, cfg):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(clip_dir, f"{clip_id}.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, FPS_OUT, (TARGET_W, TARGET_H))
    for frame_num in clip_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))
            writer.write(frame)
        else:
            writer.write(np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8))
    writer.release()


def write_visualization(cap, clip_frames, clip_data, frame_map, id_map, clip_id, vis_dir, cfg):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(vis_dir, f"{clip_id}.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, FPS_OUT, (TARGET_W, TARGET_H))

    rng = np.random.default_rng(42)
    max_pid = max(id_map.values())
    colors = {pid: tuple(map(int, rng.integers(50, 255, 3))) for pid in range(1, max_pid + 1)}

    history = {pid: [] for pid in range(1, max_pid + 1)}

    for frame_num in clip_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))

        new_fid = frame_map[frame_num]
        rows = clip_data[clip_data[:, 0].astype(int) == frame_num]

        for row in rows:
            pid = id_map[int(row[1])]
            x, y = int(round(row[2])), int(round(row[3]))
            history[pid].append((x, y))

        for pid in range(1, max_pid + 1):
            if len(history[pid]) >= 2:
                pts = np.array(history[pid], dtype=np.int32)
                cv2.polylines(frame, [pts], False, colors[pid], 2, cv2.LINE_AA)
            if history[pid]:
                cx, cy = history[pid][-1]
                cv2.circle(frame, (cx, cy), 4, colors[pid], -1)
                cv2.putText(frame, str(pid), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[pid], 1, cv2.LINE_AA)

        writer.write(frame)
    writer.release()


def generate_backgrounds(opentraj_dir, output_dir):
    scenario_dir = os.path.join(output_dir, 'scenarios')
    os.makedirs(scenario_dir, exist_ok=True)

    for scene_name, cfg in SCENES.items():
        video_path = os.path.join(opentraj_dir, cfg['video'])
        obsmat_path = os.path.join(opentraj_dir, cfg['obsmat'])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  {scene_name}: cannot open video, skipping")
            continue

        frame_ids, ped_ids, px, py = convert_coordinates(scene_name, cfg, opentraj_dir)
        mask = (px >= 0) & (px < TARGET_W) & (py >= 0) & (py < TARGET_H)
        frame_ids, ped_ids, px, py = frame_ids[mask], ped_ids[mask], px[mask], py[mask]

        unique_frames = np.sort(np.unique(frame_ids))
        sampled_frames = unique_frames[::3]

        frames_list = []
        for frame_num in sampled_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (TARGET_W, TARGET_H))

            mask_img = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
            rows_mask = frame_ids == frame_num
            for x, y in zip(px[rows_mask], py[rows_mask]):
                cv2.circle(mask_img, (int(round(x)), int(round(y))), 15, 255, -1)

            inpainted = cv2.inpaint(frame, mask_img, 3, cv2.INPAINT_TELEA)
            frames_list.append(inpainted)

        cap.release()

        if frames_list:
            background = np.median(np.array(frames_list), axis=0).astype(np.uint8)
            out_path = os.path.join(scenario_dir, f"{scene_name}.png")
            cv2.imwrite(out_path, background)
            print(f"  {scene_name}: background saved ({len(frames_list)} frames used)")
        else:
            print(f"  {scene_name}: no frames available")


def generate_maps(output_dir):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, PolygonSelector
    from matplotlib.path import Path as MplPath

    scenario_dir = os.path.join(output_dir, 'scenarios')
    scenes = ['eth', 'hotel', 'univ', 'zara1', 'zara2']

    for scene_name in scenes:
        bg_path = os.path.join(scenario_dir, f"{scene_name}.png")
        map_path = os.path.join(scenario_dir, f"{scene_name}.npy")

        if not os.path.exists(bg_path):
            print(f"  {scene_name}: no background image, run --generate_backgrounds first")
            continue

        bg = cv2.imread(bg_path)
        bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

        if os.path.exists(map_path):
            area = np.load(map_path)
            print(f"  {scene_name}: loaded existing map")
        else:
            area = np.ones((TARGET_H, TARGET_W), dtype=np.uint8)

        state = {'mode': 'add', 'area': area, 'selector': None}

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(bg_rgb)
        overlay = np.zeros((*bg_rgb.shape[:2], 4))
        overlay[:, :, 2] = 1.0
        overlay[:, :, 3] = 0.3 * state['area']
        mask_display = ax.imshow(overlay)
        ax.set_title(f"{scene_name} - Mode: ADD walkable area (close window to save)")

        def update_display():
            overlay = np.zeros((*bg_rgb.shape[:2], 4))
            overlay[:, :, 2] = 1.0
            overlay[:, :, 3] = 0.3 * state['area']
            mask_display.set_data(overlay)
            fig.canvas.draw_idle()

        def on_select(verts):
            path = MplPath(verts)
            yy, xx = np.mgrid[:TARGET_H, :TARGET_W]
            points = np.column_stack([xx.ravel(), yy.ravel()])
            polygon_mask = path.contains_points(points).reshape(TARGET_H, TARGET_W)
            if state['mode'] == 'add':
                state['area'] = np.clip(state['area'] + polygon_mask.astype(np.uint8), 0, 1)
            else:
                state['area'] = state['area'] * (~polygon_mask).astype(np.uint8)
            update_display()

        def toggle_mode(event):
            state['mode'] = 'remove' if state['mode'] == 'add' else 'add'
            ax.set_title(f"{scene_name} - Mode: {'ADD' if state['mode'] == 'add' else 'REMOVE'} walkable area")
            fig.canvas.draw_idle()

        ax_btn = plt.axes([0.4, 0.01, 0.2, 0.04])
        btn = Button(ax_btn, 'Toggle Add/Remove')
        btn.on_clicked(toggle_mode)

        state['selector'] = PolygonSelector(ax, on_select, useblit=True)
        plt.show()

        np.save(map_path, state['area'])
        print(f"  {scene_name}: map saved to {map_path}")


def main():
    parser = argparse.ArgumentParser(description="ETH-UCY preprocessing")
    parser.add_argument('--opentraj_dir', type=str, default='../OpenTraj')
    parser.add_argument('--output_dir', type=str, default='data')
    parser.add_argument('--generate_backgrounds', action='store_true')
    parser.add_argument('--generate_maps', action='store_true')
    args = parser.parse_args()

    if args.generate_backgrounds:
        print("Generating background images...")
        generate_backgrounds(args.opentraj_dir, args.output_dir)
        return

    if args.generate_maps:
        print("Interactive walkable area annotation...")
        generate_maps(args.output_dir)
        return

    print(f"Processing ETH-UCY → {args.output_dir}/")
    print(f"Target resolution: {TARGET_W}×{TARGET_H}, clip length: {CLIP_LEN}")
    total = 0
    for scene_name, cfg in SCENES.items():
        n = process_scene(scene_name, cfg, args.opentraj_dir, args.output_dir)
        total += n
    print(f"\nTotal: {total} clips")


if __name__ == '__main__':
    main()
