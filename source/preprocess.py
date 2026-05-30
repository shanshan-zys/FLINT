"""
ETH-UCY preprocessing: raw data -> unified pixel-coordinate clips

Output structure per scene:
  data/processed/{scene}/trajectories/{scene}XX.txt
  data/processed/{scene}/videos/{scene}XX.mp4
  data/processed/{scene}/visualization/{scene}XX.mp4
  data/processed/{scene}/{scene}.png   (background)
  data/processed/{scene}/{scene}.npy   (walkable area)

Usage:
  python preprocess.py --target_w 640 --target_h 480 --clip_len 25 --fps 2.5
  python preprocess.py --generate_backgrounds --target_w 640 --target_h 480
  python preprocess.py --generate_maps --target_w 640 --target_h 480
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path

SCENES = {
    'eth':   {'native_res': (640, 480), 'step': 6},
    'hotel': {'native_res': (720, 576), 'step': 10},
    'zara1': {'native_res': (720, 576), 'step': 10},
    'zara2': {'native_res': (720, 576), 'step': 10},
    'univ':  {'native_res': (720, 576), 'step': 10},
}


def load_obsmat(path):
    data = np.loadtxt(path)
    frame_ids = data[:, 0].astype(int)
    ped_ids = data[:, 1].astype(int)
    pos_x = data[:, 2]
    pos_y = data[:, 4]
    return frame_ids, ped_ids, pos_x, pos_y


def world_to_pixel(H_path, pos_x, pos_y, scene_name, native_w, native_h):
    H = np.loadtxt(H_path)
    H_inv = np.linalg.inv(H)
    n = len(pos_x)

    if scene_name == 'univ':
        world = np.vstack([pos_y, pos_x, np.ones(n)])
    else:
        world = np.vstack([pos_x, pos_y, np.ones(n)])

    pixel = H_inv @ world
    pixel /= pixel[2:3, :]
    
    if scene_name == 'univ':
        px = native_w + pixel[0] - 30
        py = native_h + pixel[1] - 30
    else:
        px = pixel[1]
        py = pixel[0]

    return px, py


def convert_coordinates(scene_name, cfg, original_dir, target_w, target_h):
    scene_dir = os.path.join(original_dir, scene_name)
    obsmat_path = os.path.join(scene_dir, 'obsmat.txt')
    frame_ids, ped_ids, pos_x, pos_y = load_obsmat(obsmat_path)

    native_w, native_h = cfg['native_res']
    H_path = os.path.join(scene_dir, 'H.txt')
    px, py = world_to_pixel(H_path, pos_x, pos_y, scene_name, native_w, native_h)

    if native_w != target_w or native_h != target_h:
        px = px * target_w / native_w
        py = py * target_h / native_h

    return frame_ids, ped_ids, px, py


def remove_out_of_bounds(frame_ids, ped_ids, px, py, target_w, target_h):
    px = np.round(px, 1)
    py = np.round(py, 1)
    mask = (px >= 0) & (px < target_w) & (py >= 0) & (py < target_h)
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


def check_clip_quality(clip_data, clip_frames, clip_len):
    if len(clip_frames) < clip_len:
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


def process_scene(scene_name, cfg, original_dir, output_dir,
                  target_w, target_h, clip_len, fps):
    scene_out = os.path.join(output_dir, scene_name)
    traj_dir = os.path.join(scene_out, 'trajectories')
    vid_dir = os.path.join(scene_out, 'videos')
    vis_dir = os.path.join(scene_out, 'visualization')
    os.makedirs(traj_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    frame_ids, ped_ids, px, py = convert_coordinates(
        scene_name, cfg, original_dir, target_w, target_h)
    frame_ids, ped_ids, px, py = remove_out_of_bounds(
        frame_ids, ped_ids, px, py, target_w, target_h)

    data = np.column_stack([frame_ids, ped_ids, px, py])
    unique_frames = np.sort(np.unique(frame_ids))
    step = cfg['step']
    segments = find_continuous_segments(unique_frames, step)

    scene_dir = os.path.join(original_dir, scene_name)
    video_path = os.path.join(scene_dir, 'video.avi')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Warning: cannot open video {video_path}")
        cap = None

    clip_num = 0
    total_clips = 0
    discarded = 0

    for seg_frames in segments:
        for start in range(0, len(seg_frames) - clip_len + 1, clip_len):
            clip_frames = seg_frames[start:start + clip_len]
            if len(clip_frames) < clip_len:
                discarded += 1
                continue

            mask = np.isin(data[:, 0].astype(int), clip_frames)
            clip_data = data[mask]

            if not check_clip_quality(clip_data, clip_frames, clip_len):
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
                write_clip_video(cap, clip_frames, clip_id, vid_dir,
                                 target_w, target_h, fps)
                write_visualization(cap, clip_frames, clip_data, frame_map,
                                    id_map, clip_id, vis_dir,
                                    target_w, target_h, fps)

            total_clips += 1

    if cap is not None:
        cap.release()

    print(f"  {scene_name}: {len(segments)} segments, {total_clips} clips, "
          f"{discarded} discarded, {len(np.unique(ped_ids))} total peds")
    return total_clips


def write_clip_video(cap, clip_frames, clip_id, out_dir,
                     target_w, target_h, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(out_dir, f"{clip_id}.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (target_w, target_h))
    for frame_num in clip_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (target_w, target_h))
            writer.write(frame)
        else:
            writer.write(np.zeros((target_h, target_w, 3), dtype=np.uint8))
    writer.release()


def write_visualization(cap, clip_frames, clip_data, frame_map, id_map,
                        clip_id, out_dir, target_w, target_h, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(out_dir, f"{clip_id}.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (target_w, target_h))

    rng = np.random.default_rng(42)
    max_pid = max(id_map.values())
    colors = {pid: tuple(map(int, rng.integers(50, 255, 3)))
              for pid in range(1, max_pid + 1)}

    history = {pid: [] for pid in range(1, max_pid + 1)}

    for frame_num in clip_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (target_w, target_h))

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
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[pid],
                            1, cv2.LINE_AA)

        writer.write(frame)
    writer.release()


def generate_backgrounds(original_dir, output_dir, target_w, target_h):
    for scene_name, cfg in SCENES.items():
        scene_dir = os.path.join(original_dir, scene_name)
        video_path = os.path.join(scene_dir, 'video.avi')
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  {scene_name}: cannot open video, skipping")
            continue

        frame_ids, ped_ids, px, py = convert_coordinates(
            scene_name, cfg, original_dir, target_w, target_h)
        mask = (px >= 0) & (px < target_w) & (py >= 0) & (py < target_h)
        frame_ids = frame_ids[mask]
        px, py = px[mask], py[mask]

        unique_frames = np.sort(np.unique(frame_ids))
        sampled_frames = unique_frames[::3]

        frames_list = []
        for frame_num in sampled_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (target_w, target_h))

            mask_img = np.zeros((target_h, target_w), dtype=np.uint8)
            rows_mask = frame_ids == frame_num
            for x, y in zip(px[rows_mask], py[rows_mask]):
                cv2.circle(mask_img, (int(round(x)), int(round(y))), 15, 255, -1)

            inpainted = cv2.inpaint(frame, mask_img, 3, cv2.INPAINT_TELEA)
            frames_list.append(inpainted)

        cap.release()

        if frames_list:
            scene_out = os.path.join(output_dir, scene_name)
            os.makedirs(scene_out, exist_ok=True)
            background = np.median(np.array(frames_list), axis=0).astype(np.uint8)
            out_path = os.path.join(scene_out, f"{scene_name}.png")
            cv2.imwrite(out_path, background)
            print(f"  {scene_name}: background saved ({len(frames_list)} frames)")
        else:
            print(f"  {scene_name}: no frames available")


def generate_maps(output_dir, target_w, target_h):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, PolygonSelector
    from matplotlib.path import Path as MplPath

    for scene_name in SCENES:
        scene_out = os.path.join(output_dir, scene_name)
        bg_path = os.path.join(scene_out, f"{scene_name}.png")
        map_path = os.path.join(scene_out, f"{scene_name}.npy")

        if not os.path.exists(bg_path):
            print(f"  {scene_name}: no background image, "
                  "run --generate_backgrounds first")
            continue

        bg = cv2.imread(bg_path)
        bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

        if os.path.exists(map_path):
            area = np.load(map_path)
            print(f"  {scene_name}: loaded existing map")
        else:
            area = np.ones((target_h, target_w), dtype=np.uint8)

        state = {'mode': 'add', 'area': area, 'selector': None}

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(bg_rgb)
        overlay = np.zeros((*bg_rgb.shape[:2], 4))
        overlay[:, :, 2] = 1.0
        overlay[:, :, 3] = 0.3 * state['area']
        mask_display = ax.imshow(overlay)
        ax.set_title(f"{scene_name} - Mode: ADD walkable area "
                     "(close window to save)")

        def update_display():
            overlay = np.zeros((*bg_rgb.shape[:2], 4))
            overlay[:, :, 2] = 1.0
            overlay[:, :, 3] = 0.3 * state['area']
            mask_display.set_data(overlay)
            fig.canvas.draw_idle()

        def on_select(verts):
            path = MplPath(verts)
            yy, xx = np.mgrid[:target_h, :target_w]
            points = np.column_stack([xx.ravel(), yy.ravel()])
            polygon_mask = path.contains_points(points).reshape(target_h,
                                                                target_w)
            if state['mode'] == 'add':
                state['area'] = np.clip(
                    state['area'] + polygon_mask.astype(np.uint8), 0, 1)
            else:
                state['area'] = (state['area']
                                 * (~polygon_mask).astype(np.uint8))
            update_display()

        def toggle_mode(event):
            state['mode'] = 'remove' if state['mode'] == 'add' else 'add'
            ax.set_title(
                f"{scene_name} - Mode: "
                f"{'ADD' if state['mode'] == 'add' else 'REMOVE'} "
                "walkable area")
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
    parser.add_argument('--original_dir', type=str, default='data/original')
    parser.add_argument('--output_dir', type=str, default='data/processed')
    parser.add_argument('--target_w', type=int, default=640)
    parser.add_argument('--target_h', type=int, default=480)
    parser.add_argument('--clip_len', type=int, default=25)
    parser.add_argument('--fps', type=float, default=2.5)
    parser.add_argument('--generate_backgrounds', action='store_true')
    parser.add_argument('--generate_maps', action='store_true')
    args = parser.parse_args()

    if args.generate_backgrounds:
        print("Generating background images...")
        generate_backgrounds(args.original_dir, args.output_dir,
                             args.target_w, args.target_h)
        return

    if args.generate_maps:
        print("Interactive walkable area annotation...")
        generate_maps(args.output_dir, args.target_w, args.target_h)
        return

    print(f"Processing ETH-UCY: {args.original_dir} -> {args.output_dir}")
    print(f"Target: {args.target_w}x{args.target_h}, clip length: {args.clip_len}")
    total = 0
    for scene_name, cfg in SCENES.items():
        n = process_scene(scene_name, cfg, args.original_dir, args.output_dir,
                          args.target_w, args.target_h, args.clip_len, args.fps)
        total += n
    print(f"\nTotal: {total} clips")


if __name__ == '__main__':
    main()
