"""
标注工具：
  --prepare_scene_prompts  生成 5 个场景描述的标注 prompt（只需标 5 次）
  --prepare_prompts        生成每个 clip 的人群动态标注 prompt
  --api_annotate           调用 Gemini API 自动标注（支持发送视频）
  --merge_scenes           将场景描述合并到 clip 标注中
"""

import os
import re
import json
import argparse
import numpy as np
from pathlib import Path


SCENE_DESCRIPTION_PROMPT = """You are an expert in pedestrian scene analysis.

Below are trajectory samples from the "{subset}" scene of the ETH-UCY pedestrian dataset.
Image resolution: 480x640 pixels (H x W). Recording is from a fixed overhead/elevated camera.

Here are representative trajectories from multiple clips in this scene to give you spatial context:

{sample_trajectories}

Describe the SCENE (not the people) in ONE paragraph (30-50 words). Focus on:
- Type of location (plaza, street, hotel entrance, campus pathway, etc.)
- Layout of walkable areas (sidewalks, pathways, open space)
- Key landmarks or boundaries (buildings, fences, roads, storefronts)
- Overall geometry (narrow corridor, wide open area, intersection, etc.)

This description will be reused for ALL clips in this scene, so describe the fixed environment only."""


CROWD_DYNAMICS_PROMPT = """You are analyzing a pedestrian trajectory clip from a surveillance dataset.
The scene image resolution is 480x640 pixels (H x W). Each clip lasts 10 seconds (25 frames at 2.5 FPS).

Scene context: {scenario_description}

Below are all pedestrian trajectories in this clip, given as pixel coordinates.
Absent positions (pedestrian not visible) are omitted.

{trajectory_text}

Write a detailed paragraph (100-150 words) describing the crowd dynamics. Cover:
1. Overall density and number of pedestrians
2. Dominant movement directions (use spatial terms: left-to-right, top-to-bottom, etc.)
3. Entry/exit regions of the scene
4. Group behaviors, interactions, near-misses, or avoidance maneuvers
5. Speed patterns (fast, slow, stationary)
6. Any notable temporal changes during the clip

Be specific about positions using terms like "upper-left", "center", "bottom-right"."""


def _extract_subset(clip_id):
    m = re.match(r'(eth|hotel|univ|zara1|zara2)', clip_id)
    return m.group(1) if m else None


def load_clip(txt_path):
    clip_id = Path(txt_path).stem
    subset = _extract_subset(clip_id)
    data = np.loadtxt(txt_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    frames = data[:, 0].astype(int)
    ped_ids = data[:, 1].astype(int)
    positions = data[:, 2:4]
    unique_peds = sorted(set(ped_ids))
    unique_frames = sorted(set(frames))
    num_frames = max(unique_frames)

    trajectories = {}
    initial_states = {}
    for pid in unique_peds:
        pid_mask = ped_ids == pid
        pid_frames = frames[pid_mask]
        pid_pos = positions[pid_mask]
        traj = []
        for f in range(1, num_frames + 1):
            idx = np.where(pid_frames == f)[0]
            traj.append(pid_pos[idx[0]].tolist() if len(idx) > 0 else [-1.0, -1.0])
        trajectories[int(pid)] = traj

        if traj[0][0] >= 0:
            vx = traj[1][0] - traj[0][0] if len(traj) > 1 and traj[1][0] >= 0 else 0.0
            vy = traj[1][1] - traj[0][1] if len(traj) > 1 and traj[1][0] >= 0 else 0.0
            initial_states[int(pid)] = {"position": traj[0], "velocity": [vx, vy]}

    return {
        "clip_id": clip_id,
        "subset": subset,
        "num_frames": num_frames,
        "num_pedestrians": len(unique_peds),
        "trajectories": trajectories,
        "initial_states": initial_states,
    }


def load_all_clips(data_dir, subset_filter=None):
    clips = []
    for f in sorted(Path(data_dir).glob("*.txt")):
        clip = load_clip(str(f))
        if subset_filter and clip["subset"] != subset_filter:
            continue
        clips.append(clip)
    return clips


def _format_traj_text(clip):
    lines = ["Trajectory coordinates (pixel, 480x640):"]
    for pid, traj in clip["trajectories"].items():
        valid = [(t, x, y) for t, (x, y) in enumerate(traj, 1) if x >= 0]
        if not valid:
            continue
        coords = " ".join(f"({x:.0f},{y:.0f})" for _, x, y in valid)
        f_start, f_end = valid[0][0], valid[-1][0]
        lines.append(f"Ped {pid} (frames {f_start}-{f_end}): {coords}")
    return "\n".join(lines)


def prepare_scene_prompts(data_dir, output_dir, clip_length=25):
    os.makedirs(output_dir, exist_ok=True)
    subsets = ["eth", "hotel", "univ", "zara1", "zara2"]

    for subset in subsets:
        clips = load_all_clips(data_dir, subset_filter=subset)
        if not clips:
            print(f"  {subset}: no data found, skipping")
            continue

        sample_clips = clips[:min(3, len(clips))]
        sample_text = ""
        for i, clip in enumerate(sample_clips):
            sample_text += f"\n--- Clip {i + 1} ({clip['num_pedestrians']} pedestrians) ---\n"
            sample_text += _format_traj_text(clip) + "\n"

        prompt = SCENE_DESCRIPTION_PROMPT.format(
            subset=subset,
            sample_trajectories=sample_text,
        )

        fname = f"scene_{subset}.txt"
        with open(os.path.join(output_dir, fname), "w") as f:
            f.write(prompt)
        print(f"  {subset}: scene prompt → {fname}")

    print(f"\n5 scene prompt files saved to {output_dir}/")
    print("Annotate each file (paste to Gemini or edit manually),")
    print("then save results to scene_descriptions.json:")
    print('  {"eth": "...", "hotel": "...", "univ": "...", "zara1": "...", "zara2": "..."}')


def format_clip_prompt(clip, scene_desc=""):
    traj_text = _format_traj_text(clip)
    return CROWD_DYNAMICS_PROMPT.format(
        scenario_description=scene_desc,
        trajectory_text=traj_text,
    )


def prepare_clip_prompts(data_dir, output_dir, scene_desc_path=None, clip_length=25):
    os.makedirs(output_dir, exist_ok=True)
    subsets = ["eth", "hotel", "univ", "zara1", "zara2"]

    scene_descs = {}
    if scene_desc_path and os.path.exists(scene_desc_path):
        with open(scene_desc_path) as f:
            scene_descs = json.load(f)
        print(f"Loaded scene descriptions from {scene_desc_path}")
    else:
        print("Warning: no scene_descriptions.json provided, prompts will lack scene context")

    total = 0
    for subset in subsets:
        clips = load_all_clips(data_dir, subset_filter=subset)
        scene_desc = scene_descs.get(subset, "")
        for clip in clips:
            prompt_text = format_clip_prompt(clip, scene_desc)
            fname = f"{clip['clip_id']}.txt"
            with open(os.path.join(output_dir, fname), "w") as f:
                f.write(prompt_text)
            total += 1
        print(f"  {subset}: {len(clips)} clips")

    print(f"Total: {total} clip prompt files saved to {output_dir}/")


def annotate_with_api(data_dir, output_path, scene_desc_path=None,
                      api_key=None, model="gemini-2.5-flash", clip_length=25,
                      video_dir=None):
    import google.generativeai as genai
    genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
    client = genai.GenerativeModel(model)

    subsets = ["eth", "hotel", "univ", "zara1", "zara2"]

    scene_descs = {}
    if scene_desc_path and os.path.exists(scene_desc_path):
        with open(scene_desc_path) as f:
            scene_descs = json.load(f)

    all_annotations = []
    checkpoint_path = output_path + ".checkpoint.json"
    done_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            all_annotations = json.load(f)
            done_ids = {a["clip_id"] for a in all_annotations}
        print(f"Resuming from checkpoint: {len(done_ids)} already done")

    for subset in subsets:
        clips = load_all_clips(data_dir, subset_filter=subset)
        scene_desc = scene_descs.get(subset, "")

        for clip in clips:
            if clip["clip_id"] in done_ids:
                continue

            prompt_text = format_clip_prompt(clip, scene_desc)

            try:
                contents = [prompt_text]
                if video_dir:
                    video_path = Path(video_dir) / f"{clip['clip_id']}.mp4"
                    if video_path.exists():
                        video_file = genai.upload_file(str(video_path))
                        contents = [video_file, prompt_text]

                resp = client.generate_content(
                    contents,
                    generation_config={"temperature": 0.3, "max_output_tokens": 500},
                )
                crowd_dynamics = resp.text.strip()
            except Exception as e:
                print(f"  Error on {clip['clip_id']}: {e}")
                crowd_dynamics = ""

            clip["crowd_dynamics"] = crowd_dynamics
            clip["scenario_description"] = scene_desc
            all_annotations.append(clip)

            print(f"  [{len(all_annotations)}] {clip['clip_id']}: {len(crowd_dynamics)} chars")

            if len(all_annotations) % 10 == 0:
                with open(checkpoint_path, "w") as f:
                    json.dump(all_annotations, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=2)
    print(f"Annotations saved to {output_path}: {len(all_annotations)} clips")


def merge_scenes(annotations_path, scene_desc_path, output_path=None):
    with open(annotations_path) as f:
        annotations = json.load(f)
    with open(scene_desc_path) as f:
        scene_descs = json.load(f)

    for ann in annotations:
        ann["scenario_description"] = scene_descs.get(ann["subset"], "")

    out = output_path or annotations_path
    with open(out, "w") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    print(f"Merged scene descriptions into {len(annotations)} clips → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLINT annotation tools")
    parser.add_argument("--prepare_scene_prompts", action="store_true",
                        help="Generate 5 scene description prompts (annotate once per scene)")
    parser.add_argument("--prepare_prompts", action="store_true",
                        help="Generate per-clip crowd dynamics prompts")
    parser.add_argument("--api_annotate", action="store_true",
                        help="Annotate crowd dynamics via Gemini API")
    parser.add_argument("--merge_scenes", action="store_true",
                        help="Merge scene descriptions into clip annotations")
    parser.add_argument("--data_dir", type=str, default="data/trajectories")
    parser.add_argument("--video_dir", type=str, default="data/clips",
                        help="Video clip directory (for multimodal API annotation)")
    parser.add_argument("--output", type=str, default="prompts/")
    parser.add_argument("--scene_descriptions", type=str, default="scene_descriptions.json",
                        help="JSON file with {subset: description} for 5 scenes")
    parser.add_argument("--annotations", type=str, default=None,
                        help="Existing annotations JSON (for --merge_scenes)")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--clip_length", type=int, default=25)
    args = parser.parse_args()

    if args.prepare_scene_prompts:
        prepare_scene_prompts(args.data_dir, args.output, args.clip_length)
    elif args.prepare_prompts:
        prepare_clip_prompts(args.data_dir, args.output, args.scene_descriptions, args.clip_length)
    elif args.api_annotate:
        annotate_with_api(args.data_dir, args.output, args.scene_descriptions,
                          args.api_key, args.model, args.clip_length, args.video_dir)
    elif args.merge_scenes:
        assert args.annotations, "--annotations required for --merge_scenes"
        merge_scenes(args.annotations, args.scene_descriptions)
    else:
        parser.print_help()
