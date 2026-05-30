"""
Annotation tools:
  --annotate_scenario  Annotate scenario descriptions via API (background image + trajectory samples)
  --annotate_crowd     Annotate crowd dynamics via API (video + trajectory data)
"""

import os
import re
import json
import time
import base64
import argparse
import numpy as np
from pathlib import Path
from openai import OpenAI


MODEL = "gemini-3.0-flash"

SCENARIO_PROMPT = """You are an expert in pedestrian scene analysis.

This is a background image from the "{subset}" scene of the ETH-UCY pedestrian dataset.
Image resolution: 640x480 pixels (Width x Height). The coordinate origin is at the top-left corner, with X increasing to the right and Y increasing downward.
The recording is from a fixed overhead/elevated camera.

Below are trajectory samples from the first clip of this scene to provide spatial context. Coordinates are in (X, Y) pixel format:

{sample_trajectories}

### Scenario Description
Describe the scenario shown in this background image in one concise paragraph (30-60 words), focusing specifically on:
- Type of location (plaza, sidewalk, hotel entrance, campus pathway, intersection, etc.)
- Layout and geometry of walkable areas (narrow corridor, wide open space, L-shaped path, etc.)
- Key landmarks or boundaries visible (buildings, fences, roads, storefronts, vegetation, etc.)

Describe only the fixed environment, not the pedestrians.

Example output:
"This is a wide, open university campus plaza featuring a major concrete pathway running diagonally from the top-left corner to the bottom-right region, serving as the primary walkable corridor. The space is strictly bounded by a prominent university building facade along the entire left margin and elevated grass lawns on the upper-right corner, restricting any pedestrian traversal beyond these physical boundaries."
"""


CROWD_PROMPT = """You are an expert in pedestrian crowd dynamics analysis.

This is a video clip "{clip_id}" from the "{subset}" scene of the ETH-UCY pedestrian dataset.
Video resolution: 640x480 pixels (Width x Height). The coordinate origin is at the top-left corner, with X increasing to the right and Y increasing downward.
Each clip lasts ~10 seconds (25 frames at 2.5 FPS).

Below are all pedestrian trajectories in this clip as pixel coordinates (X, Y). Absent positions are omitted:

{trajectory_text}

### Crowd Description
Provide a detailed description of the crowd motions and behaviors shown in the video in one paragraph of no more than 150 words. Focus on:
- Overall density (sparse, moderate, dense) and population size (small, medium, large)
- Dominant crowd flow directions (e.g., left to right, bidirectional, top-to-bottom)
- Main sources and destinations (e.g., from left edge, toward bottom-right corner)
- Collective dynamics (e.g., converging, diverging, lane formation, stopping)
- Density changes over time (e.g., increasing, stable, decreasing)
- Approximate frame range where main crowd activities occur

Do NOT describe individual pedestrian coordinates. Describe overall flow patterns and notable local dynamics.

Example output:
"The crowd exhibits a moderate density with approximately twenty active pedestrians establishing a bidirectional flow pattern. A larger group of fifteen individuals enters from the top-left boundary, moving vertically downward toward the bottom-left corner at a stable, casual pace. Simultaneously, a smaller group of five pedestrians enters from the bottom-right zone, traveling upward. Near the center of the coordinate system (around frames 10-18), these counter-flowing streams converge, creating a local lane formation where agents execute smooth, non-collision avoidance maneuvers to filter through gaps. Pedestrians along the right margin decelerate slightly to yield, while the main downward stream maintains its velocity. The temporal change is stable as both groups systematically clear the intersection by frame 22, leaving the central region completely sparse."
"""


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
    for pid in unique_peds:
        pid_mask = ped_ids == pid
        pid_frames = frames[pid_mask]
        pid_pos = positions[pid_mask]
        traj = []
        for f in range(1, num_frames + 1):
            idx = np.where(pid_frames == f)[0]
            traj.append(pid_pos[idx[0]].tolist() if len(idx) > 0 else [-1.0, -1.0])
        trajectories[int(pid)] = traj

    return {
        "clip_id": clip_id,
        "subset": subset,
        "num_frames": num_frames,
        "num_pedestrians": len(unique_peds),
        "trajectories": trajectories,
    }


def load_all_clips(data_dir, subset):
    traj_dir = Path(data_dir) / subset / "trajectories"
    if not traj_dir.exists():
        return []
    clips = []
    for f in sorted(traj_dir.glob("*.txt")):
        clips.append(load_clip(str(f)))
    return clips


def _format_traj_text(clip):
    lines = []
    for pid, traj in clip["trajectories"].items():
        valid = [(t, x, y) for t, (x, y) in enumerate(traj, 1) if x >= 0]
        if not valid:
            continue
        coords = " ".join(f"({x:.0f},{y:.0f})" for _, x, y in valid)
        f_start, f_end = valid[0][0], valid[-1][0]
        lines.append(f"Ped {pid} (frames {f_start}-{f_end}): {coords}")
    return "\n".join(lines)


def _encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _encode_video_base64(video_path):
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _load_existing_json(json_path):
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return {}


def _save_json(json_path, data):
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def annotate_scenario(data_dir, api_key, base_url, subsets):
    client = OpenAI(api_key=api_key, base_url=base_url)

    for subset in subsets:
        json_path = Path(data_dir) / subset / f"{subset}.json"
        result = _load_existing_json(json_path)

        if "scenario" in result:
            print(f"  {subset}: scenario already annotated, skipping")
            continue

        image_path = Path(data_dir) / subset / f"{subset}.png"
        if not image_path.exists():
            print(f"  {subset}: no background image found, skipping")
            continue

        clips = load_all_clips(data_dir, subset)
        if not clips:
            print(f"  {subset}: no trajectory data found, skipping")
            continue

        first_clip = clips[0]
        sample_text = _format_traj_text(first_clip)

        prompt = SCENARIO_PROMPT.format(
            subset=subset,
            sample_trajectories=sample_text,
        )

        image_b64 = _encode_image_base64(image_path)

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
                temperature=0.2,
                max_tokens=300,
            )
            description = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  {subset}: API error - {e}")
            continue

        result["scenario"] = description
        _save_json(json_path, result)
        print(f"  {subset}: scenario saved ({len(description)} chars)")
        time.sleep(1)


def annotate_crowd(data_dir, api_key, base_url, subsets):
    client = OpenAI(api_key=api_key, base_url=base_url)

    for subset in subsets:
        json_path = Path(data_dir) / subset / f"{subset}.json"
        result = _load_existing_json(json_path)

        if "crowd" not in result:
            result["crowd"] = {}

        clips = load_all_clips(data_dir, subset)
        if not clips:
            print(f"  {subset}: no trajectory data found, skipping")
            continue

        for clip in clips:
            clip_id = clip["clip_id"]
            if clip_id in result["crowd"]:
                print(f"  {clip_id}: already annotated, skipping")
                continue

            video_path = Path(data_dir) / subset / "videos" / f"{clip_id}.mp4"
            if not video_path.exists():
                print(f"  {clip_id}: no video found, skipping")
                continue

            traj_text = _format_traj_text(clip)
            prompt = CROWD_PROMPT.format(
                clip_id=clip_id,
                subset=subset,
                trajectory_text=traj_text,
            )

            video_b64 = _encode_video_base64(video_path)

            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    temperature=0.2,
                    max_tokens=500,
                )
                description = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"  {clip_id}: API error - {e}")
                continue

            result["crowd"][clip_id] = description
            _save_json(json_path, result)
            print(f"  {clip_id}: crowd saved ({len(description)} chars)")
            time.sleep(1)

        print(f"  {subset}: {len(result['crowd'])} clips annotated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLINT annotation tools")
    parser.add_argument("--annotate_scenario", action="store_true",
                        help="Annotate scenario descriptions via API (background image + trajectory samples)")
    parser.add_argument("--annotate_crowd", action="store_true",
                        help="Annotate crowd dynamics via API (video + trajectory data)")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--subsets", type=str, nargs="+",
                        default=["eth", "hotel", "univ", "zara1", "zara2"])
    args = parser.parse_args()

    if args.annotate_scenario:
        print("=== Annotating Scenarios ===")
        annotate_scenario(args.data_dir, args.api_key, args.base_url, args.subsets)

    if args.annotate_crowd:
        print("=== Annotating Crowd Dynamics ===")
        annotate_crowd(args.data_dir, args.api_key, args.base_url, args.subsets)
