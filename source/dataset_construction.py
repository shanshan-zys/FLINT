"""
Dataset construction: load clip data + coordinate tokenizer + build SFT samples.

Main output uses coordinate tokens <x_i><y_j>; metadata.raw_prompt contains the
ablation version with plain x,y;x,y;... format (separate instruction + output).
"""

import os
import re
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict


ABSENT = -1.0


class CoordTokenizer:

    def __init__(self, resolution=(480, 640), bin_size=5):
        self.H, self.W = resolution
        self.bin_size = bin_size
        self.x_bins = self.W // bin_size
        self.y_bins = self.H // bin_size

        self.x_tokens = [f"<x_{i}>" for i in range(self.x_bins + 1)]
        self.y_tokens = [f"<y_{j}>" for j in range(self.y_bins + 1)]
        self.vocab = self.x_tokens + self.y_tokens

        self.x_centers = np.array(
            [0.0] + [(i - 0.5) * bin_size for i in range(1, self.x_bins + 1)]
        )
        self.y_centers = np.array(
            [0.0] + [(j - 0.5) * bin_size for j in range(1, self.y_bins + 1)]
        )

    def tokenize(self, trajectories: np.ndarray) -> List[str]:
        N, T, _ = trajectories.shape
        tokens = []
        for t in range(T):
            for n in range(N):
                x, y = trajectories[n, t]
                if x == ABSENT or y == ABSENT:
                    tokens.extend(["<x_0>", "<y_0>"])
                else:
                    xi = int(np.clip(np.round(x / self.bin_size), 1, self.x_bins))
                    yj = int(np.clip(np.round(y / self.bin_size), 1, self.y_bins))
                    tokens.extend([f"<x_{xi}>", f"<y_{yj}>"])
        return tokens

    def detokenize(self, tokens: List[str], num_agents: int, num_steps: int) -> np.ndarray:
        traj = np.full((num_agents, num_steps, 2), ABSENT)
        idx = 0
        for t in range(num_steps):
            for n in range(num_agents):
                if idx + 1 >= len(tokens):
                    break
                x_tok, y_tok = tokens[idx], tokens[idx + 1]
                idx += 2
                try:
                    xi = int(x_tok.strip("<>").split("_")[1])
                    yj = int(y_tok.strip("<>").split("_")[1])
                except (ValueError, IndexError):
                    continue
                if xi == 0 or yj == 0:
                    continue
                traj[n, t, 0] = self.x_centers[xi]
                traj[n, t, 1] = self.y_centers[yj]
        return traj


def encode_raw_numbers(trajectories: np.ndarray) -> str:
    N, T, _ = trajectories.shape
    parts = []
    for t in range(T):
        for n in range(N):
            x, y = trajectories[n, t]
            if x == ABSENT or y == ABSENT:
                parts.append("0,0")
            else:
                parts.append(f"{int(round(x))},{int(round(y))}")
    return ";".join(parts)


def decode_raw_numbers(text: str, num_agents: int, num_steps: int) -> np.ndarray:
    traj = np.full((num_agents, num_steps, 2), ABSENT)
    parts = text.strip().split(";")
    idx = 0
    for t in range(num_steps):
        for n in range(num_agents):
            if idx >= len(parts):
                break
            pair = parts[idx].strip()
            idx += 1
            try:
                x_str, y_str = pair.split(",")
                x, y = int(x_str), int(y_str)
            except (ValueError, IndexError):
                continue
            if x == 0 and y == 0:
                continue
            traj[n, t, 0] = float(x)
            traj[n, t, 1] = float(y)
    return traj


def _extract_subset(clip_id):
    m = re.match(r'(eth|hotel|univ|zara1|zara2)', clip_id)
    return m.group(1) if m else None


def load_clip_file(txt_path):
    clip_id = Path(txt_path).stem
    subset = _extract_subset(clip_id)
    data = np.loadtxt(txt_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    frames = data[:, 0].astype(int)
    ped_ids = data[:, 1].astype(int)
    positions = data[:, 2:4]

    population = int(ped_ids.max())
    num_frames = int(frames.max())

    trajectories = {}
    initial_states = {}
    for pid in range(1, population + 1):
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
        "num_pedestrians": population,
        "trajectories": trajectories,
        "initial_states": initial_states,
    }


SFT_INSTRUCTION_COORD = """You are required to perform a crowd simulation task by generating individual pedestrian trajectories across continuous time frames within a given scenario. You will be provided with a walkable area map, a scenario description, a crowd dynamics description, population count, frame count, and initial states.

The walkable area is represented as a {grid_w}x{grid_h} (Width x Height) binary grid, downsampled from the original {W}x{H} pixel scenario by a factor of {grid_size}. Each cell corresponds to a {grid_size}x{grid_size} pixel region. Cells marked 1 are walkable; cells marked 0 are obstacles. All generated positions must remain within walkable cells.

Not all individuals are present in the first frame. Some may enter the scene from walkable edges at later frames; infer their appearance time and entry position from the crowd dynamics description. Once an individual exits the scene, they are permanently absent and must not reappear in any subsequent frame. The total population, movement duration, motion patterns, and flow directions must be consistent with the provided descriptions.

Generate trajectories as a single unbroken string of coordinate tokens. Output is organized frame by frame: for each frame, output all pedestrians' positions sequentially (Ped 1, Ped 2, ..., Ped N), then proceed to the next frame. Each position is represented as a token pair <x_i><y_j>, where the coordinate origin is at the top-left corner, x increases from left to right (tokens <x_1> to <x_{x_bins}>, bin size {bin_size}), and y increases from top to bottom (tokens <y_1> to <y_{y_bins}>). Use <x_0><y_0> as a placeholder for frames where the individual has not yet appeared or has already left.

Example for N=3, T=2 (Ped 3 not yet appeared in frame 1, enters in frame 2):
Frame 1: Ped1=<x_12><y_34> Ped2=<x_55><y_21> Ped3=<x_0><y_0>
Frame 2: Ped1=<x_13><y_35> Ped2=<x_54><y_22> Ped3=<x_99><y_88>
Actual output: <x_12><y_34><x_55><y_21><x_0><y_0><x_13><y_35><x_54><y_22><x_99><y_88>

Do not add any explanations, spaces, newlines, or text outside the token string."""

SFT_INSTRUCTION_RAW = """You are required to perform a crowd simulation task by generating individual pedestrian trajectories across continuous time frames within a given scenario. You will be provided with a walkable area map, a scenario description, a crowd dynamics description, population count, frame count, and initial states.

The walkable area is represented as a {grid_w}x{grid_h} (Width x Height) binary grid, downsampled from the original {W}x{H} pixel scenario by a factor of {grid_size}. Each cell corresponds to a {grid_size}x{grid_size} pixel region. Cells marked 1 are walkable; cells marked 0 are obstacles. All generated positions must remain within walkable cells.

Not all individuals are present in the first frame. Some may enter the scene from walkable edges at later frames; infer their appearance time and entry position from the crowd dynamics description. Once an individual exits the scene, they are permanently absent and must not reappear in any subsequent frame. The total population, movement duration, motion patterns, and flow directions must be consistent with the provided descriptions.

Generate trajectories as semicolon-separated coordinate pairs. Output is organized frame by frame: for each frame, output all pedestrians' positions sequentially (Ped 1, Ped 2, ..., Ped N), then proceed to the next frame. Each position is an x,y pair in pixel coordinates, where the coordinate origin is at the top-left corner, x increases from left to right (0 to {W}), and y increases from top to bottom (0 to {H}). Use 0,0 as a placeholder for frames where the individual has not yet appeared or has already left.

Example for N=3, T=2 (Ped 3 not yet appeared in frame 1, enters in frame 2):
Frame 1: Ped1=58,170 Ped2=274,105 Ped3=0,0
Frame 2: Ped1=63,173 Ped2=268,108 Ped3=495,440
Actual output: 58,170;274,105;0,0;63,173;268,108;495,440

Do not add any explanations, spaces, newlines, or text outside the coordinate string."""

SFT_INPUT_TEMPLATE = """Walkable area:
{walkable_grid}

Scenario: {scenario_description}

Crowd dynamics: {crowd_description}

Population: {num_pedestrians}
Frames: {num_frames}

Initial states (frame 1):
{initial_states}"""


def _format_initial_states(initial_states: Dict, ped_ids: List) -> str:
    lines = []
    for pid in ped_ids:
        state = initial_states.get(pid, initial_states.get(int(pid), None))
        if state:
            pos = state["position"]
            vel = state.get("velocity", [0, 0])
            lines.append(f"  Ped {pid}: pos=({pos[0]:.1f},{pos[1]:.1f}), vel=({vel[0]:.1f},{vel[1]:.1f})")
        else:
            lines.append(f"  Ped {pid}: enters later (not in frame 1)")
    return "\n".join(lines)


def _load_walkable_grid(data_dir, subset, grid_size=10):
    map_path = os.path.join(data_dir, subset, f"{subset}.npy")
    if not os.path.exists(map_path):
        return None, ""
    map_full = np.load(map_path)
    map_grid = map_full[::grid_size, ::grid_size]
    grid_str = "\n".join("".join(str(int(x)) for x in row) for row in map_grid)
    return map_full, grid_str


def build_sft_dataset(data_dir: str, output_path: str,
                      bin_size: int = 5, resolution: tuple = (480, 640), map_grid_size: int = 10):
    H, W = resolution
    tokenizer = CoordTokenizer(resolution=resolution, bin_size=bin_size)
    grid_h, grid_w = H // map_grid_size, W // map_grid_size

    fmt_kwargs = dict(
        W=W, H=H, grid_size=map_grid_size, grid_w=grid_w, grid_h=grid_h,
    )
    instruction_coord = SFT_INSTRUCTION_COORD.format(
        **fmt_kwargs, bin_size=bin_size,
        x_bins=tokenizer.x_bins, y_bins=tokenizer.y_bins,
    )
    instruction_raw = SFT_INSTRUCTION_RAW.format(**fmt_kwargs)

    subsets = ["eth", "hotel", "univ", "zara1", "zara2"]
    sft_data = []

    for subset in subsets:
        json_path = Path(data_dir) / subset / f"{subset}.json"
        if not json_path.exists():
            print(f"  {subset}: no annotation json, skipping")
            continue

        with open(json_path) as f:
            ann = json.load(f)

        scenario_desc = ann.get("scenario", "")
        crowd_descs = ann.get("crowd", {})

        map_full, grid_str = _load_walkable_grid(data_dir, subset, map_grid_size)
        if map_full is None:
            print(f"  {subset}: no walkable area map, skipping")
            continue

        traj_dir = Path(data_dir) / subset / "trajectories"
        if not traj_dir.exists():
            print(f"  {subset}: no trajectories dir, skipping")
            continue

        clip_count = 0
        for txt_file in sorted(traj_dir.glob("*.txt")):
            clip = load_clip_file(str(txt_file))
            clip_id = clip["clip_id"]
            crowd_desc = crowd_descs.get(clip_id, "")

            ped_ids = sorted(clip["trajectories"].keys(), key=lambda x: int(x))
            num_peds = len(ped_ids)
            num_frames = clip["num_frames"]

            traj_array = np.full((num_peds, num_frames, 2), ABSENT)
            for i, pid in enumerate(ped_ids):
                for t, pos in enumerate(clip["trajectories"][pid]):
                    traj_array[i, t] = pos

            initial_states_text = _format_initial_states(clip["initial_states"], ped_ids)

            sft_input = SFT_INPUT_TEMPLATE.format(
                scenario_description=scenario_desc,
                crowd_description=crowd_desc,
                num_pedestrians=num_peds,
                num_frames=num_frames,
                walkable_grid=grid_str,
                initial_states=initial_states_text,
            )

            tokens = tokenizer.tokenize(traj_array)
            output_text = "".join(tokens)

            raw_output = encode_raw_numbers(traj_array)

            sft_data.append({
                "instruction": instruction_coord,
                "input": sft_input,
                "output": output_text,
                "metadata": {
                    "clip_name": clip_id,
                    "population": num_peds,
                    "frames": num_frames,
                    "walkable_area": map_full.tolist(),
                    "scenario_description": scenario_desc,
                    "crowd_description": crowd_desc,
                    "trajectory": traj_array.tolist(),
                    "raw_prompt": {
                        "instruction": instruction_raw,
                        "input": sft_input,
                        "output": raw_output,
                    },
                },
            })
            clip_count += 1

        print(f"  {subset}: {clip_count} clips")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    print(f"Dataset: {len(sft_data)} samples -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SFT dataset from processed clips")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="data/eth-ucy/eth-ucy.json")
    parser.add_argument("--bin_size", type=int, default=5)
    parser.add_argument("--resolution_h", type=int, default=480)
    parser.add_argument("--resolution_w", type=int, default=640)
    parser.add_argument("--map_grid_size", type=int, default=10)
    args = parser.parse_args()

    build_sft_dataset(
        data_dir=args.data_dir,
        output_path=args.output,
        bin_size=args.bin_size,
        resolution=(args.resolution_h, args.resolution_w),
        map_grid_size=args.map_grid_size,
    )
