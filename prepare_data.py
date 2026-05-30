"""
数据准备：加载 clip 数据 + coordinate tokenizer + SFT 数据集构建

支持两种输出模式：
  1. coord tokens: <x_i><y_j> 特殊 token（主模型）
  2. raw numbers: 45,30;12,55;0,0 逗号分隔坐标（消融对比）
"""

import os
import re
import json
import yaml
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


def load_clip_file(txt_path, clip_length=25):
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


def load_all_trajectories(data_dir: str, clip_length: int = 25) -> List[Dict]:
    txt_files = sorted(Path(data_dir).glob("*.txt"))
    all_clips = []
    counts = {}
    for f in txt_files:
        clip = load_clip_file(str(f), clip_length)
        all_clips.append(clip)
        subset = clip["subset"]
        counts[subset] = counts.get(subset, 0) + 1
    for subset, count in sorted(counts.items()):
        print(f"  {subset}: {count} clips")
    print(f"Total: {len(all_clips)} clips")
    return all_clips


SFT_INSTRUCTION_COORD = """You are a crowd simulation engine. Given scene context, walkable area, and initial conditions, generate realistic pedestrian trajectories as coordinate tokens.

{walkable_area}

Output rules:
1. Output a SINGLE unbroken string of coordinate tokens, no spaces between tokens
2. Use <x_i><y_j> for each pedestrian's position at each timestep
3. Use <x_0><y_0> when a pedestrian has not appeared or has left the scene
4. Order: for each timestep t=1..T, output all N pedestrians' positions sequentially
5. Coordinate system: 640x480 image, x increases rightward (1 to {x_bins}), y increases downward (1 to {y_bins})
6. Pedestrians must stay within the walkable area
7. Do NOT add any explanation, only output the token string"""

SFT_INSTRUCTION_RAW = """You are a crowd simulation engine. Given scene context, walkable area, and initial conditions, generate realistic pedestrian trajectories as numeric coordinates.

{walkable_area}

Output rules:
1. Output coordinates as comma-separated x,y pairs, with positions separated by semicolons
2. Use 0,0 when a pedestrian has not appeared or has left the scene
3. Order: for each timestep t=1..T, output all N pedestrians' positions sequentially
4. Coordinate system: pixel coordinates in a 640x480 image (x: 0-639, y: 0-479)
5. Pedestrians must stay within the walkable area
6. Do NOT add any explanation, only output the coordinate string
Example: 45,30;12,55;0,0;78,120"""

SFT_INPUT_TEMPLATE = """Scenario: {scenario_description}

Crowd dynamics: {crowd_dynamics}

Number of pedestrians: {num_pedestrians}
Number of timesteps: {num_timesteps}

Initial states:
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
            lines.append(f"  Ped {pid}: not present in first frame")
    return "\n".join(lines)


def _load_walkable_grid(scenario_dir, subset, grid_size=10):
    map_path = os.path.join(scenario_dir, f"{subset}.npy")
    if not os.path.exists(map_path):
        return ""
    map_full = np.load(map_path)
    map_grid = map_full[::grid_size, ::grid_size]
    h, w = map_grid.shape
    grid_str = "\n".join("".join(str(int(x)) for x in row) for row in map_grid)
    return (f"The walkable area is represented as a {h}x{w} grid "
            f"(each cell = {grid_size}x{grid_size} pixels in the 480x640 image).\n"
            f"1 = walkable, 0 = obstacle:\n{grid_str}")


def build_sft_dataset(annotations_path: str, config: dict, output_path: str, use_coord_tokens: bool = True):
    with open(annotations_path) as f:
        annotations = json.load(f)

    scenario_dir = config["data"].get("scenario_dir", "./data/scenarios")
    grid_size = config["data"].get("map_grid_size", 10)

    tokenizer = None
    if use_coord_tokens:
        tokenizer = CoordTokenizer(
            resolution=tuple(config["data"]["resolution"]),
            bin_size=config["tokenizer"]["bin_size"],
        )

    walkable_cache = {}
    sft_data = []
    for ann in annotations:
        subset = ann.get("subset", "")
        if subset not in walkable_cache:
            walkable_cache[subset] = _load_walkable_grid(scenario_dir, subset, grid_size)
        walkable_area = walkable_cache[subset]

        if use_coord_tokens:
            instruction = SFT_INSTRUCTION_COORD.format(
                walkable_area=walkable_area,
                x_bins=config["tokenizer"]["x_bins"],
                y_bins=config["tokenizer"]["y_bins"],
            )
        else:
            instruction = SFT_INSTRUCTION_RAW.format(walkable_area=walkable_area)

        trajectories = ann["trajectories"]
        ped_ids = sorted(trajectories.keys(), key=lambda x: int(x))
        num_peds = len(ped_ids)
        num_steps = max(len(v) for v in trajectories.values())

        traj_array = np.full((num_peds, num_steps, 2), ABSENT)
        for i, pid in enumerate(ped_ids):
            for t, pos in enumerate(trajectories[pid]):
                traj_array[i, t] = pos

        if use_coord_tokens:
            tokens = tokenizer.tokenize(traj_array)
            response = "".join(tokens)
        else:
            response = encode_raw_numbers(traj_array)

        initial_states_text = _format_initial_states(
            ann.get("initial_states", {}), ped_ids
        )

        sft_input = SFT_INPUT_TEMPLATE.format(
            scenario_description=ann.get("scenario_description", ""),
            crowd_dynamics=ann.get("crowd_dynamics", ""),
            num_pedestrians=num_peds,
            num_timesteps=num_steps,
            initial_states=initial_states_text,
        )

        sft_data.append({
            "instruction": instruction,
            "input": sft_input,
            "output": response,
            "metadata": {
                "clip_id": ann["clip_id"],
                "subset": subset,
                "num_pedestrians": num_peds,
                "num_timesteps": num_steps,
                "use_coord_tokens": use_coord_tokens,
            },
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"SFT dataset: {len(sft_data)} samples → {output_path}")
    return sft_data


def split_train_test(sft_path: str, train_ratio: float = 0.8, seed: int = 42):
    with open(sft_path) as f:
        data = json.load(f)

    def base_clip_id(clip_id):
        return clip_id[:clip_id.rfind("_var")] if "_var" in clip_id else clip_id

    base_ids = sorted(set(base_clip_id(d["metadata"]["clip_id"]) for d in data))
    np.random.seed(seed)
    np.random.shuffle(base_ids)

    split_idx = int(len(base_ids) * train_ratio)
    train_ids = set(base_ids[:split_idx])

    train_data = [d for d in data if base_clip_id(d["metadata"]["clip_id"]) in train_ids]
    test_data = [d for d in data if base_clip_id(d["metadata"]["clip_id"]) not in train_ids]

    output_dir = str(Path(sft_path).parent)
    train_path = f"{output_dir}/train.json"
    test_path = f"{output_dir}/test.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Split: {len(train_data)} train, {len(test_data)} test")
    return train_path, test_path


def test_tokenizer(config):
    print("=== Tokenizer Roundtrip Test ===")
    tok = CoordTokenizer(
        resolution=tuple(config["data"]["resolution"]),
        bin_size=config["tokenizer"]["bin_size"],
    )
    print(f"Resolution: {tok.W}x{tok.H}, bin_size: {tok.bin_size}")
    print(f"Vocab size: {len(tok.vocab)} ({tok.x_bins + 1} x-tokens + {tok.y_bins + 1} y-tokens)")

    np.random.seed(42)
    N, T = 5, 25
    traj = np.random.rand(N, T, 2) * np.array([tok.W, tok.H])
    traj[0, 10:15] = ABSENT
    traj[2, 0:3] = ABSENT

    tokens = tok.tokenize(traj)
    recovered = tok.detokenize(tokens, N, T)

    errors = []
    for n in range(N):
        for t in range(T):
            if traj[n, t, 0] == ABSENT:
                assert recovered[n, t, 0] == ABSENT, f"Absent mismatch at ({n},{t})"
            else:
                err = np.sqrt((traj[n, t, 0] - recovered[n, t, 0]) ** 2 +
                              (traj[n, t, 1] - recovered[n, t, 1]) ** 2)
                errors.append(err)

    errors = np.array(errors)
    print(f"Roundtrip error: mean={errors.mean():.2f}, max={errors.max():.2f}, "
          f"expected max={tok.bin_size * np.sqrt(2) / 2:.2f}")

    raw = encode_raw_numbers(traj)
    recovered_raw = decode_raw_numbers(raw, N, T)
    raw_errors = []
    for n in range(N):
        for t in range(T):
            if traj[n, t, 0] == ABSENT:
                continue
            err = np.sqrt((traj[n, t, 0] - recovered_raw[n, t, 0]) ** 2 +
                          (traj[n, t, 1] - recovered_raw[n, t, 1]) ** 2)
            raw_errors.append(err)
    raw_errors = np.array(raw_errors)
    print(f"Raw number roundtrip error: mean={raw_errors.mean():.2f}, max={raw_errors.max():.2f}")
    print("=== Test PASSED ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--annotations", type=str, help="Annotations JSON path")
    parser.add_argument("--output", type=str, default="data/sft.json")
    parser.add_argument("--no_coord_tokens", action="store_true")
    parser.add_argument("--split", action="store_true", help="Split into train/test")
    parser.add_argument("--test_tokenizer", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.test_tokenizer:
        test_tokenizer(config)
    elif args.split:
        split_train_test(args.annotations or args.output, config["data"]["train_ratio"])
    elif args.annotations:
        build_sft_dataset(args.annotations, config, args.output, not args.no_coord_tokens)
    else:
        parser.print_help()
