"""
SFT 数据集构建

将标注好的数据转换为 Alpaca 格式的 SFT 训练数据:
{
  "instruction": 系统指令 (指定输出格式),
  "input": 场景上下文 + 人群描述 + 初始状态,
  "output": coordinate token 序列
}
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict

from tokenizer import build_tokenizer
from data.annotation.prompts import (
    SFT_INSTRUCTION_ABSOLUTE,
    SFT_INSTRUCTION_RELATIVE,
    SFT_INSTRUCTION_POLAR,
    SFT_INPUT_TEMPLATE,
)


def build_sft_dataset(
    annotations_path: str,
    config: dict,
    output_path: str,
    walkable_areas: Dict[str, str] = None,
):
    """
    构建SFT训练数据

    Args:
        annotations_path: 标注JSON路径
        config: 配置字典
        output_path: 输出路径
        walkable_areas: {subset: walkable_area_string} 字典
    """
    tokenizer = build_tokenizer(config)
    walkable_areas = walkable_areas or {}

    with open(annotations_path) as f:
        annotations = json.load(f)

    # 选择对应的instruction模板
    mode = config["tokenizer"]["mode"]
    if mode == "absolute":
        cfg_tok = config["tokenizer"]["absolute"]
        instruction = SFT_INSTRUCTION_ABSOLUTE.format(
            x_bins=cfg_tok["x_bins"], y_bins=cfg_tok["y_bins"]
        )
    elif mode == "relative":
        instruction = SFT_INSTRUCTION_RELATIVE
    elif mode == "polar":
        instruction = SFT_INSTRUCTION_POLAR
    else:
        raise ValueError(f"Unknown mode: {mode}")

    sft_data = []
    for ann in annotations:
        trajectories = ann["trajectories"]  # {pid: [[x,y], ...]}
        ped_ids = sorted(trajectories.keys(), key=lambda x: int(x))
        num_peds = len(ped_ids)
        num_steps = max(len(v) for v in trajectories.values())

        # 构建轨迹数组 (N, T, 2)
        traj_array = np.full((num_peds, num_steps, 2), -1.0)
        for i, pid in enumerate(ped_ids):
            traj = trajectories[pid]
            for t, pos in enumerate(traj):
                traj_array[i, t] = pos

        # tokenize
        tokens = tokenizer.tokenize(traj_array)
        response = "".join(tokens)

        # 构建初始状态描述
        initial_states_text = _format_initial_states(ann.get("initial_states", {}), ped_ids)

        # 构建input
        grid_h = config["data"]["resolution"][0] // config["tokenizer"].get("absolute", {}).get("bin_size", 5)
        grid_w = config["data"]["resolution"][1] // config["tokenizer"].get("absolute", {}).get("bin_size", 5)
        walkable = walkable_areas.get(ann["subset"], "1" * (grid_h * grid_w))

        sft_input = SFT_INPUT_TEMPLATE.format(
            grid_h=grid_h,
            grid_w=grid_w,
            walkable_area=walkable,
            scenario_description=ann["scenario_description"],
            crowd_dynamics=ann["crowd_dynamics"],
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
                "dataset": ann["dataset"],
                "subset": ann["subset"],
                "num_pedestrians": num_peds,
                "num_timesteps": num_steps,
                "token_count": len(tokens),
                "tokenizer_mode": mode,
                "is_augmented": ann.get("is_augmented", False),
            },
        })

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    # 统计
    token_counts = [d["metadata"]["token_count"] for d in sft_data]
    print(f"SFT dataset built: {len(sft_data)} samples")
    print(f"  Token count: min={min(token_counts)}, max={max(token_counts)}, "
          f"avg={np.mean(token_counts):.0f}")
    print(f"  Saved to {output_path}")

    return sft_data


def _format_initial_states(initial_states: Dict, ped_ids: List[str]) -> str:
    lines = []
    for pid in ped_ids:
        state = initial_states.get(pid, initial_states.get(int(pid), None))
        if state:
            pos = state["position"]
            vel = state.get("velocity", [0, 0])
            lines.append(
                f"  Ped {pid}: pos=({pos[0]:.1f}, {pos[1]:.1f}), "
                f"vel=({vel[0]:.1f}, {vel[1]:.1f})"
            )
        else:
            lines.append(f"  Ped {pid}: not present in first frame")
    return "\n".join(lines)


def split_train_test(
    sft_path: str,
    train_ratio: float = 0.9,
    output_dir: str = None,
    seed: int = 42,
):
    """按比例切分训练/测试集, 确保同一clip的augmented变体不跨集"""
    with open(sft_path) as f:
        data = json.load(f)

    # 提取原始clip_id (去掉 _varX 后缀)
    def base_clip_id(clip_id):
        if "_var" in clip_id:
            return clip_id[:clip_id.rfind("_var")]
        return clip_id

    base_ids = sorted(set(base_clip_id(d["metadata"]["clip_id"]) for d in data))
    np.random.seed(seed)
    np.random.shuffle(base_ids)

    split_idx = int(len(base_ids) * train_ratio)
    train_ids = set(base_ids[:split_idx])

    train_data = [d for d in data if base_clip_id(d["metadata"]["clip_id"]) in train_ids]
    test_data = [d for d in data if base_clip_id(d["metadata"]["clip_id"]) not in train_ids]

    output_dir = output_dir or str(Path(sft_path).parent)
    train_path = f"{output_dir}/train.json"
    test_path = f"{output_dir}/test.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"Split: {len(train_data)} train, {len(test_data)} test")
    print(f"  Train clips: {len(train_ids)}, Test clips: {len(base_ids) - len(train_ids)}")
    return train_path, test_path
