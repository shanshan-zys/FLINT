#!/usr/bin/env python
"""
Unified baseline runner for FLINT.

Runs baseline models on the ETH-UCY-Text test split and produces result JSON
compatible with evaluation.py.  All configuration is at the top of this file.

Usage:
  python baseline/run_baseline.py                     # run ALL enabled models
  python baseline/run_baseline.py --only api           # run API models only
  python baseline/run_baseline.py --only local         # run local models only
  python baseline/run_baseline.py --only gpt-5         # run a single model
  python baseline/run_baseline.py --only "gpt-5,gemini-3.0-flash"  # subset
"""

import os
import sys
import re
import json
import time
import random
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))

from dataset_construction import decode_raw_numbers

# ====================================================================
# Configuration  (edit here)
# ====================================================================

DATA_PATH = "data/eth-ucy-text.json"
OUTPUT_BASE = "outputs"
IMAGE_DIR = "data/processed"          # scene images for VLM
SEED = 42
NUM_SAMPLES = 1                       # K samples per clip
TEMPERATURE = 0.0
MAX_MODEL_LEN = 8192                  # vLLM max context length
DISABLE_THINKING = True               # disable Qwen3 thinking mode

# API settings
API_KEY = "bsk-9ea517c77d92cbda3e05b174e0b4d236"
API_BASE_URL = "http://llmapi.bilibili.co/v1"

# ── Model registry ────────────────────────────────────────────────
# Each entry: (model_key, model_type, model_name/path)
#   model_type: "api" | "local_llm" | "local_vlm" | "local_plm"
#
# Comment out any line to skip that model.
MODEL_DIR = "models"                  # local models root (relative to FLINT/)

MODELS = [
    # --- PLMs ---
    ("bert-base", "local_plm", f"{MODEL_DIR}/bert-base"),
    ("t5-base", "local_plm", f"{MODEL_DIR}/t5-base"),
    # --- Lightweight LLMs ---
    ("llama3-8b", "local_llm", f"{MODEL_DIR}/llama-3.1-8b"),
    ("qwen3-8b", "local_llm", f"{MODEL_DIR}/qwen3-8b"),
    # --- VLMs ---
    ("llava-next-7b", "local_vlm", f"{MODEL_DIR}/llava-next-7b"),
    ("qwen25-vl-7b", "local_vlm", f"{MODEL_DIR}/qwen2.5-vl-7b"),
    # --- General-purpose APIs ---
    ("deepseek-v4-flash", "api", "deepseek-v4-flash"),
    ("gemini3flash", "api", "gemini-3.0-flash"),
    ("gpt5", "api",  "gpt-5"),
    ("claude_opus46", "api", "claude-opus-4-6"),
]

# ====================================================================
# Helpers  (do not edit below unless needed)
# ====================================================================

ABSENT = -1.0


def get_test_split(data_path, seed=42):
    with open(data_path) as f:
        raw = json.load(f)
    indices = list(range(len(raw)))
    random.Random(seed).shuffle(indices)
    split = int(len(indices) * 0.8)
    test_idx = indices[split:]
    return [raw[i] for i in test_idx]


def format_alpaca_prompt(instruction, input_text):
    return (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"{input_text}\n\n"
        "### Response:\n"
    )


def strip_think(text):
    """Remove <think>...</think> from generated text."""
    m = re.match(r'<think>.*?</think>\s*', text, re.DOTALL)
    if m:
        return text[m.end():]
    return text


def parse_trajectory(generated_text, population, frames):
    """Parse raw number format output into [N][T][2] list."""
    generated_text = strip_think(generated_text)
    try:
        traj = decode_raw_numbers(generated_text, population, frames)
    except Exception:
        traj = np.full((population, frames, 2), ABSENT)

    result = []
    for n in range(population):
        ped_traj = []
        for t in range(frames):
            x, y = traj[n, t, 0], traj[n, t, 1]
            if x == ABSENT or y == ABSENT:
                ped_traj.append([-1.0, -1.0])
            else:
                ped_traj.append([float(x), float(y)])
        result.append(ped_traj)
    return result


def get_scene_from_clip(clip_name):
    """Extract scene name from clip_name (e.g., 'eth01' -> 'eth')."""
    return re.match(r'([a-z]+)\d+', clip_name).group(1)


def compact_json(obj, indent=2):
    """Compact JSON writer matching inference.py format."""
    COMPACT_KEYS = {"output_trajectories", "walkable_area", "output"}

    def _serialize(o, level):
        pad = " " * (indent * level)
        pad_inner = " " * (indent * (level + 1))
        if isinstance(o, dict):
            items = []
            for k, v in o.items():
                if k in COMPACT_KEYS:
                    items.append(
                        f'{pad_inner}{json.dumps(k)}: '
                        f'{json.dumps(v, separators=(",", ": "))}')
                else:
                    items.append(
                        f'{pad_inner}{json.dumps(k)}: '
                        f'{_serialize(v, level + 1)}')
            return "{\n" + ",\n".join(items) + f"\n{pad}}}"
        elif isinstance(o, list) and o and isinstance(o[0], dict):
            items = [f"{pad_inner}{_serialize(item, level + 1)}" for item in o]
            return "[\n" + ",\n".join(items) + f"\n{pad}]"
        else:
            return json.dumps(o)
    return _serialize(obj, 0) + "\n"


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}min"


# ====================================================================
# Model factory
# ====================================================================

def create_model(model_type, model_name):
    """Instantiate a model by type."""
    if model_type == "api":
        from baseline.api_models import APIModel
        return APIModel(
            model_name=model_name,
            api_key=API_KEY,
            base_url=API_BASE_URL,
            temperature=TEMPERATURE,
            disable_thinking=DISABLE_THINKING,
        )
    elif model_type == "local_llm":
        from baseline.local_models import LocalLLM
        return LocalLLM(
            model_name=model_name,
            temperature=TEMPERATURE,
            max_model_len=MAX_MODEL_LEN,
            disable_thinking=DISABLE_THINKING,
        )
    elif model_type == "local_vlm":
        from baseline.local_models import LocalVLM
        return LocalVLM(
            model_name=model_name,
            temperature=TEMPERATURE,
            max_model_len=MAX_MODEL_LEN,
            disable_thinking=DISABLE_THINKING,
        )
    elif model_type == "local_plm":
        from baseline.local_models import LocalPLM
        return LocalPLM(
            model_name=model_name,
            temperature=TEMPERATURE,
            disable_thinking=DISABLE_THINKING,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ====================================================================
# Run one model on test split
# ====================================================================

def run_one_model(model_key, model_type, model_name, test_samples):
    """Run inference for a single model and save results + stats."""
    task_name = f"baseline_{model_key}"
    print(f"\n{'=' * 70}")
    print(f"  Model: {model_name}  ({model_type})")
    print(f"  Task:  {task_name}")
    print(f"{'=' * 70}")

    # Create model
    model = create_model(model_type, model_name)
    model_info = model.get_model_info()
    for k, v in model_info.items():
        print(f"  {k}: {v}")
    print()

    is_vlm = model_type == "local_vlm"
    results = []
    per_sample_times = []
    total_start = time.time()

    for idx, sample in enumerate(test_samples):
        meta = sample["metadata"]
        population = meta["population"]
        frames = meta["frames"]
        clip_name = meta["clip_name"]

        raw_prompt = meta["raw_prompt"]
        instruction = raw_prompt["instruction"]
        input_text = raw_prompt["input"]

        if is_vlm:
            scene = get_scene_from_clip(clip_name)
            image_path = os.path.join(IMAGE_DIR, scene, f"{scene}.png")
            prompt = model.build_prompt(instruction, input_text)
        else:
            image_path = None
            prompt = format_alpaca_prompt(instruction, input_text)

        max_tokens = min(population * frames * 20, 4096)

        sample_start = time.time()

        output_trajectories = []
        for k in range(NUM_SAMPLES):
            generated = model.generate(prompt, max_tokens, image_path)
            traj = parse_trajectory(generated, population, frames)
            output_trajectories.append(traj)

            if idx == 0 and k == 0:
                print(f"  [DEBUG] First 200 chars: {generated[:200]}")

        sample_time = time.time() - sample_start
        per_sample_times.append(sample_time)

        print(f"  [{idx + 1}/{len(test_samples)}] {clip_name} "
              f"(pop={population}, frames={frames}) "
              f"time={format_time(sample_time)}")

        results.append({
            "clip_id": clip_name,
            "output_trajectories": output_trajectories,
            "metadata": {
                "population": population,
                "frames": frames,
                "walkable_area": meta["walkable_area"],
                "scenario_description": meta["scenario_description"],
                "crowd_description": meta["crowd_description"],
                "output": meta["trajectory"],
            }
        })

    total_time = time.time() - total_start

    # ── Save results ──
    out_dir = os.path.join(OUTPUT_BASE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task_name}.json")
    with open(out_path, "w") as f:
        f.write(compact_json(results))

    # ── Save stats ──
    stats = {
        "model_key": model_key,
        "model_name": model_name,
        "model_type": model_type,
        "model_info": model_info,
        "num_samples": len(test_samples),
        "num_generations_per_sample": NUM_SAMPLES,
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_sample_seconds": round(np.mean(per_sample_times), 2),
        "min_time_per_sample_seconds": round(np.min(per_sample_times), 2),
        "max_time_per_sample_seconds": round(np.max(per_sample_times), 2),
        "disable_thinking": DISABLE_THINKING,
        "temperature": TEMPERATURE,
    }
    stats_path = os.path.join(out_dir, f"{task_name}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Print summary ──
    print(f"\n  Total time:       {format_time(total_time)}")
    print(f"  Avg time/sample:  {format_time(np.mean(per_sample_times))}")
    print(f"  Results saved:    {out_path}")
    print(f"  Stats saved:      {stats_path}")

    return task_name


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run FLINT baseline models")
    parser.add_argument(
        "--only", type=str, default=None,
        help='Filter: "api", "local", "local_llm", "local_vlm", "local_plm", '
             'or comma-separated model keys/names '
             '(e.g. "gpt-5,qwen3_8b"). Default: run all.')
    args = parser.parse_args()

    # ── Filter models ──
    if args.only is None:
        selected = MODELS
    elif args.only == "api":
        selected = [(k, t, n) for k, t, n in MODELS if t == "api"]
    elif args.only == "local":
        selected = [(k, t, n) for k, t, n in MODELS if t != "api"]
    elif args.only in ("local_llm", "local_vlm", "local_plm"):
        selected = [(k, t, n) for k, t, n in MODELS if t == args.only]
    else:
        keys = set(s.strip() for s in args.only.split(","))
        selected = [(k, t, n) for k, t, n in MODELS
                     if k in keys or n in keys]

    if not selected:
        print(f"No models matched --only={args.only!r}")
        print(f"Available: {[k for k, _, _ in MODELS]}")
        return

    print(f"Models to run ({len(selected)}):")
    for key, mtype, mname in selected:
        print(f"  {key:20s}  {mtype:12s}  {mname}")

    # ── Load test split (shared across all models) ──
    test_samples = get_test_split(DATA_PATH, seed=SEED)
    print(f"\nTest split (seed={SEED}): {len(test_samples)} samples\n")

    # ── Run each model ──
    completed = []
    for key, mtype, mname in selected:
        try:
            task_name = run_one_model(key, mtype, mname, test_samples)
            completed.append(task_name)
        except Exception as e:
            print(f"\n  ERROR running {key} ({mname}): {e}")
            import traceback
            traceback.print_exc()
            continue

    # ── Final summary ──
    print(f"\n{'=' * 70}")
    print(f"All done. Completed {len(completed)}/{len(selected)} models:")
    for name in completed:
        print(f"  {name}")
    print(f"\nRun evaluation with:")
    print(f"  python source/evaluation.py --task_name <task_name> "
          f"--output_base {OUTPUT_BASE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
