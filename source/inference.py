"""
FLINT inference with vLLM: batch test-set generation or interactive terminal mode.

Requires merged model (run merge_lora.py first).
Uses VLLM_USE_V1=0 for per-request logits processor support.

Usage:
  # Batch mode (test split)
  VLLM_USE_V1=0 python source/inference.py batch \
      --data data/eth-ucy-text.json \
      --model outputs/merged/coord_ce-ep30-.../epoch_25 \
      --task_name coord_ce-ep30-...-25

  # Interactive mode
  VLLM_USE_V1=0 python source/inference.py interactive \
      --model outputs/merged/coord_ce-ep30-.../epoch_25
"""

import os
import re
import json
import random
import argparse
import numpy as np

os.environ["VLLM_USE_V1"] = "0"

from vllm import LLM, SamplingParams

from dataset_construction import CoordTokenizer, decode_raw_numbers, encode_raw_numbers


ABSENT = -1.0


class CoordConstrainedProcessor:
    """vLLM V0 logits processor: x/y alternation + teacher-forced initial states."""

    def __init__(self, x_token_ids, y_token_ids, forced_tokens=None):
        self.x_ids = set(x_token_ids)
        self.y_ids = set(y_token_ids)
        self.forced_tokens = forced_tokens or {}
        self._x_mask = None
        self._y_mask = None

    def __call__(self, prompt_token_ids, output_token_ids, scores):
        import torch
        if self._x_mask is None or self._x_mask.shape[0] != scores.shape[0]:
            vocab_size = scores.shape[0]
            self._x_mask = torch.full((vocab_size,), float("-inf"), device=scores.device)
            self._y_mask = torch.full((vocab_size,), float("-inf"), device=scores.device)
            for tid in self.x_ids:
                self._x_mask[tid] = 0.0
            for tid in self.y_ids:
                self._y_mask[tid] = 0.0

        generated_len = len(output_token_ids)

        if generated_len in self.forced_tokens:
            mask = torch.full_like(scores, float("-inf"))
            mask[self.forced_tokens[generated_len]] = 0.0
            return scores + mask

        if generated_len % 2 == 0:
            return scores + self._x_mask
        else:
            return scores + self._y_mask


def format_alpaca_prompt(sample: dict) -> str:
    return (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n"
        "### Instruction:\n"
        f"{sample['instruction']}\n\n"
        "### Input:\n"
        f"{sample['input']}\n\n"
        "### Response:\n"
    )


def get_test_split(data_path, seed=42):
    with open(data_path) as f:
        raw = json.load(f)
    indices = list(range(len(raw)))
    random.Random(seed).shuffle(indices)
    split = int(len(indices) * 0.8)
    test_idx = indices[split:]
    return [raw[i] for i in test_idx]


def parse_output_trajectory(generated_text, use_coord_tokens, coord_tok, population, frames):
    if use_coord_tokens:
        tokens = re.findall(r'<[xy]_\d+>', generated_text)
        traj = coord_tok.detokenize(tokens, population, frames)
    else:
        traj = decode_raw_numbers(generated_text, population, frames)

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


COMPACT_KEYS = {"output_trajectories", "walkable_area", "output"}


def compact_json(obj, indent=2):
    def _serialize(o, level):
        pad = " " * (indent * level)
        pad_inner = " " * (indent * (level + 1))
        if isinstance(o, dict):
            items = []
            for k, v in o.items():
                if k in COMPACT_KEYS:
                    items.append(f'{pad_inner}{json.dumps(k)}: {json.dumps(v, separators=(",", ": "))}')
                else:
                    items.append(f'{pad_inner}{json.dumps(k)}: {_serialize(v, level + 1)}')
            return "{\n" + ",\n".join(items) + f"\n{pad}}}"
        elif isinstance(o, list) and o and isinstance(o[0], dict):
            items = [f"{pad_inner}{_serialize(item, level + 1)}" for item in o]
            return "[\n" + ",\n".join(items) + f"\n{pad}]"
        else:
            return json.dumps(o)
    return _serialize(obj, 0) + "\n"


def strip_think(text):
    """Remove <think>...</think> from generated text."""
    m = re.match(r'<think>.*?</think>\s*', text, re.DOTALL)
    if m:
        return text[m.end():]
    return text


def _build_prompt(sample, use_coord):
    """Build prompt from sample, selecting coord or raw instruction."""
    if use_coord:
        instruction = sample["instruction"]
        input_text = sample["input"]
    else:
        meta = sample["metadata"]
        instruction = meta["instruction"]
        input_text = meta["input"]
    return format_alpaca_prompt({"instruction": instruction, "input": input_text})


def _build_forced_tokens_coord(gt_traj, coord_tok, tokenizer, population, frames):
    """Build {output_position: token_id} for each agent's first valid frame.

    Token ordering is agent-first: agent0-frame0-x, agent0-frame0-y,
    agent0-frame1-x, ..., agent1-frame0-x, ...
    Position = n * frames * 2 + t * 2 (for x), +1 (for y).
    """
    forced = {}
    traj_array = np.array(gt_traj)  # (N, T, 2)
    for n in range(population):
        for t in range(frames):
            x, y = traj_array[n, t, 0], traj_array[n, t, 1]
            if x == ABSENT or y == ABSENT:
                continue
            xi = int(np.clip(np.round(x / coord_tok.bin_size), 1, coord_tok.x_bins))
            yj = int(np.clip(np.round(y / coord_tok.bin_size), 1, coord_tok.y_bins))
            x_token = f"<x_{xi}>"
            y_token = f"<y_{yj}>"
            x_tid = tokenizer.convert_tokens_to_ids(x_token)
            y_tid = tokenizer.convert_tokens_to_ids(y_token)
            pos_base = n * frames * 2 + t * 2
            forced[pos_base] = x_tid
            forced[pos_base + 1] = y_tid
            break  # only first valid frame per agent
    return forced


# ====================================================================
# Batch mode
# ====================================================================
def cmd_batch(args):
    use_coord = not args.raw_mode
    resolution = (args.resolution_h, args.resolution_w)

    test_samples = get_test_split(args.data, seed=args.seed)
    print(f"Test split (seed={args.seed}): {len(test_samples)} samples")

    coord_tok = None
    if use_coord:
        coord_tok = CoordTokenizer(resolution=resolution, bin_size=args.bin_size)

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_seq_length,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )

    tokenizer_obj = llm.get_tokenizer()

    x_ids, y_ids = None, None
    if use_coord:
        x_ids = tokenizer_obj.convert_tokens_to_ids(coord_tok.x_tokens)
        y_ids = tokenizer_obj.convert_tokens_to_ids(coord_tok.y_tokens)

    prompts = []
    max_tokens_list = []
    forced_tokens_list = []
    for sample in test_samples:
        meta = sample["metadata"]
        prompts.append(_build_prompt(sample, use_coord))
        expected_tokens = meta["population"] * meta["frames"] * 2
        max_tokens_list.append(expected_tokens)
        if use_coord:
            forced = _build_forced_tokens_coord(
                sample["output"], coord_tok, tokenizer_obj,
                meta["population"], meta["frames"],
            )
        else:
            forced = {}
        forced_tokens_list.append(forced)

    for k_sample in range(args.num_samples):
        print(f"\nSample {k_sample+1}/{args.num_samples}")

        sampling_params_list = []
        for idx, max_tok in enumerate(max_tokens_list):
            sp_kwargs = dict(
                max_tokens=max_tok,
                min_tokens=max_tok,
            )
            if args.num_samples > 1:
                sp_kwargs["temperature"] = args.temperature
                sp_kwargs["top_p"] = args.top_p
            else:
                sp_kwargs["temperature"] = 0.0
            if use_coord:
                processor = CoordConstrainedProcessor(
                    x_ids, y_ids, forced_tokens=forced_tokens_list[idx],
                )
                sp_kwargs["logits_processors"] = [processor]
            sampling_params_list.append(SamplingParams(**sp_kwargs))

        outputs = llm.generate(prompts, sampling_params=sampling_params_list)

        if k_sample == 0:
            results = []
            for idx, sample in enumerate(test_samples):
                meta = sample["metadata"]
                results.append({
                    "clip_id": meta["clip_name"],
                    "output_trajectories": [],
                    "metadata": {
                        "population": meta["population"],
                        "frames": meta["frames"],
                        "walkable_area": meta["walkable_area"],
                        "scenario_description": meta["scenario_description"],
                        "crowd_description": meta["crowd_description"],
                        "output": sample["output"],
                    }
                })

        for idx, output in enumerate(outputs):
            meta = test_samples[idx]["metadata"]
            generated_text = output.outputs[0].text.strip()
            generated_text = strip_think(generated_text)

            if idx == 0 and k_sample == 0:
                if use_coord:
                    coord_tokens = re.findall(r'<[xy]_\d+>', generated_text)
                    print(f"[DEBUG] First clip: coord tokens={len(coord_tokens)}, "
                          f"expected={meta['population'] * meta['frames'] * 2}")
                print(f"[DEBUG] First 300 chars: {generated_text[:300]}")

            traj = parse_output_trajectory(
                generated_text, use_coord, coord_tok,
                meta["population"], meta["frames"]
            )
            results[idx]["output_trajectories"].append(traj)

        print(f"  Generated {len(outputs)} clips")

    out_dir = os.path.join(args.output_base, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.task_name}.json")
    with open(out_path, "w") as f:
        f.write(compact_json(results))
    print(f"\nResults saved: {out_path} ({len(results)} samples, K={args.num_samples})")


# ====================================================================
# Interactive mode
# ====================================================================
def cmd_interactive(args):
    use_coord = not args.raw_mode
    resolution = (args.resolution_h, args.resolution_w)

    coord_tok = None
    if use_coord:
        coord_tok = CoordTokenizer(resolution=resolution, bin_size=args.bin_size)

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_seq_length,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
    )

    logits_processor = None
    if use_coord:
        tokenizer_obj = llm.get_tokenizer()
        x_ids = tokenizer_obj.convert_tokens_to_ids(coord_tok.x_tokens)
        y_ids = tokenizer_obj.convert_tokens_to_ids(coord_tok.y_tokens)
        logits_processor = CoordConstrainedProcessor(x_ids, y_ids)

    sp_kwargs = dict(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.temperature > 0 else 0.0,
    )
    if args.temperature > 0:
        sp_kwargs["top_p"] = args.top_p
    if logits_processor is not None:
        sp_kwargs["logits_processors"] = [logits_processor]
    sampling_params = SamplingParams(**sp_kwargs)

    print(f"\nFLINT Interactive Mode ({'coord' if use_coord else 'raw'})")
    print("Paste your prompt and press Enter twice to submit. Type 'quit' to exit.\n")

    while True:
        try:
            lines = []
            while True:
                line = input(">>> " if not lines else "... ")
                if line.strip().lower() in ("quit", "exit") and not lines:
                    print("Exiting.")
                    return
                if line == "" and lines and lines[-1] == "":
                    lines.pop()
                    break
                lines.append(line)
            prompt = "\n".join(lines)
            if not prompt.strip():
                continue

            outputs = llm.generate([prompt], sampling_params=[sampling_params])
            generated = outputs[0].outputs[0].text.strip()
            generated = strip_think(generated)

            print(f"\n--- Response ---")
            print(generated)
            print(f"--- End ---\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return


# ====================================================================
# CLI
# ====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLINT inference (vLLM)")
    subparsers = parser.add_subparsers(dest="mode", help="Run mode")

    # --- Shared args ---
    def add_common_args(p):
        p.add_argument("--model", type=str, required=True, help="Path to merged model")
        p.add_argument("--max_seq_length", type=int, default=8192)
        p.add_argument("--max_new_tokens", type=int, default=4096)
        p.add_argument("--temperature", type=float, default=0.0)
        p.add_argument("--top_p", type=float, default=0.9)
        p.add_argument("--raw_mode", action="store_true")
        p.add_argument("--bin_size", type=int, default=5)
        p.add_argument("--resolution_h", type=int, default=480)
        p.add_argument("--resolution_w", type=int, default=640)

    # --- batch ---
    batch_p = subparsers.add_parser("batch", help="Batch inference on test split")
    add_common_args(batch_p)
    batch_p.add_argument("--data", type=str, required=True)
    batch_p.add_argument("--task_name", type=str, required=True)
    batch_p.add_argument("--output_base", type=str, default="./outputs")
    batch_p.add_argument("--seed", type=int, default=42)
    batch_p.add_argument("--num_samples", type=int, default=1)

    # --- interactive ---
    interactive_p = subparsers.add_parser("interactive", help="Interactive terminal mode")
    add_common_args(interactive_p)

    args = parser.parse_args()
    if args.mode == "batch":
        cmd_batch(args)
    elif args.mode == "interactive":
        cmd_interactive(args)
    else:
        parser.print_help()
