"""
FLINT inference: load trained LoRA model, generate trajectories on test split.
"""

import os
import re
import json
import random
import torch
import argparse
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from dataset_construction import CoordTokenizer, decode_raw_numbers


ABSENT = -1.0


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


def load_model(backbone, checkpoint_path, use_coord_tokens, resolution, bin_size, max_seq_length):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        backbone,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    coord_tok = None
    if use_coord_tokens:
        coord_tok = CoordTokenizer(resolution=resolution, bin_size=bin_size)
        model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    return model, tokenizer, coord_tok


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


COMPACT_KEYS = {"output_trajectories", "walkable_area", "trajectory"}


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


def run_inference(model, tokenizer, coord_tok, test_samples, args, use_coord, task_name, checkpoint_label):
    """Run inference on test split, save results."""
    print(f"\nRunning inference: {task_name} (checkpoint: {checkpoint_label})")

    results = []
    for idx, sample in enumerate(test_samples):
        meta = sample["metadata"]

        if args.raw_mode:
            fields = meta["raw_prompt"]
        else:
            fields = sample

        prompt = format_alpaca_prompt(fields)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=args.max_seq_length).to(model.device)

        all_trajectories = []
        for k in range(args.num_samples):
            gen_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
            )
            if args.num_samples > 1:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            new_ids = output_ids[0][inputs.input_ids.shape[1]:]
            think_token = tokenizer.convert_tokens_to_ids("<think>")
            end_think_token = tokenizer.convert_tokens_to_ids("</think>")
            if think_token is not None and think_token in new_ids:
                try:
                    end_pos = new_ids.tolist().index(end_think_token)
                    new_ids = new_ids[end_pos + 1:]
                except ValueError:
                    pass

            generated_text = tokenizer.decode(
                new_ids, skip_special_tokens=True,
            ).strip()

            if idx == 0 and k == 0:
                raw_text = tokenizer.decode(
                    output_ids[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=False,
                ).strip()
                print(f"[DEBUG] Raw generated (first 500 chars):\n{raw_text[:500]}")
                print(f"[DEBUG] Cleaned text (first 500 chars):\n{generated_text[:500]}")
                if use_coord:
                    coord_tokens = re.findall(r'<[xy]_\d+>', generated_text)
                    print(f"[DEBUG] Coord tokens found: {len(coord_tokens)}, "
                          f"expected: {meta['population'] * meta['frames'] * 2}")
                else:
                    parts = generated_text.strip().split(";")
                    print(f"[DEBUG] Raw number parts found: {len(parts)}, "
                          f"expected: {meta['population'] * meta['frames']}")

            traj = parse_output_trajectory(
                generated_text, use_coord, coord_tok,
                meta["population"], meta["frames"]
            )
            all_trajectories.append(traj)

        result = {
            "clip_id": meta["clip_name"],
            "output_trajectories": all_trajectories,
            "metadata": {
                "population": meta["population"],
                "frames": meta["frames"],
                "walkable_area": meta["walkable_area"],
                "scenario_description": meta["scenario_description"],
                "crowd_description": meta["crowd_description"],
                "trajectory": meta["trajectory"],
            }
        }
        results.append(result)

        print(f"[{idx+1}/{len(test_samples)}] {meta['clip_name']} "
              f"(pop={meta['population']}, frames={meta['frames']}, "
              f"samples={args.num_samples})")

    out_dir = os.path.join(args.output_base, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{task_name}.json")
    with open(out_path, "w") as f:
        f.write(compact_json(results))
    print(f"Results saved: {out_path} ({len(results)} samples)")


def generate(args):
    use_coord = not args.raw_mode
    resolution = (args.resolution_h, args.resolution_w)

    test_samples = get_test_split(args.data, seed=args.seed)
    print(f"Test split (seed={args.seed}): {len(test_samples)} samples")

    eval_epochs = [int(x.strip()) for x in args.eval_epochs.split(",")]
    print(f"Eval epochs: {eval_epochs}")

    for ep in eval_epochs:
        checkpoint_path = os.path.join(
            args.output_base, "checkpoint", args.train_task, f"epoch_{ep}"
        )
        if not os.path.isdir(checkpoint_path):
            print(f"\nSkipping epoch {ep}: checkpoint not found at {checkpoint_path}")
            continue

        task_name = f"{args.task_name_base}-{ep}"

        print(f"\n{'='*60}")
        print(f"  Epoch {ep}: loading {checkpoint_path}")
        print(f"{'='*60}")

        model, tokenizer, coord_tok = load_model(
            args.backbone, checkpoint_path, use_coord, resolution,
            args.bin_size, args.max_seq_length
        )
        print(f"Sampling: num_samples={args.num_samples}, "
              f"do_sample={'True' if args.num_samples > 1 else 'False'}, "
              f"temperature={args.temperature}, top_p={args.top_p}")

        run_inference(model, tokenizer, coord_tok, test_samples,
                      args, use_coord, task_name, f"epoch_{ep}")

        del model
        torch.cuda.empty_cache()

    print("\nAll inference complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLINT inference")

    parser.add_argument("--data", type=str, required=True,
                        help="Path to eth-ucy-text.json")
    parser.add_argument("--task_name_base", type=str, required=True,
                        help="Base task name (epoch suffix appended automatically)")
    parser.add_argument("--train_task", type=str, required=True,
                        help="Training task name (for checkpoint path)")
    parser.add_argument("--eval_epochs", type=str, default="20",
                        help="Comma-separated epochs to evaluate, e.g. '0,5,10,15,20'")
    parser.add_argument("--output_base", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--backbone", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=4096)

    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of trajectory samples per clip")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--raw_mode", action="store_true",
                        help="Use raw number format (no coord tokens)")
    parser.add_argument("--bin_size", type=int, default=5)
    parser.add_argument("--resolution_h", type=int, default=480)
    parser.add_argument("--resolution_w", type=int, default=640)

    args = parser.parse_args()
    generate(args)
