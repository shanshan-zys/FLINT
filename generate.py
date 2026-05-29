"""
FLINT 推理生成：加载训练好的模型，生成轨迹
"""

import os
import re
import json
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path

from prepare_data import CoordTokenizer, decode_raw_numbers


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def format_alpaca_prompt(sample: dict) -> str:
    return (
        "### Instruction:\n"
        f"{sample['instruction']}\n\n"
        "### Input:\n"
        f"{sample['input']}\n\n"
        "### Response:\n"
    )


def load_model(model_path: str, config: dict):
    from unsloth import FastLanguageModel

    model, llm_tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config["training"]["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, llm_tokenizer


def generate_samples(model, llm_tokenizer, prompt: str, num_samples: int,
                     temperature: float, top_p: float, max_new_tokens: int):
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(model.device)
    results = []

    for _ in range(num_samples):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.1,
            )
        generated = llm_tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=False,
        )
        results.append(generated.strip())

    return results


def generate_from_test(args):
    config = load_config(args.config)
    model, llm_tokenizer = load_model(args.model_path, config)

    with open(args.test_data) as f:
        test_data = json.load(f)

    use_coord = test_data[0]["metadata"].get("use_coord_tokens", True)
    coord_tok = None
    if use_coord:
        coord_tok = CoordTokenizer(
            resolution=tuple(config["data"]["resolution"]),
            bin_size=config["tokenizer"]["bin_size"],
        )

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for idx, sample in enumerate(test_data):
        prompt = format_alpaca_prompt(sample)
        num_agents = sample["metadata"]["num_pedestrians"]
        num_steps = sample["metadata"]["num_timesteps"]

        print(f"[{idx + 1}/{len(test_data)}] {sample['metadata']['clip_id']} "
              f"({num_agents} peds, {num_steps} steps) × {args.num_samples} samples")

        gen_texts = generate_samples(
            model, llm_tokenizer, prompt,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=config["evaluation"]["top_p"],
            max_new_tokens=config["training"]["max_seq_length"] // 2,
        )

        sample_results = {
            "clip_id": sample["metadata"]["clip_id"],
            "num_pedestrians": num_agents,
            "num_timesteps": num_steps,
            "use_coord_tokens": use_coord,
            "raw_generations": gen_texts,
            "trajectories": [],
        }

        for gen_text in gen_texts:
            if use_coord:
                tokens = re.findall(r'<[^>]+>', gen_text)
                traj = coord_tok.detokenize(tokens, num_agents, num_steps)
            else:
                traj = decode_raw_numbers(gen_text, num_agents, num_steps)
            sample_results["trajectories"].append(traj.tolist())

        all_results.append(sample_results)

    output_path = os.path.join(args.output_dir, "generated.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nGenerated {len(all_results)} clips → {output_path}")


def generate_custom(args):
    config = load_config(args.config)
    model, llm_tokenizer = load_model(args.model_path, config)

    from prepare_data import SFT_INSTRUCTION_COORD
    instruction = SFT_INSTRUCTION_COORD.format(
        x_bins=config["tokenizer"]["x_bins"],
        y_bins=config["tokenizer"]["y_bins"],
    )

    sft_input = (
        f"Scenario: Custom scene\n\n"
        f"Crowd dynamics: {args.custom_prompt}\n\n"
        f"Number of pedestrians: {args.num_agents}\n"
        f"Number of timesteps: {args.num_steps}\n\n"
        f"Initial states:\n  (random initialization)"
    )

    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{sft_input}\n\n"
        f"### Response:\n"
    )

    print(f"Custom prompt: {args.custom_prompt}")
    print(f"Generating {args.num_samples} samples ({args.num_agents} agents, {args.num_steps} steps)")

    gen_texts = generate_samples(
        model, llm_tokenizer, prompt,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=config["evaluation"]["top_p"],
        max_new_tokens=config["training"]["max_seq_length"] // 2,
    )

    coord_tok = CoordTokenizer(
        resolution=tuple(config["data"]["resolution"]),
        bin_size=config["tokenizer"]["bin_size"],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "custom_prompt": args.custom_prompt,
        "num_agents": args.num_agents,
        "num_steps": args.num_steps,
        "raw_generations": gen_texts,
        "trajectories": [],
    }
    for gen_text in gen_texts:
        tokens = re.findall(r'<[^>]+>', gen_text)
        traj = coord_tok.detokenize(tokens, args.num_agents, args.num_steps)
        results["trajectories"].append(traj.tolist())

    output_path = os.path.join(args.output_dir, "custom_generated.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Custom generation → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--custom_prompt", type=str, default=None)
    parser.add_argument("--num_agents", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output_dir", type=str, default="results/main")
    args = parser.parse_args()

    if args.custom_prompt:
        generate_custom(args)
    elif args.test_data:
        generate_from_test(args)
    else:
        parser.error("Provide --test_data or --custom_prompt")
