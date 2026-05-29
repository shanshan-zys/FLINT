"""
主评估脚本

整合所有评估指标, 输出完整评估报告
"""

import os
import re
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from tokenizer import build_tokenizer
from evaluate.diversity import compute_diversity_from_raw_generations, compute_all_diversity_metrics
from evaluate.quality import compute_all_quality_metrics
from evaluate.llm_judge import LLMJudge, multi_judge_evaluation, compute_trajectory_stats
from evaluate.controllability import compute_all_controllability


def evaluate_model(
    model_path: str,
    config_path: str,
    test_data_path: str,
    output_dir: str,
    num_diversity_samples: int = 20,
    run_llm_judge: bool = True,
    run_controllability: bool = False,
):
    """
    完整评估流程:
    1. 对每个测试样本生成1条轨迹 → 质量评估
    2. 对每个测试样本生成K条轨迹 → 多样性评估
    3. LLM Judge 评估
    4. 可控性评估 (可选)
    """
    from train.train_sft import generate_trajectories, load_config

    config = load_config(config_path)
    coord_tokenizer = build_tokenizer(config)

    with open(test_data_path) as f:
        test_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    results = {"model": model_path, "config": config_path, "num_test": len(test_data)}

    # ================================================================
    # Step 1: 质量评估 (每个样本生成1条)
    # ================================================================
    print("=" * 60)
    print("Step 1: Quality Evaluation")
    print("=" * 60)

    quality_metrics_all = []
    generated_trajectories = []

    for idx, sample in enumerate(test_data):
        print(f"  [{idx+1}/{len(test_data)}] Generating...")
        gen_tokens = generate_trajectories(
            model_path, config, sample, num_samples=1, temperature=0.1,
        )[0]

        tokens = re.findall(r'<[^>]+>', gen_tokens)
        num_agents = sample["metadata"]["num_pedestrians"]
        num_steps = sample["metadata"]["num_timesteps"]
        pred_traj = coord_tokenizer.detokenize(tokens, num_agents, num_steps)
        generated_trajectories.append(pred_traj)

        # groundtruth
        gt_tokens = re.findall(r'<[^>]+>', sample["output"])
        gt_traj = coord_tokenizer.detokenize(gt_tokens, num_agents, num_steps)

        quality = compute_all_quality_metrics(pred_traj, gt_traj)
        quality["clip_id"] = sample["metadata"]["clip_id"]
        quality_metrics_all.append(quality)

    # 平均质量
    avg_quality = {}
    for key in quality_metrics_all[0]:
        if key == "clip_id":
            continue
        vals = [m[key] for m in quality_metrics_all]
        avg_quality[key] = float(np.mean(vals))
    results["quality"] = avg_quality
    print(f"\n  Quality metrics (avg):")
    for k, v in avg_quality.items():
        print(f"    {k}: {v:.4f}")

    # ================================================================
    # Step 2: 多样性评估 (每个样本生成K条)
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"Step 2: Diversity Evaluation (K={num_diversity_samples})")
    print("=" * 60)

    diversity_metrics_all = []
    multi_sample_trajs = {}

    for idx, sample in enumerate(test_data):
        print(f"  [{idx+1}/{len(test_data)}] Generating {num_diversity_samples} samples...")
        gen_tokens_list = generate_trajectories(
            model_path, config, sample,
            num_samples=num_diversity_samples,
            temperature=config["evaluation"]["temperature"],
            top_p=config["evaluation"]["top_p"],
        )

        num_agents = sample["metadata"]["num_pedestrians"]
        num_steps = sample["metadata"]["num_timesteps"]

        # 解码所有样本
        all_trajs = []
        for gen_tokens in gen_tokens_list:
            tokens = re.findall(r'<[^>]+>', gen_tokens)
            traj = coord_tokenizer.detokenize(tokens, num_agents, num_steps)
            all_trajs.append(traj)

        multi_sample_trajs[idx] = all_trajs

        # 按agent计算多样性
        trajs_per_agent = defaultdict(list)
        for traj in all_trajs:
            for n in range(num_agents):
                trajs_per_agent[n].append(traj[n])

        diversity = compute_all_diversity_metrics(trajs_per_agent)
        diversity["clip_id"] = sample["metadata"]["clip_id"]
        diversity_metrics_all.append(diversity)

    # 平均多样性
    avg_diversity = {}
    for key in diversity_metrics_all[0]:
        if key == "clip_id":
            continue
        vals = [m[key] for m in diversity_metrics_all]
        avg_diversity[key] = float(np.mean(vals))
    results["diversity"] = avg_diversity
    print(f"\n  Diversity metrics (avg):")
    for k, v in avg_diversity.items():
        print(f"    {k}: {v:.4f}")

    # ================================================================
    # Step 3: LLM Judge (可选)
    # ================================================================
    if run_llm_judge:
        print(f"\n{'=' * 60}")
        print("Step 3: LLM Judge Evaluation")
        print("=" * 60)

        judges = []
        for judge_model in config["evaluation"]["llm_judges"]:
            if "gpt" in judge_model:
                judges.append(LLMJudge("openai", judge_model))
            elif "deepseek" in judge_model:
                judges.append(LLMJudge("deepseek", judge_model))
            elif "gemini" in judge_model:
                judges.append(LLMJudge("google", judge_model))

        if judges:
            judge_results = multi_judge_evaluation(
                judges, test_data, generated_trajectories, multi_sample_trajs,
            )
            results["llm_judge"] = {
                k: v for k, v in judge_results.items()
                if not isinstance(v, list)
            }
            print(f"\n  LLM Judge scores:")
            for k, v in results["llm_judge"].items():
                print(f"    {k}: {v:.2f}")

    # ================================================================
    # 保存结果
    # ================================================================
    result_path = os.path.join(output_dir, "evaluation_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_path}")

    return results


def compare_models(result_paths: List[str], output_path: str = None):
    """对比多个模型的评估结果"""
    all_results = []
    for path in result_paths:
        with open(path) as f:
            all_results.append(json.load(f))

    # 生成对比表格
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Quality
    print("\n--- Quality Metrics ---")
    header = f"{'Model':<30}"
    quality_keys = list(all_results[0].get("quality", {}).keys())
    for k in quality_keys:
        header += f"  {k:>12}"
    print(header)
    for r in all_results:
        row = f"{Path(r['model']).name:<30}"
        for k in quality_keys:
            v = r.get("quality", {}).get(k, 0)
            row += f"  {v:>12.4f}"
        print(row)

    # Diversity
    print("\n--- Diversity Metrics ---")
    header = f"{'Model':<30}"
    div_keys = list(all_results[0].get("diversity", {}).keys())
    for k in div_keys:
        header += f"  {k:>12}"
    print(header)
    for r in all_results:
        row = f"{Path(r['model']).name:<30}"
        for k in div_keys:
            v = r.get("diversity", {}).get(k, 0)
            row += f"  {v:>12.4f}"
        print(row)

    if output_path:
        with open(output_path, "w") as f:
            json.dump({"models": all_results}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--no_llm_judge", action="store_true")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        config_path=args.config,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        num_diversity_samples=args.num_samples,
        run_llm_judge=not args.no_llm_judge,
    )
