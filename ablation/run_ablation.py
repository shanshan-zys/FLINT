"""
消融实验 Runner

支持的消融维度:
1. Tokenizer模式: absolute vs relative vs polar
2. LLM backbone: Qwen3 vs LLaMA
3. 物理loss组合: none / collision / smoothness / walkability / all
4. 采样温度对多样性的影响
5. 数据集组合: ETH-UCY only / + SDD / + augmented descriptions
"""

import os
import json
import yaml
import itertools
import subprocess
from pathlib import Path
from typing import List, Dict
from copy import deepcopy


def run_single_experiment(
    config: dict,
    experiment_name: str,
    train_path: str,
    test_path: str,
    base_output_dir: str,
) -> str:
    """运行单个实验, 返回结果路径"""
    output_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    # 保存这个实验的配置
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # 训练
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")

    from train.train_sft import train
    train(config_path, train_path, test_path)

    # 评估
    model_path = os.path.join(
        output_dir,
        f"{Path(config['training']['backbone']).name}_{config['tokenizer']['mode']}",
        "final",
    )
    from evaluate.run_eval import evaluate_model
    results = evaluate_model(
        model_path=model_path,
        config_path=config_path,
        test_data_path=test_path,
        output_dir=os.path.join(output_dir, "eval"),
        num_diversity_samples=config["evaluation"]["num_samples_per_prompt"],
        run_llm_judge=False,  # 消融实验默认不跑LLM judge (太贵)
    )

    return os.path.join(output_dir, "eval", "evaluation_results.json")


def ablation_tokenizer(
    base_config_path: str,
    train_path: str,
    test_path: str,
    output_dir: str,
):
    """
    消融1: Tokenizer模式对比
    absolute vs relative vs polar
    """
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    result_paths = []
    for mode in ["absolute", "relative", "polar"]:
        config = deepcopy(base_config)
        config["tokenizer"]["mode"] = mode
        name = f"tokenizer_{mode}"
        path = run_single_experiment(config, name, train_path, test_path, output_dir)
        result_paths.append(path)

    from evaluate.run_eval import compare_models
    compare_models(result_paths, os.path.join(output_dir, "ablation_tokenizer.json"))


def ablation_physics_loss(
    base_config_path: str,
    train_path: str,
    test_path: str,
    output_dir: str,
):
    """
    消融2: 物理loss组合
    """
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    variants = {
        "no_physics": {"enabled": False, "collision_weight": 0, "smoothness_weight": 0, "walkability_weight": 0},
        "collision_only": {"enabled": True, "collision_weight": 0.1, "smoothness_weight": 0, "walkability_weight": 0},
        "smoothness_only": {"enabled": True, "collision_weight": 0, "smoothness_weight": 0.05, "walkability_weight": 0},
        "walkability_only": {"enabled": True, "collision_weight": 0, "smoothness_weight": 0, "walkability_weight": 0.2},
        "all_physics": {"enabled": True, "collision_weight": 0.1, "smoothness_weight": 0.05, "walkability_weight": 0.2},
    }

    result_paths = []
    for name, loss_cfg in variants.items():
        config = deepcopy(base_config)
        config["training"]["physics_loss"] = loss_cfg
        path = run_single_experiment(config, f"physics_{name}", train_path, test_path, output_dir)
        result_paths.append(path)

    from evaluate.run_eval import compare_models
    compare_models(result_paths, os.path.join(output_dir, "ablation_physics.json"))


def ablation_temperature(
    model_path: str,
    config_path: str,
    test_path: str,
    output_dir: str,
):
    """
    消融3: 采样温度对多样性-质量 tradeoff 的影响
    不需要重新训练, 只需要改推理参数
    """
    from evaluate.run_eval import evaluate_model

    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2]
    results = []

    for temp in temperatures:
        print(f"\n--- Temperature = {temp} ---")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        config["evaluation"]["temperature"] = temp

        temp_config_path = os.path.join(output_dir, f"config_temp{temp}.yaml")
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)

        result = evaluate_model(
            model_path=model_path,
            config_path=temp_config_path,
            test_data_path=test_path,
            output_dir=os.path.join(output_dir, f"temp_{temp}"),
            num_diversity_samples=20,
            run_llm_judge=False,
        )
        result["temperature"] = temp
        results.append(result)

    # 保存和可视化
    with open(os.path.join(output_dir, "ablation_temperature.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot_diversity_quality_tradeoff(results, output_dir)


def ablation_backbone(
    base_config_path: str,
    train_path: str,
    test_path: str,
    output_dir: str,
):
    """
    消融4: LLM backbone对比
    """
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    backbones = base_config["ablation"]["backbones"]
    result_paths = []
    for backbone in backbones:
        config = deepcopy(base_config)
        config["training"]["backbone"] = backbone
        name = f"backbone_{Path(backbone).name}"
        path = run_single_experiment(config, name, train_path, test_path, output_dir)
        result_paths.append(path)

    from evaluate.run_eval import compare_models
    compare_models(result_paths, os.path.join(output_dir, "ablation_backbone.json"))


def _plot_diversity_quality_tradeoff(results: List[Dict], output_dir: str):
    """绘制 diversity vs quality 的 tradeoff 图"""
    try:
        import matplotlib.pyplot as plt

        temps = [r["temperature"] for r in results]
        jade = [r.get("quality", {}).get("JADE", 0) for r in results]
        atd = [r.get("diversity", {}).get("ATD", 0) for r in results]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()

        ax1.plot(temps, jade, 'b-o', label='JADE (↓ better)', linewidth=2)
        ax2.plot(temps, atd, 'r-s', label='ATD (↑ more diverse)', linewidth=2)

        ax1.set_xlabel('Sampling Temperature', fontsize=12)
        ax1.set_ylabel('JADE (Quality)', color='b', fontsize=12)
        ax2.set_ylabel('ATD (Diversity)', color='r', fontsize=12)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

        plt.title('Diversity-Quality Tradeoff across Temperature')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "diversity_quality_tradeoff.pdf"), dpi=300)
        plt.close()
        print(f"  Plot saved to {output_dir}/diversity_quality_tradeoff.pdf")
    except ImportError:
        print("  matplotlib not available, skipping plot")


def run_all_ablations(
    config_path: str,
    train_path: str,
    test_path: str,
    output_dir: str,
):
    """运行全部消融实验"""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "#" * 60)
    print("# Ablation 1: Tokenizer Mode")
    print("#" * 60)
    ablation_tokenizer(config_path, train_path, test_path,
                       os.path.join(output_dir, "tokenizer"))

    print("\n" + "#" * 60)
    print("# Ablation 2: Physics Loss")
    print("#" * 60)
    ablation_physics_loss(config_path, train_path, test_path,
                          os.path.join(output_dir, "physics"))

    print("\n" + "#" * 60)
    print("# Ablation 3: Backbone")
    print("#" * 60)
    ablation_backbone(config_path, train_path, test_path,
                      os.path.join(output_dir, "backbone"))

    print("\n\nAll ablations complete!")
    print(f"Results in {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ablation_results")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["all", "tokenizer", "physics", "backbone", "temperature"])
    parser.add_argument("--model_path", type=str, default=None,
                        help="For temperature ablation only")
    args = parser.parse_args()

    if args.ablation == "all":
        run_all_ablations(args.config, args.train_data, args.test_data, args.output_dir)
    elif args.ablation == "tokenizer":
        ablation_tokenizer(args.config, args.train_data, args.test_data, args.output_dir)
    elif args.ablation == "physics":
        ablation_physics_loss(args.config, args.train_data, args.test_data, args.output_dir)
    elif args.ablation == "backbone":
        ablation_backbone(args.config, args.train_data, args.test_data, args.output_dir)
    elif args.ablation == "temperature":
        assert args.model_path, "--model_path required for temperature ablation"
        ablation_temperature(args.model_path, args.config, args.test_data, args.output_dir)
