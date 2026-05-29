"""
下游任务: 用FLINT生成多样轨迹作为数据增强, 提升轨迹预测模型性能

实验设计:
┌─────────────────────────────────────────────────────────────────┐
│  Group A (Baseline):     原始 ETH-UCY 训练集                     │
│  Group B (+FLINT):       原始 + FLINT 生成的多样轨迹              │
│  Group C (+Traditional): 原始 + SFM/SPDiff 生成的轨迹            │
│  Group D (+Random):      原始 + 随机扰动轨迹 (控制组)             │
└─────────────────────────────────────────────────────────────────┘

如果 B 显著优于 A/C/D → 证明 FLINT 的多样性是"有意义的真实多样性"
"""

import os
import re
import json
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


# ====================================================================
# Step 1: 生成增强数据
# ====================================================================
def generate_augmentation_data(
    model_path: str,
    config_path: str,
    annotations_path: str,
    output_path: str,
    num_variants_per_sample: int = 5,
    diverse_prompts: List[str] = None,
):
    """
    用FLINT模型生成多样化轨迹作为增强数据

    策略:
    1. 对每个原始样本, 用不同temperature采样多次
    2. 对每个原始样本, 用augmented text描述生成
    3. 合并所有生成轨迹
    """
    from train.train_sft import generate_trajectories, load_config
    from tokenizer import build_tokenizer

    config = load_config(config_path)
    coord_tokenizer = build_tokenizer(config)

    with open(annotations_path) as f:
        annotations = json.load(f)

    augmented_trajectories = []

    for idx, sample in enumerate(annotations):
        print(f"[{idx+1}/{len(annotations)}] Generating augmentation for {sample.get('metadata', {}).get('clip_id', idx)}...")

        num_agents = sample.get("metadata", {}).get("num_pedestrians", 1)
        num_steps = sample.get("metadata", {}).get("num_timesteps", 25)

        # 方式1: 多温度采样
        for temp in [0.5, 0.8, 1.0]:
            gen_tokens_list = generate_trajectories(
                model_path, config, sample,
                num_samples=max(1, num_variants_per_sample // 3),
                temperature=temp,
            )
            for gen_tokens in gen_tokens_list:
                tokens = re.findall(r'<[^>]+>', gen_tokens)
                traj = coord_tokenizer.detokenize(tokens, num_agents, num_steps)
                augmented_trajectories.append({
                    "trajectories": traj.tolist(),
                    "source": "flint",
                    "temperature": temp,
                    "clip_id": sample.get("metadata", {}).get("clip_id", str(idx)),
                    "num_pedestrians": num_agents,
                    "num_timesteps": num_steps,
                })

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(augmented_trajectories, f)
    print(f"Generated {len(augmented_trajectories)} augmented samples → {output_path}")
    return augmented_trajectories


def generate_traditional_augmentation(
    trajectories: List[np.ndarray],
    method: str = "sfm",
    num_variants: int = 5,
) -> List[np.ndarray]:
    """
    用传统方法 (Social Force Model) 生成增强数据

    作为对比组: 证明传统方法的增强因为缺乏多样性而效果差
    """
    augmented = []
    for traj in trajectories:
        N, T, _ = traj.shape
        for _ in range(num_variants):
            if method == "sfm":
                aug = _social_force_simulation(traj)
            elif method == "noise":
                aug = _random_noise_augmentation(traj)
            else:
                aug = traj.copy()
            augmented.append(aug)
    return augmented


def _social_force_simulation(initial_traj: np.ndarray) -> np.ndarray:
    """简化版Social Force Model模拟"""
    N, T, _ = initial_traj.shape
    traj = initial_traj.copy()
    tau = 0.5  # relaxation time
    A = 2.0    # repulsive force strength
    B = 1.0    # repulsive force range

    for t in range(1, T):
        for n in range(N):
            if traj[n, t - 1, 0] < 0:
                continue
            # 期望速度 (朝groundtruth方向)
            if traj[n, t, 0] >= 0:
                v_desired = traj[n, t] - traj[n, t - 1]
            else:
                v_desired = np.zeros(2)

            # Social force (排斥)
            f_social = np.zeros(2)
            for m in range(N):
                if m == n or traj[m, t - 1, 0] < 0:
                    continue
                diff = traj[n, t - 1] - traj[m, t - 1]
                dist = np.linalg.norm(diff) + 1e-6
                f_social += A * np.exp(-dist / B) * diff / dist

            # 加随机扰动
            noise = np.random.randn(2) * 2.0

            # 更新位置
            v = v_desired + f_social * 0.1 + noise
            traj[n, t] = traj[n, t - 1] + v

    return traj


def _random_noise_augmentation(traj: np.ndarray, noise_std: float = 5.0) -> np.ndarray:
    """随机噪声增强 (控制组, 证明多样性不是随机噪声就能提供的)"""
    aug = traj.copy()
    mask = aug[:, :, 0] >= 0
    noise = np.random.randn(*aug.shape) * noise_std
    aug[mask] += noise[mask]
    return aug


# ====================================================================
# Step 2: Social-STGCNN 轨迹预测模型 (简化版)
# ====================================================================
class SocialSTGCNN:
    """
    简化版 Social-STGCNN 轨迹预测模型

    观测8帧 → 预测12帧
    这里用简化实现, 实际使用时可替换为完整的开源实现
    """

    def __init__(self, obs_len: int = 8, pred_len: int = 12, device: str = "cuda"):
        import torch
        import torch.nn as nn

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.device = device

        # 简化网络: GRU + MLP
        self.encoder = nn.GRU(input_size=2, hidden_size=64, num_layers=2, batch_first=True)
        self.decoder = nn.GRU(input_size=2, hidden_size=64, num_layers=2, batch_first=True)
        self.output_layer = nn.Linear(64, 2)
        self.social_layer = nn.Linear(64 * 2, 64)

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.output_layer = self.output_layer.to(device)
        self.social_layer = self.social_layer.to(device)

    def parameters(self):
        import itertools
        return itertools.chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.output_layer.parameters(),
            self.social_layer.parameters(),
        )

    def train_mode(self):
        self.encoder.train()
        self.decoder.train()
        self.output_layer.train()
        self.social_layer.train()

    def eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()
        self.output_layer.eval()
        self.social_layer.eval()

    def forward(self, obs_traj: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            obs_traj: (batch, obs_len, 2)
        Returns:
            pred_traj: (batch, pred_len, 2)
        """
        import torch
        _, hidden = self.encoder(obs_traj)

        # 解码
        last_pos = obs_traj[:, -1:, :]  # (batch, 1, 2)
        predictions = []
        input_pos = last_pos

        for _ in range(self.pred_len):
            output, hidden = self.decoder(input_pos, hidden)
            delta = self.output_layer(output)
            next_pos = input_pos + delta
            predictions.append(next_pos)
            input_pos = next_pos

        return torch.cat(predictions, dim=1)  # (batch, pred_len, 2)


# ====================================================================
# Step 3: 训练和评估 Pipeline
# ====================================================================
def prepare_prediction_data(
    trajectories: List[np.ndarray],
    obs_len: int = 8,
    pred_len: int = 12,
) -> List[Dict]:
    """
    将轨迹数据转换为 (观测, 预测) 对

    每条轨迹: 前obs_len帧作为观测, 后pred_len帧作为预测目标
    """
    samples = []
    seq_len = obs_len + pred_len

    for traj_data in trajectories:
        if isinstance(traj_data, dict):
            traj = np.array(traj_data["trajectories"])
        else:
            traj = traj_data

        if traj.ndim == 2:
            traj = traj[np.newaxis]  # (1, T, 2)

        N, T, _ = traj.shape
        for n in range(N):
            # 找连续有效的片段
            valid = traj[n, :, 0] >= 0
            start = 0
            while start < T:
                if not valid[start]:
                    start += 1
                    continue
                end = start
                while end < T and valid[end]:
                    end += 1
                seg_len = end - start
                if seg_len >= seq_len:
                    for s in range(start, end - seq_len + 1):
                        obs = traj[n, s:s + obs_len].copy()
                        pred = traj[n, s + obs_len:s + seq_len].copy()
                        samples.append({"obs": obs, "pred": pred})
                start = end

    return samples


def train_prediction_model(
    train_samples: List[Dict],
    val_samples: List[Dict],
    model_config: Dict = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    训练轨迹预测模型并评估

    Returns:
        {"ADE": float, "FDE": float}
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    obs_len = model_config.get("obs_len", 8) if model_config else 8
    pred_len = model_config.get("pred_len", 12) if model_config else 12

    # 准备数据
    train_obs = torch.tensor(
        np.array([s["obs"] for s in train_samples]), dtype=torch.float32
    )
    train_pred = torch.tensor(
        np.array([s["pred"] for s in train_samples]), dtype=torch.float32
    )
    val_obs = torch.tensor(
        np.array([s["obs"] for s in val_samples]), dtype=torch.float32
    )
    val_pred = torch.tensor(
        np.array([s["pred"] for s in val_samples]), dtype=torch.float32
    )

    train_loader = DataLoader(
        TensorDataset(train_obs, train_pred),
        batch_size=batch_size, shuffle=True,
    )

    model = SocialSTGCNN(obs_len=obs_len, pred_len=pred_len, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 训练
    model.train_mode()
    for epoch in range(epochs):
        total_loss = 0
        for batch_obs, batch_pred in train_loader:
            batch_obs = batch_obs.to(device)
            batch_pred = batch_pred.to(device)

            pred = model.forward(batch_obs)
            loss = criterion(pred, batch_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # 评估
    model.eval_mode()
    with torch.no_grad():
        val_obs_dev = val_obs.to(device)
        val_pred_dev = val_pred.to(device)
        pred_output = model.forward(val_obs_dev)

        # ADE
        displacement = torch.sqrt(
            ((pred_output - val_pred_dev) ** 2).sum(dim=-1)
        )
        ade = displacement.mean().item()

        # FDE
        fde = displacement[:, -1].mean().item()

    return {"ADE": ade, "FDE": fde}


# ====================================================================
# Step 4: 完整实验
# ====================================================================
def run_augmentation_experiment(
    original_data_dir: str,
    flint_augmented_path: str,
    config_path: str,
    output_dir: str,
    augment_ratios: List[float] = [0.0, 0.5, 1.0, 2.0],
):
    """
    完整的数据增强实验

    对比:
    A. 原始训练集 (ratio=0)
    B. 原始 + FLINT增强 (不同ratio)
    C. 原始 + SFM增强 (对照)
    D. 原始 + 随机噪声增强 (控制组)
    """
    from data.preprocessing.data_loader import load_eth_ucy_trajectories

    with open(config_path) as f:
        config = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)

    # 加载原始数据
    print("Loading original ETH-UCY data...")
    original_trajectories = []
    for subset in ["eth", "hotel", "univ", "zara1", "zara2"]:
        clips = load_eth_ucy_trajectories(original_data_dir, subset)
        for clip in clips:
            traj_list = list(clip["trajectories"].values())
            N = len(traj_list)
            T = len(traj_list[0])
            traj_array = np.array(traj_list).reshape(N, T, 2)
            original_trajectories.append(traj_array)

    # 加载FLINT增强数据
    print("Loading FLINT augmented data...")
    with open(flint_augmented_path) as f:
        flint_data = json.load(f)
    flint_trajectories = [np.array(d["trajectories"]) for d in flint_data]

    # 准备预测数据
    obs_len = config["downstream"]["trajectory_prediction"]["obs_len"]
    pred_len = config["downstream"]["trajectory_prediction"]["pred_len"]

    original_samples = prepare_prediction_data(original_trajectories, obs_len, pred_len)

    # 切分原始数据为 train/val
    np.random.seed(42)
    indices = np.random.permutation(len(original_samples))
    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_original = [original_samples[i] for i in train_indices]
    val_samples = [original_samples[i] for i in val_indices]

    print(f"Original: {len(train_original)} train, {len(val_samples)} val")

    results = {}

    for ratio in augment_ratios:
        num_augment = int(len(train_original) * ratio)
        if num_augment == 0 and ratio > 0:
            num_augment = 1

        # ---- Group A: 原始 (ratio=0时) ----
        if ratio == 0:
            print(f"\n--- Group A: Original only ---")
            r = train_prediction_model(
                train_original, val_samples,
                model_config=config["downstream"]["trajectory_prediction"],
            )
            results["original"] = r
            print(f"  ADE={r['ADE']:.4f}, FDE={r['FDE']:.4f}")
            continue

        # ---- Group B: + FLINT ----
        print(f"\n--- Group B: Original + FLINT (ratio={ratio}) ---")
        flint_samples = prepare_prediction_data(
            flint_trajectories[:num_augment * 2], obs_len, pred_len
        )
        np.random.shuffle(flint_samples)
        flint_samples = flint_samples[:num_augment]
        train_b = train_original + flint_samples
        r_b = train_prediction_model(
            train_b, val_samples,
            model_config=config["downstream"]["trajectory_prediction"],
        )
        results[f"flint_ratio{ratio}"] = r_b
        print(f"  ADE={r_b['ADE']:.4f}, FDE={r_b['FDE']:.4f}")

        # ---- Group C: + SFM ----
        print(f"\n--- Group C: Original + SFM (ratio={ratio}) ---")
        sfm_trajs = generate_traditional_augmentation(
            original_trajectories[:num_augment], method="sfm", num_variants=1,
        )
        sfm_samples = prepare_prediction_data(sfm_trajs, obs_len, pred_len)[:num_augment]
        train_c = train_original + sfm_samples
        r_c = train_prediction_model(
            train_c, val_samples,
            model_config=config["downstream"]["trajectory_prediction"],
        )
        results[f"sfm_ratio{ratio}"] = r_c
        print(f"  ADE={r_c['ADE']:.4f}, FDE={r_c['FDE']:.4f}")

        # ---- Group D: + Random noise ----
        print(f"\n--- Group D: Original + Random (ratio={ratio}) ---")
        noise_trajs = generate_traditional_augmentation(
            original_trajectories[:num_augment], method="noise", num_variants=1,
        )
        noise_samples = prepare_prediction_data(noise_trajs, obs_len, pred_len)[:num_augment]
        train_d = train_original + noise_samples
        r_d = train_prediction_model(
            train_d, val_samples,
            model_config=config["downstream"]["trajectory_prediction"],
        )
        results[f"noise_ratio{ratio}"] = r_d
        print(f"  ADE={r_d['ADE']:.4f}, FDE={r_d['FDE']:.4f}")

    # 保存结果
    result_path = os.path.join(output_dir, "augmentation_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_augmentation_summary(results)
    _plot_augmentation_results(results, output_dir)

    return results


def _print_augmentation_summary(results: Dict):
    """打印增强实验结果摘要"""
    print("\n" + "=" * 70)
    print("AUGMENTATION EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"{'Group':<25} {'ADE':>10} {'FDE':>10} {'ADE Δ%':>10} {'FDE Δ%':>10}")
    print("-" * 70)

    base_ade = results.get("original", {}).get("ADE", 1)
    base_fde = results.get("original", {}).get("FDE", 1)

    for name, r in sorted(results.items()):
        ade_delta = (r["ADE"] - base_ade) / base_ade * 100
        fde_delta = (r["FDE"] - base_fde) / base_fde * 100
        print(f"  {name:<25} {r['ADE']:>8.4f} {r['FDE']:>8.4f} "
              f"{ade_delta:>+8.1f}% {fde_delta:>+8.1f}%")


def _plot_augmentation_results(results: Dict, output_dir: str):
    """绘制增强效果对比图"""
    try:
        import matplotlib.pyplot as plt

        groups = {"FLINT": {}, "SFM": {}, "Random": {}}
        for name, r in results.items():
            if name == "original":
                continue
            for prefix, group in [("flint", "FLINT"), ("sfm", "SFM"), ("noise", "Random")]:
                if name.startswith(prefix):
                    ratio = float(name.split("ratio")[1])
                    groups[group][ratio] = r

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        colors = {"FLINT": "tab:blue", "SFM": "tab:orange", "Random": "tab:gray"}

        base_ade = results.get("original", {}).get("ADE", 1)
        base_fde = results.get("original", {}).get("FDE", 1)

        for metric_idx, (metric, base_val) in enumerate([("ADE", base_ade), ("FDE", base_fde)]):
            ax = axes[metric_idx]
            ax.axhline(y=base_val, color='red', linestyle='--', label='Original (no aug)')

            for group_name, group_data in groups.items():
                if not group_data:
                    continue
                ratios = sorted(group_data.keys())
                values = [group_data[r][metric] for r in ratios]
                ax.plot(ratios, values, '-o', color=colors[group_name],
                        label=f'+{group_name}', linewidth=2, markersize=8)

            ax.set_xlabel('Augmentation Ratio', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric} vs Augmentation Ratio', fontsize=13)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "augmentation_comparison.pdf"), dpi=300)
        plt.close()
        print(f"\nPlot saved to {output_dir}/augmentation_comparison.pdf")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_data", type=str, required=True,
                        help="ETH-UCY raw data directory")
    parser.add_argument("--flint_augmented", type=str, required=True,
                        help="FLINT generated augmentation data JSON")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output_dir", type=str, default="./downstream_results")
    args = parser.parse_args()

    run_augmentation_experiment(
        original_data_dir=args.original_data,
        flint_augmented_path=args.flint_augmented,
        config_path=args.config,
        output_dir=args.output_dir,
    )
