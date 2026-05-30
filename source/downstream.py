"""
下游任务：轨迹预测数据增强

对比：原始数据集 vs 原始 + FLINT 增强数据集
"""

import os
import json
import argparse
import numpy as np
from typing import List, Dict

from prepare_data import load_all_trajectories


# ====================================================================
# 数据准备
# ====================================================================
def prepare_prediction_samples(trajectories: List[Dict], obs_len=8, pred_len=12):
    seq_len = obs_len + pred_len
    samples = []

    for clip in trajectories:
        if isinstance(clip, dict) and "trajectories" in clip:
            traj_dict = clip["trajectories"]
            ped_ids = sorted(traj_dict.keys(), key=lambda x: int(x))
            N = len(ped_ids)
            T = max(len(traj_dict[pid]) for pid in ped_ids)
            traj_array = np.full((N, T, 2), -1.0)
            for i, pid in enumerate(ped_ids):
                for t, pos in enumerate(traj_dict[pid]):
                    traj_array[i, t] = pos
        else:
            traj_array = np.array(clip)
            if traj_array.ndim == 2:
                traj_array = traj_array[np.newaxis]

        N, T, _ = traj_array.shape
        for n in range(N):
            valid_segs = []
            start = None
            for t in range(T):
                if traj_array[n, t, 0] >= 0:
                    if start is None:
                        start = t
                else:
                    if start is not None:
                        valid_segs.append((start, t))
                        start = None
            if start is not None:
                valid_segs.append((start, T))

            for seg_start, seg_end in valid_segs:
                seg_len = seg_end - seg_start
                if seg_len >= seq_len:
                    for s in range(seg_start, seg_end - seq_len + 1):
                        obs = traj_array[n, s:s + obs_len].copy()
                        pred = traj_array[n, s + obs_len:s + seq_len].copy()
                        samples.append({"obs": obs, "pred": pred})

    return samples


def load_flint_generated(results_path: str) -> List:
    with open(results_path) as f:
        data = json.load(f)
    clips = []
    for item in data:
        for traj_list in item["trajectories"]:
            traj = np.array(traj_list)
            n_agents = item["num_pedestrians"]
            n_steps = item["num_timesteps"]
            if traj.ndim == 2:
                traj = traj.reshape(n_agents, n_steps, 2)
            clips.append(traj)
    return clips


# ====================================================================
# GRU 轨迹预测器
# ====================================================================
class TrajectoryPredictor:

    def __init__(self, obs_len=8, pred_len=12, hidden=64, layers=2, device="cuda"):
        import torch
        import torch.nn as nn

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.device = device

        self.encoder = nn.GRU(2, hidden, layers, batch_first=True).to(device)
        self.decoder = nn.GRU(2, hidden, layers, batch_first=True).to(device)
        self.out_fc = nn.Linear(hidden, 2).to(device)

    def parameters(self):
        import itertools
        return itertools.chain(
            self.encoder.parameters(),
            self.decoder.parameters(),
            self.out_fc.parameters(),
        )

    def forward(self, obs):
        import torch
        _, hidden = self.encoder(obs)
        inp = obs[:, -1:, :]
        preds = []
        for _ in range(self.pred_len):
            out, hidden = self.decoder(inp, hidden)
            delta = self.out_fc(out)
            inp = inp + delta
            preds.append(inp)
        return torch.cat(preds, dim=1)


def train_and_eval(train_samples, val_samples, config, device="cuda"):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    obs_len = config["downstream"]["obs_len"]
    pred_len = config["downstream"]["pred_len"]
    epochs = config["downstream"]["predictor_epochs"]
    lr = config["downstream"]["predictor_lr"]
    batch_size = config["downstream"]["predictor_batch_size"]

    train_obs = torch.tensor(np.array([s["obs"] for s in train_samples]), dtype=torch.float32)
    train_pred = torch.tensor(np.array([s["pred"] for s in train_samples]), dtype=torch.float32)
    val_obs = torch.tensor(np.array([s["obs"] for s in val_samples]), dtype=torch.float32)
    val_pred = torch.tensor(np.array([s["pred"] for s in val_samples]), dtype=torch.float32)

    loader = DataLoader(TensorDataset(train_obs, train_pred), batch_size=batch_size, shuffle=True)

    model = TrajectoryPredictor(obs_len, pred_len, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.encoder.train()
    model.decoder.train()
    model.out_fc.train()
    for epoch in range(epochs):
        total_loss = 0
        for bo, bp in loader:
            bo, bp = bo.to(device), bp.to(device)
            pred = model.forward(bo)
            loss = criterion(pred, bp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader):.4f}")

    model.encoder.eval()
    model.decoder.eval()
    model.out_fc.eval()
    with torch.no_grad():
        vo = val_obs.to(device)
        vp = val_pred.to(device)
        pred_out = model.forward(vo)
        disp = torch.sqrt(((pred_out - vp) ** 2).sum(dim=-1))
        ade = disp.mean().item()
        fde = disp[:, -1].mean().item()

    return {"ADE": ade, "FDE": fde}


# ====================================================================
# Main experiment
# ====================================================================
def run_downstream(args):
    config = {
        "downstream": {
            "obs_len": args.obs_len,
            "pred_len": args.pred_len,
            "predictor_epochs": args.predictor_epochs,
            "predictor_lr": args.predictor_lr,
            "predictor_batch_size": args.predictor_batch_size,
        },
    }

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    obs_len = args.obs_len
    pred_len = args.pred_len

    print("Loading original ETH-UCY data...")
    original_clips = load_all_trajectories(args.original_data)
    original_samples = prepare_prediction_samples(original_clips, obs_len, pred_len)

    np.random.seed(42)
    indices = np.random.permutation(len(original_samples))
    split = int(0.8 * len(indices))
    train_orig = [original_samples[i] for i in indices[:split]]
    val_samples = [original_samples[i] for i in indices[split:]]
    print(f"Original: {len(train_orig)} train, {len(val_samples)} val")

    print("Loading FLINT augmented data...")
    flint_clips = load_flint_generated(os.path.join(args.flint_results, "generated.json"))
    flint_samples = prepare_prediction_samples(
        [{"trajectories": {str(n): c[n].tolist() for n in range(c.shape[0])}}
         for c in flint_clips],
        obs_len, pred_len,
    )
    print(f"FLINT augmented: {len(flint_samples)} samples")

    results = {}

    print("\n--- Baseline: original only ---")
    results["baseline"] = train_and_eval(train_orig, val_samples, config, device)
    print(f"  ADE={results['baseline']['ADE']:.4f}, FDE={results['baseline']['FDE']:.4f}")

    print("\n--- +FLINT augmentation ---")
    train_aug = train_orig + flint_samples
    results["+FLINT"] = train_and_eval(train_aug, val_samples, config, device)
    print(f"  ADE={results['+FLINT']['ADE']:.4f}, FDE={results['+FLINT']['FDE']:.4f}")

    # Summary
    base_ade = results["baseline"]["ADE"]
    base_fde = results["baseline"]["FDE"]
    ade_delta = (results["+FLINT"]["ADE"] - base_ade) / base_ade * 100
    fde_delta = (results["+FLINT"]["FDE"] - base_fde) / base_fde * 100

    print(f"\n{'='*60}")
    print(f"{'Method':<20} {'ADE':>10} {'FDE':>10} {'ADE Δ%':>10} {'FDE Δ%':>10}")
    print(f"{'-'*60}")
    print(f"{'baseline':<20} {base_ade:>10.4f} {base_fde:>10.4f} {'—':>10} {'—':>10}")
    print(f"{'+FLINT':<20} {results['+FLINT']['ADE']:>10.4f} {results['+FLINT']['FDE']:>10.4f} "
          f"{ade_delta:>+9.1f}% {fde_delta:>+9.1f}%")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "downstream_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {args.output_dir}/downstream_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flint_results", type=str, required=True)
    parser.add_argument("--original_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval/downstream")
    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--predictor_epochs", type=int, default=50)
    parser.add_argument("--predictor_lr", type=float, default=1e-3)
    parser.add_argument("--predictor_batch_size", type=int, default=64)
    args = parser.parse_args()
    run_downstream(args)
