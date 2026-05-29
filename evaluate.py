"""
FLINT 评估：质量指标 + 多样性指标 + LLM Judge

用法：
  python evaluate.py --results_dir results/main --test_data data/test.json --config config.yaml
  python evaluate.py --compare eval/main eval/ablation_raw_numbers ...
"""

import os
import re
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from scipy.spatial.distance import pdist
from scipy.stats import entropy

ABSENT = -1.0


# ====================================================================
# Quality metrics
# ====================================================================
def joint_average_displacement_error(pred: np.ndarray, gt: np.ndarray) -> float:
    N, T, _ = pred.shape
    total, count = 0.0, 0
    for n in range(N):
        for t in range(T):
            if gt[n, t, 0] != ABSENT and pred[n, t, 0] != ABSENT:
                total += np.sqrt((pred[n, t, 0] - gt[n, t, 0]) ** 2 +
                                 (pred[n, t, 1] - gt[n, t, 1]) ** 2)
                count += 1
    return total / max(count, 1)


def joint_final_displacement_error(pred: np.ndarray, gt: np.ndarray) -> float:
    N, T, _ = pred.shape
    total, count = 0.0, 0
    for n in range(N):
        for t in range(T - 1, -1, -1):
            if gt[n, t, 0] != ABSENT:
                if pred[n, t, 0] != ABSENT:
                    total += np.sqrt((pred[n, t, 0] - gt[n, t, 0]) ** 2 +
                                     (pred[n, t, 1] - gt[n, t, 1]) ** 2)
                    count += 1
                break
    return total / max(count, 1)


def collision_rate(traj: np.ndarray, threshold: float = 10.0) -> float:
    N, T, _ = traj.shape
    total_pairs, collisions = 0, 0
    for t in range(T):
        for i in range(N):
            if traj[i, t, 0] == ABSENT:
                continue
            for j in range(i + 1, N):
                if traj[j, t, 0] == ABSENT:
                    continue
                dist = np.sqrt((traj[i, t, 0] - traj[j, t, 0]) ** 2 +
                               (traj[i, t, 1] - traj[j, t, 1]) ** 2)
                total_pairs += 1
                if dist < threshold:
                    collisions += 1
    return collisions / max(total_pairs, 1)


def trajectory_smoothness(traj: np.ndarray) -> float:
    N, T, _ = traj.shape
    all_jerk = []
    for n in range(N):
        valid = []
        for t in range(T):
            if traj[n, t, 0] != ABSENT:
                valid.append(traj[n, t])
        if len(valid) < 4:
            continue
        pos = np.array(valid)
        vel = np.diff(pos, axis=0)
        acc = np.diff(vel, axis=0)
        jerk = np.sqrt((np.diff(acc, axis=0) ** 2).sum(axis=-1))
        all_jerk.extend(jerk.tolist())
    return float(np.mean(all_jerk)) if all_jerk else 0.0


def compute_quality(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    return {
        "JADE": joint_average_displacement_error(pred, gt),
        "JFDE": joint_final_displacement_error(pred, gt),
        "Collision_Rate": collision_rate(pred),
        "Smoothness": trajectory_smoothness(pred),
    }


# ====================================================================
# Diversity metrics
# ====================================================================
def _valid(traj):
    mask = (traj[:, 0] != ABSENT) & (traj[:, 1] != ABSENT)
    return traj[mask]


def average_trajectory_diversity(trajs: List[np.ndarray]) -> float:
    K = len(trajs)
    if K < 2:
        return 0.0
    total, count = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            min_len = min(len(trajs[i]), len(trajs[j]))
            if min_len == 0:
                continue
            diff = trajs[i][:min_len] - trajs[j][:min_len]
            total += np.sqrt((diff ** 2).sum(axis=-1) + 1e-8).mean()
            count += 1
    return total / max(count, 1)


def final_displacement_diversity(trajs: List[np.ndarray]) -> float:
    endpoints = []
    for t in trajs:
        v = _valid(t)
        if len(v) > 0:
            endpoints.append(v[-1])
    if len(endpoints) < 2:
        return 0.0
    return float(np.std(np.array(endpoints), axis=0).mean())


def spatial_coverage(trajs: List[np.ndarray], grid_size: int = 20,
                     bounds=(0, 0, 480, 360)) -> float:
    x_min, y_min, x_max, y_max = bounds
    visited = set()
    for t in trajs:
        for x, y in _valid(t):
            gi = int(np.clip((x - x_min) / (x_max - x_min) * grid_size, 0, grid_size - 1))
            gj = int(np.clip((y - y_min) / (y_max - y_min) * grid_size, 0, grid_size - 1))
            visited.add((gi, gj))
    return len(visited) / (grid_size * grid_size)


def endpoint_entropy(trajs: List[np.ndarray], num_bins: int = 10,
                     bounds=(0, 0, 480, 360)) -> float:
    x_min, y_min, x_max, y_max = bounds
    endpoints = []
    for t in trajs:
        v = _valid(t)
        if len(v) > 0:
            endpoints.append(v[-1])
    if len(endpoints) < 2:
        return 0.0
    eps = np.array(endpoints)
    gi = np.clip(((eps[:, 0] - x_min) / (x_max - x_min) * num_bins).astype(int), 0, num_bins - 1)
    gj = np.clip(((eps[:, 1] - y_min) / (y_max - y_min) * num_bins).astype(int), 0, num_bins - 1)
    cell_ids = gi * num_bins + gj
    counts = np.bincount(cell_ids, minlength=num_bins * num_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(entropy(probs))


def mode_count(trajs: List[np.ndarray], eps_scale: float = 0.6, min_samples: int = 2) -> int:
    from sklearn.cluster import DBSCAN
    features = []
    for t in trajs:
        v = _valid(t)
        if len(v) < 2:
            continue
        disps = np.diff(v, axis=0)
        speed = np.sqrt((disps ** 2).sum(axis=-1)).mean()
        direction = np.arctan2(v[-1, 1] - v[0, 1], v[-1, 0] - v[0, 0])
        features.append([v[0, 0], v[0, 1], v[-1, 0], v[-1, 1], speed, direction * 50])
    if len(features) < 3:
        return 1
    features = np.array(features)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    labels = DBSCAN(eps=eps_scale, min_samples=min_samples).fit_predict(features)
    n = len(set(labels)) - (1 if -1 in labels else 0)
    return max(n, 1)


def self_nearest_distance(trajs: List[np.ndarray]) -> float:
    K = len(trajs)
    if K < 2:
        return 0.0
    dist_mat = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            ml = min(len(trajs[i]), len(trajs[j]))
            if ml == 0:
                continue
            d = np.sqrt(((trajs[i][:ml] - trajs[j][:ml]) ** 2).sum(axis=-1) + 1e-8).mean()
            dist_mat[i, j] = dist_mat[j, i] = d
    nn = []
    for i in range(K):
        row = dist_mat[i].copy()
        row[i] = float("inf")
        nn.append(row.min())
    return float(np.mean(nn))


def compute_diversity(trajs_per_agent: Dict[int, List[np.ndarray]]) -> Dict[str, float]:
    all_atd, all_fdd, all_cov, all_ent, all_mc, all_snd = [], [], [], [], [], []
    for _, trajs in trajs_per_agent.items():
        if len(trajs) < 2:
            continue
        all_atd.append(average_trajectory_diversity(trajs))
        all_fdd.append(final_displacement_diversity(trajs))
        all_cov.append(spatial_coverage(trajs))
        all_ent.append(endpoint_entropy(trajs))
        all_mc.append(mode_count(trajs))
        all_snd.append(self_nearest_distance(trajs))
    return {
        "ATD": float(np.mean(all_atd)) if all_atd else 0.0,
        "FDD": float(np.mean(all_fdd)) if all_fdd else 0.0,
        "Coverage": float(np.mean(all_cov)) if all_cov else 0.0,
        "Endpoint_Entropy": float(np.mean(all_ent)) if all_ent else 0.0,
        "Mode_Count": float(np.mean(all_mc)) if all_mc else 0.0,
        "Self_Nearest_Dist": float(np.mean(all_snd)) if all_snd else 0.0,
    }


# ====================================================================
# LLM Judge
# ====================================================================
JUDGE_QUALITY_PROMPT = """You are an expert evaluator for crowd simulation trajectories. Score on 1-10 scale.

## Scene Context
{scene_context}

## Crowd Dynamics Description
{crowd_dynamics}

## Generated Trajectory Summary
{num_pedestrians} pedestrians, {num_timesteps} timesteps.
{trajectory_stats}

## Criteria
1. **Semantic Alignment (SA)**: Do crowd dynamics match the description?
2. **Event Alignment (EA)**: Are key events captured?
3. **Motion Plausibility (MP)**: Are motions smooth and natural?
4. **Interaction Realism (IR)**: Plausible social distancing?
5. **Trajectory Integrity (TI)**: Continuous and complete?

Output ONLY JSON: {{"SA": X, "EA": X, "MP": X, "IR": X, "TI": X, "reasoning": "..."}}"""

JUDGE_DIVERSITY_PROMPT = """You are evaluating DIVERSITY of {num_samples} trajectory sets from the SAME description.

## Scene: {scene_context}
## Description: {crowd_dynamics}
## Samples Summary:
{samples_summary}

## Criteria
1. **Trajectory Diversity (TD)**: How different are the samples?
2. **Plausible Diversity (PD)**: Are diverse outputs still plausible?
3. **Behavioral Range (BR)**: How many distinct behavior patterns?

Output ONLY JSON: {{"TD": X, "PD": X, "BR": X, "reasoning": "..."}}"""


class LLMJudge:
    def __init__(self, provider: str, model: str, api_key=None, base_url=None):
        self.provider = provider
        self.model = model
        if provider in ("openai", "deepseek"):
            from openai import OpenAI
            kwargs = {}
            if provider == "deepseek":
                kwargs["base_url"] = base_url or "https://api.deepseek.com/v1"
                kwargs["api_key"] = api_key or os.environ.get("DEEPSEEK_API_KEY")
            else:
                kwargs["api_key"] = api_key or os.environ.get("OPENAI_API_KEY")
            self.client = OpenAI(**kwargs)
        elif provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
            self.client = genai.GenerativeModel(model)

    def _call(self, prompt: str) -> str:
        if self.provider in ("openai", "deepseek"):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500, temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        else:
            resp = self.client.generate_content(
                prompt, generation_config={"temperature": 0.1, "max_output_tokens": 500},
            )
            return resp.text.strip()

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{[^}]+\}', text)
            return json.loads(m.group()) if m else {}

    def score_quality(self, scene, dynamics, stats, n_peds, n_steps) -> dict:
        prompt = JUDGE_QUALITY_PROMPT.format(
            scene_context=scene, crowd_dynamics=dynamics,
            trajectory_stats=stats, num_pedestrians=n_peds, num_timesteps=n_steps,
        )
        return self._parse_json(self._call(prompt))

    def score_diversity(self, scene, dynamics, summary, n_samples) -> dict:
        prompt = JUDGE_DIVERSITY_PROMPT.format(
            scene_context=scene, crowd_dynamics=dynamics,
            samples_summary=summary, num_samples=n_samples,
        )
        return self._parse_json(self._call(prompt))


def compute_trajectory_stats(traj: np.ndarray) -> str:
    N, T, _ = traj.shape
    speeds = []
    for n in range(N):
        for t in range(1, T):
            if traj[n, t, 0] >= 0 and traj[n, t - 1, 0] >= 0:
                speeds.append(np.sqrt((traj[n, t, 0] - traj[n, t - 1, 0]) ** 2 +
                                       (traj[n, t, 1] - traj[n, t - 1, 1]) ** 2))
    active = [sum(1 for n in range(N) if traj[n, t, 0] >= 0) for t in range(T)]
    s = f"- Avg speed: {np.mean(speeds):.1f} px/frame\n" if speeds else ""
    s += f"- Active peds/frame: {np.mean(active):.1f} ({min(active)}-{max(active)})\n"
    return s


# ====================================================================
# Main evaluation pipeline
# ====================================================================
def evaluate_model(results_dir, test_data_path, config, output_dir, run_llm_judge=True):
    from prepare_data import CoordTokenizer, decode_raw_numbers

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(results_dir, "generated.json")) as f:
        gen_results = json.load(f)
    with open(test_data_path) as f:
        test_data = json.load(f)

    test_lookup = {d["metadata"]["clip_id"]: d for d in test_data}

    coord_tok = CoordTokenizer(
        resolution=tuple(config["data"]["resolution"]),
        bin_size=config["tokenizer"]["bin_size"],
    )

    quality_all = []
    diversity_all = []

    print(f"Evaluating {len(gen_results)} clips from {results_dir}")

    for gr in gen_results:
        clip_id = gr["clip_id"]
        n_agents = gr["num_pedestrians"]
        n_steps = gr["num_timesteps"]
        use_coord = gr.get("use_coord_tokens", True)

        td = test_lookup.get(clip_id)
        if not td:
            continue

        # groundtruth
        if use_coord:
            gt_tokens = re.findall(r'<[^>]+>', td["output"])
            gt_traj = coord_tok.detokenize(gt_tokens, n_agents, n_steps)
        else:
            gt_traj = decode_raw_numbers(td["output"], n_agents, n_steps)

        # quality: use first generated sample
        pred_traj = np.array(gr["trajectories"][0])
        if pred_traj.ndim == 2:
            pred_traj = pred_traj.reshape(n_agents, n_steps, 2)
        q = compute_quality(pred_traj, gt_traj)
        q["clip_id"] = clip_id
        quality_all.append(q)

        # diversity: use all K samples
        trajs_per_agent = defaultdict(list)
        for traj_list in gr["trajectories"]:
            traj = np.array(traj_list)
            if traj.ndim == 2:
                traj = traj.reshape(n_agents, n_steps, 2)
            for n in range(n_agents):
                trajs_per_agent[n].append(traj[n])
        d = compute_diversity(trajs_per_agent)
        d["clip_id"] = clip_id
        diversity_all.append(d)

    # Aggregate
    results = {"model": results_dir}
    if quality_all:
        keys = [k for k in quality_all[0] if k != "clip_id"]
        results["quality"] = {k: float(np.mean([q[k] for q in quality_all])) for k in keys}
        print("\nQuality:")
        for k, v in results["quality"].items():
            print(f"  {k}: {v:.4f}")

    if diversity_all:
        keys = [k for k in diversity_all[0] if k != "clip_id"]
        results["diversity"] = {k: float(np.mean([d[k] for d in diversity_all])) for k in keys}
        print("\nDiversity:")
        for k, v in results["diversity"].items():
            print(f"  {k}: {v:.4f}")

    # LLM Judge
    if run_llm_judge:
        judges_cfg = config["evaluation"].get("llm_judges", [])
        judges = []
        for jm in judges_cfg:
            if "deepseek" in jm:
                judges.append(LLMJudge("deepseek", jm))
            elif "gpt" in jm:
                judges.append(LLMJudge("openai", jm))
            elif "gemini" in jm:
                judges.append(LLMJudge("google", jm))

        if judges:
            print(f"\nLLM Judge ({len(judges)} judges)...")
            q_scores, d_scores = [], []
            for gr in gen_results[:10]:
                clip_id = gr["clip_id"]
                td = test_lookup.get(clip_id)
                if not td:
                    continue
                pred = np.array(gr["trajectories"][0])
                if pred.ndim == 2:
                    pred = pred.reshape(gr["num_pedestrians"], gr["num_timesteps"], 2)
                stats = compute_trajectory_stats(pred)

                for judge in judges:
                    try:
                        qs = judge.score_quality(
                            td.get("input", "")[:200], td.get("input", "")[200:400],
                            stats, gr["num_pedestrians"], gr["num_timesteps"],
                        )
                        q_scores.append(qs)
                    except Exception as e:
                        print(f"  Judge error: {e}")

            if q_scores:
                for dim in ["SA", "EA", "MP", "IR", "TI"]:
                    vals = [s.get(dim, 0) for s in q_scores if dim in s]
                    if vals:
                        results.setdefault("llm_judge", {})[dim] = float(np.mean(vals))
                print("  LLM Judge scores:", results.get("llm_judge", {}))

    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {output_dir}/evaluation_results.json")
    return results


def compare_models(result_dirs: List[str], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    for d in result_dirs:
        path = os.path.join(d, "evaluation_results.json")
        if os.path.exists(path):
            with open(path) as f:
                all_results.append(json.load(f))

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    # Quality
    if all_results and "quality" in all_results[0]:
        print("\n--- Quality ---")
        keys = list(all_results[0]["quality"].keys())
        header = f"{'Model':<35}" + "".join(f"  {k:>12}" for k in keys)
        print(header)
        for r in all_results:
            row = f"{Path(r['model']).name:<35}"
            for k in keys:
                row += f"  {r.get('quality', {}).get(k, 0):>12.4f}"
            print(row)

    # Diversity
    if all_results and "diversity" in all_results[0]:
        print("\n--- Diversity ---")
        keys = list(all_results[0]["diversity"].keys())
        header = f"{'Model':<35}" + "".join(f"  {k:>12}" for k in keys)
        print(header)
        for r in all_results:
            row = f"{Path(r['model']).name:<35}"
            for k in keys:
                row += f"  {r.get('diversity', {}).get(k, 0):>12.4f}"
            print(row)

    with open(os.path.join(output_dir, "comparison.json"), "w") as f:
        json.dump({"models": all_results}, f, indent=2)
    print(f"\nComparison → {output_dir}/comparison.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--output_dir", type=str, default="eval/main")
    parser.add_argument("--no_llm_judge", action="store_true")
    parser.add_argument("--compare", nargs="+", type=str,
                        help="Compare multiple eval result dirs")
    args = parser.parse_args()

    if args.compare:
        compare_models(args.compare, args.output_dir)
    elif args.results_dir and args.test_data:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        evaluate_model(args.results_dir, args.test_data, config, args.output_dir,
                       run_llm_judge=not args.no_llm_judge)
    else:
        parser.print_help()
