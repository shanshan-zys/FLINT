"""
FLINT evaluation: visualization + metrics + LLM judge + diversity.

Usage:
  python evaluation.py --task_name main-ep20-... --metrics --vis_video --vis_image
  python evaluation.py --task_name main-ep20-... --llm_judge --llm_provider deepseek
  python evaluation.py --task_name main-ep20-... --diversity
"""

import os
import re
import json
import time
import argparse
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.stats import entropy


ABSENT = -1.0
PENALTY_POS = np.array([1000.0, 1000.0])


def _extract_subset(clip_id):
    m = re.match(r'(eth|hotel|univ|zara1|zara2)', clip_id)
    return m.group(1) if m else "unknown"


def load_results(output_base, task_name):
    path = os.path.join(output_base, "results", f"{task_name}.json")
    with open(path) as f:
        raw = json.load(f)
    samples = []
    for item in raw:
        population = item["metadata"]["population"]
        frames = item["metadata"]["frames"]
        gt = np.array(item["metadata"]["output"], dtype=np.float64)
        preds = []
        for traj_list in item["output_trajectories"]:
            preds.append(np.array(traj_list, dtype=np.float64))
        samples.append({
            "clip_id": item["clip_id"],
            "population": population,
            "frames": frames,
            "gt": gt,
            "preds": preds,
            "walkable_area": item["metadata"]["walkable_area"],
            "scenario_description": item["metadata"]["scenario_description"],
            "crowd_description": item["metadata"]["crowd_description"],
        })
    return samples


# ====================================================================
# JADE / JFDE (KD-tree nearest-neighbor)
# ====================================================================
def compute_jade(pred, gt, tolerant=True, gap_tolerance=3):
    """
    tolerant=True: 按ID对应收集pred点，缺失时向前填充（gap<=gap_tolerance），
                   pred_pts数量=当前帧gt有效人数（与reference一致）。
    tolerant=False: 严格模式，缺失直接用penalty，pred_pts包含所有人。
    """
    N, T, _ = gt.shape
    total_dist, total_count = 0.0, 0
    for t in range(T):
        gt_pts = []
        pred_pts = []
        if tolerant:
            for n in range(N):
                if gt[n, t, 0] == ABSENT or gt[n, t, 1] == ABSENT:
                    continue
                gt_pts.append(gt[n, t])
                if n < pred.shape[0] and pred[n, t, 0] != ABSENT and pred[n, t, 1] != ABSENT:
                    pred_pts.append(pred[n, t])
                else:
                    filled = False
                    if n < pred.shape[0]:
                        for prev_t in range(t - 1, max(t - gap_tolerance - 1, -1), -1):
                            if pred[n, prev_t, 0] != ABSENT and pred[n, prev_t, 1] != ABSENT:
                                gt_gap = sum(1 for tt in range(prev_t + 1, t)
                                             if gt[n, tt, 0] != ABSENT)
                                if gt_gap <= gap_tolerance:
                                    pred_pts.append(pred[n, prev_t])
                                    filled = True
                                    break
                    if not filled:
                        pred_pts.append(PENALTY_POS)
        else:
            for n in range(N):
                if gt[n, t, 0] != ABSENT and gt[n, t, 1] != ABSENT:
                    gt_pts.append(gt[n, t])
            if not gt_pts:
                continue
            for n in range(pred.shape[0]):
                if pred[n, t, 0] == ABSENT or pred[n, t, 1] == ABSENT:
                    pred_pts.append(PENALTY_POS)
                else:
                    pred_pts.append(pred[n, t])
        if not gt_pts:
            continue
        gt_arr = np.array(gt_pts)
        pred_arr = np.array(pred_pts)
        if len(pred_arr) == 0:
            total_dist += 1000.0 * len(gt_arr)
            total_count += len(gt_arr)
            continue
        tree = cKDTree(pred_arr)
        dists, _ = tree.query(gt_arr, k=1)
        total_dist += np.sum(dists)
        total_count += len(dists)
    return total_dist / max(total_count, 1)


def compute_jfde(pred, gt, tolerant=True, gap_tolerance=3):
    """
    tolerant=True: 按ID对应，缺失时向前填充（与reference一致）。
    tolerant=False: 严格模式，缺失直接用penalty。
    """
    N, T, _ = gt.shape
    last_frame = T - 1
    gt_pts = []
    pred_pts = []
    if tolerant:
        for n in range(N):
            if gt[n, last_frame, 0] == ABSENT or gt[n, last_frame, 1] == ABSENT:
                continue
            gt_pts.append(gt[n, last_frame])
            if n < pred.shape[0] and pred[n, last_frame, 0] != ABSENT and pred[n, last_frame, 1] != ABSENT:
                pred_pts.append(pred[n, last_frame])
            else:
                filled = False
                if n < pred.shape[0]:
                    for prev_t in range(last_frame - 1, max(last_frame - gap_tolerance - 1, -1), -1):
                        if pred[n, prev_t, 0] != ABSENT and pred[n, prev_t, 1] != ABSENT:
                            gt_gap = sum(1 for tt in range(prev_t + 1, last_frame)
                                         if gt[n, tt, 0] != ABSENT)
                            if gt_gap <= gap_tolerance:
                                pred_pts.append(pred[n, prev_t])
                                filled = True
                                break
                if not filled:
                    pred_pts.append(PENALTY_POS)
    else:
        for n in range(N):
            if gt[n, last_frame, 0] != ABSENT and gt[n, last_frame, 1] != ABSENT:
                gt_pts.append(gt[n, last_frame])
        for n in range(pred.shape[0]):
            if pred[n, last_frame, 0] == ABSENT or pred[n, last_frame, 1] == ABSENT:
                pred_pts.append(PENALTY_POS)
            else:
                pred_pts.append(pred[n, last_frame])
    if not gt_pts:
        return 0.0
    gt_arr = np.array(gt_pts)
    pred_arr = np.array(pred_pts)
    if len(pred_arr) == 0:
        return 1000.0
    tree = cKDTree(pred_arr)
    dists, _ = tree.query(gt_arr, k=1)
    return float(np.mean(dists))


def collision_rate(traj, threshold=10.0):
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


def trajectory_smoothness(traj):
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


def compute_metrics(pred, gt, tolerant=True):
    return {
        "JADE": compute_jade(pred, gt, tolerant=tolerant),
        "JFDE": compute_jfde(pred, gt, tolerant=tolerant),
        "Collision_Rate": collision_rate(pred),
        "Smoothness": trajectory_smoothness(pred),
    }


# ====================================================================
# Individual motion metrics (L, D, V, A, E, steerE, C)
# ====================================================================
def _compute_individual(traj, collision_radius=3.0):
    N, T, _ = traj.shape
    per_ped = []
    for n in range(N):
        valid_idx = [t for t in range(T) if traj[n, t, 0] != ABSENT]
        if len(valid_idx) < 2:
            continue
        pos = traj[n, valid_idx]
        duration = valid_idx[-1] - valid_idx[0]
        if duration == 0:
            duration = 1
        deltas = np.diff(pos, axis=0)
        step_lens = np.linalg.norm(deltas, axis=1)
        L = np.sum(step_lens) / duration

        start, end = pos[0], pos[-1]
        if np.allclose(start, end):
            D = 0.0
        else:
            line_vec = end - start
            norm_line = np.linalg.norm(line_vec)
            v = pos - start
            proj = (v.dot(line_vec) / norm_line ** 2)[:, None] * line_vec
            perp = np.linalg.norm(v - proj, axis=1)
            D = np.sum(perp)

        speeds = step_lens
        V = np.sum(np.abs(np.diff(speeds))) / duration

        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        dtheta = np.diff(angles)
        dtheta = np.mod(dtheta + np.pi, 2 * np.pi) - np.pi
        A = np.sum(np.abs(dtheta)) * 180.0 / np.pi / duration

        es, ew, m = 2.23, 1.26, 70
        E = np.sum(m * (es + ew * speeds ** 2)) / duration

        delta_v = np.diff(np.vstack(([[0, 0]], deltas)), axis=0)
        steerE = np.sum(m * (es + ew * np.sum(delta_v ** 2, axis=1))) / duration

        per_ped.append({"L": L, "D": D, "V": V, "A": A, "E": E, "steerE": steerE})

    C = _collision_rate_individual(traj, collision_radius)
    return per_ped, C


def _collision_rate_individual(traj, radius=3.0):
    N, T, _ = traj.shape
    rates = []
    for t in range(T):
        active = [(traj[n, t, 0], traj[n, t, 1]) for n in range(N) if traj[n, t, 0] != ABSENT]
        if not active:
            continue
        collided = set()
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                d = np.sqrt((active[i][0] - active[j][0]) ** 2 +
                            (active[i][1] - active[j][1]) ** 2)
                if d < 2 * radius:
                    collided.add(i)
                    collided.add(j)
        rates.append(len(collided) / len(active))
    return float(np.mean(rates)) if rates else 0.0


def compute_individual_metrics(pred, gt, collision_radius=3.0):
    pred_peds, pred_C = _compute_individual(pred, collision_radius)
    gt_peds, gt_C = _compute_individual(gt, collision_radius)

    keys = ["L", "D", "V", "A", "E", "steerE"]
    pred_mean = {k: float(np.mean([p[k] for p in pred_peds])) for k in keys} if pred_peds else {k: 0.0 for k in keys}
    gt_mean = {k: float(np.mean([p[k] for p in gt_peds])) for k in keys} if gt_peds else {k: 0.0 for k in keys}

    pred_mean["C"] = pred_C
    gt_mean["C"] = gt_C
    delta = {k: pred_mean[k] - gt_mean[k] for k in keys + ["C"]}
    return {"pred": pred_mean, "gt": gt_mean, "delta": delta}


# ====================================================================
# Video visualization
# ====================================================================
def visualize_video(sample, data_dir, output_dir, use_background=False):
    import cv2
    clip_id = sample["clip_id"]
    subset = _extract_subset(clip_id)
    pred = sample["preds"][0]
    N, T, _ = pred.shape

    if use_background:
        bg_path = os.path.join(data_dir, subset, f"{subset}.png")
        bg = cv2.imread(bg_path)
        if bg is None:
            print(f"  Warning: background not found: {bg_path}, skipping")
            return
        bg = cv2.resize(bg, (640, 480))
        frame_source = [bg.copy() for _ in range(T)]
    else:
        video_path = os.path.join(data_dir, subset, "videos", f"{clip_id}.mp4")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Warning: video not found: {video_path}, skipping")
            return
        frame_source = []
        for _ in range(T):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (640, 480))
            frame_source.append(frame)
        cap.release()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{clip_id}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, 2.5, (640, 480))

    rng = np.random.default_rng(42)
    colors = {n: tuple(map(int, rng.integers(50, 255, 3))) for n in range(N)}
    history = {n: [] for n in range(N)}

    for t in range(T):
        frame = frame_source[t].copy()
        for n in range(N):
            if pred[n, t, 0] != ABSENT:
                history[n].append((int(round(pred[n, t, 0])), int(round(pred[n, t, 1]))))
        for n in range(N):
            if len(history[n]) >= 2:
                pts = np.array(history[n], dtype=np.int32)
                cv2.polylines(frame, [pts], False, colors[n], 2, cv2.LINE_AA)
            if history[n]:
                cx, cy = history[n][-1]
                cv2.circle(frame, (cx, cy), 4, colors[n], -1)
                cv2.putText(frame, str(n + 1), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[n], 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


# ====================================================================
# Image visualization (side-by-side: GT left, Pred right)
# ====================================================================
def _draw_trajectories(img, traj, N, T, colors, label=None):
    import cv2
    for n in range(N):
        pts = []
        for t in range(T):
            if traj[n, t, 0] != ABSENT:
                pts.append((int(round(traj[n, t, 0])), int(round(traj[n, t, 1]))))
        if len(pts) >= 2:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts_arr], False, colors[n], 2, cv2.LINE_AA)
        if pts:
            cv2.circle(img, pts[0], 5, colors[n], -1)
            cv2.circle(img, pts[-1], 5, colors[n], 2)
    if label:
        cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)


def visualize_image(sample, data_dir, output_dir):
    import cv2
    clip_id = sample["clip_id"]
    subset = _extract_subset(clip_id)
    pred = sample["preds"][0]
    gt = sample["gt"]
    N, T, _ = pred.shape

    bg_path = os.path.join(data_dir, subset, f"{subset}.png")
    bg = cv2.imread(bg_path)
    if bg is None:
        print(f"  Warning: background not found: {bg_path}, skipping")
        return
    bg = cv2.resize(bg, (640, 480))

    rng = np.random.default_rng(42)
    colors = {n: tuple(map(int, rng.integers(50, 255, 3))) for n in range(N)}

    img_gt = bg.copy()
    img_pred = bg.copy()
    _draw_trajectories(img_gt, gt, N, T, colors, "GT")
    _draw_trajectories(img_pred, pred, N, T, colors, "Pred")

    gap = np.full((480, 10, 3), 40, dtype=np.uint8)
    canvas = np.hstack([img_gt, gap, img_pred])

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{clip_id}.png")
    cv2.imwrite(out_path, canvas)


# ====================================================================
# LLM Judge
# ====================================================================
def _traj_to_text(traj, population, frames):
    lines = []
    for t in range(frames):
        for n in range(population):
            if traj[n, t, 0] != ABSENT:
                x, y = int(round(traj[n, t, 0])), int(round(traj[n, t, 1]))
                lines.append(f"{t+1} {n+1} {x} {y}")
    return "\n".join(lines)


LLM_JUDGE_PROMPT = """You are an expert in crowd trajectory and behavior analysis. Your task is to evaluate and rate individual trajectories generated from textual descriptions on a scale of 1 to 10 based on the given grading criteria. Your judgment must be objective and impartial, combining knowledge of motion physics and semantic understanding.

### Textual Descriptions
{textual_descriptions}

### Generated Trajectories
The generated individual trajectories are stored in the data format that contains multiple lines with four values <frame, id, x, y>, where each line represents the position (x,y) of the id-th individual at time-step frame.
{generated_trajectories}

### Reference Trajectories
{reference_trajectories}

### Grading Criteria
You should score the generated trajectories on a scale of 1 to 10 (10 being best) for each of the following five dimensions.
1. Semantic Alignment (SA): How well do the generated crowd dynamics match the textual description?
2. Event Alignment (EA): Are the key events described in the text captured in the generated trajectories?
3. Motion Plausibility (MP): Are the individual motions smooth, natural, and physically plausible?
4. Interaction Realism (IR): Do pedestrians maintain plausible social distancing and exhibit realistic collision avoidance?
5. Trajectory Integrity (TI): Are trajectories continuous and complete, without sudden jumps or premature terminations?

### Output Instruction
You should reason step by step. First, extract key motion requirements from the textual descriptions. Then, check the generated trajectories against these requirements. Last, output a single formatted JSON object strictly following the output format below. Do not include any other texts, explanations, or markdown formattings outside the JSON.

{{
  "scores": {{
    "semantic_alignment": <integer_1_to_10>,
    "event_alignment": <integer_1_to_10>,
    "motion_plausibility": <integer_1_to_10>,
    "interaction_realism": <integer_1_to_10>,
    "trajectory_integrity": <integer_1_to_10>
  }},
  "overall_score": <float_or_integer_1_to_10>,
  "reasoning": "<A concise paragraph explaining your scoring rationale, referencing specific criteria and metrics.>"
}}"""


class LLMJudge:
    def __init__(self, provider, model=None, api_key=None, base_url=None):
        self.provider = provider
        if provider == "deepseek":
            from openai import OpenAI
            self.client = OpenAI(
                base_url=base_url or "https://api.deepseek.com/v1",
                api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            )
            self.model = model or "deepseek-chat"
        elif provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            )
            self.model = model or "gpt-4o"
        elif provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
            self.model = model or "gemini-2.0-flash"
            self.client = genai.GenerativeModel(self.model)

    def _call(self, prompt):
        if self.provider in ("deepseek", "openai"):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800, temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        else:
            resp = self.client.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 800},
            )
            return resp.text.strip()

    def _parse(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*\}', text)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            return {}

    def evaluate(self, sample, pred):
        population, frames = sample["population"], sample["frames"]
        gt = sample["gt"]

        textual = (f"Scenario: {sample['scenario_description']}\n\n"
                   f"Crowd dynamics: {sample['crowd_description']}")
        gen_text = _traj_to_text(pred, population, frames)
        ref_text = _traj_to_text(gt, population, frames)

        prompt = LLM_JUDGE_PROMPT.format(
            textual_descriptions=textual,
            generated_trajectories=gen_text,
            reference_trajectories=ref_text,
        )
        response = self._call(prompt)
        return self._parse(response)


# ====================================================================
# Diversity metrics
# ====================================================================
def _valid(traj):
    mask = (traj[:, 0] != ABSENT) & (traj[:, 1] != ABSENT)
    return traj[mask]


def average_trajectory_diversity(trajs):
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


def final_displacement_diversity(trajs):
    endpoints = []
    for t in trajs:
        v = _valid(t)
        if len(v) > 0:
            endpoints.append(v[-1])
    if len(endpoints) < 2:
        return 0.0
    return float(np.std(np.array(endpoints), axis=0).mean())


def spatial_coverage(trajs, grid_size=20, bounds=(0, 0, 640, 480)):
    x_min, y_min, x_max, y_max = bounds
    visited = set()
    for t in trajs:
        for x, y in _valid(t):
            gi = int(np.clip((x - x_min) / (x_max - x_min) * grid_size, 0, grid_size - 1))
            gj = int(np.clip((y - y_min) / (y_max - y_min) * grid_size, 0, grid_size - 1))
            visited.add((gi, gj))
    return len(visited) / (grid_size * grid_size)


def endpoint_entropy(trajs, num_bins=10, bounds=(0, 0, 640, 480)):
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


def mode_count(trajs, eps_scale=0.6, min_samples=2):
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


def self_nearest_distance(trajs):
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


def compute_diversity(preds):
    K = len(preds)
    if K < 2:
        return None
    N = preds[0].shape[0]
    trajs_per_agent = defaultdict(list)
    for pred in preds:
        for n in range(N):
            trajs_per_agent[n].append(_valid(pred[n]))

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
# Print helpers
# ====================================================================
def _print_table(title, per_clip, keys, group_key="clip_id"):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    by_subset = defaultdict(list)
    for item in per_clip:
        subset = _extract_subset(item[group_key])
        by_subset[subset].append(item)

    header = f"{'Subset':<12}" + "".join(f"{k:>14}" for k in keys)
    print(header)
    print("-" * len(header))
    for subset in sorted(by_subset):
        items = by_subset[subset]
        row = f"{subset:<12}"
        for k in keys:
            row += f"{np.mean([it[k] for it in items]):>14.4f}"
        print(row)

    row = f"{'Overall':<12}"
    for k in keys:
        row += f"{np.mean([it[k] for it in per_clip]):>14.4f}"
    print("-" * len(header))
    print(row)


# ====================================================================
# Main
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description="FLINT Evaluation")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_base", type=str, default="./outputs")

    parser.add_argument("--vis_video", action="store_true")
    parser.add_argument("--vis_image", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--individual_metrics", action="store_true")
    parser.add_argument("--llm_judge", action="store_true")
    parser.add_argument("--diversity", action="store_true")

    parser.add_argument("--use_background", action="store_true")
    parser.add_argument("--collision_radius", type=float, default=3.0)
    parser.add_argument("--tolerant", action="store_true", default=True,
                        help="Tolerant JADE/JFDE: ID-matched with gap filling (default)")
    parser.add_argument("--no_tolerant", action="store_true",
                        help="Strict JADE/JFDE: no gap filling, all preds in pool")

    parser.add_argument("--llm_provider", type=str, default="deepseek",
                        choices=["deepseek", "openai", "google"])
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--llm_api_key", type=str, default=None)
    parser.add_argument("--llm_base_url", type=str, default=None)

    args = parser.parse_args()

    tolerant = not args.no_tolerant

    samples = load_results(args.output_base, args.task_name)
    K = len(samples[0]["preds"])
    task_dir = os.path.join(args.output_base, "results", args.task_name)
    vis_dir = os.path.join(task_dir, "visualization")

    print(f"Task: {args.task_name}")
    print(f"Clips: {len(samples)}, Samples/clip: {K}")
    print(f"JADE/JFDE mode: {'tolerant' if tolerant else 'strict'}")

    # ── Metrics ───────────────────────────────────────────────────
    if args.metrics:
        print("\nComputing JADE/JFDE/Collision/Smoothness...")
        per_clip = []
        for sample in samples:
            clip_metrics_list = []
            for pred in sample["preds"]:
                clip_metrics_list.append(compute_metrics(pred, sample["gt"], tolerant=tolerant))
            avg = {}
            for key in clip_metrics_list[0]:
                avg[key] = float(np.mean([m[key] for m in clip_metrics_list]))
            avg["clip_id"] = sample["clip_id"]
            per_clip.append(avg)

        keys = ["JADE", "JFDE", "Collision_Rate", "Smoothness"]
        _print_table("Quality Metrics", per_clip, keys)

        os.makedirs(task_dir, exist_ok=True)
        agg = {k: float(np.mean([p[k] for p in per_clip])) for k in keys}
        out = {"per_clip": per_clip, "aggregate": agg}

        if args.individual_metrics:
            print("\nComputing individual motion metrics (L/D/V/A/E/steerE/C)...")
            indiv_per_clip = []
            for sample in samples:
                indiv_list = []
                for pred in sample["preds"]:
                    indiv_list.append(compute_individual_metrics(
                        pred, sample["gt"], args.collision_radius))
                avg_delta = {}
                for key in indiv_list[0]["delta"]:
                    avg_delta[key] = float(np.mean([m["delta"][key] for m in indiv_list]))
                avg_pred = {}
                for key in indiv_list[0]["pred"]:
                    avg_pred[key] = float(np.mean([m["pred"][key] for m in indiv_list]))
                avg_gt = {}
                for key in indiv_list[0]["gt"]:
                    avg_gt[key] = float(np.mean([m["gt"][key] for m in indiv_list]))
                indiv_per_clip.append({
                    "clip_id": sample["clip_id"],
                    "pred": avg_pred, "gt": avg_gt, "delta": avg_delta,
                })

            delta_keys = ["L", "D", "V", "A", "E", "steerE", "C"]
            flat_for_print = []
            for item in indiv_per_clip:
                flat = {"clip_id": item["clip_id"]}
                for k in delta_keys:
                    flat[k] = item["delta"][k]
                flat_for_print.append(flat)
            _print_table("Individual Metrics (delta = pred - gt)", flat_for_print, delta_keys)

            out["individual_metrics"] = {
                "per_clip": indiv_per_clip,
                "aggregate_delta": {k: float(np.mean([p["delta"][k] for p in indiv_per_clip])) for k in delta_keys},
            }

        metrics_path = os.path.join(task_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nMetrics saved: {metrics_path}")

    # ── Video visualization ───────────────────────────────────────
    if args.vis_video:
        print("\nGenerating visualization videos...")
        for i, sample in enumerate(samples):
            visualize_video(sample, args.data_dir, vis_dir, args.use_background)
            print(f"  [{i+1}/{len(samples)}] {sample['clip_id']}.mp4")
        print(f"Videos saved: {vis_dir}/")

    # ── Image visualization ───────────────────────────────────────
    if args.vis_image:
        print("\nGenerating trajectory images...")
        for i, sample in enumerate(samples):
            visualize_image(sample, args.data_dir, vis_dir)
            print(f"  [{i+1}/{len(samples)}] {sample['clip_id']}.png")
        print(f"Images saved: {vis_dir}/")

    # ── LLM Judge ─────────────────────────────────────────────────
    if args.llm_judge:
        print(f"\nRunning LLM Judge ({args.llm_provider})...")
        judge = LLMJudge(args.llm_provider, args.llm_model,
                         args.llm_api_key, args.llm_base_url)
        judge_results = []
        for i, sample in enumerate(samples):
            clip_scores = []
            for pred in sample["preds"]:
                try:
                    score = judge.evaluate(sample, pred)
                    clip_scores.append(score)
                except Exception as e:
                    print(f"  Error on {sample['clip_id']}: {e}")
                time.sleep(1)

            if clip_scores:
                scores_with_values = [s for s in clip_scores if "scores" in s]
                if scores_with_values:
                    avg_scores = {}
                    for dim in ["semantic_alignment", "event_alignment", "motion_plausibility",
                                "interaction_realism", "trajectory_integrity"]:
                        vals = [s["scores"].get(dim, 0) for s in scores_with_values]
                        avg_scores[dim] = float(np.mean(vals))
                    overall = [s.get("overall_score", 0) for s in scores_with_values]
                    judge_results.append({
                        "clip_id": sample["clip_id"],
                        "scores": avg_scores,
                        "overall_score": float(np.mean(overall)),
                        "num_samples": len(scores_with_values),
                    })

            print(f"  [{i+1}/{len(samples)}] {sample['clip_id']}")

        if judge_results:
            dims = ["semantic_alignment", "event_alignment", "motion_plausibility",
                    "interaction_realism", "trajectory_integrity"]
            agg = {d: float(np.mean([r["scores"][d] for r in judge_results])) for d in dims}
            agg["overall_score"] = float(np.mean([r["overall_score"] for r in judge_results]))

            flat = []
            for r in judge_results:
                flat.append({"clip_id": r["clip_id"],
                             "SA": r["scores"]["semantic_alignment"],
                             "EA": r["scores"]["event_alignment"],
                             "MP": r["scores"]["motion_plausibility"],
                             "IR": r["scores"]["interaction_realism"],
                             "TI": r["scores"]["trajectory_integrity"],
                             "Overall": r["overall_score"]})
            _print_table("LLM Judge Scores", flat, ["SA", "EA", "MP", "IR", "TI", "Overall"])

            os.makedirs(task_dir, exist_ok=True)
            judge_path = os.path.join(task_dir, "llm_judge.json")
            with open(judge_path, "w") as f:
                json.dump({"per_clip": judge_results, "aggregate": agg}, f, indent=2)
            print(f"LLM Judge saved: {judge_path}")

    # ── Diversity ─────────────────────────────────────────────────
    if args.diversity:
        if K < 2:
            print("\nDiversity: skipped (only 1 sample per clip, need K>=2)")
        else:
            print("\nComputing diversity metrics...")
            div_per_clip = []
            for sample in samples:
                d = compute_diversity(sample["preds"])
                if d is not None:
                    d["clip_id"] = sample["clip_id"]
                    div_per_clip.append(d)

            if div_per_clip:
                div_keys = ["ATD", "FDD", "Coverage", "Endpoint_Entropy", "Mode_Count", "Self_Nearest_Dist"]
                _print_table("Diversity Metrics", div_per_clip, div_keys)

                agg = {k: float(np.mean([p[k] for p in div_per_clip])) for k in div_keys}
                os.makedirs(task_dir, exist_ok=True)
                div_path = os.path.join(task_dir, "diversity.json")
                with open(div_path, "w") as f:
                    json.dump({"per_clip": div_per_clip, "aggregate": agg}, f, indent=2)
                print(f"Diversity saved: {div_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
