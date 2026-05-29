"""
LLM-as-Judge 评估

用多个LLM对生成轨迹进行5维度打分:
1. Semantic Alignment (SA)
2. Event Alignment (EA)
3. Motion Plausibility (MP)
4. Interaction Realism (IR)
5. Trajectory Integrity (TI)

增加多样性维度:
6. Behavioral Diversity (BD) — 同一prompt多次生成的结果多样性
"""

import os
import json
from typing import List, Dict, Optional


JUDGE_PROMPT = """You are an expert evaluator for crowd simulation trajectories. Score the generated trajectories on a scale of 1-10 for each criterion.

## Scene Context
{scene_context}

## Crowd Dynamics Description
{crowd_dynamics}

## Generated Trajectory Summary
Number of pedestrians: {num_pedestrians}
Number of timesteps: {num_timesteps}
Trajectory statistics:
{trajectory_stats}

## Scoring Criteria

1. **Semantic Alignment (SA)**: How well do the crowd dynamics match the textual description?
   - 10: Perfect match with all described behaviors
   - 1: No relation to the description

2. **Event Alignment (EA)**: Do trajectories capture key events (meeting, splitting, stopping)?
   - 10: All described events are present
   - 1: No events captured

3. **Motion Plausibility (MP)**: Are velocity/acceleration changes smooth and natural?
   - 10: Perfectly natural human walking
   - 1: Physically impossible motions

4. **Interaction Realism (IR)**: Do interactions maintain plausible social distancing?
   - 10: Realistic social behaviors
   - 1: Constant collisions or no awareness

5. **Trajectory Integrity (TI)**: Are trajectories continuous and complete?
   - 10: All trajectories smooth and complete
   - 1: Broken, teleporting, or missing trajectories

Output ONLY valid JSON in this exact format:
{{"SA": X, "EA": X, "MP": X, "IR": X, "TI": X, "reasoning": "brief explanation"}}
"""


DIVERSITY_JUDGE_PROMPT = """You are evaluating the DIVERSITY of crowd simulation outputs. You are given {num_samples} different trajectory sets generated from the SAME textual description.

## Scene Context
{scene_context}

## Crowd Dynamics Description
{crowd_dynamics}

## Generated Samples Summary
{samples_summary}

## Scoring Criteria

1. **Trajectory Diversity (TD)**: How different are the generated trajectories across samples?
   - 10: Each sample shows distinctly different movement patterns
   - 1: All samples are nearly identical

2. **Plausible Diversity (PD)**: Are the diverse outputs all still plausible?
   - 10: All variants are realistic and match the description
   - 1: Diversity comes from noise/errors, not meaningful variation

3. **Behavioral Range (BR)**: How many distinct behavioral patterns are represented?
   - 10: Wide range of movement strategies (different speeds, paths, interactions)
   - 1: Single dominant pattern with minor variations

Output ONLY valid JSON:
{{"TD": X, "PD": X, "BR": X, "reasoning": "brief explanation"}}
"""


class LLMJudge:
    def __init__(self, provider: str, model: str, api_key: Optional[str] = None):
        self.provider = provider
        self.model = model
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        elif provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
            self.client = genai.GenerativeModel(model)
        elif provider == "deepseek":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
            )

    def _call_api(self, prompt: str) -> str:
        if self.provider in ("openai", "deepseek"):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        elif self.provider == "google":
            resp = self.client.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 500},
            )
            return resp.text.strip()

    def score_quality(
        self,
        scene_context: str,
        crowd_dynamics: str,
        trajectory_stats: str,
        num_pedestrians: int,
        num_timesteps: int,
    ) -> Dict:
        prompt = JUDGE_PROMPT.format(
            scene_context=scene_context,
            crowd_dynamics=crowd_dynamics,
            num_pedestrians=num_pedestrians,
            num_timesteps=num_timesteps,
            trajectory_stats=trajectory_stats,
        )
        response = self._call_api(prompt)
        try:
            scores = json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                scores = {"SA": 1, "EA": 1, "MP": 1, "IR": 1, "TI": 1, "reasoning": "parse error"}
        return scores

    def score_diversity(
        self,
        scene_context: str,
        crowd_dynamics: str,
        samples_summary: str,
        num_samples: int,
    ) -> Dict:
        prompt = DIVERSITY_JUDGE_PROMPT.format(
            scene_context=scene_context,
            crowd_dynamics=crowd_dynamics,
            samples_summary=samples_summary,
            num_samples=num_samples,
        )
        response = self._call_api(prompt)
        try:
            scores = json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                scores = {"TD": 1, "PD": 1, "BR": 1, "reasoning": "parse error"}
        return scores


def compute_trajectory_stats(traj: "np.ndarray") -> str:
    """生成轨迹统计摘要 (给LLM judge看)"""
    import numpy as np
    N, T, _ = traj.shape
    speeds = []
    for n in range(N):
        for t in range(1, T):
            if traj[n, t, 0] >= 0 and traj[n, t - 1, 0] >= 0:
                d = np.sqrt(
                    (traj[n, t, 0] - traj[n, t - 1, 0]) ** 2 +
                    (traj[n, t, 1] - traj[n, t - 1, 1]) ** 2
                )
                speeds.append(d)

    active_counts = []
    for t in range(T):
        count = sum(1 for n in range(N) if traj[n, t, 0] >= 0)
        active_counts.append(count)

    stats = f"- Average speed: {np.mean(speeds):.1f} px/frame\n"
    stats += f"- Speed range: [{np.min(speeds):.1f}, {np.max(speeds):.1f}]\n" if speeds else ""
    stats += f"- Active pedestrians per frame: {np.mean(active_counts):.1f} (range {min(active_counts)}-{max(active_counts)})\n"

    # 主方向
    if speeds:
        directions = []
        for n in range(N):
            valid = [(traj[n, t, 0], traj[n, t, 1])
                     for t in range(T) if traj[n, t, 0] >= 0]
            if len(valid) >= 2:
                dx = valid[-1][0] - valid[0][0]
                dy = valid[-1][1] - valid[0][1]
                angle = np.degrees(np.arctan2(dy, dx))
                directions.append(angle)
        if directions:
            stats += f"- Dominant direction: {np.mean(directions):.0f}° (std={np.std(directions):.0f}°)\n"

    return stats


def multi_judge_evaluation(
    judges: List[LLMJudge],
    test_samples: List[Dict],
    generated_trajectories: List["np.ndarray"],
    multi_sample_trajectories: Optional[Dict] = None,
) -> Dict:
    """
    多LLM judge综合评估

    Args:
        judges: LLM judge列表
        test_samples: 测试样本
        generated_trajectories: 生成的轨迹
        multi_sample_trajectories: {sample_idx: [K条轨迹]} 用于多样性评估

    Returns:
        综合评估结果
    """
    import numpy as np

    all_quality_scores = []
    all_diversity_scores = []

    for idx, (sample, gen_traj) in enumerate(zip(test_samples, generated_trajectories)):
        stats = compute_trajectory_stats(gen_traj)

        for judge in judges:
            # 质量评估
            q_scores = judge.score_quality(
                scene_context=sample.get("scenario_description", ""),
                crowd_dynamics=sample.get("crowd_dynamics", ""),
                trajectory_stats=stats,
                num_pedestrians=gen_traj.shape[0],
                num_timesteps=gen_traj.shape[1],
            )
            q_scores["judge"] = judge.model
            q_scores["sample_idx"] = idx
            all_quality_scores.append(q_scores)

            # 多样性评估 (如果有多次采样)
            if multi_sample_trajectories and idx in multi_sample_trajectories:
                multi_trajs = multi_sample_trajectories[idx]
                summaries = []
                for si, st in enumerate(multi_trajs):
                    summaries.append(f"Sample {si+1}: {compute_trajectory_stats(st)}")
                samples_summary = "\n".join(summaries)

                d_scores = judge.score_diversity(
                    scene_context=sample.get("scenario_description", ""),
                    crowd_dynamics=sample.get("crowd_dynamics", ""),
                    samples_summary=samples_summary,
                    num_samples=len(multi_trajs),
                )
                d_scores["judge"] = judge.model
                d_scores["sample_idx"] = idx
                all_diversity_scores.append(d_scores)

    # 聚合
    result = {
        "quality_scores": all_quality_scores,
        "diversity_scores": all_diversity_scores,
    }

    # 计算平均分
    quality_dims = ["SA", "EA", "MP", "IR", "TI"]
    for dim in quality_dims:
        vals = [s[dim] for s in all_quality_scores if dim in s]
        result[f"avg_{dim}"] = float(np.mean(vals)) if vals else 0.0
    result["avg_LLM_Score"] = float(np.mean([
        np.mean([s.get(d, 0) for d in quality_dims])
        for s in all_quality_scores
    ])) if all_quality_scores else 0.0

    diversity_dims = ["TD", "PD", "BR"]
    for dim in diversity_dims:
        vals = [s[dim] for s in all_diversity_scores if dim in s]
        result[f"avg_{dim}"] = float(np.mean(vals)) if vals else 0.0

    return result
