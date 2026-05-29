"""
API 标注脚本

支持三种模式:
1. openai: GPT-4o / GPT-4o-mini (图像+文本)
2. google: Gemini-1.5-Flash / Pro (最便宜的图像标注)
3. hybrid: 本地VLM初标 + API精修 (性价比最优)
"""

import os
import json
import time
import base64
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from .prompts import (
    SCENARIO_DESCRIPTION_PROMPT,
    CROWD_DYNAMICS_PROMPT,
    CROWD_DYNAMICS_VISUAL_PROMPT,
    CROWD_DYNAMICS_TEXT_ONLY_PROMPT,
    DIVERSE_DESCRIPTION_PROMPT,
    QUALITY_CHECK_PROMPT,
)


@dataclass
class AnnotationSample:
    clip_id: str
    dataset: str
    subset: str
    frame_start: int
    frame_end: int
    num_pedestrians: int
    trajectories: Dict  # {ped_id: [(x, y), ...]}
    background_image_path: Optional[str] = None
    frame_paths: Optional[List[str]] = None


def format_trajectories_text(trajectories: Dict, max_peds: int = 30) -> str:
    """将轨迹格式化为文本"""
    lines = []
    for pid, coords in list(trajectories.items())[:max_peds]:
        coord_str = ", ".join([f"({x:.0f},{y:.0f})" for x, y in coords])
        lines.append(f"  Ped {pid}: [{coord_str}]")
    if len(trajectories) > max_peds:
        lines.append(f"  ... and {len(trajectories) - max_peds} more pedestrians")
    return "\n".join(lines)


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ====================================================================
# OpenAI API
# ====================================================================
class OpenAIAnnotator:
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def annotate_scenario(self, image_path: str) -> str:
        img_b64 = encode_image_base64(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": SCENARIO_DESCRIPTION_PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                        "detail": "low",  # low detail = 更便宜
                    }},
                ],
            }],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def annotate_crowd_dynamics(
        self,
        sample: AnnotationSample,
        scenario_desc: str,
        use_images: bool = False,
    ) -> str:
        traj_text = format_trajectories_text(sample.trajectories)

        if use_images and sample.frame_paths:
            # 带图像的标注 (更贵但更准)
            prompt = CROWD_DYNAMICS_VISUAL_PROMPT.format(
                num_frames=len(sample.frame_paths),
                scenario_description=scenario_desc,
                num_pedestrians=sample.num_pedestrians,
            )
            content = [{"type": "text", "text": prompt}]
            # 只用首/中/末3帧, 省token
            selected_frames = [
                sample.frame_paths[0],
                sample.frame_paths[len(sample.frame_paths) // 2],
                sample.frame_paths[-1],
            ]
            for fp in selected_frames:
                img_b64 = encode_image_base64(fp)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"},
                })
        else:
            # 纯文本标注 (最便宜)
            prompt = CROWD_DYNAMICS_PROMPT.format(
                duration=(sample.frame_end - sample.frame_start) / 2.5,
                num_frames=sample.frame_end - sample.frame_start + 1,
                fps=2.5,
                num_pedestrians=sample.num_pedestrians,
                trajectory_text=traj_text,
            )
            content = [{"type": "text", "text": prompt}]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=500,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def generate_diverse_descriptions(
        self,
        scenario_desc: str,
        original_dynamics: str,
        num_variants: int = 5,
    ) -> List[str]:
        """为同一场景生成多样化描述 (多样性增强的关键)"""
        prompt = DIVERSE_DESCRIPTION_PROMPT.format(
            num_variants=num_variants,
            scenario_description=scenario_desc,
            original_dynamics=original_dynamics,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.9,  # 高温度 → 更多样
        )
        text = response.choices[0].message.content.strip()
        variants = [v.strip() for v in text.split("---") if v.strip()]
        return variants


# ====================================================================
# Google Gemini API (最便宜的图像标注方案)
# ====================================================================
class GeminiAnnotator:
    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        import google.generativeai as genai
        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)

    def annotate_scenario(self, image_path: str) -> str:
        from PIL import Image
        img = Image.open(image_path)
        response = self.model.generate_content(
            [SCENARIO_DESCRIPTION_PROMPT, img],
            generation_config={"temperature": 0.3, "max_output_tokens": 200},
        )
        return response.text.strip()

    def annotate_crowd_dynamics(
        self,
        sample: AnnotationSample,
        scenario_desc: str,
        use_images: bool = False,
    ) -> str:
        traj_text = format_trajectories_text(sample.trajectories)
        prompt = CROWD_DYNAMICS_PROMPT.format(
            duration=(sample.frame_end - sample.frame_start) / 2.5,
            num_frames=sample.frame_end - sample.frame_start + 1,
            fps=2.5,
            num_pedestrians=sample.num_pedestrians,
            trajectory_text=traj_text,
        )
        parts = [prompt]
        if use_images and sample.frame_paths:
            from PIL import Image
            parts.append(Image.open(sample.frame_paths[0]))
        response = self.model.generate_content(
            parts,
            generation_config={"temperature": 0.3, "max_output_tokens": 500},
        )
        return response.text.strip()

    def generate_diverse_descriptions(
        self,
        scenario_desc: str,
        original_dynamics: str,
        num_variants: int = 5,
    ) -> List[str]:
        prompt = DIVERSE_DESCRIPTION_PROMPT.format(
            num_variants=num_variants,
            scenario_description=scenario_desc,
            original_dynamics=original_dynamics,
        )
        response = self.model.generate_content(
            [prompt],
            generation_config={"temperature": 0.9, "max_output_tokens": 1500},
        )
        text = response.text.strip()
        return [v.strip() for v in text.split("---") if v.strip()]


# ====================================================================
# 本地 VLM 标注 (完全免费, 需要GPU)
# ====================================================================
class LocalVLMAnnotator:
    """用 Qwen2.5-VL-7B 做本地标注, 零API成本"""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()

    def annotate_scenario(self, image_path: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": SCENARIO_DESCRIPTION_PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image_path], return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=200, temperature=0.3)
        output_text = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        return output_text.strip()

    def annotate_crowd_dynamics(
        self,
        sample: AnnotationSample,
        scenario_desc: str,
        use_images: bool = False,
    ) -> str:
        traj_text = format_trajectories_text(sample.trajectories)
        prompt = CROWD_DYNAMICS_TEXT_ONLY_PROMPT.format(
            scenario_description=scenario_desc,
            width=480, height=360,
            duration=(sample.frame_end - sample.frame_start) / 2.5,
            num_frames=sample.frame_end - sample.frame_start + 1,
            trajectory_text=traj_text,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=500, temperature=0.3)
        output_text = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        return output_text.strip()


# ====================================================================
# 批量标注 Pipeline
# ====================================================================
def build_annotator(config: dict):
    provider = config["annotation"]["provider"]
    model = config["annotation"]["model"]
    if provider == "openai":
        return OpenAIAnnotator(model=model)
    elif provider == "google":
        return GeminiAnnotator(model=model)
    elif provider == "local":
        return LocalVLMAnnotator(model_name=config["annotation"]["local_model"])
    else:
        raise ValueError(f"Unknown provider: {provider}")


def annotate_dataset(
    samples: List[AnnotationSample],
    config: dict,
    output_path: str,
    scenario_cache: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    批量标注数据集

    推荐流程:
    1. 先用 Gemini-Flash 标注全部 (~$1/1000clip)
    2. 对10%样本用 GPT-4o 做质检
    3. 对质检不合格的重新用 GPT-4o 标注
    """
    annotator = build_annotator(config)
    scenario_cache = scenario_cache or {}
    results = []

    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Annotating {sample.clip_id}...")

        # 场景描述 (同一场景只标注一次)
        scene_key = f"{sample.dataset}_{sample.subset}"
        if scene_key not in scenario_cache:
            if sample.background_image_path:
                scenario_cache[scene_key] = annotator.annotate_scenario(
                    sample.background_image_path
                )
            else:
                scenario_cache[scene_key] = f"A pedestrian scene from the {sample.subset} subset"

        scenario_desc = scenario_cache[scene_key]

        # 人群动态描述
        crowd_desc = annotator.annotate_crowd_dynamics(
            sample, scenario_desc, use_images=bool(sample.frame_paths)
        )

        result = {
            "clip_id": sample.clip_id,
            "dataset": sample.dataset,
            "subset": sample.subset,
            "scenario_description": scenario_desc,
            "crowd_dynamics": crowd_desc,
            "num_pedestrians": sample.num_pedestrians,
            "frame_range": [sample.frame_start, sample.frame_end],
            "trajectories": {
                str(k): [[float(x), float(y)] for x, y in v]
                for k, v in sample.trajectories.items()
            },
        }
        results.append(result)

        # 每10个样本保存一次 (防止中断丢失)
        if (i + 1) % 10 == 0:
            _save_results(results, output_path)
            print(f"  Saved {len(results)} annotations to {output_path}")

        time.sleep(0.5)  # rate limit

    _save_results(results, output_path)
    print(f"Done! {len(results)} annotations saved to {output_path}")
    return results


def _save_results(results: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ====================================================================
# 多样性增强: 为已有标注生成变体描述
# ====================================================================
def augment_with_diverse_descriptions(
    annotations_path: str,
    output_path: str,
    config: dict,
    num_variants: int = 5,
):
    """
    对每个已有标注, 生成多样化的text描述变体
    这是证明"多样性"的关键数据来源
    """
    with open(annotations_path) as f:
        annotations = json.load(f)

    annotator = build_annotator(config)
    augmented = []

    for ann in annotations:
        augmented.append(ann)  # 保留原始
        if hasattr(annotator, "generate_diverse_descriptions"):
            variants = annotator.generate_diverse_descriptions(
                scenario_desc=ann["scenario_description"],
                original_dynamics=ann["crowd_dynamics"],
                num_variants=num_variants,
            )
            for vi, variant in enumerate(variants):
                aug_ann = ann.copy()
                aug_ann["clip_id"] = f"{ann['clip_id']}_var{vi}"
                aug_ann["crowd_dynamics"] = variant
                aug_ann["is_augmented"] = True
                augmented.append(aug_ann)

    _save_results(augmented, output_path)
    print(f"Augmented {len(annotations)} → {len(augmented)} samples")
    return augmented
