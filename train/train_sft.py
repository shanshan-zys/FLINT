"""
FLINT 训练 Pipeline

基于 Unsloth + LoRA 的 SFT 训练, 支持:
1. 标准 cross-entropy loss
2. 可选的物理约束 auxiliary loss
3. 多种 coordinate tokenizer
4. 多种 LLM backbone
"""

import os
import sys
import json
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_model_and_tokenizer(config: dict):
    """
    加载模型和tokenizer, 添加 coordinate tokens, 配置 LoRA

    使用 Unsloth 加速 (4bit量化 + LoRA)
    """
    from unsloth import FastLanguageModel

    backbone = config["training"]["backbone"]
    max_seq_length = config["training"]["max_seq_length"]
    lora_cfg = config["training"]["lora"]

    model, llm_tokenizer = FastLanguageModel.from_pretrained(
        model_name=backbone,
        max_seq_length=max_seq_length,
        dtype=None,  # auto
        load_in_4bit=True,
    )

    # 添加 coordinate tokens
    from tokenizer import build_tokenizer
    coord_tokenizer = build_tokenizer(config)
    new_tokens = coord_tokenizer.vocab
    num_added = llm_tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(llm_tokenizer))
    print(f"Added {num_added} coordinate tokens (mode={coord_tokenizer.mode})")

    # 配置 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    return model, llm_tokenizer, coord_tokenizer


def format_alpaca_prompt(sample: dict) -> str:
    """将SFT数据格式化为Alpaca prompt"""
    return (
        "### Instruction:\n"
        f"{sample['instruction']}\n\n"
        "### Input:\n"
        f"{sample['input']}\n\n"
        "### Response:\n"
        f"{sample['output']}"
    )


def load_sft_dataset(data_path: str, llm_tokenizer) -> Dataset:
    """加载并tokenize SFT数据"""
    with open(data_path) as f:
        raw_data = json.load(f)

    texts = [format_alpaca_prompt(sample) for sample in raw_data]
    dataset = Dataset.from_dict({"text": texts})
    return dataset


def train(config_path: str, train_path: str, test_path: Optional[str] = None):
    """主训练函数"""
    config = load_config(config_path)
    tc = config["training"]

    # 准备模型
    model, llm_tokenizer, coord_tokenizer = prepare_model_and_tokenizer(config)

    # 加载数据
    train_dataset = load_sft_dataset(train_path, llm_tokenizer)
    eval_dataset = None
    if test_path and os.path.exists(test_path):
        eval_dataset = load_sft_dataset(test_path, llm_tokenizer)

    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples: {len(eval_dataset)}")

    # 训练配置
    output_dir = os.path.join(
        tc["output_dir"],
        f"{Path(tc['backbone']).name}_{coord_tokenizer.mode}",
    )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=tc["epochs"],
        per_device_train_batch_size=tc["batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler"],
        warmup_ratio=tc["warmup_ratio"],
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_seq_length=tc["max_seq_length"],
        dataset_text_field="text",
        packing=False,
        report_to="tensorboard",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=llm_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # 如果启用物理loss, 包装trainer
    if tc["physics_loss"]["enabled"]:
        trainer = wrap_with_physics_loss(trainer, config, coord_tokenizer, llm_tokenizer)

    print(f"\nStarting training...")
    print(f"  Backbone: {tc['backbone']}")
    print(f"  Tokenizer: {coord_tokenizer.mode}")
    print(f"  Physics loss: {tc['physics_loss']['enabled']}")
    print(f"  Output: {output_dir}")

    trainer.train()

    # 保存
    model.save_pretrained(os.path.join(output_dir, "final"))
    llm_tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"\nTraining complete. Model saved to {output_dir}/final")


def wrap_with_physics_loss(trainer, config, coord_tokenizer, llm_tokenizer):
    """
    包装 SFTTrainer, 在 compute_loss 中加入物理约束

    实现方式: monkey-patch trainer.compute_loss
    """
    from train.physics_loss import PhysicsAwareLoss

    tc = config["training"]["physics_loss"]
    physics_loss_fn = PhysicsAwareLoss(
        collision_weight=tc["collision_weight"],
        smoothness_weight=tc["smoothness_weight"],
        walkability_weight=tc["walkability_weight"],
    ).to(trainer.model.device)

    # 获取 coordinate token 的 id 范围
    coord_tokens = coord_tokenizer.vocab
    coord_token_ids = [llm_tokenizer.convert_tokens_to_ids(t) for t in coord_tokens]
    x_token_ids = [tid for t, tid in zip(coord_tokens, coord_token_ids) if t.startswith("<x_")]
    y_token_ids = [tid for t, tid in zip(coord_tokens, coord_token_ids) if t.startswith("<y_")]

    original_compute_loss = trainer.compute_loss

    def compute_loss_with_physics(model, inputs, return_outputs=False, **kwargs):
        # 先算标准 CE loss
        if return_outputs:
            loss, outputs = original_compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            loss = original_compute_loss(model, inputs, return_outputs=False, **kwargs)
            outputs = model(**inputs)

        # 提取 coordinate token 位置的 logits 计算物理loss
        # 这是近似方案: 从完整logits中提取x/y token的分数
        logits = outputs.logits  # (B, seq_len, vocab_size)

        if len(x_token_ids) > 0 and len(y_token_ids) > 0:
            x_logits = logits[:, :, x_token_ids]  # (B, seq_len, num_x_tokens)
            y_logits = logits[:, :, y_token_ids]  # (B, seq_len, num_y_tokens)

            # 简化: 对整个序列的平均值计算 smoothness
            x_pred = torch.softmax(x_logits, dim=-1)
            y_pred = torch.softmax(y_logits, dim=-1)

            x_centers = physics_loss_fn.x_centers.to(x_pred.device)
            y_centers = physics_loss_fn.y_centers.to(y_pred.device)
            x_coords = (x_pred * x_centers).sum(dim=-1)  # (B, seq_len)
            y_coords = (y_pred * y_centers).sum(dim=-1)

            # smoothness loss on predicted coordinates
            if x_coords.shape[1] > 3:
                vx = x_coords[:, 1:] - x_coords[:, :-1]
                vy = y_coords[:, 1:] - y_coords[:, :-1]
                ax = vx[:, 1:] - vx[:, :-1]
                ay = vy[:, 1:] - vy[:, :-1]
                jerk = torch.sqrt(ax ** 2 + ay ** 2 + 1e-6)
                physics_loss = tc["smoothness_weight"] * jerk.mean()
                loss = loss + physics_loss

        if return_outputs:
            return loss, outputs
        return loss

    trainer.compute_loss = compute_loss_with_physics
    print("Physics-aware loss wrapper enabled")
    return trainer


# ====================================================================
# 推理/生成
# ====================================================================
def generate_trajectories(
    model_path: str,
    config: dict,
    prompt_data: dict,
    num_samples: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
):
    """
    用训练好的模型生成轨迹

    Args:
        model_path: 模型路径
        config: 配置
        prompt_data: SFT格式的输入数据 (不含output)
        num_samples: 生成次数 (多样性评估时 >1)
        temperature: 采样温度
        top_p: nucleus sampling

    Returns:
        list of token strings
    """
    from unsloth import FastLanguageModel

    model, llm_tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config["training"]["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    prompt = (
        "### Instruction:\n"
        f"{prompt_data['instruction']}\n\n"
        "### Input:\n"
        f"{prompt_data['input']}\n\n"
        "### Response:\n"
    )

    inputs = llm_tokenizer(prompt, return_tensors="pt").to(model.device)
    results = []

    for _ in range(num_samples):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config["training"]["max_seq_length"] // 2,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLINT Training")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, default=None)
    args = parser.parse_args()

    train(args.config, args.train_data, args.test_data)
