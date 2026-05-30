"""
FLINT 训练：Unsloth + LoRA SFT

支持：
  --use_coord_tokens / --no_coord_tokens  选择 tokenizer 模式
  --physics none / collision / all        选择物理 loss
"""

import os
import json
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback

from prepare_data import CoordTokenizer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_model_and_tokenizer(config: dict, use_coord_tokens: bool = True):
    from unsloth import FastLanguageModel

    tc = config["training"]
    model, llm_tokenizer = FastLanguageModel.from_pretrained(
        model_name=tc["backbone"],
        max_seq_length=tc["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
    )

    if use_coord_tokens:
        coord_tok = CoordTokenizer(
            resolution=tuple(config["data"]["resolution"]),
            bin_size=config["tokenizer"]["bin_size"],
        )
        num_added = llm_tokenizer.add_tokens(coord_tok.vocab)
        model.resize_token_embeddings(len(llm_tokenizer))
        print(f"Added {num_added} coordinate tokens")
    else:
        coord_tok = None

    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    return model, llm_tokenizer, coord_tok


def format_alpaca(sample: dict) -> str:
    return (
        "### Instruction:\n"
        f"{sample['instruction']}\n\n"
        "### Input:\n"
        f"{sample['input']}\n\n"
        "### Response:\n"
        f"{sample['output']}"
    )


def load_sft_dataset(data_path: str) -> Dataset:
    with open(data_path) as f:
        raw = json.load(f)
    texts = [format_alpaca(s) for s in raw]
    return Dataset.from_dict({"text": texts})


# ====================================================================
# Physics-aware loss wrapper
# ====================================================================
class PhysicsLoss:

    def __init__(self, config, coord_tok, llm_tokenizer, device):
        self.device = device
        tc = config["training"]["physics_loss"]
        self.collision_weight = tc.get("collision_weight", 0.1)
        self.smoothness_weight = tc.get("smoothness_weight", 0.05)
        self.walkable_weight = tc.get("walkable_weight", 0.05)
        self.collision_threshold = tc.get("collision_threshold", 10.0)

        bin_size = config["tokenizer"]["bin_size"]
        x_bins = config["tokenizer"]["x_bins"]
        y_bins = config["tokenizer"]["y_bins"]

        self.x_centers = torch.tensor(
            [(i - 0.5) * bin_size for i in range(1, x_bins + 1)],
            dtype=torch.float32, device=device,
        )
        self.y_centers = torch.tensor(
            [(j - 0.5) * bin_size for j in range(1, y_bins + 1)],
            dtype=torch.float32, device=device,
        )

        coord_tokens = coord_tok.vocab
        all_ids = [llm_tokenizer.convert_tokens_to_ids(t) for t in coord_tokens]
        self.x_token_ids = [tid for t, tid in zip(coord_tokens, all_ids) if t.startswith("<x_") and not t.startswith("<x_0>")]
        self.y_token_ids = [tid for t, tid in zip(coord_tokens, all_ids) if t.startswith("<y_") and not t.startswith("<y_0>")]

        self.walkable_map = None
        if self.walkable_weight > 0:
            scenario_dir = config["data"].get("scenario_dir", "./data/scenarios")
            grid_size = config["data"].get("map_grid_size", 10)
            self._load_walkable_maps(scenario_dir, grid_size, x_bins, y_bins, bin_size)

    def _load_walkable_maps(self, scenario_dir, grid_size, x_bins, y_bins, bin_size):
        merged = np.ones((480, 640), dtype=np.float32)
        loaded_any = False
        for scene in ["eth", "hotel", "univ", "zara1", "zara2"]:
            map_path = os.path.join(scenario_dir, f"{scene}.npy")
            if os.path.exists(map_path):
                loaded_any = True
        if not loaded_any:
            self.walkable_weight = 0
            return
        walkable_grid = np.ones((y_bins, x_bins), dtype=np.float32)
        for scene in ["eth", "hotel", "univ", "zara1", "zara2"]:
            map_path = os.path.join(scenario_dir, f"{scene}.npy")
            if os.path.exists(map_path):
                scene_map = np.load(map_path).astype(np.float32)
                scene_grid = scene_map[bin_size//2::bin_size, bin_size//2::bin_size]
                h = min(scene_grid.shape[0], y_bins)
                w = min(scene_grid.shape[1], x_bins)
                walkable_grid[:h, :w] = np.minimum(walkable_grid[:h, :w], scene_grid[:h, :w])
        self.walkable_map = torch.tensor(walkable_grid, dtype=torch.float32, device=self.device)

    def compute(self, logits: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=self.device)

        if not self.x_token_ids or not self.y_token_ids:
            return loss

        x_logits = logits[:, :, self.x_token_ids]
        y_logits = logits[:, :, self.y_token_ids]

        x_probs = torch.softmax(x_logits, dim=-1)
        y_probs = torch.softmax(y_logits, dim=-1)
        x_coords = (x_probs * self.x_centers.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        y_coords = (y_probs * self.y_centers.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        if self.smoothness_weight > 0 and x_coords.shape[1] > 3:
            vx = x_coords[:, 1:] - x_coords[:, :-1]
            vy = y_coords[:, 1:] - y_coords[:, :-1]
            ax = vx[:, 1:] - vx[:, :-1]
            ay = vy[:, 1:] - vy[:, :-1]
            jerk = torch.sqrt(ax ** 2 + ay ** 2 + 1e-6)
            loss = loss + self.smoothness_weight * jerk.mean()

        if self.collision_weight > 0 and x_coords.shape[1] > 2:
            diffs_x = x_coords[:, :, None] - x_coords[:, None, :]
            diffs_y = y_coords[:, :, None] - y_coords[:, None, :]
            dists = torch.sqrt(diffs_x ** 2 + diffs_y ** 2 + 1e-6)
            mask = torch.triu(torch.ones_like(dists[0]), diagonal=1).bool()
            penalty = torch.relu(self.collision_threshold - dists[:, mask])
            loss = loss + self.collision_weight * penalty.mean()

        if self.walkable_weight > 0 and self.walkable_map is not None:
            x_bin_probs = x_probs
            y_bin_probs = y_probs
            walkable_prob = torch.einsum('bti,btj,ij->bt', y_bin_probs, x_bin_probs, self.walkable_map)
            unwalkable_penalty = torch.relu(0.5 - walkable_prob)
            loss = loss + self.walkable_weight * unwalkable_penalty.mean()

        return loss


def wrap_with_physics(trainer, config, coord_tok, llm_tokenizer):
    physics = PhysicsLoss(config, coord_tok, llm_tokenizer, trainer.model.device)
    original_compute_loss = trainer.compute_loss

    def compute_loss_with_physics(model, inputs, return_outputs=False, **kwargs):
        loss, outputs = original_compute_loss(model, inputs, return_outputs=True, **kwargs)
        p_loss = physics.compute(outputs.logits)
        loss = loss + p_loss
        return (loss, outputs) if return_outputs else loss

    trainer.compute_loss = compute_loss_with_physics
    print("Physics-aware loss enabled (smoothness + collision)")
    return trainer


# ====================================================================
# Main training
# ====================================================================
def train(args):
    config = load_config(args.config)
    tc = config["training"]
    use_coord = not args.no_coord_tokens

    model, llm_tokenizer, coord_tok = prepare_model_and_tokenizer(config, use_coord)

    train_dataset = load_sft_dataset(args.train_data)
    eval_dataset = load_sft_dataset(args.test_data) if args.test_data and os.path.exists(args.test_data) else None

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset) if eval_dataset else 0}")

    output_dir = args.output_dir or os.path.join(tc["output_dir"], "main")

    callbacks = []
    if eval_dataset and tc.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=tc["early_stopping_patience"],
        ))

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=tc["epochs"],
        per_device_train_batch_size=tc["batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler"],
        warmup_ratio=tc["warmup_ratio"],
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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
        callbacks=callbacks,
    )

    if args.physics != "none" and use_coord and coord_tok:
        trainer = wrap_with_physics(trainer, config, coord_tok, llm_tokenizer)

    print(f"\nTraining: backbone={tc['backbone']}, coord_tokens={use_coord}, "
          f"physics={args.physics}, output={output_dir}")

    trainer.train()

    best_dir = os.path.join(output_dir, "best")
    model.save_pretrained(best_dir)
    llm_tokenizer.save_pretrained(best_dir)
    print(f"Model saved to {best_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--use_coord_tokens", action="store_true", default=True)
    parser.add_argument("--no_coord_tokens", action="store_true")
    parser.add_argument("--physics", type=str, default="none",
                        choices=["none", "collision", "all"])
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    train(args)
