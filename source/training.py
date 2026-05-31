"""
FLINT training: LoRA SFT with physics-aware loss.

Supports:
  --use_coord_tokens / --no_coord_tokens   tokenizer mode
  --use_raw_prompt                         tokenizer ablation (metadata.raw_prompt)
  --physics none / collision / all         physics loss selection
  --task_name                              names output dirs under loss/ and checkpoint/
"""

import os
import json
import random
import torch
import argparse
import numpy as np

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

from dataset_construction import CoordTokenizer


# ====================================================================
# Alpaca format
# ====================================================================
def format_alpaca(sample: dict) -> str:
    return (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n"
        "### Instruction:\n"
        f"{sample['instruction']}\n\n"
        "### Input:\n"
        f"{sample['input']}\n\n"
        "### Response:\n"
        f"{sample['output']}"
    )


# ====================================================================
# Data loading: single JSON, seed-42 split 8:2
# ====================================================================
def load_and_split(data_path, seed=42, use_raw_prompt=False):
    with open(data_path) as f:
        raw = json.load(f)

    indices = list(range(len(raw)))
    random.Random(seed).shuffle(indices)
    split = int(len(indices) * 0.8)
    train_idx, eval_idx = indices[:split], indices[split:]

    def build(idx_list):
        texts, sample_indices, metadata_list = [], [], []
        for new_i, orig_i in enumerate(idx_list):
            s = raw[orig_i]
            if use_raw_prompt:
                fields = s["metadata"]["raw_prompt"]
            else:
                fields = s
            texts.append(format_alpaca(fields))
            sample_indices.append(new_i)

            meta = s["metadata"]
            metadata_list.append({
                "clip_name": meta["clip_name"],
                "population": meta["population"],
                "frames": meta["frames"],
                "walkable_area": meta["walkable_area"],
                "trajectory": meta["trajectory"],
            })
        ds = Dataset.from_dict({"text": texts, "sample_idx": sample_indices})
        return ds, metadata_list

    train_ds, train_meta = build(train_idx)
    eval_ds, eval_meta = build(eval_idx)
    print(f"Data split (seed={seed}): train={len(train_ds)}, eval={len(eval_ds)}")
    return train_ds, eval_ds, train_meta, eval_meta


# ====================================================================
# Model and tokenizer
# ====================================================================
def prepare_model_and_tokenizer(backbone, use_coord_tokens,
                                resolution, bin_size):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        backbone,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(
        backbone, trust_remote_code=True,
    )
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    coord_tok = None
    if use_coord_tokens:
        coord_tok = CoordTokenizer(resolution=resolution, bin_size=bin_size)
        num_added = llm_tokenizer.add_tokens(coord_tok.vocab)
        model.resize_token_embeddings(len(llm_tokenizer))
        print(f"Added {num_added} coordinate tokens")

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, llm_tokenizer, coord_tok


# ====================================================================
# Physics-aware loss (differentiable, per-sample, masked)
# ====================================================================
class PhysicsLoss:

    def __init__(self, coord_tok, llm_tokenizer, metadata_list, config, device):
        self.device = device
        self.collision_weight = config["collision_weight"]
        self.smoothness_weight = config["smoothness_weight"]
        self.walkable_weight = config["walkable_weight"]
        self.collision_threshold = config["collision_threshold"]

        bin_size = coord_tok.bin_size
        x_bins = coord_tok.x_bins
        y_bins = coord_tok.y_bins

        # soft-position centers (exclude bin 0 which is the absent placeholder)
        self.x_centers = torch.tensor(
            [(i - 0.5) * bin_size for i in range(1, x_bins + 1)],
            dtype=torch.float32, device=device,
        )
        self.y_centers = torch.tensor(
            [(j - 0.5) * bin_size for j in range(1, y_bins + 1)],
            dtype=torch.float32, device=device,
        )

        # token id sets for fast lookup
        all_tokens = coord_tok.vocab
        all_ids = [llm_tokenizer.convert_tokens_to_ids(t) for t in all_tokens]
        self.x_token_ids = []
        self.y_token_ids = []
        self.x0_id = llm_tokenizer.convert_tokens_to_ids("<x_0>")
        self.y0_id = llm_tokenizer.convert_tokens_to_ids("<y_0>")

        for tok, tid in zip(all_tokens, all_ids):
            if tok.startswith("<x_") and tok != "<x_0>":
                self.x_token_ids.append(tid)
            elif tok.startswith("<y_") and tok != "<y_0>":
                self.y_token_ids.append(tid)

        self.x_id_set = set(self.x_token_ids + [self.x0_id])
        self.y_id_set = set(self.y_token_ids + [self.y0_id])

        # precompute walkable grids per scene (deduplicate by clip prefix)
        self.walkable_grids = {}
        if self.walkable_weight > 0:
            self._build_walkable_grids(metadata_list, bin_size, x_bins, y_bins)

        self.metadata_list = metadata_list

    def _build_walkable_grids(self, metadata_list, bin_size, x_bins, y_bins):
        seen = set()
        for meta in metadata_list:
            clip = meta["clip_name"]
            scene = clip.rstrip("0123456789")
            if scene in seen:
                continue
            seen.add(scene)
            wa = meta["walkable_area"]
            wa_np = np.array(wa, dtype=np.float32)
            grid = wa_np[bin_size // 2::bin_size, bin_size // 2::bin_size]
            h = min(grid.shape[0], y_bins)
            w = min(grid.shape[1], x_bins)
            full_grid = np.ones((y_bins, x_bins), dtype=np.float32)
            full_grid[:h, :w] = grid[:h, :w]
            self.walkable_grids[scene] = torch.tensor(
                full_grid, dtype=torch.float32, device=self.device
            )

    def _get_walkable_grid(self, clip_name):
        scene = clip_name.rstrip("0123456789")
        return self.walkable_grids.get(scene)

    def compute(self, logits, labels, sample_indices):
        """Compute physics losses over the batch. Returns dict of scalar tensors."""
        zero = torch.tensor(0.0, device=self.device, requires_grad=True)
        collision_loss = zero
        smoothness_loss = zero
        walkable_loss = zero
        count = 0

        B = logits.shape[0]
        for b in range(B):
            idx = sample_indices[b].item()
            if idx < 0 or idx >= len(self.metadata_list):
                continue
            meta = self.metadata_list[idx]
            N = meta["population"]
            T = meta["frames"]

            lab = labels[b]  # (seq_len,)
            log = logits[b]  # (seq_len, vocab)

            # find x-token and y-token positions in labels
            x_positions = []
            y_positions = []
            for pos in range(lab.shape[0]):
                lid = lab[pos].item()
                if lid == -100:
                    continue
                if lid in self.x_id_set:
                    x_positions.append(pos)
                elif lid in self.y_id_set:
                    y_positions.append(pos)

            n_pairs = min(len(x_positions), len(y_positions))
            expected = N * T
            if n_pairs < expected:
                expected = n_pairs
            if expected == 0:
                continue

            x_positions = x_positions[:expected]
            y_positions = y_positions[:expected]

            # HF shift: logits[t] predicts labels[t+1], so use logits[pos-1]
            x_logit_pos = [p - 1 for p in x_positions]
            y_logit_pos = [p - 1 for p in y_positions]

            # skip if any logit position would be negative
            if any(p < 0 for p in x_logit_pos + y_logit_pos):
                continue

            # soft positions via softmax (differentiable)
            x_logits_sel = log[x_logit_pos][:, self.x_token_ids]  # (expected, x_bins)
            y_logits_sel = log[y_logit_pos][:, self.y_token_ids]  # (expected, y_bins)
            x_probs = torch.softmax(x_logits_sel, dim=-1)
            y_probs = torch.softmax(y_logits_sel, dim=-1)
            soft_x = (x_probs * self.x_centers.unsqueeze(0)).sum(dim=-1)  # (expected,)
            soft_y = (y_probs * self.y_centers.unsqueeze(0)).sum(dim=-1)

            # absent mask: label is <x_0> or <y_0>
            x_labels = lab[x_positions]
            y_labels = lab[y_positions]
            valid = (x_labels != self.x0_id) & (y_labels != self.y0_id)  # (expected,)

            # try to reshape to (T, N)
            actual_T = expected // N if N > 0 else 0
            if actual_T == 0 or actual_T * N != expected:
                continue

            soft_x = soft_x[:actual_T * N].reshape(actual_T, N)
            soft_y = soft_y[:actual_T * N].reshape(actual_T, N)
            valid_mask = valid[:actual_T * N].reshape(actual_T, N)

            # smoothness loss: jerk per pedestrian across valid frames
            if self.smoothness_weight > 0 and actual_T > 2:
                for n in range(N):
                    vm = valid_mask[:, n]
                    valid_idx = torch.where(vm)[0]
                    if len(valid_idx) < 3:
                        continue
                    sx = soft_x[valid_idx, n]
                    sy = soft_y[valid_idx, n]
                    vx = sx[1:] - sx[:-1]
                    vy = sy[1:] - sy[:-1]
                    ax = vx[1:] - vx[:-1]
                    ay = vy[1:] - vy[:-1]
                    jerk = torch.sqrt(ax ** 2 + ay ** 2 + 1e-6)
                    smoothness_loss = smoothness_loss + self.smoothness_weight * jerk.mean()

            # collision loss: pairwise distance at each frame
            if self.collision_weight > 0 and N > 1:
                for t in range(actual_T):
                    vm = valid_mask[t]
                    valid_peds = torch.where(vm)[0]
                    if len(valid_peds) < 2:
                        continue
                    px = soft_x[t, valid_peds]
                    py = soft_y[t, valid_peds]
                    dx = px.unsqueeze(0) - px.unsqueeze(1)
                    dy = py.unsqueeze(0) - py.unsqueeze(1)
                    dists = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
                    triu_mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
                    penalty = torch.relu(self.collision_threshold - dists[triu_mask])
                    if penalty.numel() > 0:
                        collision_loss = collision_loss + self.collision_weight * penalty.mean()

            # walkable loss
            if self.walkable_weight > 0:
                w_grid = self._get_walkable_grid(meta["clip_name"])
                if w_grid is not None:
                    for pos_i in range(expected):
                        if not valid[pos_i]:
                            continue
                        xp = x_probs[pos_i]  # (x_bins,)
                        yp = y_probs[pos_i]  # (y_bins,)
                        w_prob = torch.einsum("i,j,ij->", yp, xp, w_grid)
                        walkable_loss = walkable_loss + self.walkable_weight * torch.relu(0.5 - w_prob)

            count += 1

        if count > 0:
            collision_loss = collision_loss / count
            smoothness_loss = smoothness_loss / count
            walkable_loss = walkable_loss / count

        return {
            "collision": collision_loss,
            "smoothness": smoothness_loss,
            "walkable": walkable_loss,
        }


# ====================================================================
# Custom data collator that preserves sample_idx
# ====================================================================
class CollatorWithIndex:
    def __init__(self, base_collator):
        self.base_collator = base_collator

    def __call__(self, features):
        indices = [f.pop("sample_idx", -1) for f in features]
        batch = self.base_collator(features)
        batch["sample_indices"] = torch.tensor(indices, dtype=torch.long)
        return batch


# ====================================================================
# Custom trainer with physics loss + per-step logging
# ====================================================================
class FLINTTrainer(SFTTrainer):

    def __init__(self, *args, physics_loss_fn=None, task_name="default",
                 output_base="./outputs", **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_loss_fn = physics_loss_fn
        self.task_name = task_name
        self.output_base = output_base
        self._step_losses = {"ce": [], "collision": [], "smoothness": [], "walkable": []}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_indices = inputs.pop("sample_indices", None)

        outputs = model(**inputs)
        ce_loss = outputs.loss

        c_loss = torch.tensor(0.0, device=ce_loss.device)
        s_loss = torch.tensor(0.0, device=ce_loss.device)
        w_loss = torch.tensor(0.0, device=ce_loss.device)

        if self.physics_loss_fn is not None and sample_indices is not None:
            labels = inputs.get("labels")
            if labels is not None:
                p = self.physics_loss_fn.compute(outputs.logits, labels, sample_indices)
                c_loss = p["collision"]
                s_loss = p["smoothness"]
                w_loss = p["walkable"]

        total = ce_loss + c_loss + s_loss + w_loss

        self._step_losses["ce"].append(ce_loss.item())
        self._step_losses["collision"].append(c_loss.item())
        self._step_losses["smoothness"].append(s_loss.item())
        self._step_losses["walkable"].append(w_loss.item())

        step = self.state.global_step
        print(f"[Step {step}] CE: {ce_loss.item():.4f} | "
              f"Collision: {c_loss.item():.4f} | "
              f"Smoothness: {s_loss.item():.4f} | "
              f"Walkable: {w_loss.item():.4f} | "
              f"Total: {total.item():.4f}")

        return (total, outputs) if return_outputs else total


# ====================================================================
# Callbacks: loss logging + checkpoint saving
# ====================================================================
class LossLogCallback(TrainerCallback):

    def __init__(self, task_name, output_base):
        self.task_name = task_name
        self.output_base = output_base

    def log_epoch(self, trainer, epoch):
        losses = trainer._step_losses
        if not losses["ce"]:
            return
        means = {k: sum(v) / len(v) if v else 0.0 for k, v in losses.items()}
        total = sum(means.values())

        loss_dir = os.path.join(self.output_base, "loss")
        os.makedirs(loss_dir, exist_ok=True)
        log_path = os.path.join(loss_dir, f"{self.task_name}.txt")
        line = (f"Epoch {epoch}: ce={means['ce']:.6f} "
                f"collision={means['collision']:.6f} "
                f"smoothness={means['smoothness']:.6f} "
                f"walkable={means['walkable']:.6f} "
                f"total={total:.6f}\n")
        with open(log_path, "a") as f:
            f.write(line)
        print(f"[Epoch {epoch} avg] {line.strip()}")

        # reset
        for k in losses:
            losses[k].clear()


class CheckpointCallback(TrainerCallback):

    def __init__(self, task_name, output_base, save_interval=5):
        self.task_name = task_name
        self.output_base = output_base
        self.save_interval = save_interval
        self._last_saved_epoch = -1

    def _save(self, trainer, epoch):
        ckpt_dir = os.path.join(
            self.output_base, "checkpoint", self.task_name, f"epoch_{epoch}"
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        trainer.model.save_pretrained(ckpt_dir)
        trainer.processing_class.save_pretrained(ckpt_dir)
        self._last_saved_epoch = epoch
        print(f"Checkpoint saved: {ckpt_dir}")

    def save_if_needed(self, trainer, epoch):
        if epoch % self.save_interval == 0:
            self._save(trainer, epoch)

    def save_final(self, trainer, epoch):
        if self._last_saved_epoch != epoch:
            self._save(trainer, epoch)


# ====================================================================
# Epoch tracking callback (drives loss-log and checkpoint)
# ====================================================================
class EpochEndCallback(TrainerCallback):

    def __init__(self, loss_cb, ckpt_cb):
        self.loss_cb = loss_cb
        self.ckpt_cb = ckpt_cb

    def on_epoch_end(self, _args, state, _control, **_kwargs):
        epoch = int(round(state.epoch))
        if hasattr(self, "_trainer"):
            self.loss_cb.log_epoch(self._trainer, epoch)
            self.ckpt_cb.save_if_needed(self._trainer, epoch)

    def on_train_end(self, _args, state, _control, **_kwargs):
        epoch = int(round(state.epoch))
        if hasattr(self, "_trainer"):
            self.ckpt_cb.save_final(self._trainer, epoch)


# ====================================================================
# Main training
# ====================================================================
def train(args):
    use_coord = not args.no_coord_tokens
    resolution = (args.resolution_h, args.resolution_w)

    # load data
    train_ds, _, train_meta, _ = load_and_split(
        args.data, seed=args.seed, use_raw_prompt=args.use_raw_prompt
    )

    # model
    backbone = args.backbone
    if os.path.isdir(backbone):
        print(f"Loading base model from local path: {backbone}")
    else:
        print(f"Loading base model from HuggingFace: {backbone}")

    model, llm_tokenizer, coord_tok = prepare_model_and_tokenizer(
        backbone, use_coord, resolution, args.bin_size
    )

    # physics loss
    physics_loss_fn = None
    if args.physics != "none" and use_coord and coord_tok is not None:
        physics_config = {
            "collision_weight": args.collision_weight if args.physics in ("collision", "all") else 0.0,
            "smoothness_weight": args.smoothness_weight if args.physics == "all" else 0.0,
            "walkable_weight": args.walkable_weight if args.physics == "all" else 0.0,
            "collision_threshold": args.collision_threshold,
        }
        physics_loss_fn = PhysicsLoss(
            coord_tok, llm_tokenizer, train_meta, physics_config, model.device
        )
        enabled = [k for k, v in physics_config.items() if "weight" in k and v > 0]
        print(f"Physics loss enabled: {enabled}")

    # callbacks
    loss_cb = LossLogCallback(args.task_name, args.output_base)
    ckpt_cb = CheckpointCallback(args.task_name, args.output_base, save_interval=5)
    epoch_cb = EpochEndCallback(loss_cb, ckpt_cb)

    callbacks = [epoch_cb]

    training_args = SFTConfig(
        output_dir=os.path.join(args.output_base, "runs", args.task_name),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_length=args.max_seq_length,
        packing=False,
        report_to="tensorboard",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = FLINTTrainer(
        model=model,
        processing_class=llm_tokenizer,
        train_dataset=train_ds,
        args=training_args,
        callbacks=callbacks,
        physics_loss_fn=physics_loss_fn,
        task_name=args.task_name,
        output_base=args.output_base,
    )

    # wrap collator to pass sample_idx through
    trainer.data_collator = CollatorWithIndex(trainer.data_collator)

    # attach trainer ref so callbacks can access it
    epoch_cb._trainer = trainer

    print(f"\nTraining: task={args.task_name}, backbone={backbone}, "
          f"coord_tokens={use_coord}, physics={args.physics}, "
          f"early_stopping={args.early_stopping}")

    trainer.train()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLINT training")

    # data
    parser.add_argument("--data", type=str, required=True,
                        help="Path to eth-ucy-text.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_raw_prompt", action="store_true",
                        help="Use metadata.raw_prompt for tokenizer ablation")

    # task naming and output
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--output_base", type=str, default="./outputs")

    # tokenizer
    parser.add_argument("--use_coord_tokens", action="store_true", default=True)
    parser.add_argument("--no_coord_tokens", action="store_true")
    parser.add_argument("--bin_size", type=int, default=5)
    parser.add_argument("--resolution_h", type=int, default=480)
    parser.add_argument("--resolution_w", type=int, default=640)

    # model
    parser.add_argument("--backbone", type=str, default="Qwen/Qwen3-8B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=8192)

    # training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    # physics loss
    parser.add_argument("--physics", type=str, default="none",
                        choices=["none", "collision", "all"])
    parser.add_argument("--collision_weight", type=float, default=0.001)
    parser.add_argument("--smoothness_weight", type=float, default=0.05)
    parser.add_argument("--walkable_weight", type=float, default=0.05)
    parser.add_argument("--collision_threshold", type=float, default=10.0)

    args = parser.parse_args()
    train(args)
