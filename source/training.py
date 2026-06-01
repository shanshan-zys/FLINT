"""
FLINT training: LoRA SFT with physics-aware loss.

Supports:
  --use_coord_tokens / --no_coord_tokens   tokenizer mode
  --use_raw_prompt                         tokenizer ablation (metadata.raw_prompt)
  --physics none / collision / all         physics loss selection
  --task_name                              names output dirs under loss/ and checkpoint/
  --peek_inference                        generate one sample per epoch for observation
"""

import os
import re
import json
import random
import torch
import argparse
import numpy as np

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

from dataset_construction import CoordTokenizer, decode_raw_numbers


ABSENT = -1.0


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


def format_alpaca_prompt(sample: dict) -> str:
    return (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n"
        "### Instruction:\n"
        f"{sample['instruction']}\n\n"
        "### Input:\n"
        f"{sample['input']}\n\n"
        "### Response:\n"
    )


# ====================================================================
# Data loading: single JSON, seed-42 split 8:2
# Returns pre-tokenized dataset with input_ids constructed manually
# to ensure coord tokens are single tokens (not BPE-split)
# ====================================================================
def load_and_split(data_path, tokenizer, coord_tok, seed=42,
                   use_raw_prompt=False, use_coord_tokens=True, max_seq_length=8192):
    with open(data_path) as f:
        raw = json.load(f)

    indices = list(range(len(raw)))
    random.Random(seed).shuffle(indices)
    split = int(len(indices) * 0.8)
    train_idx, eval_idx = indices[:split], indices[split:]

    def build(idx_list):
        all_input_ids, all_response_starts, sample_indices, metadata_list = [], [], [], []
        for new_i, orig_i in enumerate(idx_list):
            s = raw[orig_i]
            if use_raw_prompt:
                fields = s["metadata"]["raw_prompt"]
            else:
                fields = s

            # Tokenize prompt (instruction + input + "### Response:\n")
            prompt_text = format_alpaca_prompt(fields)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            # Tokenize response: coord tokens via convert_tokens_to_ids (not BPE)
            if use_coord_tokens:
                output_text = fields["output"]
                output_tokens = re.findall(r'<[xy]_\d+>', output_text)
                response_ids = tokenizer.convert_tokens_to_ids(output_tokens)
            else:
                # raw mode: normal tokenization is fine for plain numbers
                response_ids = tokenizer.encode(fields["output"], add_special_tokens=False)

            response_start = len(prompt_ids)

            # Combine: prompt + response + EOS
            input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]

            # Truncate if needed
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]

            all_input_ids.append(input_ids)
            all_response_starts.append(response_start)
            sample_indices.append(new_i)

            meta = s["metadata"]
            metadata_list.append({
                "clip_name": meta["clip_name"],
                "population": meta["population"],
                "frames": meta["frames"],
                "walkable_area": meta["walkable_area"],
                "trajectory": meta["trajectory"],
            })

        ds = Dataset.from_dict({
            "input_ids": all_input_ids,
            "response_start": all_response_starts,
            "sample_idx": sample_indices,
        })
        return ds, metadata_list

    train_ds, train_meta = build(train_idx)
    eval_ds, eval_meta = build(eval_idx)
    print(f"Data split (seed={seed}): train={len(train_ds)}, eval={len(eval_ds)}")

    # Print debug info for first sample
    first_ids = train_ds[0]["input_ids"]
    print(f"[DEBUG] First train sample: {len(first_ids)} tokens")
    if use_coord_tokens and coord_tok:
        new_token_start = len(tokenizer) - len(coord_tok.vocab)
        coord_count = sum(1 for t in first_ids if t >= new_token_start)
        print(f"[DEBUG] Coord tokens in first sample: {coord_count}")

    return train_ds, eval_ds, train_meta, eval_meta, raw, eval_idx


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
        self.regression_weight = config.get("regression_weight", 0.0)
        self.collision_threshold = config["collision_threshold"]

        self.bin_size = coord_tok.bin_size
        bin_size = self.bin_size
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

        # token_id → bin center (for regression loss GT)
        self.x_id_to_center = {}
        for i, tok in enumerate(coord_tok.x_tokens[1:], start=1):
            tid = llm_tokenizer.convert_tokens_to_ids(tok)
            self.x_id_to_center[tid] = (i - 0.5) * bin_size
        self.y_id_to_center = {}
        for j, tok in enumerate(coord_tok.y_tokens[1:], start=1):
            tid = llm_tokenizer.convert_tokens_to_ids(tok)
            self.y_id_to_center[tid] = (j - 0.5) * bin_size

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
        regression_loss = zero
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

            # regression loss: SmoothL1 between soft prediction and GT bin center
            if self.regression_weight > 0:
                gt_x = torch.tensor(
                    [self.x_id_to_center.get(lab[p].item(), 0.0) for p in x_positions[:expected]],
                    device=self.device, dtype=torch.float32
                )
                gt_y = torch.tensor(
                    [self.y_id_to_center.get(lab[p].item(), 0.0) for p in y_positions[:expected]],
                    device=self.device, dtype=torch.float32
                )
                flat_soft_x = soft_x.reshape(-1)
                flat_soft_y = soft_y.reshape(-1)
                flat_valid = valid_mask.reshape(-1)
                if flat_valid.sum() > 0:
                    reg_x = torch.nn.functional.smooth_l1_loss(
                        flat_soft_x[flat_valid], gt_x[flat_valid], beta=self.bin_size)
                    reg_y = torch.nn.functional.smooth_l1_loss(
                        flat_soft_y[flat_valid], gt_y[flat_valid], beta=self.bin_size)
                    regression_loss = regression_loss + self.regression_weight * (reg_x + reg_y)

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
            regression_loss = regression_loss / count

        return {
            "collision": collision_loss,
            "smoothness": smoothness_loss,
            "walkable": walkable_loss,
            "regression": regression_loss,
        }


# ====================================================================
# Custom data collator: completion-only loss + sample_idx passthrough
# ====================================================================
class CompletionOnlyCollator:
    """Masks prompt labels to -100 using pre-computed response_start.
    Also preserves sample_idx for physics loss."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        sample_indices = [f.pop("sample_idx", -1) for f in features]
        response_starts = [f.pop("response_start", 0) for f in features]

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = torch.full((len(features), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        labels = torch.full((len(features), max_len), -100, dtype=torch.long)

        for i, f in enumerate(features):
            ids = f["input_ids"]
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            # Only compute loss on response tokens (after prompt)
            resp_start = response_starts[i]
            labels[i, resp_start:seq_len] = input_ids[i, resp_start:seq_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sample_indices": torch.tensor(sample_indices, dtype=torch.long),
        }


# ====================================================================
# Custom trainer with physics loss + per-step logging
# ====================================================================
class FLINTTrainer(Trainer):

    def __init__(self, *args, physics_loss_fn=None, task_name="default",
                 output_base="./outputs", **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_loss_fn = physics_loss_fn
        self.task_name = task_name
        self.output_base = output_base
        self._step_losses = {"ce": [], "collision": [], "smoothness": [], "walkable": [], "regression": []}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_indices = inputs.pop("sample_indices", None)

        outputs = model(**inputs)
        ce_loss = outputs.loss

        c_loss = torch.tensor(0.0, device=ce_loss.device)
        s_loss = torch.tensor(0.0, device=ce_loss.device)
        w_loss = torch.tensor(0.0, device=ce_loss.device)
        r_loss = torch.tensor(0.0, device=ce_loss.device)

        if self.physics_loss_fn is not None and sample_indices is not None:
            labels = inputs.get("labels")
            if labels is not None:
                p = self.physics_loss_fn.compute(outputs.logits, labels, sample_indices)
                c_loss = p["collision"]
                s_loss = p["smoothness"]
                w_loss = p["walkable"]
                r_loss = p["regression"]

        total = ce_loss + c_loss + s_loss + w_loss + r_loss

        self._step_losses["ce"].append(ce_loss.item())
        self._step_losses["collision"].append(c_loss.item())
        self._step_losses["smoothness"].append(s_loss.item())
        self._step_losses["walkable"].append(w_loss.item())
        self._step_losses["regression"].append(r_loss.item())

        step = self.state.global_step
        print(f"[Step {step}] CE: {ce_loss.item():.4f} | "
              f"Coll: {c_loss.item():.4f} | Smooth: {s_loss.item():.4f} | "
              f"Walk: {w_loss.item():.4f} | Reg: {r_loss.item():.4f} | "
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
                f"regression={means['regression']:.6f} "
                f"total={total:.6f}\n")
        with open(log_path, "a") as f:
            f.write(line)
        print(f"[Epoch {epoch} avg] {line.strip()}")

        # reset
        for k in losses:
            losses[k].clear()

    def peek_prediction(self, trainer, epoch):
        """Forward first training sample, print argmax next-token prediction."""
        model = trainer.model
        model.eval()

        ds = trainer.train_dataset
        sample = ds[0]
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0)
        input_ids = input_ids.to(model.device)
        resp_start = sample["response_start"]

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0]  # (seq_len, vocab)

        # argmax at response positions (logits[t] predicts token t+1)
        pred_ids = logits[resp_start - 1: -1].argmax(dim=-1).tolist()
        gt_ids = sample["input_ids"][resp_start:]

        tokenizer = getattr(trainer, "processing_class", None) or trainer.tokenizer

        # Show first 30 predicted vs ground truth tokens
        n_show = min(30, len(pred_ids), len(gt_ids))
        pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids[:n_show])
        gt_tokens = tokenizer.convert_ids_to_tokens(gt_ids[:n_show])

        # Count how many coord tokens in full prediction
        coord_pattern = re.compile(r'<[xy]_\d+>')
        n_coord_pred = sum(1 for t in tokenizer.convert_ids_to_tokens(pred_ids) if coord_pattern.match(t))
        n_correct = sum(1 for p, g in zip(pred_ids, gt_ids) if p == g)

        print(f"[Epoch {epoch} peek] response_len={len(gt_ids)}, "
              f"coord_tokens_in_pred={n_coord_pred}/{len(pred_ids)}, "
              f"exact_match={n_correct}/{len(gt_ids)} ({100*n_correct/max(len(gt_ids),1):.1f}%)")
        print(f"  GT  (first {n_show}): {gt_tokens}")
        print(f"  Pred(first {n_show}): {pred_tokens}")

        model.train()


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
        trainer.model.save_pretrained(ckpt_dir, save_embedding_layers=True)
        tokenizer = getattr(trainer, "processing_class", None) or trainer.tokenizer
        tokenizer.save_pretrained(ckpt_dir)
        self._last_saved_epoch = epoch
        print(f"Checkpoint saved: {ckpt_dir}")

    def save_if_needed(self, trainer, epoch):
        if epoch % self.save_interval == 0:
            self._save(trainer, epoch)
            return True
        return False

    def save_final(self, trainer, epoch):
        if self._last_saved_epoch != epoch:
            self._save(trainer, epoch)


# ====================================================================
# Mid-training inference
# ====================================================================
def parse_output_trajectory(generated_text, use_coord_tokens, coord_tok, population, frames):
    if use_coord_tokens:
        tokens = re.findall(r'<[xy]_\d+>', generated_text)
        traj = coord_tok.detokenize(tokens, population, frames)
    else:
        traj = decode_raw_numbers(generated_text, population, frames)

    result = []
    for n in range(population):
        ped_traj = []
        for t in range(frames):
            x, y = traj[n, t, 0], traj[n, t, 1]
            if x == ABSENT or y == ABSENT:
                ped_traj.append([-1.0, -1.0])
            else:
                ped_traj.append([float(x), float(y)])
        result.append(ped_traj)
    return result


COMPACT_KEYS = {"output_trajectories", "walkable_area", "trajectory"}


def compact_json(obj, indent=2):
    def _serialize(o, level):
        pad = " " * (indent * level)
        pad_inner = " " * (indent * (level + 1))
        if isinstance(o, dict):
            items = []
            for k, v in o.items():
                if k in COMPACT_KEYS:
                    items.append(f'{pad_inner}{json.dumps(k)}: {json.dumps(v, separators=(",", ": "))}')
                else:
                    items.append(f'{pad_inner}{json.dumps(k)}: {_serialize(v, level + 1)}')
            return "{\n" + ",\n".join(items) + f"\n{pad}}}"
        elif isinstance(o, list) and o and isinstance(o[0], dict):
            items = [f"{pad_inner}{_serialize(item, level + 1)}" for item in o]
            return "[\n" + ",\n".join(items) + f"\n{pad}]"
        else:
            return json.dumps(o)
    return _serialize(obj, 0) + "\n"


def peek_generate_one(trainer, tokenizer, coord_tok, sample,
                      use_coord, use_raw_prompt, epoch, max_new_tokens, max_seq_length):
    """Generate one sample unconstrained, print result to observe learning progress."""
    model = trainer.model
    was_training = model.training
    model.eval()
    model.to(dtype=torch.bfloat16)

    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    meta = sample["metadata"]
    if use_raw_prompt:
        fields = meta["raw_prompt"]
    else:
        fields = sample

    prompt = format_alpaca_prompt(fields)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=max_seq_length).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=False)

    new_ids = output_ids[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    if use_coord:
        coord_tokens = re.findall(r'<[xy]_\d+>', generated_text)
        print(f"[Epoch {epoch} generate] clip={meta['clip_name']} | "
              f"coord_tokens={len(coord_tokens)}, expected={meta['population'] * meta['frames'] * 2}")
        print(f"  First 60 tokens: {coord_tokens[:60]}")
    else:
        parts = generated_text.strip().split(";")
        print(f"[Epoch {epoch} generate] clip={meta['clip_name']} | "
              f"parts={len(parts)}, expected={meta['population'] * meta['frames']}")
        print(f"  First 200 chars: {generated_text[:200]}")

    if was_training:
        model.train()


# ====================================================================
# Epoch tracking callback (drives loss-log, checkpoint, and inference)
# ====================================================================
class EpochEndCallback(TrainerCallback):

    def __init__(self, loss_cb, ckpt_cb, infer_fn=None):
        self.loss_cb = loss_cb
        self.ckpt_cb = ckpt_cb
        self.infer_fn = infer_fn

    def on_epoch_end(self, _args, state, _control, **_kwargs):
        epoch = int(round(state.epoch))
        if hasattr(self, "_trainer"):
            self.loss_cb.log_epoch(self._trainer, epoch)
            self.loss_cb.peek_prediction(self._trainer, epoch)
            saved = self.ckpt_cb.save_if_needed(self._trainer, epoch)
            if saved and self.infer_fn is not None:
                self.infer_fn(self._trainer, epoch)

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

    # model (must be before data loading so tokenizer is available)
    backbone = args.backbone
    if os.path.isdir(backbone):
        print(f"Loading base model from local path: {backbone}")
    else:
        print(f"Loading base model from HuggingFace: {backbone}")

    model, llm_tokenizer, coord_tok = prepare_model_and_tokenizer(
        backbone, use_coord, resolution, args.bin_size
    )

    # load data (pre-tokenized with coord tokens handled correctly)
    train_ds, _, train_meta, _, raw_data, test_idx = load_and_split(
        args.data, llm_tokenizer, coord_tok, seed=args.seed,
        use_raw_prompt=args.use_raw_prompt, use_coord_tokens=use_coord,
        max_seq_length=args.max_seq_length,
    )

    # physics loss (enabled whenever any weight > 0 and coord tokens are used)
    physics_loss_fn = None
    if use_coord and coord_tok is not None:
        physics_config = {
            "collision_weight": args.collision_weight,
            "smoothness_weight": args.smoothness_weight,
            "walkable_weight": args.walkable_weight,
            "regression_weight": args.regression_weight,
            "collision_threshold": args.collision_threshold,
        }
        any_active = any(v > 0 for k, v in physics_config.items() if "weight" in k)
        if any_active:
            physics_loss_fn = PhysicsLoss(
                coord_tok, llm_tokenizer, train_meta, physics_config, model.device
            )
            enabled = [k for k, v in physics_config.items() if "weight" in k and v > 0]
            print(f"Physics loss enabled: {enabled}")

    # prepare inference peek (generate one sample per epoch for observation)
    infer_fn = None
    if args.peek_inference:
        test_samples = [raw_data[i] for i in test_idx]
        print(f"Peek inference enabled (1 sample per epoch)")

        def infer_fn(trainer, epoch):
            peek_generate_one(
                trainer, llm_tokenizer, coord_tok, test_samples[0],
                use_coord, args.use_raw_prompt, epoch, args.max_new_tokens,
                args.max_seq_length,
            )

    # callbacks
    loss_cb = LossLogCallback(args.task_name, args.output_base)
    ckpt_cb = CheckpointCallback(args.task_name, args.output_base, save_interval=5)
    epoch_cb = EpochEndCallback(loss_cb, ckpt_cb, infer_fn=infer_fn)

    callbacks = [epoch_cb]

    training_args = TrainingArguments(
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
        report_to="tensorboard",
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = FLINTTrainer(
        model=model,
        tokenizer=llm_tokenizer,
        train_dataset=train_ds,
        args=training_args,
        callbacks=callbacks,
        physics_loss_fn=physics_loss_fn,
        task_name=args.task_name,
        output_base=args.output_base,
    )

    # Collator uses pre-computed response_start to mask prompt labels
    trainer.data_collator = CompletionOnlyCollator(llm_tokenizer.pad_token_id)

    # Verify label masking on first sample
    sample_feature = train_ds[0]
    test_batch = trainer.data_collator([{k: v for k, v in sample_feature.items()}])
    n_loss = (test_batch["labels"][0] != -100).sum().item()
    n_all = (test_batch["attention_mask"][0] == 1).sum().item()
    print(f"[Collator] First sample: {n_all} tokens, {n_loss} with loss "
          f"({100*n_loss/max(n_all,1):.1f}% response)")
    if n_loss == 0:
        print("[WARNING] No tokens have loss! Check response_start values.")

    # attach trainer ref so callbacks can access it
    epoch_cb._trainer = trainer

    print(f"\nTraining: task={args.task_name}, backbone={backbone}, "
          f"coord_tokens={use_coord}")

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
    parser.add_argument("--max_new_tokens", type=int, default=4096)

    # training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)

    # loss weights (set to 0 to disable)
    parser.add_argument("--collision_weight", type=float, default=0.0)
    parser.add_argument("--smoothness_weight", type=float, default=0.0)
    parser.add_argument("--walkable_weight", type=float, default=0.0)
    parser.add_argument("--regression_weight", type=float, default=0.0)
    parser.add_argument("--collision_threshold", type=float, default=10.0)

    # mid-training observation
    parser.add_argument("--peek_inference", action="store_true",
                        help="Generate one sample per epoch for observation (no file saved)")

    args = parser.parse_args()
    train(args)
