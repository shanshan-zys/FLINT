"""
Merge LoRA adapter into base model for fast inference with vLLM.

Usage:
  python source/merge_lora.py \
      --backbone /mnt/oss-write/opensource_models/Qwen3-8B \
      --checkpoint outputs/checkpoint/ablation_no_physics-ep30-c0.001-s0.01-w0.01-r0.01/epoch_25 \
      --output outputs/merged/ablation_no_physics-ep30-c0.001-s0.01-w0.01-r0.01/epoch_25
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def _get_vocab_size_from_adapter(checkpoint_path):
    """Detect vocab size from saved adapter weights (handles coord token case)."""
    import safetensors.torch
    from pathlib import Path
    adapter_path = Path(checkpoint_path)
    for fname in ["adapter_model.safetensors", "adapter_model.bin"]:
        fpath = adapter_path / fname
        if fpath.exists():
            if fname.endswith(".safetensors"):
                state = safetensors.torch.load_file(str(fpath), device="cpu")
            else:
                state = torch.load(str(fpath), map_location="cpu")
            for key, tensor in state.items():
                if "embed_tokens" in key and "weight" in key:
                    return tensor.shape[0]
    return None


def merge(args):
    print(f"Loading tokenizer from: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    vocab_size = _get_vocab_size_from_adapter(args.checkpoint)
    target_size = vocab_size if vocab_size else len(tokenizer)
    print(f"Vocab size: tokenizer={len(tokenizer)}, adapter={vocab_size}, using={target_size}")

    print(f"Loading base model: {args.backbone}")
    model = AutoModelForCausalLM.from_pretrained(
        args.backbone,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(target_size)

    print(f"Loading LoRA adapter: {args.checkpoint}")
    model = PeftModel.from_pretrained(model, args.checkpoint)

    print("Merging weights...")
    model = model.merge_and_unload()

    os.makedirs(args.output, exist_ok=True)
    print(f"Saving merged model: {args.output}")
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA into base model")
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LoRA checkpoint (with tokenizer)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for merged model")
    args = parser.parse_args()
    merge(args)
