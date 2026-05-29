"""
API 成本估算工具

帮助选择最经济的标注方案
"""


def estimate_cost(
    num_samples: int,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    use_images: bool = False,
    num_images_per_sample: int = 3,
):
    """
    估算标注成本

    定价参考 (2024-2025):
    ┌─────────────────────┬──────────────┬───────────────┬────────────────┐
    │  Model              │  Input $/1M  │  Output $/1M  │  Image (low)   │
    ├─────────────────────┼──────────────┼───────────────┼────────────────┤
    │  GPT-4o             │  $2.50       │  $10.00       │  ~85 tokens    │
    │  GPT-4o-mini        │  $0.15       │  $0.60        │  ~85 tokens    │
    │  Gemini-1.5-Flash   │  $0.075      │  $0.30        │  ~258 tokens   │
    │  Gemini-1.5-Pro     │  $1.25       │  $5.00        │  ~258 tokens   │
    │  DeepSeek-V3        │  $0.27       │  $1.10        │  N/A           │
    │  Qwen2.5-VL (local) │  FREE        │  FREE         │  FREE          │
    └─────────────────────┴──────────────┴───────────────┴────────────────┘
    """

    pricing = {
        "gpt-4o": {"input": 2.50, "output": 10.00, "image_tokens": 85},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60, "image_tokens": 85},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30, "image_tokens": 258},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00, "image_tokens": 258},
        "deepseek-chat": {"input": 0.27, "output": 1.10, "image_tokens": 0},
    }

    if model not in pricing:
        print(f"Unknown model: {model}")
        return

    p = pricing[model]

    # 估算每个样本的token数
    input_tokens_per_sample = 500  # prompt + trajectory text
    output_tokens_per_sample = 200  # 描述

    if use_images:
        input_tokens_per_sample += num_images_per_sample * p["image_tokens"]

    total_input = num_samples * input_tokens_per_sample
    total_output = num_samples * output_tokens_per_sample

    input_cost = total_input / 1_000_000 * p["input"]
    output_cost = total_output / 1_000_000 * p["output"]
    total_cost = input_cost + output_cost

    print(f"\n{'='*50}")
    print(f"Cost Estimation: {model}")
    print(f"{'='*50}")
    print(f"Samples: {num_samples}")
    print(f"Use images: {use_images}")
    print(f"Input tokens/sample: {input_tokens_per_sample:,}")
    print(f"Output tokens/sample: {output_tokens_per_sample:,}")
    print(f"Total input tokens: {total_input:,}")
    print(f"Total output tokens: {total_output:,}")
    print(f"Input cost: ${input_cost:.4f}")
    print(f"Output cost: ${output_cost:.4f}")
    print(f"TOTAL COST: ${total_cost:.4f}")
    print(f"{'='*50}")

    return total_cost


def compare_all_providers(num_samples: int = 1000):
    """对比所有方案的成本"""
    print("\n" + "=" * 60)
    print(f"COST COMPARISON for {num_samples} samples")
    print("=" * 60)

    configs = [
        ("GPT-4o (text+img)", "gpt-4o", True),
        ("GPT-4o (text only)", "gpt-4o", False),
        ("GPT-4o-mini (text+img)", "gpt-4o-mini", True),
        ("GPT-4o-mini (text only)", "gpt-4o-mini", False),
        ("Gemini Flash (text+img)", "gemini-1.5-flash", True),
        ("Gemini Flash (text only)", "gemini-1.5-flash", False),
        ("DeepSeek-V3 (text only)", "deepseek-chat", False),
        ("Qwen2.5-VL (local)", None, False),
    ]

    print(f"\n{'Provider':<30} {'Cost':>10}")
    print("-" * 42)

    for name, model, use_img in configs:
        if model is None:
            print(f"  {name:<30} {'$0.00':>10}  (需要GPU)")
            continue
        cost = estimate_cost(num_samples, model=model, use_images=use_img)
        print(f"  {name:<30} ${cost:>8.2f}")

    print("\n推荐: Gemini-1.5-Flash (text+img) → ~$0.10/1000样本")
    print("      如果有GPU, Qwen2.5-VL 完全免费")


if __name__ == "__main__":
    compare_all_providers(1000)
