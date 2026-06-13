"""
Local baseline models:
  - LocalLLM:  LLaMA-3.1-8B, Qwen3-8B  (vLLM)
  - LocalVLM:  LLaVA-NeXT-7B, Qwen2.5-VL-7B  (vLLM + image)
  - LocalPLM:  T5-Base, BERT-Base  (HuggingFace transformers)
"""

import os
os.environ.setdefault("VLLM_USE_V1", "0")


# ====================================================================
# VLM prompt templates
# ====================================================================
LLAVA_PROMPT_TEMPLATE = (
    "<image>\n"
    "Below is an instruction that describes a task, "
    "paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

QWEN_VL_PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
    "Below is an instruction that describes a task, "
    "paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# ====================================================================
# Shared helper: count params from HuggingFace config
# ====================================================================
def _count_params_from_config(model_name):
    """Estimate parameter count from HuggingFace model config."""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(config, "num_parameters"):
            return config.num_parameters
        hidden = getattr(config, "hidden_size", 0)
        layers = getattr(config, "num_hidden_layers", 0)
        vocab = getattr(config, "vocab_size", 0)
        if hidden and layers:
            return 12 * layers * hidden * hidden + vocab * hidden
    except Exception:
        pass
    return None


def _format_params(n):
    """Format parameter count as human-readable string."""
    if n is None:
        return "unknown"
    if n > 1e9:
        return f"{n / 1e9:.1f}B"
    return f"{n / 1e6:.0f}M"


# ====================================================================
# LocalLLM: vLLM-based LLM (LLaMA, Qwen3)
# ====================================================================
class LocalLLM:
    """Local LLM loaded with vLLM for raw-format generation."""

    def __init__(self, model_name, temperature=0.0,
                 max_model_len=8192, gpu_memory_utilization=0.9,
                 disable_thinking=False):
        from vllm import LLM
        self.model_name = model_name
        self.temperature = temperature
        self.disable_thinking = disable_thinking

        print(f"Loading local LLM: {model_name}")
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )
        self._model_params = _count_params_from_config(model_name)

    def _maybe_disable_thinking(self, prompt):
        if not self.disable_thinking:
            return prompt
        if "qwen" in self.model_name.lower():
            return prompt + "/no_think"
        return prompt

    def generate(self, prompt, max_tokens, image_path=None):
        from vllm import SamplingParams
        if self.disable_thinking:
            prompt = self._maybe_disable_thinking(prompt)

        sp = SamplingParams(max_tokens=max_tokens, temperature=self.temperature)
        outputs = self.llm.generate([prompt], sampling_params=[sp])
        return outputs[0].outputs[0].text.strip()

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "type": "local_llm",
            "params": _format_params(self._model_params),
            "disable_thinking": self.disable_thinking,
        }


# ====================================================================
# LocalVLM: vLLM-based VLM (LLaVA, Qwen-VL)
# ====================================================================
class LocalVLM:
    """Local VLM loaded with vLLM for multimodal inference."""

    def __init__(self, model_name, temperature=0.0,
                 max_model_len=8192, gpu_memory_utilization=0.9,
                 disable_thinking=False):
        from vllm import LLM
        self.model_name = model_name
        self.temperature = temperature
        self.disable_thinking = disable_thinking
        self.is_qwen_vl = "qwen" in model_name.lower() and "vl" in model_name.lower()

        print(f"Loading local VLM: {model_name}")
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1},
        )
        self._model_params = _count_params_from_config(model_name)

    def build_prompt(self, instruction, input_text):
        """Build VLM-specific prompt with image placeholder."""
        if self.is_qwen_vl:
            return QWEN_VL_PROMPT_TEMPLATE.format(
                instruction=instruction, input=input_text)
        else:
            return LLAVA_PROMPT_TEMPLATE.format(
                instruction=instruction, input=input_text)

    def generate(self, prompt, max_tokens, image_path=None):
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=max_tokens, temperature=self.temperature)

        if image_path and os.path.exists(image_path):
            mm_data = {"image": image_path}
        else:
            mm_data = None

        outputs = self.llm.generate(
            [{"prompt": prompt, "multi_modal_data": mm_data}],
            sampling_params=[sp],
        )
        return outputs[0].outputs[0].text.strip()

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "type": "local_vlm",
            "params": _format_params(self._model_params),
            "disable_thinking": self.disable_thinking,
        }


# ====================================================================
# LocalPLM: HuggingFace transformers (T5, BERT)
# ====================================================================
class LocalPLM:
    """Pre-trained language model (T5/BERT) for trajectory generation.

    T5: encoder-decoder, uses model.generate() natively.
    BERT: encoder-only, will produce poor results (demonstrates limitation).
    """

    def __init__(self, model_name, temperature=0.0,
                 device="cuda", disable_thinking=False):
        import torch
        from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                                  AutoModelForCausalLM)

        self.model_name = model_name
        self.temperature = temperature
        self.device = device if torch.cuda.is_available() else "cpu"
        self.is_t5 = "t5" in model_name.lower()

        print(f"Loading local PLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)

        if self.is_t5:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(self.device)

        self.model.eval()
        self._model_params = sum(p.numel() for p in self.model.parameters())

    def generate(self, prompt, max_tokens, image_path=None):
        import torch
        max_input_len = 512 if self.is_t5 else 1024
        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=max_input_len,
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": min(max_tokens, 2048),
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        if self.is_t5:
            generated = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True)
        else:
            input_len = inputs["input_ids"].shape[1]
            generated = self.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True)

        return generated.strip()

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "type": "local_plm",
            "params": _format_params(self._model_params),
            "params_exact": self._model_params,
        }
