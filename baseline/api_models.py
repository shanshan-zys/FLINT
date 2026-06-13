"""
API-based baseline models (OpenAI-compatible):
  GPT-5, DeepSeek-V4-Flash, Gemini-3.0-Flash, Claude Opus 4.6
"""

import time
import base64
from openai import OpenAI


# Known parameter counts (approximate, for reference table)
KNOWN_PARAMS = {
    "gpt-5": "~2000B (estimated)",
    "deepseek-v4-flash": "~236B MoE",
    "gemini-3.0-flash": "unknown",
    "claude-opus-4-6": "unknown",
}


class APIModel:
    """OpenAI-compatible API model for zero-shot trajectory generation."""

    def __init__(self, model_name, api_key, base_url,
                 temperature=0.0, max_retries=3, retry_delay=5.0,
                 disable_thinking=False):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.disable_thinking = disable_thinking

    def generate(self, prompt, max_tokens, image_path=None):
        messages = self._build_messages(prompt, image_path)

        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                )
                if self.disable_thinking:
                    kwargs["extra_body"] = {"enable_thinking": False}

                resp = self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                return text.strip()
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"  API error (attempt {attempt+1}): {e}, "
                          f"retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    print(f"  API failed after {self.max_retries} attempts: {e}")
                    return ""

    def _build_messages(self, prompt, image_path=None):
        if image_path:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            content = [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        return [{"role": "user", "content": content}]

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "type": "api",
            "params": KNOWN_PARAMS.get(self.model_name, "unknown"),
            "disable_thinking": self.disable_thinking,
        }
