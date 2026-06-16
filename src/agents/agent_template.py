#!/usr/bin/env python
# coding=utf-8

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
import re
from functools import lru_cache

DEFAULT_DIFFUSIONGEMMA_MODEL = "google/diffusiongemma-26B-A4B-it"
DEFAULT_GEMMA4_MODEL = "google/gemma-4-26B-A4B-it"
DEFAULT_BACKEND = "diffusiongemma"


def agent_kwargs_from_env() -> dict:
    """Build agent constructor kwargs from environment variables.

    DiffusionGemma is the default backend. Set MODEL_BACKEND=gemma4 to use
    Gemma 4, or MODEL_BACKEND=openai to use an OpenAI-compatible server.
    """
    backend = os.environ.get("MODEL_BACKEND", DEFAULT_BACKEND).lower()
    if backend == "diffusiongemma":
        model = os.environ.get("DIFFUSIONGEMMA_MODEL_ID") or DEFAULT_DIFFUSIONGEMMA_MODEL
    elif backend == "gemma4":
        model = os.environ.get("GEMMA4_MODEL_ID") or DEFAULT_GEMMA4_MODEL
    else:
        model = os.environ.get("MODEL_NAME")
    return {
        "backend": backend,
        "api_key": os.environ.get("API_KEY"),
        "base_url": os.environ.get("BASE_URL"),
        "model": model,
    }


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


class _DiffusionGemmaRuntime:
    def __init__(self, model_id: str):
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "DiffusionGemma backend requires transformers, torch, and accelerate. "
                "Install them with: uv pip install -U transformers torch accelerate"
            ) from exc

        model_cls = getattr(transformers, "DiffusionGemmaForBlockDiffusion", None)
        if model_cls is None:
            model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
        if model_cls is None:
            raise RuntimeError(
                "Installed transformers does not expose DiffusionGemmaForBlockDiffusion "
                "or AutoModelForMultimodalLM. Upgrade transformers."
            )

        dtype = os.environ.get("DIFFUSIONGEMMA_DTYPE", "auto")
        device_map = os.environ.get("DIFFUSIONGEMMA_DEVICE_MAP", "auto")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = model_cls.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()
        self._torch = torch

    def generate(self, messages: list[dict], max_new_tokens: int) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = inputs.to(model_device)

        with self._torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated = output[0]
        decode_candidates = []
        input_ids = inputs.get("input_ids")
        if (
            input_ids is not None
            and hasattr(generated, "shape")
            and generated.shape[-1] > input_ids.shape[-1]
        ):
            decode_candidates.append(generated[input_ids.shape[-1]:])
        decode_candidates.append(generated)

        for candidate in decode_candidates:
            decoded = self.processor.decode(candidate, skip_special_tokens=True)
            text = _clean_generated_text(decoded)
            if text:
                return text
        return ""


class _Gemma4Runtime:
    def __init__(self, model_id: str):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Gemma 4 backend requires transformers, torch, and accelerate. "
                "Install them with: uv pip install -U transformers torch accelerate"
            ) from exc

        dtype = os.environ.get("GEMMA4_DTYPE", "auto")
        device_map = os.environ.get("GEMMA4_DEVICE_MAP", "auto")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()
        self._torch = torch

    def generate(self, messages: list[dict], max_new_tokens: int) -> str:
        enable_thinking = _env_bool("GEMMA4_ENABLE_THINKING", False)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self.processor(text=text, return_tensors="pt")
        model_device = getattr(self.model, "device", None)
        if model_device is not None:
            inputs = inputs.to(model_device)
        input_len = inputs["input_ids"].shape[-1]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": _env_float("GEMMA4_TEMPERATURE", 1.0),
            "top_p": _env_float("GEMMA4_TOP_P", 0.95),
            "top_k": _env_int("GEMMA4_TOP_K", 64),
            "do_sample": _env_bool("GEMMA4_DO_SAMPLE", True),
        }

        with self._torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)

        response = self.processor.decode(
            output[0][input_len:],
            skip_special_tokens=False,
        ).strip()
        if hasattr(self.processor, "parse_response"):
            parsed = self.processor.parse_response(response)
            if isinstance(parsed, str):
                response = parsed
            elif isinstance(parsed, dict):
                response = parsed.get("content") or parsed.get("answer") or response
            elif isinstance(parsed, (list, tuple)) and parsed:
                response = str(parsed[-1])
        return _strip_thinking_channel(str(response))


@lru_cache(maxsize=2)
def _get_diffusiongemma_runtime(model_id: str) -> _DiffusionGemmaRuntime:
    return _DiffusionGemmaRuntime(model_id)


@lru_cache(maxsize=2)
def _get_gemma4_runtime(model_id: str) -> _Gemma4Runtime:
    return _Gemma4Runtime(model_id)


def _strip_thinking_channel(text: str) -> str:
    """Remove empty or populated DiffusionGemma thinking-channel tags."""
    text = re.sub(r"<\|channel\>thought\s*.*?<channel\|>", "", text, flags=re.S)
    return text.strip()


def _decoded_to_text(value) -> str:
    """Normalize processor.decode outputs across Gemma processor versions."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "content" in value:
            return _decoded_to_text(value.get("content"))
        parts = [_decoded_to_text(item) for item in value.values()]
        return "\n".join(part for part in parts if part)
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            if isinstance(item, dict) and item.get("role") in {"assistant", "model"}:
                text = _decoded_to_text(item.get("content"))
                if text:
                    return text
        parts = [_decoded_to_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value)


def _clean_generated_text(decoded) -> str:
    """Extract only the final model answer from decoded chat-template text."""
    text = _strip_thinking_channel(_decoded_to_text(decoded))
    for marker in (
        "\nmodel\nthought\n",
        "\nassistant\nthought\n",
        "\nmodel\n",
        "\nassistant\n",
    ):
        if marker in text:
            text = text.rsplit(marker, 1)[1].strip()
            break
    if text.startswith("thought\n"):
        text = text[len("thought\n"):].strip()
    return text


class OpenAIAgent:
    """Compatibility agent wrapper with DiffusionGemma as the default backend."""

    def __init__(self, api_key=None, base_url=None, model=None, backend=None):
        self.backend = (backend or os.environ.get("MODEL_BACKEND") or DEFAULT_BACKEND).lower()
        self.api_key = api_key or os.environ.get("API_KEY")
        self.base_url = base_url or os.environ.get("BASE_URL")
        if self.backend == "diffusiongemma":
            self.model = model or os.environ.get("DIFFUSIONGEMMA_MODEL_ID") or DEFAULT_DIFFUSIONGEMMA_MODEL
        elif self.backend == "gemma4":
            self.model = model or os.environ.get("GEMMA4_MODEL_ID") or DEFAULT_GEMMA4_MODEL
        else:
            self.model = model or os.environ.get("MODEL_NAME")
        self._client = None

    @property
    def client(self):
        if self.backend != "openai":
            raise RuntimeError("client is only available for MODEL_BACKEND=openai")
        if self._client is None:
            if not self.api_key or not self.base_url:
                raise RuntimeError("MODEL_BACKEND=openai requires API_KEY and BASE_URL")
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _complete(self, user_content: str, max_new_tokens: int | None = None) -> str:
        messages = [{"role": "user", "content": user_content}]
        max_tokens = max_new_tokens or _env_int("DIFFUSIONGEMMA_MAX_NEW_TOKENS", 512)

        if self.backend == "diffusiongemma":
            runtime = _get_diffusiongemma_runtime(self.model)
            return runtime.generate(messages, max_tokens)

        if self.backend == "gemma4":
            runtime = _get_gemma4_runtime(self.model)
            return runtime.generate(messages, max_tokens)

        if self.backend == "openai":
            if not self.model:
                raise RuntimeError("MODEL_BACKEND=openai requires MODEL_NAME")
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                }
            )
            return completion.choices[0].message.content

        raise ValueError(
            f"Unsupported MODEL_BACKEND={self.backend!r}. "
            "Use 'diffusiongemma', 'gemma4', or 'openai'."
        )

    def generate(self, prompt):
        return self._complete(prompt)
