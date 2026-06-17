from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor


DEFAULT_PROMPT = """You are a research-query decomposition engine.

Task:
Decompose the user query into several focused academic search subqueries.

User query:
diffusion LLM

Requirements:
- Return exactly 6 subqueries.
- Each subquery must be useful for finding papers, surveys, or technical reports.
- Cover complementary aspects of the topic: definitions, model families, training methods, inference and sampling, benchmarks, and limitations.
- Keep each subquery concise, search-engine friendly, and in English.
- Do not answer the research question.
- Do not include explanations, citations, or markdown.
- Return only valid JSON with this schema:
{
  "subqueries": [
    "subquery 1",
    "subquery 2",
    "subquery 3",
    "subquery 4",
    "subquery 5",
    "subquery 6"
  ]
}
"""


def strip_thinking_channel(text: str) -> str:
    text = re.sub(r"<\|channel\>thought\s*.*?<channel\|>", "", text, flags=re.S)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


def decoded_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "content" in value:
            return decoded_to_text(value.get("content"))
        return "\n".join(part for part in (decoded_to_text(item) for item in value.values()) if part)
    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            if isinstance(item, dict) and item.get("role") in {"assistant", "model"}:
                text = decoded_to_text(item.get("content"))
                if text:
                    return text
        return "\n".join(part for part in (decoded_to_text(item) for item in value) if part)
    return str(value)


def clean_generated_text(decoded: Any) -> str:
    text = strip_thinking_channel(decoded_to_text(decoded))
    for marker in (
        "\nassistant\nanalysis\n",
        "\nmodel\nthought\n",
        "\nassistant\nthought\n",
        "\nassistant\n",
        "\nmodel\n",
    ):
        if marker in text:
            text = text.rsplit(marker, 1)[1].strip()
            break
    for prefix in ("analysis\n", "thought\n", "assistant\n", "model\n"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def move_to_model_device(batch: Any, model: Any) -> Any:
    model_device = getattr(model, "device", None)
    if model_device is not None and str(model_device) not in {"cpu", "meta"}:
        return batch.to(model_device) if hasattr(batch, "to") else batch

    device_map = getattr(model, "hf_device_map", {}) or {}
    for device in device_map.values():
        if isinstance(device, int):
            return batch.to(f"cuda:{device}") if hasattr(batch, "to") else batch
        if isinstance(device, str) and device.startswith("cuda"):
            return batch.to(device) if hasattr(batch, "to") else batch
    return batch


def diffusion_model_class() -> Any:
    model_cls = getattr(transformers, "DiffusionGemmaForBlockDiffusion", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
    if model_cls is None:
        raise RuntimeError("DiffusionGemma model class is not available in transformers")
    return model_cls


def prepare_diffusion_inputs(processor: Any, prompt: str, model: Any) -> Any:
    messages = [{"role": "user", "content": prompt}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return move_to_model_device(inputs, model)


def prepare_ar_inputs(processor: Any, prompt: str, model: Any) -> Any:
    messages = [{"role": "user", "content": prompt}]
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        text = processor.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        text = processor.apply_chat_template(messages, **template_kwargs)
    inputs = processor(text=text, return_tensors="pt")
    return move_to_model_device(inputs, model)


def decode_output(processor: Any, inputs: Any, output: Any) -> tuple[str, int | None]:
    first = output[0]
    input_ids = inputs.get("input_ids")
    generated_token_count = None
    decode_candidates: list[Any] = []
    if input_ids is not None and hasattr(first, "shape"):
        generated_token_count = max(0, int(first.shape[-1] - input_ids.shape[-1]))
        if first.shape[-1] > input_ids.shape[-1]:
            decode_candidates.append(first[input_ids.shape[-1] :])
    decode_candidates.append(first)

    for candidate in decode_candidates:
        decoded = processor.decode(candidate, skip_special_tokens=True)
        text = clean_generated_text(decoded)
        if text:
            return text, generated_token_count

    for candidate in decode_candidates:
        decoded = processor.decode(candidate, skip_special_tokens=False)
        text = clean_generated_text(decoded)
        if text:
            return text, generated_token_count

    return "", generated_token_count


def load_model_and_processor(kind: str, model_id: str, dtype: str, device_map: str) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(model_id)
    if kind == "diffusion":
        model = diffusion_model_class().from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
        )
    elif kind == "ar":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=device_map,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device_map,
            )
    else:
        raise ValueError(f"Unsupported model kind: {kind}")
    model.eval()
    return processor, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["diffusion", "ar"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-file")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--do-sample", action="store_true")
    args = parser.parse_args()

    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT

    load_start = time.perf_counter()
    processor, model = load_model_and_processor(args.kind, args.model_id, args.dtype, args.device_map)
    synchronize_cuda()
    load_ms = (time.perf_counter() - load_start) * 1000.0

    records: list[dict[str, Any]] = []
    for run in range(1, args.repeats + 1):
        prep_start = time.perf_counter()
        inputs = (
            prepare_diffusion_inputs(processor, prompt, model)
            if args.kind == "diffusion"
            else prepare_ar_inputs(processor, prompt, model)
        )
        synchronize_cuda()
        prep_ms = (time.perf_counter() - prep_start) * 1000.0

        generate_kwargs: dict[str, Any] = {"max_new_tokens": args.max_new_tokens}
        if args.kind == "ar":
            generate_kwargs["do_sample"] = args.do_sample
            if args.do_sample:
                if args.temperature is not None:
                    generate_kwargs["temperature"] = args.temperature
                if args.top_p is not None:
                    generate_kwargs["top_p"] = args.top_p
                if args.top_k is not None:
                    generate_kwargs["top_k"] = args.top_k

        synchronize_cuda()
        generation_start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, **generate_kwargs)
        synchronize_cuda()
        generation_ms = (time.perf_counter() - generation_start) * 1000.0

        decode_start = time.perf_counter()
        text, generated_tokens = decode_output(processor, inputs, output)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0

        record = {
            "run": run,
            "generation_ms": generation_ms,
            "prep_ms_excluded": prep_ms,
            "decode_ms_excluded": decode_ms,
            "response_chars": len(text),
            "generated_tokens": generated_tokens,
            "response_text": text,
        }
        records.append(record)
        print(
            f"{args.model_id} raw-hf run {run}: generation={generation_ms:.1f} ms, "
            f"chars={len(text)}, generated_tokens={generated_tokens}",
            flush=True,
        )

    generation_latencies = [record["generation_ms"] for record in records]
    result = {
        "model_id": args.model_id,
        "kind": args.kind,
        "timing_scope": "torch model.generate only; model load, prompt read, tokenization, decode, and file writes excluded",
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "load_ms_excluded": load_ms,
        "generation_ms": generation_latencies,
        "mean_generation_ms": statistics.fmean(generation_latencies),
        "median_generation_ms": statistics.median(generation_latencies),
        "min_generation_ms": min(generation_latencies),
        "max_generation_ms": max(generation_latencies),
        "records": records,
    }
    Path(args.output_file).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "model_id",
        "kind",
        "mean_generation_ms",
        "median_generation_ms",
        "min_generation_ms",
        "max_generation_ms",
        "generation_ms",
        "load_ms_excluded",
    )}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
