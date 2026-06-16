from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoProcessor


SAMPLE_STEPS = {1, 2, 3, 4, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512}


PROMPT_TEMPLATE = """You are a research-query decomposition engine.

Task:
Decompose the user query into several focused academic search subqueries.

User query:
{query}

Requirements:
- Return exactly 6 subqueries.
- Each subquery must be useful for finding papers, surveys, or technical reports.
- Cover complementary aspects of the topic: definitions, model families, training methods, inference and sampling, benchmarks, and limitations.
- Keep each subquery concise, search-engine friendly, and in English.
- Do not answer the research question.
- Do not include explanations, citations, or markdown.
- Return only valid JSON with this schema:
{{
  "subqueries": [
    "subquery 1",
    "subquery 2",
    "subquery 3",
    "subquery 4",
    "subquery 5",
    "subquery 6"
  ]
}}
"""


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def diffusion_model_class() -> Any:
    model_cls = getattr(transformers, "DiffusionGemmaForBlockDiffusion", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
    if model_cls is None:
        raise RuntimeError("DiffusionGemma model class is not available in transformers")
    return model_cls


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


def strip_thinking_channel(text: str) -> str:
    text = re.sub(r"<\|channel\>thought\s*.*?<channel\|>", "", text, flags=re.S)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


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
    return text.strip()


def find_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stack = 0
    start: int | None = None
    for index, char in enumerate(text):
        if char == "{":
            if stack == 0:
                start = index
            stack += 1
        elif char == "}":
            if stack:
                stack -= 1
                if stack == 0 and start is not None:
                    candidates.append(text[start : index + 1])
                    start = None
    if not candidates:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            candidates.append(text[first : last + 1])
    return candidates


def parse_subquery_json(text: str) -> dict[str, Any] | None:
    for candidate in find_json_candidates(text):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        subqueries = obj.get("subqueries") if isinstance(obj, dict) else None
        if isinstance(subqueries, list) and all(isinstance(item, str) for item in subqueries):
            return {"subqueries": [item.strip() for item in subqueries if item.strip()]}
    return None


def canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        text = clean_generated_text(processor.decode(candidate, skip_special_tokens=True))
        if text:
            return text, generated_token_count

    for candidate in decode_candidates:
        text = clean_generated_text(processor.decode(candidate, skip_special_tokens=False))
        if text:
            return text, generated_token_count

    return "", generated_token_count


class DraftProbeStreamer:
    _takes_logits = False

    def __init__(
        self,
        *,
        tokenizer: Any,
        query_id: str,
        source_query: str,
        start_time: float,
        stable_steps: int,
        events_file: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.query_id = query_id
        self.source_query = source_query
        self.start_time = start_time
        self.stable_steps = stable_steps
        self.events_file = events_file
        self.draft_step = 0
        self.last_exact6_canonical: str | None = None
        self.exact6_streak = 0
        self.first_valid: dict[str, Any] | None = None
        self.first_exact6: dict[str, Any] | None = None
        self.first_stable_exact6: dict[str, Any] | None = None
        self.valid_json_count = 0
        self.exact6_count = 0
        self.last_text = ""
        self.last_obj: dict[str, Any] | None = None

    def _write_event(
        self,
        *,
        event_type: str,
        elapsed_ms: float,
        text: str,
        obj: dict[str, Any] | None,
    ) -> None:
        event = {
            "query_id": self.query_id,
            "source_query": self.source_query,
            "draft_step": self.draft_step,
            "elapsed_ms": elapsed_ms,
            "event_type": event_type,
            "text_sha256": sha256_text(text),
            "text_chars": len(text),
            "text": text,
            "parsed_subqueries": obj.get("subqueries") if obj else None,
        }
        self.events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.events_file.flush()

    def put_draft(self, value: Any, **kwargs: Any) -> None:
        self.draft_step += 1
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        if len(value.shape) > 1:
            value = value[0]
        text = clean_generated_text(self.tokenizer.decode(value, skip_special_tokens=True))
        self.last_text = text

        obj = parse_subquery_json(text)
        exact6 = bool(obj and len(obj.get("subqueries", [])) == 6)
        event_types: list[str] = []

        if obj:
            self.valid_json_count += 1
            self.last_obj = obj
            event_types.append("valid_json")
            if self.first_valid is None:
                self.first_valid = {
                    "draft_step": self.draft_step,
                    "elapsed_ms": elapsed_ms,
                    "subqueries": obj["subqueries"],
                    "text": text,
                }
                event_types.append("first_valid_json")

        if exact6 and obj:
            self.exact6_count += 1
            event_types.append("exact6")
            current_canonical = canonical_json(obj)
            if self.first_exact6 is None:
                self.first_exact6 = {
                    "draft_step": self.draft_step,
                    "elapsed_ms": elapsed_ms,
                    "subqueries": obj["subqueries"],
                    "text": text,
                }
                event_types.append("first_exact6")

            if current_canonical == self.last_exact6_canonical:
                self.exact6_streak += 1
            else:
                self.last_exact6_canonical = current_canonical
                self.exact6_streak = 1

            if self.first_stable_exact6 is None and self.exact6_streak >= self.stable_steps:
                self.first_stable_exact6 = {
                    "draft_step": self.draft_step,
                    "elapsed_ms": elapsed_ms,
                    "stable_steps": self.stable_steps,
                    "subqueries": obj["subqueries"],
                    "text": text,
                }
                event_types.append(f"first_stable_exact6_{self.stable_steps}")
        else:
            self.last_exact6_canonical = None
            self.exact6_streak = 0

        should_sample = self.draft_step in SAMPLE_STEPS
        if should_sample:
            event_types.append("sample")

        if event_types:
            self._write_event(
                event_type=",".join(sorted(set(event_types))),
                elapsed_ms=elapsed_ms,
                text=text,
                obj=obj,
            )

    def put(self, value: Any) -> None:
        return None

    def end(self) -> None:
        return None


def load_queries(path: Path, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({"id": str(obj["id"]), "query": str(obj["query"])})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def summarize_numbers(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-file", required=True)
    parser.add_argument("--events-file", required=True)
    parser.add_argument("--results-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--model-id", default="google/diffusiongemma-26B-A4B-it")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--stable-steps", type=int, default=2)
    args = parser.parse_args()

    queries = load_queries(Path(args.queries_file), args.limit)
    print(f"loaded_queries={len(queries)}", flush=True)

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = diffusion_model_class().from_pretrained(
        args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    model.eval()
    synchronize_cuda()
    load_ms = (time.perf_counter() - load_start) * 1000.0
    print(f"model_loaded_ms={load_ms:.1f}", flush=True)

    results_path = Path(args.results_file)
    events_path = Path(args.events_file)
    summary_path = Path(args.summary_file)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as results_file, events_path.open(
        "w", encoding="utf-8"
    ) as events_file:
        for index, item in enumerate(queries, start=1):
            query_id = item["id"]
            source_query = item["query"]
            prompt = PROMPT_TEMPLATE.format(query=source_query)
            messages = [{"role": "user", "content": prompt}]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = move_to_model_device(inputs, model)

            synchronize_cuda()
            start_time = time.perf_counter()
            streamer = DraftProbeStreamer(
                tokenizer=processor.tokenizer,
                query_id=query_id,
                source_query=source_query,
                start_time=start_time,
                stable_steps=args.stable_steps,
                events_file=events_file,
            )

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    streamer=streamer,
                )
            synchronize_cuda()
            full_generation_ms = (time.perf_counter() - start_time) * 1000.0
            final_text, generated_tokens = decode_output(processor, inputs, output)
            final_obj = parse_subquery_json(final_text)
            final_exact6 = bool(final_obj and len(final_obj.get("subqueries", [])) == 6)
            final_canonical = canonical_json(final_obj) if final_obj else None

            row = {
                "query_id": query_id,
                "query_index": index,
                "source_query": source_query,
                "draft_steps": streamer.draft_step,
                "valid_json_count": streamer.valid_json_count,
                "exact6_count": streamer.exact6_count,
                "generated_tokens": generated_tokens,
                "full_generation_ms": full_generation_ms,
                "first_valid_json": streamer.first_valid,
                "first_exact6": streamer.first_exact6,
                "first_stable_exact6": streamer.first_stable_exact6,
                "final_valid_json": final_obj is not None,
                "final_exact6": final_exact6,
                "final_subqueries": final_obj.get("subqueries") if final_obj else None,
                "final_text": final_text,
                "first_exact6_equals_final": (
                    bool(streamer.first_exact6)
                    and final_canonical is not None
                    and canonical_json({"subqueries": streamer.first_exact6["subqueries"]}) == final_canonical
                ),
                "first_stable_exact6_equals_final": (
                    bool(streamer.first_stable_exact6)
                    and final_canonical is not None
                    and canonical_json({"subqueries": streamer.first_stable_exact6["subqueries"]})
                    == final_canonical
                ),
            }
            rows.append(row)
            results_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            results_file.flush()

            first_exact_ms = row["first_exact6"]["elapsed_ms"] if row["first_exact6"] else None
            stable_ms = row["first_stable_exact6"]["elapsed_ms"] if row["first_stable_exact6"] else None
            print(
                f"{index:03d}/{len(queries)} {query_id}: drafts={streamer.draft_step}, "
                f"full={full_generation_ms:.1f}ms, first_exact6={first_exact_ms}, stable={stable_ms}, "
                f"final_exact6={final_exact6}",
                flush=True,
            )

    first_valid_ms = [r["first_valid_json"]["elapsed_ms"] for r in rows if r["first_valid_json"]]
    first_exact6_ms = [r["first_exact6"]["elapsed_ms"] for r in rows if r["first_exact6"]]
    stable_ms = [r["first_stable_exact6"]["elapsed_ms"] for r in rows if r["first_stable_exact6"]]
    full_ms = [r["full_generation_ms"] for r in rows]
    summary = {
        "model_id": args.model_id,
        "queries_file": args.queries_file,
        "total_queries": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "stable_steps": args.stable_steps,
        "load_ms_excluded": load_ms,
        "final_exact6_count": sum(1 for r in rows if r["final_exact6"]),
        "first_valid_json_count": sum(1 for r in rows if r["first_valid_json"]),
        "first_exact6_count": sum(1 for r in rows if r["first_exact6"]),
        "first_stable_exact6_count": sum(1 for r in rows if r["first_stable_exact6"]),
        "first_exact6_equals_final_count": sum(1 for r in rows if r["first_exact6_equals_final"]),
        "first_stable_exact6_equals_final_count": sum(
            1 for r in rows if r["first_stable_exact6_equals_final"]
        ),
        "draft_steps": summarize_numbers([float(r["draft_steps"]) for r in rows]),
        "full_generation_ms": summarize_numbers(full_ms),
        "first_valid_json_ms": summarize_numbers(first_valid_ms),
        "first_exact6_ms": summarize_numbers(first_exact6_ms),
        "first_stable_exact6_ms": summarize_numbers(stable_ms),
    }
    if first_exact6_ms:
        summary["median_full_over_first_exact6"] = (
            summary["full_generation_ms"]["median"] / summary["first_exact6_ms"]["median"]
        )
    if stable_ms:
        summary["median_full_over_first_stable_exact6"] = (
            summary["full_generation_ms"]["median"] / summary["first_stable_exact6_ms"]["median"]
        )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
