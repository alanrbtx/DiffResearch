from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.agents.agent_template import (
    DEFAULT_DIFFUSIONGEMMA_MODEL,
    _get_diffusiongemma_runtime,
    agent_kwargs_from_env,
)
from src.agents.agents_collection import ExtractionAgent, PlanningAgent, RelevanceAgent, SummarizationAgent
from src.web_tools.search_engine import ArXiv, SemanticScholar
from src.web_tools.visit_site import visit_site


DECOMPOSITION_PROMPT_TEMPLATE = """You are a research-query decomposition engine.

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


@dataclass
class DecompositionResult:
    subqueries: list[str]
    mode: str
    elapsed_ms: float
    draft_step: int | None
    final_text: str | None


class EarlySubqueriesReady(Exception):
    pass


def synchronize_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


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
        elif char == "}" and stack:
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
            cleaned = [item.strip() for item in subqueries if item.strip()]
            return {"subqueries": cleaned}
    return None


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


class FastSubqueryStreamer:
    _takes_logits = False

    def __init__(self, *, decoder: Any, start_time: float, early_stop: bool) -> None:
        self.decoder = decoder
        self.start_time = start_time
        self.early_stop = early_stop
        self.draft_step = 0
        self.first_exact6: dict[str, Any] | None = None
        self.last_text = ""

    def put_draft(self, value: Any, **kwargs: Any) -> None:
        self.draft_step += 1
        if hasattr(value, "shape") and len(value.shape) > 1:
            value = value[0]
        text = clean_generated_text(self.decoder.decode(value, skip_special_tokens=True))
        self.last_text = text
        obj = parse_subquery_json(text)
        if not obj or len(obj.get("subqueries", [])) != 6:
            return

        elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
        if self.first_exact6 is None:
            self.first_exact6 = {
                "draft_step": self.draft_step,
                "elapsed_ms": elapsed_ms,
                "subqueries": obj["subqueries"],
                "text": text,
            }
        if self.early_stop:
            raise EarlySubqueriesReady()

    def put(self, value: Any) -> None:
        return None

    def end(self) -> None:
        return None


def decompose_fast_mode(
    *,
    query: str,
    model_id: str,
    max_new_tokens: int,
    early_stop: bool,
) -> DecompositionResult:
    runtime = _get_diffusiongemma_runtime(model_id)
    processor = runtime.processor
    model = runtime.model
    prompt = DECOMPOSITION_PROMPT_TEMPLATE.format(query=query)
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
    streamer = FastSubqueryStreamer(decoder=processor, start_time=start_time, early_stop=early_stop)

    try:
        with runtime._torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
            )
    except EarlySubqueriesReady:
        synchronize_cuda()
        assert streamer.first_exact6 is not None
        return DecompositionResult(
            subqueries=streamer.first_exact6["subqueries"],
            mode="first_exact6_draft",
            elapsed_ms=streamer.first_exact6["elapsed_ms"],
            draft_step=streamer.first_exact6["draft_step"],
            final_text=streamer.first_exact6["text"],
        )

    synchronize_cuda()
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    final_text, _generated_tokens = decode_output(processor, inputs, output)
    final_obj = parse_subquery_json(final_text)
    if final_obj and len(final_obj.get("subqueries", [])) == 6:
        return DecompositionResult(
            subqueries=final_obj["subqueries"],
            mode="final_exact6",
            elapsed_ms=elapsed_ms,
            draft_step=streamer.draft_step,
            final_text=final_text,
        )
    if streamer.first_exact6 is not None:
        return DecompositionResult(
            subqueries=streamer.first_exact6["subqueries"],
            mode="first_exact6_draft_no_stop",
            elapsed_ms=streamer.first_exact6["elapsed_ms"],
            draft_step=streamer.first_exact6["draft_step"],
            final_text=streamer.first_exact6["text"],
        )
    raise RuntimeError(
        "DiffusionGemma did not produce valid JSON with exactly 6 subqueries. "
        f"Last draft text: {streamer.last_text!r}. Final text: {final_text!r}"
    )


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).casefold().strip()


def collect_search_results(
    *,
    subqueries: list[str],
    arxiv_per_query: int,
    s2_per_query: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arxiv = ArXiv(max_results=arxiv_per_query)
    s2 = SemanticScholar(max_results=s2_per_query)
    raw_results: list[dict[str, Any]] = []

    for query_index, subquery in enumerate(subqueries, start=1):
        print(f"\n||SEARCH {query_index}/6|| {subquery}", flush=True)

        print("  [ArXiv] searching...", flush=True)
        arxiv_results = arxiv.search(subquery, top_n=arxiv_per_query)
        print(f"  [ArXiv] {len(arxiv_results)} results", flush=True)
        for result in arxiv_results:
            row = dict(result)
            row["source"] = "arxiv"
            row["subquery"] = subquery
            row["subquery_index"] = query_index
            raw_results.append(row)

        print("  [Semantic Scholar] searching...", flush=True)
        s2_results = s2.search(subquery, top_n=s2_per_query)
        print(f"  [Semantic Scholar] {len(s2_results)} results", flush=True)
        for result in s2_results:
            row = dict(result)
            row["source"] = "S2"
            row["subquery"] = subquery
            row["subquery_index"] = query_index
            raw_results.append(row)

    seen_titles: set[str] = set()
    unique_results: list[dict[str, Any]] = []
    for result in raw_results:
        title_key = normalize_title(str(result.get("title", "")))
        if not title_key or title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        unique_results.append(result)

    return raw_results, unique_results


def build_source_material(
    *,
    prompt: str,
    search_results: list[dict[str, Any]],
    rel_agent: RelevanceAgent,
    ext_agent: ExtractionAgent,
    use_relevance: bool,
    use_squeeze: bool,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    result_text = ""
    references: list[str] = []
    source_records: list[dict[str, Any]] = []
    paper_num = 0

    for result in tqdm(search_results, desc="Fetching sources"):
        title = str(result.get("title", "N/A"))
        href = str(result.get("url", "N/A"))
        authors = str(result.get("authors", "N/A"))
        year = str(result.get("year", "N/A"))
        source = str(result.get("source", "unknown"))

        record = {
            "title": title,
            "url": href,
            "authors": authors,
            "year": year,
            "source": source,
            "subquery": result.get("subquery"),
            "subquery_index": result.get("subquery_index"),
            "used": False,
            "skipped_reason": None,
            "text_chars": 0,
        }

        if use_relevance and "1" not in rel_agent.generate(prompt, title):
            record["skipped_reason"] = "relevance_filter"
            source_records.append(record)
            continue

        if source == "S2":
            abstract = str(result.get("abstract", "N/A"))
            clean_text = f"Abstract:\n{abstract}" if abstract != "N/A" else "not available"
        else:
            clean_text = visit_site(href)

        if use_squeeze:
            clean_text = ext_agent.generate(prompt, clean_text)

        paper_num += 1
        result_text += f'\n\n[{paper_num}] {authors} ({year}). "{title}". {href}\n{clean_text}'
        reference = f"[{paper_num}] {authors} ({year}). {title}. {href}"
        references.append(reference)
        record.update({"used": True, "paper_num": paper_num, "text_chars": len(clean_text)})
        source_records.append(record)

    return result_text, references, source_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DiffusionGemma fast-mode deep research")
    parser.add_argument("--prompt", required=True, help="Literature review topic.")
    parser.add_argument("--output", default="diffusion_deep_report.txt")
    parser.add_argument("--metadata-output", default="diffusion_deep_metadata.json")
    parser.add_argument("--model-id", default=os.environ.get("DIFFUSIONGEMMA_MODEL_ID") or DEFAULT_DIFFUSIONGEMMA_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--arxiv-per-query", type=int, default=2)
    parser.add_argument("--s2-per-query", type=int, default=2)
    parser.add_argument("--squeeze", action="store_true", help="Extract/squeeze fetched source text with the model.")
    parser.add_argument("--relevance", action="store_true", help="Filter search results by title relevance.")
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Continue full generation after first exact-6 draft instead of exiting fast mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_kwargs = agent_kwargs_from_env()

    rel_agent = RelevanceAgent(**agent_kwargs)
    ext_agent = ExtractionAgent(**agent_kwargs)
    sum_agent = SummarizationAgent(**agent_kwargs)
    planning_agent = PlanningAgent(**agent_kwargs)

    print("\n||PLANNING AGENT|| Creating literature review plan\n", flush=True)
    plan = planning_agent.generate(args.prompt)
    print(plan, flush=True)

    print("\n||DIFFUSIONGEMMA FAST DECOMPOSITION|| Waiting for first exact-6 draft\n", flush=True)
    decomposition = decompose_fast_mode(
        query=args.prompt,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        early_stop=not args.no_early_stop,
    )
    print(
        f"Decomposition mode={decomposition.mode}, elapsed_ms={decomposition.elapsed_ms:.1f}, "
        f"draft_step={decomposition.draft_step}",
        flush=True,
    )
    for index, subquery in enumerate(decomposition.subqueries, start=1):
        print(f"  {index}. {subquery}", flush=True)

    raw_results, unique_results = collect_search_results(
        subqueries=decomposition.subqueries,
        arxiv_per_query=args.arxiv_per_query,
        s2_per_query=args.s2_per_query,
    )
    print(
        f"\n||SEARCH COMPLETE|| raw_results={len(raw_results)}, unique_titles={len(unique_results)}\n",
        flush=True,
    )

    result_text, references, source_records = build_source_material(
        prompt=args.prompt,
        search_results=unique_results,
        rel_agent=rel_agent,
        ext_agent=ext_agent,
        use_relevance=args.relevance,
        use_squeeze=args.squeeze,
    )
    print(f"\n||SOURCE MATERIAL|| used_sources={len(references)}\n", flush=True)
    if not references:
        raise RuntimeError("No source material was collected; refusing to write an unsourced review.")

    print("\n||SUMMARIZATION AGENT|| Writing literature review\n", flush=True)
    review = sum_agent.generate(
        args.prompt,
        result_text,
        references="\n".join(references),
        plan=plan,
    )

    output_path = Path(args.output)
    metadata_path = Path(args.metadata_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(review, encoding="utf-8")
    metadata = {
        "prompt": args.prompt,
        "plan": plan,
        "decomposition": asdict(decomposition),
        "search": {
            "arxiv_per_query": args.arxiv_per_query,
            "s2_per_query": args.s2_per_query,
            "raw_result_count": len(raw_results),
            "unique_result_count": len(unique_results),
            "raw_results": raw_results,
        },
        "sources": source_records,
        "references": references,
        "output": str(output_path),
        "metadata_output": str(metadata_path),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written to {output_path}", flush=True)
    print(f"Metadata written to {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
