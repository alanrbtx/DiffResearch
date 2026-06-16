# DiffResearch

DiffResearch is a lightweight framework for autonomous research workflows. It uses a native DiffusionGemma core model, academic search backends, web extraction, and multi-agent synthesis to produce grounded long-form reports without LangChain or LangGraph.

The default model backend is `google/diffusiongemma-26B-A4B-it` via Hugging Face Transformers. The repository also supports `google/gemma-4-26B-A4B-it` as a native Transformers backend, while OpenAI-compatible servers remain available as an explicit fallback.

## What It Does

- Formats a user topic into search queries.
- Searches ArXiv and Semantic Scholar for academic sources.
- Optionally uses Serper for general web search in the experimental plan-based benchmark runner.
- Extracts article or paper text with ArXiv metadata, Serper scraping, `trafilatura`, or BeautifulSoup fallback.
- Builds structured literature-review reports with inline numbered references.
- Supports benchmark runs against a sibling DeepResearchBench checkout.

## Repository Layout

```text
run_lite_deep_research.py   # single-pass literature review workflow
run_full_deep_research.py   # decomposition, judge, and plan-check workflow
dr_bench/run_dr_bench.py    # DeepResearchBench runner using ArXiv + Semantic Scholar
dr_bench/run_dr_bench_plan_based.py
                            # experimental plan-based Serper/web benchmark runner
src/agents/                 # model backend wrapper and research agents
src/web_tools/              # search engines and page extraction utilities
examples/                   # sample reports
examples/plan_based_report.txt
                            # sample report from the plan-based benchmark path
deep_research_scheem.png    # architecture diagram
```

## Setup

Install dependencies with `uv`:

```bash
uv venv
uv pip install -r requirements.txt
```

The native Hugging Face backends require current `transformers`, `torch`, and `accelerate`; these are included in `requirements.txt`.

The default backend loads DiffusionGemma locally:

```bash
export MODEL_BACKEND="diffusiongemma"
export DIFFUSIONGEMMA_MODEL_ID="google/diffusiongemma-26B-A4B-it"
```

`MODEL_BACKEND` and `DIFFUSIONGEMMA_MODEL_ID` can be omitted when using the defaults above. If the model requires authenticated access in your Hugging Face environment, login with the Hugging Face CLI or set `HF_TOKEN`.

Optional DiffusionGemma runtime knobs:

```bash
export DIFFUSIONGEMMA_DEVICE_MAP="auto"
export DIFFUSIONGEMMA_DTYPE="auto"
export DIFFUSIONGEMMA_MAX_NEW_TOKENS="512"
```

Use an OpenAI-compatible backend only when explicitly requested:

```bash
export MODEL_BACKEND="openai"
export API_KEY="your_api_key"
export BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="your-model-name"
```

Optional search keys:

```bash
export S2_API_KEY="your_semantic_scholar_key"
export SERPER_API_KEY="your_serper_key"
```

`S2_API_KEY` is optional for Semantic Scholar. `SERPER_API_KEY` is required by `dr_bench/run_dr_bench_plan_based.py` for web search and Serper scraping.

## Gemma 4 Backend

Use Gemma 4 26B A4B instead of DiffusionGemma by selecting the `gemma4` backend:

```bash
export MODEL_BACKEND="gemma4"
export GEMMA4_MODEL_ID="google/gemma-4-26B-A4B-it"
```

Optional Gemma 4 runtime and sampling knobs:

```bash
export GEMMA4_DEVICE_MAP="auto"
export GEMMA4_DTYPE="auto"
export GEMMA4_ENABLE_THINKING="0"
export GEMMA4_TEMPERATURE="1.0"
export GEMMA4_TOP_P="0.95"
export GEMMA4_TOP_K="64"
export GEMMA4_DO_SAMPLE="1"
```

The Gemma 4 backend is text-only in this repository's agent pipeline. The model itself also supports image inputs; add multimodal plumbing separately before using images in agents.

## DiffusionGemma Inference

The native backend follows the Hugging Face Transformers inference path:

```python
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

processor = AutoProcessor.from_pretrained("google/diffusiongemma-26B-A4B-it")
model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    "google/diffusiongemma-26B-A4B-it",
    dtype="auto",
    device_map="auto",
)
```

Agents call `processor.apply_chat_template(...)` and then `model.generate(...)`. The model is loaded lazily and cached once per process, so creating multiple agents does not reload the 26B checkpoint.

For Gemma 4, the native backend follows the causal-LM path:

```python
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("google/gemma-4-26B-A4B-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-26B-A4B-it",
    dtype="auto",
    device_map="auto",
)
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = processor(text=text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
```

## Usage

### Lite Literature Review

Runs one formatted academic search, gathers papers, and writes `report_2.txt`.

```bash
uv run run_lite_deep_research.py --prompt "Retrieval-augmented generation for open-domain QA"
```

Useful flags:

```bash
--relevance  # filter papers with RelevanceAgent
--squeeze    # compress fetched text with ExtractionAgent before synthesis
```

### Full Literature Review

Adds complexity detection, query decomposition, judge refinement, and plan coverage checks.

```bash
uv run run_full_deep_research.py --prompt "How has reinforcement learning from human feedback evolved for LLM alignment?"
```

Simple prompts write `report.txt`; complex prompts write `report_2.txt`.

## DeepResearchBench

The benchmark scripts expect DeepResearchBench data in a sibling directory:

```text
../deep_research_bench/data/prompt_data/query.jsonl
```

Run the main benchmark pipeline:

```bash
uv run dr_bench/run_dr_bench.py --model-name my-model
```

Resume an interrupted run:

```bash
uv run dr_bench/run_dr_bench.py --model-name my-model --resume
```

Output is written to:

```text
../deep_research_bench/data/test_data/raw_data/<model-name>.jsonl
```

## Plan-Based Benchmark Runner

`dr_bench/run_dr_bench_plan_based.py` adds `QueryFormattingAgent`, `IntentAgent`, `PlanningAgent`, optional relevance filtering, optional squeezing, and Serper web search:

```bash
uv run dr_bench/run_dr_bench_plan_based.py --model-name my-model-plan --relevance --squeeze
```

Important: although `IntentAgent` is called, the script currently overrides the result with `intent = 'web'`, so it always uses Serper/web search. In prior checks, this plan-based Serper/web-search path did not improve results.

## Agent Pipelines

`dr_bench/run_dr_bench.py`:

```text
ComplexityAgent -> DecomposeAgent -> ArXiv/SemanticScholar search -> SummarizationAgent -> JudgeAgent
```

`dr_bench/run_dr_bench_plan_based.py`:

```text
QueryFormattingAgent -> IntentAgent (ignored) -> PlanningAgent -> Serper web search -> optional RelevanceAgent/ExtractionAgent -> SummarizationAgent
```

## Notes and Limitations

- ArXiv requests are rate-limited; the search class waits between requests.
- DuckDuckGo HTML search is still present as legacy/fallback code and may be throttled.
- Several historical branches contained generated files such as `report_2.txt` and `__pycache__`; avoid committing new generated artifacts.
- Keep API keys and local environment files out of Git.

## License

Licensed under the Apache License, Version 2.0.
