# DiffResearch

**Accelerating Deep Research with Diffusion Language Models**

DiffResearch is an open multi-agent framework for studying how diffusion language models change the latency and quality of deep-research systems. The EMNLP study holds the agent scaffold, prompts, retrieval, and evaluation fixed while swapping a native DiffusionGemma backbone with a similarly sized autoregressive Gemma 4 backbone.

The repository also implements **dFast**, a decomposition mode that reads DiffusionGemma's intermediate denoising states and returns a valid set of search subqueries before generation finishes. The framework then retrieves evidence and produces long-form, citation-bearing reports without LangChain or LangGraph.

## 🔔 News

> **August 2026**: 🎉 Our paper *DiffResearch: Accelerating Deep Research with Diffusion Language Models* was **accepted to the EMNLP 2026 Industry Track!**

## Paper Overview

Deep-research agents repeatedly plan, decompose a question, retrieve evidence, and synthesize a long report. Autoregressive models perform the model-bound parts of this pipeline token by token. DiffResearch tests whether replacing only that backbone with diffusion decoding reduces latency without lowering report quality.

The paper studies three configurations on 100 Deep Research Bench tasks:

- **dFull:** DiffusionGemma with ordinary full decomposition generation.
- **dFast:** the same diffusion backbone, with subqueries extracted from intermediate denoising states.
- **AR:** a similarly sized Gemma 4 autoregressive backbone in the same agent pipeline.

### Main Findings

| Result | Finding |
| --- | --- |
| End-to-end latency | dFull and dFast are roughly **6× faster** than AR in the reported H200 setup. |
| Planning | Diffusion reduces mean planning latency from 39.32 s to about 3.6–3.7 s (**10.7×**). |
| Long-form writing | Mean writing latency falls from 59.78 s to about 5.1–5.3 s (**about 11–12×**). |
| dFast decomposition | The first valid exact-count draft is **1.99× faster at the median** than full decomposition generation in the isolated probe. |
| Report quality | RACE scores remain similar; dFast has the best available-case overall score, while paired mean-difference intervals include zero. |

dFast is a **stage-level optimization**: it accelerates decomposition, but does not provide an additional end-to-end reduction over dFull in the current pipeline. The diffusion reports are only about 10–17% shorter than the AR reports in the language-specific means, so output length alone does not explain the much larger writing-stage speedup.

A separate Mercury-backed DiffResearch instance is also reported in a public leaderboard snapshot. That result measures complete-system competitiveness and is not a causal comparison of diffusion decoding.

## dFast: Reading Intermediate Denoising States

dFast attaches a custom streamer to DiffusionGemma generation. At each denoising step it decodes the current draft, searches for a valid exact-count JSON object, and stops as soon as one appears. The released fast-mode entry point and isolated probe request exactly six non-empty subqueries. dFast does not use a separate draft model and is therefore not speculative decoding.

```text
research question
      -> planning
      -> dFast decomposition (first valid exact-6 draft)
      -> ArXiv / Semantic Scholar retrieval
      -> source extraction
      -> long-form synthesis
```

Early drafts are used only for retrieval-oriented decomposition, where wording variation is tolerable. Final report generation still runs to completion.

## What the Repository Provides

- Native Hugging Face backends for DiffusionGemma and Gemma 4, plus an explicit OpenAI-compatible fallback.
- A paper-aligned dFast deep-research entry point with report and metadata outputs.
- The saved 100-query intermediate-state probe and its per-query artifacts.
- ArXiv, Semantic Scholar, and optional Serper retrieval backends.
- General-purpose lite, iterative, and DeepResearchBench workflows.

## Repository Layout

```text
diffusion_deep_research.py  # paper-aligned dFast pipeline
early_diffusion/            # intermediate-state probe, results, and reproduction notes
run_lite_deep_research.py   # general single-pass literature review workflow
run_full_deep_research.py   # general decomposition, judge, and plan-check workflow
dr_bench/run_dr_bench.py    # DeepResearchBench runner using ArXiv + Semantic Scholar
dr_bench/run_dr_bench_plan_based.py
                            # experimental plan-based Serper/web benchmark runner
src/agents/                 # model backend wrapper and research agents
src/web_tools/              # search engines and page extraction utilities
examples/                   # sample reports
arGemma_report.txt          # saved autoregressive comparison report
dGemma_report.txt           # saved diffusion comparison report
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

## Paper-Aligned dFast Workflow

Run the diffusion-native pipeline with fast decomposition:

```bash
uv run diffusion_deep_research.py \
  --prompt "How do diffusion language models change generation speed and quality?" \
  --output diffusion_deep_report.txt \
  --metadata-output diffusion_deep_metadata.json
```

The command writes the final report separately from structured metadata containing the plan, extracted subqueries, decomposition latency, draft step, retrieved records, and references. Add `--relevance` to filter results by title or `--squeeze` to compress retrieved text before synthesis.

To wait for full decomposition generation instead of stopping at the first valid exact-count draft, add:

```bash
--no-early-stop
```

The isolated 100-query dFast experiment, saved outputs, and reproduction command are documented in [`early_diffusion/README.md`](early_diffusion/README.md). Its authoritative per-query outputs are stored in `early_diffusion/results/draft_step_results.jsonl`.

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

## Additional Workflows

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

The benchmark scripts use the same agent abstraction for diffusion and autoregressive backbones and expect DeepResearchBench data in a sibling directory:

```text
../deep_research_bench/data/prompt_data/query.jsonl
```

Run the general benchmark pipeline with DiffusionGemma:

```bash
MODEL_BACKEND=diffusiongemma \
uv run dr_bench/run_dr_bench.py --model-name diffusiongemma
```

Swap only the configured backbone to run Gemma 4:

```bash
MODEL_BACKEND=gemma4 \
uv run dr_bench/run_dr_bench.py --model-name gemma4
```

Resume an interrupted run:

```bash
uv run dr_bench/run_dr_bench.py --model-name my-model --resume
```

Output is written to:

```text
../deep_research_bench/data/test_data/raw_data/<model-name>.jsonl
```

### Reproduction Scope

The public runner above is a practical ArXiv + Semantic Scholar workflow. Matching the paper's controlled tables additionally requires the reported Semantic-Scholar-only retrieval configuration, H200 serving setup, stage-level timing instrumentation, and official RACE evaluation. The dFast intermediate-state mechanism and its isolated timing artifacts are released separately under `early_diffusion/`.

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
- Generated reports and `__pycache__` directories are development artifacts; avoid committing new ones.
- Keep API keys and local environment files out of Git.

## License

Licensed under the Apache License, Version 2.0.
