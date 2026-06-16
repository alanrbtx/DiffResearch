# DiffResearch

DiffResearch is a lightweight framework for autonomous research workflows. It uses OpenAI-compatible chat completions, academic search backends, web extraction, and multi-agent synthesis to produce grounded long-form reports without LangChain or LangGraph.

The current `main` branch is based on the `research_bench` pipeline. It focuses on academic literature-review generation and DeepResearchBench-style evaluation.

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
src/agents/                 # OpenAI-compatible agent classes and prompts
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

Configure an OpenAI-compatible backend:

```bash
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
