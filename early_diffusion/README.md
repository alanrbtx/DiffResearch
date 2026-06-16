# Early Diffusion Query Extraction

This directory contains the DiffusionGemma `put_draft` experiment for early
query extraction during diffusion decoding.

## Contents

```text
scripts/draft_step_probe.py          # raw HF DiffusionGemma draft-step probe
scripts/raw_hf_query_benchmark.py    # raw HF generation-only latency benchmark
data/query_decomposition_100.jsonl   # 100 source research queries
results/report.md                    # human-readable report
results/draft_step_summary.json      # aggregate metrics
results/draft_step_results.jsonl     # per-query final and early outputs
results/draft_step_events.jsonl      # sampled/valid draft events
results/per_query_summary.csv        # compact spreadsheet export
results/early_vs_final_differences.json
```

## Run The Draft Probe

The probe uses Hugging Face directly, without FastAPI. Model loading is
excluded from the reported draft/final latency metrics.

```bash
HF_ENDPOINT=http://huggingface.proxy \
HF_HUB_DISABLE_XET=1 \
HF_HOME=/cache/models/huggingface \
python3 early_diffusion/scripts/draft_step_probe.py \
  --queries-file early_diffusion/data/query_decomposition_100.jsonl \
  --events-file early_diffusion/results/draft_step_events.jsonl \
  --results-file early_diffusion/results/draft_step_results.jsonl \
  --summary-file early_diffusion/results/draft_step_summary.json \
  --max-new-tokens 768 \
  --stable-steps 2
```

## Run The Raw HF Latency Benchmark

```bash
HF_ENDPOINT=http://huggingface.proxy \
HF_HUB_DISABLE_XET=1 \
HF_HOME=/cache/models/huggingface \
python3 early_diffusion/scripts/raw_hf_query_benchmark.py \
  --kind diffusion \
  --model-id google/diffusiongemma-26B-A4B-it \
  --output-file early_diffusion/results/raw_hf_query_generation_metrics.json \
  --repeats 3 \
  --max-new-tokens 768
```

## Summary

On the saved 100-query run, the first valid exact-6 JSON appeared for every
query. Median latency to first exact-6 JSON was about 243 ms, versus about
485 ms for full generation. Waiting for 2 identical exact-6 draft states
matched the final answer in 91 of 100 cases, but median latency was close to
full generation.
