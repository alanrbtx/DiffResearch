# Early Diffusion Query Extraction Report

## Scope

This report summarizes an experiment with `google/diffusiongemma-26B-A4B-it` using Hugging Face `generate(..., streamer=...)` and a custom streamer with `put_draft`. The task was query decomposition: each input research query must be decomposed into exactly 6 focused academic search subqueries returned as JSON.

The goal was to check whether usable search queries can be extracted before full diffusion generation completes.

## Files

- `../data/query_decomposition_100.jsonl`
- `draft_step_results.jsonl`
- `draft_step_events.jsonl`
- `draft_step_summary.json`
- `per_query_summary.csv`
- `early_vs_final_differences.json`
- `../scripts/draft_step_probe.py`

## Method

- Dataset: 100 synthetic but realistic research queries across LLMs, retrieval, biomedicine, robotics, climate, security, multimodal generation, time series, code, and AI governance.
- Prompt: fixed query decomposition instruction, with only `User query` changed per row.
- Model: `google/diffusiongemma-26B-A4B-it`.
- Instrumentation: custom streamer with `put_draft(value, **kwargs)`. Each draft state was decoded, cleaned, and scanned for a JSON object containing `subqueries`.
- Valid output criterion: JSON object with a `subqueries` list.
- Exact-6 criterion: valid JSON with exactly 6 non-empty string subqueries.
- Stable criterion: same exact-6 JSON observed for 2 consecutive draft states.
- Timing: model loading is excluded. Timings are wall-clock milliseconds from the start of `model.generate()` to draft/final events.

## Headline Results

| Metric | Value |
|---|---:|
| Total queries | 100 |
| Final exact-6 JSON | 100/100 |
| First valid JSON during draft | 100/100 |
| First exact-6 JSON during draft | 100/100 |
| Stable exact-6 JSON, 2 draft states | 100/100 |
| First exact-6 equals final exactly | 1/100 |
| Stable exact-6 equals final exactly | 91/100 |

## Latency Summary

| Stage | Mean ms | P50 ms | P75 ms | P90 ms | P95 ms | Min ms | Max ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| First exact-6 JSON | 268.5 | 243.4 | 289.6 | 292.4 | 338.7 | 193.1 | 1553.1 |
| Stable exact-6 JSON | 475.7 | 468.4 | 502.9 | 537.0 | 581.1 | 338.7 | 1697.5 |
| Full generation | 499.7 | 484.9 | 532.8 | 579.8 | 629.9 | 339.5 | 1749.5 |

## Speedup

- Median full generation / median first exact-6 JSON: `1.99x`.
- Median full generation / median stable exact-6 JSON: `1.04x`.
- Mean full generation / mean first exact-6 JSON: `1.86x`.
- Mean full generation / mean stable exact-6 JSON: `1.05x`.

Interpretation: extracting the first exact-6 JSON gives a clear latency win, but exact string equality with the final answer is low. Waiting for 2 identical exact-6 draft states preserves most final answers exactly, but median latency is close to full generation.

## Draft Step Counts

| Metric | Value |
|---|---:|
| Mean draft steps | 8.99 |
| Median draft steps | 9.0 |
| Min draft steps | 6 |
| Max draft steps | 17 |

## Representative Examples

### Fastest first exact-6 JSON

- Query id: `q097`
- Source query: `education applications of large language models`
- First exact-6: `193.1 ms`
- Stable exact-6: `531.6 ms`
- Full generation: `532.3 ms`
- Stable equals final: `yes`

First exact-6 subqueries:

```json
[
  "survey of large language models applications education education and learning",
  "LLM architectures for for tutoring and and intelligent tutoring systems",
  "fine-tuning large language models for educational content tasks",
  "prompt engineering and and and techniques for for educational assistants",
  "bench for evaluating evaluating language model performance in educational settings",
  "limitations ethical concerns of of LL in in academic education"
]
```

Final subqueries:

```json
[
  "survey of large language model applications in education and pedagogy",
  "LLM architectures for personalized learning and intelligent tutoring systems",
  "fine-tuning large language models for specialized educational content generation",
  "prompt engineering and few-shot learning for educational AI",
  "benchmarks for evaluating LLM performance in academic contexts",
  "limitations ethical concerns and biases of LLMs in education"
]
```

### Slowest first exact-6 JSON

- Query id: `q001`
- Source query: `diffusion LLM`
- First exact-6: `1553.1 ms`
- Stable exact-6: `1697.5 ms`
- Full generation: `1749.5 ms`
- Stable equals final: `yes`

First exact-6 subqueries:

```json
[
  "diffusion models for language language survey and definitions",
  "discrete diffusion diffusion models for text generation architectures",
  "training objectives for diffusion-based language models",
  "inference acceleration and sampling algorithms for diffusion LLMs",
  "benchmarking and of diffusion diffusion vs language models",
  "limitations and challenges of diffusion vs autoregressivegressive models"
]
```

Final subqueries:

```json
[
  "diffusion models for language modeling survey and definitions",
  "discrete and continuous diffusion for text generation architecture",
  "training objectives for diffusion-based language models",
  "inference acceleration and sampling methods for diffusion LLMs",
  "benchmarking performance evaluation of diffusion large language models",
  "limitations and challenges of diffusion vs autoregressive transformers"
]
```

### Largest first-exact speedup over full generation

- Query id: `q055`
- Source query: `data poisoning attacks on foundation models`
- First exact-6: `241.5 ms`
- Stable exact-6: `537.0 ms`
- Full generation: `875.3 ms`
- Stable equals final: `no`

First exact-6 subqueries:

```json
[
  "survey of data poisoning attacks on large scale foundation models",
  "backdoordoor in in large language models and vision transformers",
  "fine-tuning data poisoning and and- learning techniques",
  "backback attacks and and attacks in generative generative models",
  "benchmarks and datasets for evaluating data poisoning robustness",
  "defense mechanisms and robustness against data poisoning in foundation models"
]
```

Final subqueries:

```json
[
  "survey of data poisoning attacks on large scale foundation models",
  "backdoor attacks on pretraining language models and vision transformers",
  "fine-tuning data poisoning vulnerabilities and adaptive learning attacks",
  "poisoning-based inference time attacks on generative AI",
  "benchmarks and datasets for evaluating data poisoning robustness",
  "defense mechanisms and mitigation of data poisoning in foundation models"
]
```

## Stable Exact-6 Differences From Final

Stable exact-6 differed from final in `9` cases. First examples:

### Difference case q038

- Query id: `q038`
- Source query: `AI methods for biodiversity monitoring`
- First exact-6: `241.9 ms`
- Stable exact-6: `436.0 ms`
- Full generation: `536.4 ms`
- Stable equals final: `no`

First exact-6 subqueries:

```json
[
  "review of artificial intelligence methods for biodiversity monitoring and conservation",
  "deep learning architectures for species identification in acoustic monitoring",
  "unsupervised learning and transfersupervised learning for ecological data data",
  "remote sensing and satellite imagery techniques for biodiversity assessment",
  "benchmarks and datasets for AI-based biodiversity assessment",
  "challenges and limitations of machine learning in ecological monitoring monitoring"
]
```

Final subqueries:

```json
[
  "review of artificial intelligence methods for biodiversity monitoring and conservation",
  "deep learning architectures for species identification and acoustic monitoring",
  "unsupervised and semi-supervised learning for ecological data analysis",
  "remote sensing and satellite imagery analysis for habitat assessment",
  "benchmarks and datasets for AI-based biodiversity monitoring",
  "challenges and limitations of machine learning in wildlife population tracking"
]
```

### Difference case q043

- Query id: `q043`
- Source query: `hybrid search combining BM25 and embeddings`
- First exact-6: `291.3 ms`
- Stable exact-6: `387.5 ms`
- Full generation: `532.7 ms`
- Stable equals final: `no`

First exact-6 subqueries:

```json
[
  "hybrid information retrieval survey BM25 and dense embeddings survey",
  "architectures for combining sparse and dense vector retrieval",
  "cross-encoder vs bi-encoder for hybrid search",
  "reciprocal rank fusion vs alpha fusion methods for hybrid search",
  "performance benchmarks of hybrid search retrieval systems",
  "limitations and challenges of lexical- in dense retrieval"
]
```

Final subqueries:

```json
[
  "hybrid information retrieval survey BM25 and dense embeddings",
  "architectural patterns for combining sparse and dense vector retrieval",
  "cross-encoder vs bi-encoder in hybrid search",
  "reciprocal rank fusion vs alpha fusion methods for hybrid search",
  "performance benchmarks of hybrid search retrieval systems",
  "limitations and challenges of lexical and semantic search fusion"
]
```

### Difference case q053

- Query id: `q053`
- Source query: `detecting synthetic media with deep learning`
- First exact-6: `291.8 ms`
- Stable exact-6: `581.0 ms`
- Full generation: `678.1 ms`
- Stable equals final: `no`

First exact-6 subqueries:

```json
[
  "survey of synthetic media detection using deep learning techniques",
  "deep learning architectures for deepfake detection in video",
  "unsupervised and self-supervised learning for synthetic image detection",
  "generalization and robustness robustness of synthetic media detectors",
  "benchmarks and datasets for synthetic media detection",
  "limitations and adversarial attacks on deep learning based detectors"
]
```

Final subqueries:

```json
[
  "survey of synthetic media detection using deep learning techniques",
  "deep learning architectures for deepfake detection in video",
  "unsupervised and self-supervised learning for synthetic image forgery",
  "generalization and robustness of synthetic media detectors models",
  "benchmarks and datasets for synthetic media evaluation",
  "limitations and adversarial attacks on deep learning based detectors"
]
```

### Difference case q055

- Query id: `q055`
- Source query: `data poisoning attacks on foundation models`
- First exact-6: `241.5 ms`
- Stable exact-6: `537.0 ms`
- Full generation: `875.3 ms`
- Stable equals final: `no`

First exact-6 subqueries:

```json
[
  "survey of data poisoning attacks on large scale foundation models",
  "backdoordoor in in large language models and vision transformers",
  "fine-tuning data poisoning and and- learning techniques",
  "backback attacks and and attacks in generative generative models",
  "benchmarks and datasets for evaluating data poisoning robustness",
  "defense mechanisms and robustness against data poisoning in foundation models"
]
```

Final subqueries:

```json
[
  "survey of data poisoning attacks on large scale foundation models",
  "backdoor attacks on pretraining language models and vision transformers",
  "fine-tuning data poisoning vulnerabilities and adaptive learning attacks",
  "poisoning-based inference time attacks on generative AI",
  "benchmarks and datasets for evaluating data poisoning robustness",
  "defense mechanisms and mitigation of data poisoning in foundation models"
]
```

### Difference case q060

- Query id: `q060`
- Source query: `formal verification of neural network robustness`
- First exact-6: `290.3 ms`
- Stable exact-6: `386.8 ms`
- Full generation: `484.1 ms`
- Stable equals final: `no`

First exact-6 subqueries:

```json
[
  "formal verification of neural network robustness survey definitions",
  "formal verification of deep neural networks and CNNs",
  "robustness training methods formal verification methods",
  "sampling-based and and sampling for neural networks",
  "benchmarks for neural network robustness verification",
  "limitations and scalability challenges of formal verification in AI"
]
```

Final subqueries:

```json
[
  "formal verification of neural network robustness survey definitions",
  "formal verification of deep neural networks and CNNs",
  "robustness training methods using formal verification",
  "sampling-based and probabilistic verification for neural networks",
  "benchmarks for neural network robustness verification",
  "limitations and scalability challenges of formal verification in AI"
]
```

## Per-Query Summary

| id | source query | drafts | first exact ms | stable ms | full ms | first=final | stable=final |
|---|---|---:|---:|---:|---:|---:|---:|
| `q001` | diffusion LLM | 8 | 1553.1 | 1697.5 | 1749.5 | no | yes |
| `q002` | retrieval augmented generation for medical question answering | 9 | 264.1 | 409.5 | 506.3 | no | yes |
| `q003` | scaling laws for multimodal transformers | 10 | 252.5 | 548.0 | 548.8 | no | yes |
| `q004` | benchmarking long context language models | 9 | 252.5 | 492.6 | 493.3 | no | yes |
| `q005` | efficient fine tuning methods for large language models | 12 | 253.2 | 640.9 | 641.8 | no | yes |
| `q006` | safety alignment in open source language models | 9 | 254.0 | 447.0 | 495.8 | no | yes |
| `q007` | tool use and planning in autonomous LLM agents | 9 | 290.5 | 482.9 | 483.7 | no | yes |
| `q008` | mixture of experts routing stability in transformers | 8 | 290.5 | 435.4 | 436.1 | no | yes |
| `q009` | speculative decoding for faster text generation | 9 | 338.7 | 435.2 | 485.9 | no | yes |
| `q010` | quantization effects on reasoning in large language models | 10 | 292.3 | 533.1 | 533.8 | no | yes |
| `q011` | graph neural networks for molecular property prediction | 7 | 243.1 | 392.6 | 393.5 | no | yes |
| `q012` | foundation models for protein structure and function | 10 | 241.6 | 531.1 | 531.9 | no | yes |
| `q013` | AI methods for drug repurposing | 8 | 193.4 | 433.8 | 434.5 | no | yes |
| `q014` | causal inference with deep learning in healthcare | 9 | 241.9 | 483.2 | 483.9 | no | yes |
| `q015` | privacy preserving federated learning for hospitals | 8 | 248.9 | 346.5 | 443.4 | no | yes |
| `q016` | medical image segmentation with vision transformers | 7 | 242.8 | 387.5 | 388.2 | no | yes |
| `q017` | uncertainty estimation in clinical AI systems | 9 | 243.0 | 484.4 | 485.2 | no | yes |
| `q018` | large language models for radiology report generation | 11 | 242.5 | 531.8 | 581.2 | no | yes |
| `q019` | self supervised learning for electronic health records | 7 | 241.7 | 386.6 | 387.3 | no | yes |
| `q020` | robustness of AI diagnosis under distribution shift | 9 | 290.9 | 488.3 | 489.0 | no | yes |
| `q021` | reinforcement learning from human feedback | 6 | 241.6 | 338.7 | 339.5 | no | yes |
| `q022` | offline reinforcement learning for robotics | 8 | 241.8 | 388.3 | 436.9 | no | yes |
| `q023` | world models for embodied AI | 9 | 292.2 | 486.2 | 486.9 | no | yes |
| `q024` | sim to real transfer for robot manipulation | 12 | 386.6 | 531.2 | 628.7 | no | yes |
| `q025` | multi agent reinforcement learning coordination | 7 | 194.2 | 386.4 | 387.2 | no | yes |
| `q026` | safe exploration in reinforcement learning | 8 | 299.1 | 452.2 | 452.9 | no | yes |
| `q027` | language guided robotic planning | 7 | 251.4 | 396.4 | 397.1 | no | yes |
| `q028` | vision language action models for robots | 10 | 290.6 | 532.6 | 533.3 | no | yes |
| `q029` | imitation learning from human demonstrations | 8 | 240.5 | 433.4 | 434.1 | no | yes |
| `q030` | benchmark datasets for embodied navigation | 12 | 242.4 | 629.7 | 630.5 | no | yes |
| `q031` | climate modeling with machine learning | 10 | 195.1 | 389.0 | 534.4 | no | yes |
| `q032` | AI for extreme weather forecasting | 11 | 243.6 | 581.9 | 582.6 | no | yes |
| `q033` | remote sensing foundation models | 7 | 193.2 | 385.7 | 386.5 | no | yes |
| `q034` | satellite image change detection | 8 | 247.5 | 440.3 | 441.0 | no | yes |
| `q035` | machine learning for carbon capture materials | 7 | 289.6 | 386.4 | 387.1 | no | yes |
| `q036` | energy efficient data center scheduling | 9 | 242.2 | 483.0 | 483.7 | no | yes |
| `q037` | deep learning for solar power forecasting | 8 | 241.8 | 435.7 | 436.4 | no | yes |
| `q038` | AI methods for biodiversity monitoring | 10 | 241.9 | 436.0 | 536.4 | no | no |
| `q039` | climate risk assessment with graph models | 9 | 241.2 | 482.8 | 484.6 | no | yes |
| `q040` | machine learning for wildfire prediction | 8 | 195.8 | 437.8 | 438.5 | no | yes |
| `q041` | neural information retrieval benchmarks | 9 | 291.1 | 483.7 | 484.5 | no | yes |
| `q042` | dense retrieval for scientific literature search | 7 | 240.9 | 385.3 | 386.0 | no | yes |
| `q043` | hybrid search combining BM25 and embeddings | 10 | 291.3 | 387.5 | 532.7 | no | no |
| `q044` | query rewriting for conversational search | 9 | 195.6 | 485.4 | 486.2 | no | yes |
| `q045` | learning to rank with large language models | 9 | 243.3 | 486.5 | 487.3 | no | yes |
| `q046` | evaluation metrics for retrieval augmented generation | 10 | 289.5 | 531.2 | 532.0 | no | yes |
| `q047` | citation recommendation in academic search | 10 | 289.9 | 530.7 | 531.4 | no | yes |
| `q048` | entity linking for knowledge graph construction | 9 | 241.7 | 482.9 | 483.6 | no | yes |
| `q049` | semantic search for code repositories | 9 | 241.6 | 434.5 | 483.3 | no | yes |
| `q050` | personalized search with privacy constraints | 7 | 242.5 | 344.7 | 393.8 | no | yes |
| `q051` | adversarial attacks on vision language models | 10 | 242.5 | 534.1 | 534.9 | no | yes |
| `q052` | watermarking generated text from language models | 9 | 291.3 | 484.2 | 484.9 | no | yes |
| `q053` | detecting synthetic media with deep learning | 13 | 291.8 | 581.0 | 678.1 | no | no |
| `q054` | red teaming large language models | 12 | 242.4 | 629.1 | 629.9 | no | yes |
| `q055` | data poisoning attacks on foundation models | 17 | 241.5 | 537.0 | 875.3 | no | no |
| `q056` | privacy leakage in language model memorization | 10 | 290.7 | 532.4 | 533.1 | no | yes |
| `q057` | jailbreak defenses for chat assistants | 9 | 241.6 | 482.5 | 483.2 | no | yes |
| `q058` | model extraction attacks on deployed APIs | 8 | 244.8 | 439.9 | 440.6 | no | yes |
| `q059` | secure inference for neural networks on edge devices | 10 | 293.2 | 437.6 | 534.6 | no | yes |
| `q060` | formal verification of neural network robustness | 9 | 290.3 | 386.8 | 484.1 | no | no |
| `q061` | diffusion models for image editing | 9 | 241.5 | 385.8 | 482.8 | no | no |
| `q062` | text to video generation evaluation | 8 | 241.7 | 434.2 | 435.3 | no | yes |
| `q063` | 3D generation with neural radiance fields and diffusion | 10 | 342.8 | 537.1 | 537.9 | no | yes |
| `q064` | controllable music generation with transformers | 7 | 243.0 | 387.9 | 388.7 | no | yes |
| `q065` | speech synthesis with discrete audio tokens | 10 | 338.0 | 532.9 | 533.7 | no | yes |
| `q066` | multimodal chain of thought reasoning | 9 | 245.4 | 346.9 | 492.2 | no | no |
| `q067` | visual question answering with large multimodal models | 7 | 244.6 | 389.9 | 390.6 | no | yes |
| `q068` | document understanding with layout aware transformers | 7 | 297.9 | 394.6 | 395.3 | no | yes |
| `q069` | OCR free visual document models | 11 | 290.3 | 484.4 | 581.7 | no | no |
| `q070` | evaluation of image captioning factuality | 8 | 242.1 | 386.8 | 435.7 | no | yes |
| `q071` | time series forecasting with foundation models | 9 | 290.0 | 492.7 | 493.5 | no | yes |
| `q072` | anomaly detection in industrial sensor data | 9 | 193.4 | 488.0 | 488.7 | no | yes |
| `q073` | graph transformers for traffic forecasting | 8 | 393.8 | 442.5 | 443.2 | yes | yes |
| `q074` | neural operators for partial differential equations | 9 | 243.3 | 485.3 | 486.0 | no | yes |
| `q075` | physics informed neural networks limitations | 9 | 243.5 | 485.3 | 486.0 | no | yes |
| `q076` | surrogate modeling for computational fluid dynamics | 7 | 245.0 | 391.3 | 392.1 | no | yes |
| `q077` | Bayesian optimization for materials discovery | 8 | 243.4 | 454.3 | 455.0 | no | yes |
| `q078` | active learning for expensive simulations | 9 | 195.9 | 486.8 | 487.6 | no | yes |
| `q079` | probabilistic forecasting for supply chains | 8 | 244.0 | 437.3 | 438.0 | no | yes |
| `q080` | foundation models for tabular data | 8 | 246.0 | 439.4 | 440.1 | no | yes |
| `q081` | AI assisted theorem proving | 9 | 247.6 | 493.6 | 494.4 | no | yes |
| `q082` | program synthesis with large language models | 8 | 242.2 | 436.4 | 437.2 | no | yes |
| `q083` | automated code repair with neural models | 8 | 195.0 | 436.6 | 437.4 | no | yes |
| `q084` | software engineering agents for bug fixing | 10 | 242.0 | 531.8 | 532.6 | no | yes |
| `q085` | unit test generation using language models | 10 | 193.3 | 531.5 | 532.3 | no | yes |
| `q086` | code retrieval augmented generation | 7 | 242.1 | 386.6 | 387.3 | no | yes |
| `q087` | benchmarking coding assistants on real repositories | 10 | 242.6 | 387.6 | 533.1 | no | no |
| `q088` | static analysis enhanced neural code models | 10 | 339.5 | 538.6 | 539.3 | no | yes |
| `q089` | security vulnerability detection in source code | 9 | 241.6 | 482.5 | 483.2 | no | yes |
| `q090` | natural language to SQL generation robustness | 8 | 246.4 | 390.8 | 439.9 | no | yes |
| `q091` | fairness in algorithmic hiring systems | 8 | 245.2 | 437.5 | 438.2 | no | yes |
| `q092` | explainability methods for deep neural networks | 9 | 242.4 | 484.1 | 484.9 | no | yes |
| `q093` | human AI collaboration in decision making | 10 | 241.4 | 532.2 | 532.9 | no | yes |
| `q094` | AI governance frameworks for foundation models | 9 | 289.1 | 434.2 | 482.8 | no | yes |
| `q095` | auditing bias in recommendation systems | 8 | 193.6 | 435.9 | 436.6 | no | yes |
| `q096` | economic impacts of generative AI adoption | 11 | 289.5 | 578.9 | 579.6 | no | yes |
| `q097` | education applications of large language models | 10 | 193.1 | 531.6 | 532.3 | no | yes |
| `q098` | mental health chatbots safety evaluation | 8 | 243.5 | 442.1 | 445.0 | no | yes |
| `q099` | human preference modeling for recommender systems | 10 | 247.1 | 489.3 | 538.3 | no | yes |
| `q100` | AI generated misinformation detection and mitigation | 9 | 242.1 | 484.3 | 485.1 | no | yes |

## Notes

- `draft_step_events.jsonl` contains sampled and valid draft events. It intentionally does not contain every raw non-candidate draft text to keep the artifact compact.
- `draft_step_results.jsonl` is the authoritative per-query artifact with final answers and first/stable draft extractions.
- `per_query_summary.csv` is a compact tabular export for spreadsheet analysis.
- The experiment did not use FastAPI or port-forwarding. It used raw HF and a custom streamer.
