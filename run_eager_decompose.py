"""
Eager decompose mode: every `check_every` denoising steps the partially-denoised
sequence is scanned for complete sub-queries (no remaining mask tokens in their span).
Each ready query is dispatched immediately to a thread pool for web search, running
in parallel with the ongoing denoising.  Model-based filtering (relevance / squeeze)
is deferred until after denoising to avoid GPU contention.
"""

import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModel

from llada_inference import add_gumbel_noise, get_num_transfer_tokens
from src.agents.agents_collection import (
    RelevanceAgent, ExtractionAgent, SummarizationAgent, PlanningAgent,
)
from src.web_tools.search_engine import ArXiv, SemanticScholar
from src.web_tools.visit_site import visit_site

MASK_ID = 126336
_MASK_PLACEHOLDER = '\x00'

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser('Eager Decompose Deep Research')
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--squeeze', action='store_true',
                    help='Use extraction agent to compress paper text')
parser.add_argument('--relevance', action='store_true',
                    help='Use relevance agent to filter papers')
parser.add_argument('--check_every', type=int, default=8,
                    help='Check for ready queries every N denoising steps')
parser.add_argument('--gen_length', type=int, default=128)
parser.add_argument('--steps', type=int, default=128)
parser.add_argument('--block_length', type=int, default=32)
parser.add_argument('--max_workers', type=int, default=4,
                    help='Max parallel search threads')


# ---------------------------------------------------------------------------
# Modified generation loop with mid-step callback
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_with_eager_dispatch(
    model, prompt, attention_mask=None,
    steps=128, gen_length=128, block_length=32,
    temperature=0., remasking='low_confidence',
    check_every=8, on_step=None,
):
    """
    LLaDA masked diffusion generation that calls on_step(x, prompt_len) every
    check_every inner denoising steps so callers can inspect partial results.
    """
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        MASK_ID, dtype=torch.long,
    ).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((prompt.shape[0], gen_length),
                       dtype=attention_mask.dtype, device=model.device),
        ], dim=-1)

    prompt_len = prompt.shape[1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    global_step = 0

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == MASK_ID)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == MASK_ID)
            logits = model(x, attention_mask=attention_mask).logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

            global_step += 1
            if on_step is not None and global_step % check_every == 0:
                on_step(x, prompt_len)

    if on_step is not None:
        on_step(x, prompt_len)

    return x


# ---------------------------------------------------------------------------
# Query detection
# ---------------------------------------------------------------------------
def _decode_gen_with_placeholder(gen_ids: list, tokenizer) -> str:
    """
    Decode the generation token IDs, replacing MASK_ID tokens with
    _MASK_PLACEHOLDER so callers can detect unfinished spans.
    Decodes token-by-token to avoid the tokenizer merging mask tokens.
    """
    parts = []
    run = []
    for tid in gen_ids:
        if tid == MASK_ID:
            if run:
                parts.append(tokenizer.decode(run, skip_special_tokens=True))
                run = []
            parts.append(_MASK_PLACEHOLDER)
        else:
            run.append(tid)
    if run:
        parts.append(tokenizer.decode(run, skip_special_tokens=True))
    return ''.join(parts)


class EagerDispatcher:
    """
    Scans the current denoised sequence for complete queries and dispatches
    web searches to a thread pool the moment they are ready.
    """

    def __init__(self, tokenizer, executor, fetch_fn):
        self.tokenizer = tokenizer
        self.executor = executor
        self.fetch_fn = fetch_fn
        self._dispatched: set[int] = set()
        self._futures: dict[int, object] = {}

    def __call__(self, x: torch.Tensor, prompt_len: int):
        gen_ids = x[0, prompt_len:].tolist()
        decoded = _decode_gen_with_placeholder(gen_ids, self.tokenizer)
        segments = decoded.split('\n\n')

        # Only consider segments that are followed by a '\n\n' separator
        # (i.e. all but the last, which may still be forming).
        for idx, seg in enumerate(segments[:-1]):
            query = seg.strip()
            if query and _MASK_PLACEHOLDER not in seg and idx not in self._dispatched:
                print(f"\n[EAGER DISPATCH] sub-query {idx + 1}: {query}\n")
                self._dispatched.add(idx)
                self._futures[idx] = self.executor.submit(self.fetch_fn, query)

    def collect(self) -> dict[int, list]:
        """Block until all dispatched searches complete; return idx -> paper_list."""
        return {idx: fut.result() for idx, fut in self._futures.items()}


# ---------------------------------------------------------------------------
# Web-only paper fetching (no model inference — safe to call from threads)
# ---------------------------------------------------------------------------
def fetch_web(query: str, arxiv_engine, s2_engine) -> list[dict]:
    """Search ArXiv + Semantic Scholar, deduplicate, and fetch raw paper content."""
    print(f"  [THREAD] fetching: {query}")
    arxiv_results = arxiv_engine.search(query)
    s2_results = s2_engine.search(query)

    seen: set[str] = set()
    papers = []
    for result in arxiv_results + s2_results:
        key = result['title'].lower().strip()
        if key in seen:
            continue
        seen.add(key)

        if result.get('source') == 'S2':
            abstract = result.get('abstract', 'N/A')
            result['_text'] = f"Abstract:\n{abstract}" if abstract != 'N/A' else 'not available'
        else:
            result['_text'] = visit_site(result['url'])

        papers.append(result)

    print(f"  [THREAD] done ({len(papers)} papers): {query}")
    return papers


# ---------------------------------------------------------------------------
# Model-based filtering + formatting (runs on main thread after denoising)
# ---------------------------------------------------------------------------
def process_papers(papers: list[dict], prompt: str, paper_offset: int,
                   rel_agent, ext_agent, use_relevance: bool, use_squeeze: bool):
    result_text = ''
    references = []
    paper_num = paper_offset

    for result in tqdm(papers, desc='processing'):
        title = result['title']
        href = result['url']
        authors = result.get('authors', 'N/A')
        year = result.get('year', 'N/A')
        text = result['_text']

        if use_relevance and '1' not in rel_agent.generate(prompt, title):
            continue
        if use_squeeze:
            text = ext_agent.generate(prompt, text)

        paper_num += 1
        result_text += f'\n\n[{paper_num}] {authors} ({year}). "{title}". {href}\n{text}'
        references.append(f'[{paper_num}] {authors} ({year}). {title}. {href}')

    return result_text, references, paper_num


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    print("Loading LLaDA model...")
    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-1.5', trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-1.5', trust_remote_code=True)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    assert tokenizer.pad_token_id != MASK_ID

    rel_agent = RelevanceAgent(model=model, tokenizer=tokenizer)
    ext_agent = ExtractionAgent(model=model, tokenizer=tokenizer)
    sum_agent = SummarizationAgent(model=model, tokenizer=tokenizer)
    planning_agent = PlanningAgent(model=model, tokenizer=tokenizer)

    arxiv_engine = ArXiv()
    s2_engine = SemanticScholar()

    print("\n\n\n||PLANNING AGENT|| Creating literature review plan\n\n\n")
    plan = planning_agent.generate(args.prompt)
    print(f"Plan:\n{plan}\n")

    # Build the decompose prompt exactly as DecomposeAgent does
    decompose_user = (
        f"Analyze the literature review topic and break it down into 2 diverse academic search queries. "
        f"Each query must target a different sub-theme or aspect of the topic to maximize coverage of the "
        f"relevant literature. "
        f"Example:\nLiterature Review Topic: \"Retrieval-Augmented Generation for open-domain QA\"\n"
        f"Search Queries:\nRetrieval-Augmented Generation dense retrieval methods open-domain question answering\n"
        f"Knowledge grounding hallucination reduction large language models RAG\n\n"
        f"Topic: {args.prompt}. Return only the search queries separated by \"\\n\\n\""
    )
    messages = [{"role": "user", "content": decompose_user}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded['input_ids'].to(args.device)
    attn_mask = encoded['attention_mask'].to(args.device)

    def make_fetch_fn(query):
        return fetch_web(query, arxiv_engine, s2_engine)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        dispatcher = EagerDispatcher(tokenizer, executor, make_fetch_fn)

        print("\n\n\n||DECOMPOSE GENERATION|| Denoising sub-queries with eager dispatch\n\n\n")
        generate_with_eager_dispatch(
            model, input_ids,
            attention_mask=attn_mask,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=0.,
            check_every=args.check_every,
            on_step=dispatcher,
        )

        print("\n\n\n||COLLECTING|| Waiting for all searches to complete\n\n\n")
        query_results = dispatcher.collect()

    if not query_results:
        print("No sub-queries were dispatched — check gen_length / check_every settings.")
        return

    # Model-based filtering + formatting (sequential, on main thread)
    all_result_text = ''
    all_references = []
    paper_num = 0
    for idx in sorted(query_results):
        papers = query_results[idx]
        batch_text, batch_refs, paper_num = process_papers(
            papers, args.prompt, paper_num,
            rel_agent, ext_agent, args.relevance, args.squeeze,
        )
        all_result_text += batch_text
        all_references.extend(batch_refs)

    print("\n\n\n||SUMMARIZATION AGENT|| Writing literature review\n\n\n")
    review = sum_agent.generate(
        args.prompt, all_result_text,
        references='\n'.join(all_references), plan=plan,
    )

    with open("report_eager.txt", "w", encoding="utf-8") as f:
        f.write(review)
    print("Report written to report_eager.txt")


if __name__ == '__main__':
    main()
