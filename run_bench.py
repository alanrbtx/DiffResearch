import os
import re
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.agents.agents_collection import SummarizationAgent, DecomposeAgent, JudgeAgent, ComplexityAgent
from src.web_tools.search_engine import make_search_engine
from src.web_tools.visit_site import visit_site

api_key = os.environ['API_KEY']
base_url = os.environ['BASE_URL']
model = os.environ['MODEL_NAME']

parser = argparse.ArgumentParser('Run DiffResearch on DeepResearchBench')
parser.add_argument('--model-name', type=str, required=True,
                    help='Output filename stem (without .jsonl)')
parser.add_argument('--resume', action='store_true',
                    help='Skip already completed task IDs')
parser.add_argument('--always-complex', action='store_true',
                    help='Skip complexity check and always use the multi-query pipeline '
                         '(recommended for DeepResearchBench queries)')
parser.add_argument('--search-delay', type=float, default=2.0,
                    help='Seconds to wait between search API requests (default: 2.0)')
parser.add_argument('--top-n-simple', type=int, default=6,
                    help='Number of sites to visit for simple (single-query) mode (default: 6)')
parser.add_argument('--top-n-complex', type=int, default=4,
                    help='Number of sites to visit per sub-query in complex mode (default: 4)')
parser.add_argument('--max-judge-iters', type=int, default=2,
                    help='Max number of judge refinement iterations (default: 2)')
parser.add_argument('--visit-workers', type=int, default=4,
                    help='Parallel workers for visiting sites within a sub-query (default: 4)')
args = parser.parse_args()

BENCH_DIR = Path(__file__).parent.parent / 'deep_research_bench'
QUERY_FILE = BENCH_DIR / 'data' / 'prompt_data' / 'query.jsonl'
OUTPUT_FILE = BENCH_DIR / 'data' / 'test_data' / 'raw_data' / f'{args.model_name}.jsonl'

LANG_NAMES = {'zh': 'Chinese', 'en': 'English'}

# Matches leading list markers: "1.", "1)", "-", "*", "•", "–"
_LIST_PREFIX = re.compile(r'^\s*(?:\d+[.)]\s*|[-*•–]\s*)')

sum_agent = SummarizationAgent(api_key=api_key, base_url=base_url, model=model)
comp_agent = ComplexityAgent(api_key=api_key, base_url=base_url, model=model)
judge_agent = JudgeAgent(api_key=api_key, base_url=base_url, model=model)
decompose_agent = DecomposeAgent(api_key=api_key, base_url=base_url, model=model)
search_engine = make_search_engine()


def log(msg: str, indent: int = 0):
    print('  ' * indent + msg, flush=True)


def clean_query(q: str) -> str:
    """Strip numbered/bulleted list prefixes that LLMs sometimes add."""
    return _LIST_PREFIX.sub('', q).strip()


def parse_judge(verdict: str) -> tuple[bool, list[str]]:
    """Parse judge reply. Returns (is_sufficient, follow_up_queries)."""
    lines = [l.strip() for l in verdict.strip().splitlines() if l.strip()]
    if not lines:
        return True, []
    if lines[0].upper().startswith('SUFFICIENT'):
        return True, []
    # INSUFFICIENT — remaining lines are follow-up queries
    queries = [clean_query(l) for l in lines[1:] if l.upper() != 'INSUFFICIENT' and l]
    return False, queries


def visit_parallel(search_results: list[dict], indent: int) -> tuple[str, list[str]]:
    """Visit all sites in parallel; return (concatenated_text, visited_urls)."""
    def fetch(idx_res):
        idx, res = idx_res
        url_short = res['url'][:70]
        log(f'Visiting [{idx + 1}/{len(search_results)}]: {url_short}', indent + 1)
        text = visit_site(res['url'], fallback_snippet=res.get('snippet', ''))
        is_fallback = text == res.get('snippet', '') and text
        status = 'snippet fallback' if is_fallback else f'{len(text)} chars'
        log(f'  -> {status}', indent + 1)
        return idx, text, res['url']

    results_map: dict[int, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=args.visit_workers) as ex:
        futures = {ex.submit(fetch, (i, r)): i for i, r in enumerate(search_results)}
        for fut in as_completed(futures):
            idx, text, url = fut.result()
            results_map[idx] = (text, url)

    result_text = ''
    urls = []
    for idx in sorted(results_map):
        text, url = results_map[idx]
        result_text += f'\n\nSite {idx + 1}:\n\n{text}'
        urls.append(url)
    return result_text, urls


def scrape_queries(
    queries: list[str], top_n: int, language: str, indent: int = 1
) -> tuple[str, list[str]]:
    """Search and scrape for each query in parallel per site.

    Returns (merged_synthesized_text, all_visited_urls).
    """
    merged = ''
    all_urls: list[str] = []

    for q_idx, q in enumerate(queries, 1):
        log(f'[{q_idx}/{len(queries)}] Search: "{q[:80]}"', indent)
        time.sleep(args.search_delay)
        search_results = search_engine.search(q, top_n=top_n)
        log(f'  -> {len(search_results)} results', indent)

        if not search_results:
            merged += f'\n\n### Sub-topic: {q}\n\n[no results]'
            continue

        result_text, urls = visit_parallel(search_results, indent)
        all_urls.extend(urls)

        log(f'Summarizing sub-topic...', indent + 1)
        partial = sum_agent.generate(q, result_text, language=language)
        log(f'  -> {len(partial)} chars', indent + 1)
        merged += f'\n\n### Sub-topic: {q}\n\n{partial}'

    return merged, all_urls


def run_research(prompt: str, language: str) -> str:
    is_complex = '1' if args.always_complex else comp_agent.generate(prompt)
    all_urls: list[str] = []

    if '1' in is_complex:
        log('Decomposing query...')
        raw = decompose_agent.generate(prompt)
        sub_queries = [clean_query(q) for q in raw.splitlines() if q.strip()]
        log(f'-> {len(sub_queries)} sub-queries:')
        for i, q in enumerate(sub_queries, 1):
            log(f'  {i}. {q[:90]}')

        log('Scraping...')
        merged_result, urls = scrape_queries(sub_queries, args.top_n_complex, language)
        all_urls.extend(urls)

        log('Generating final report...')
        final_report = sum_agent.generate(prompt, merged_result, language=language)
        log(f'-> {len(final_report)} chars')

        for iteration in range(args.max_judge_iters):
            log(f'Judge evaluating (iter {iteration + 1}/{args.max_judge_iters})...')
            verdict = judge_agent.generate(prompt, final_report)
            sufficient, follow_up = parse_judge(verdict)

            if sufficient:
                log('-> Judge: SUFFICIENT')
                break

            log(f'-> Judge: INSUFFICIENT — {len(follow_up)} follow-up queries:')
            for i, q in enumerate(follow_up, 1):
                log(f'  {i}. {q[:90]}')

            if not follow_up:
                break

            extra_result, extra_urls = scrape_queries(follow_up, args.top_n_complex, language)
            all_urls.extend(extra_urls)
            combined = merged_result + '\n\n### Follow-up Research\n\n' + extra_result
            log('Regenerating report...')
            final_report = sum_agent.generate(prompt, combined, language=language)
            log(f'-> {len(final_report)} chars')
            merged_result = combined

    else:
        log('Simple mode: single search')
        time.sleep(args.search_delay)
        search_results = search_engine.search(prompt, top_n=args.top_n_simple)
        log(f'-> {len(search_results)} results')
        result_text, urls = visit_parallel(search_results, indent=0)
        all_urls.extend(urls)
        log('Generating report...')
        final_report = sum_agent.generate(prompt, result_text, language=language)

    # Append deduplicated sources for citation evaluation
    unique_urls = list(dict.fromkeys(u for u in all_urls if u))
    if unique_urls:
        sources_section = '\n\n## Sources\n' + '\n'.join(f'- {u}' for u in unique_urls)
        final_report += sources_section

    return final_report


def load_queries():
    with open(QUERY_FILE) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done_ids():
    if not OUTPUT_FILE.exists():
        return set()
    done = set()
    with open(OUTPUT_FILE) as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)['id'])
    return done


def main():
    queries = load_queries()
    done_ids = load_done_ids() if args.resume else set()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mode = 'a' if args.resume else 'w'

    print(f'Model: {model}')
    print(f'Pipeline: always_complex={args.always_complex}, '
          f'top_n_complex={args.top_n_complex}, top_n_simple={args.top_n_simple}, '
          f'max_judge_iters={args.max_judge_iters}, search_delay={args.search_delay}s, '
          f'visit_workers={args.visit_workers}')
    print(f'Output: {OUTPUT_FILE}\n')

    with open(OUTPUT_FILE, mode, encoding='utf-8') as out_f:
        for item in queries:
            task_id = item['id']
            prompt = item['prompt']
            language = LANG_NAMES.get(item.get('language', 'en'), 'English')

            if task_id in done_ids:
                print(f'[{task_id}/100] Skipping (already done)')
                continue

            print(f'\n[{task_id}/100] [{language}] {prompt[:100]}...')
            try:
                article = run_research(prompt, language=language)
            except Exception as e:
                print(f'  ERROR on task {task_id}: {e}')
                article = f'ERROR: {e}'

            record = {'id': task_id, 'prompt': prompt, 'article': article}
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            out_f.flush()
            print(f'  Done ({len(article)} chars)')

    print(f'\nOutput saved to: {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
