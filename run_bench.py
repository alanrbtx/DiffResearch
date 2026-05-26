import os
import json
import time
import argparse
from pathlib import Path
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
                    help='Seconds to wait between DuckDuckGo requests (default: 2.0)')
parser.add_argument('--top-n-simple', type=int, default=6,
                    help='Number of sites to visit for simple (single-query) mode (default: 6)')
parser.add_argument('--top-n-complex', type=int, default=4,
                    help='Number of sites to visit per sub-query in complex mode (default: 4)')
parser.add_argument('--max-judge-iters', type=int, default=2,
                    help='Max number of judge refinement iterations (default: 2)')
args = parser.parse_args()

BENCH_DIR = Path(__file__).parent.parent / 'deep_research_bench'
QUERY_FILE = BENCH_DIR / 'data' / 'prompt_data' / 'query.jsonl'
OUTPUT_FILE = BENCH_DIR / 'data' / 'test_data' / 'raw_data' / f'{args.model_name}.jsonl'

sum_agent = SummarizationAgent(api_key=api_key, base_url=base_url, model=model)
comp_agent = ComplexityAgent(api_key=api_key, base_url=base_url, model=model)
judge_agent = JudgeAgent(api_key=api_key, base_url=base_url, model=model)
decompose_agent = DecomposeAgent(api_key=api_key, base_url=base_url, model=model)
search_engine = make_search_engine()


def log(msg: str, indent: int = 0):
    prefix = '  ' * indent
    print(f'{prefix}{msg}', flush=True)


def scrape_queries(queries: list[str], top_n: int, indent: int = 1) -> str:
    """Search and scrape websites for a list of queries; return concatenated synthesized text."""
    merged = ''
    for q_idx, q in enumerate(queries, 1):
        result_text = ''
        log(f'[{q_idx}/{len(queries)}] Search: "{q[:80]}"', indent)
        time.sleep(args.search_delay)
        search_results = search_engine.search(q, top_n=top_n)
        log(f'  -> {len(search_results)} results found', indent)

        for idx, res in enumerate(search_results):
            url_short = res['url'][:70]
            log(f'  Visiting [{idx + 1}/{len(search_results)}]: {url_short}', indent)
            clean_text = visit_site(res['url'], fallback_snippet=res.get('snippet', ''))
            chars = len(clean_text)
            status = 'snippet fallback' if clean_text == res.get('snippet', '') else f'{chars} chars'
            log(f'    -> {status}', indent)
            result_text += f'\n\nSite {idx + 1}:\n\n{clean_text}'

        log(f'  Summarizing sub-topic...', indent)
        partial = sum_agent.generate(q, result_text)
        log(f'    -> {len(partial)} chars', indent)
        merged += f'\n\n### Sub-topic: {q}\n\n{partial}'
    return merged


def run_research(prompt: str) -> str:
    is_complex = '1' if args.always_complex else comp_agent.generate(prompt)

    if '1' in is_complex:
        log('Decomposing query into sub-queries...')
        raw_queries = decompose_agent.generate(prompt)
        sub_queries = [q.strip() for q in raw_queries.split('\n') if q.strip()]
        log(f'-> {len(sub_queries)} sub-queries:')
        for i, q in enumerate(sub_queries, 1):
            log(f'  {i}. {q[:90]}')

        log('Scraping sub-queries...')
        merged_result = scrape_queries(sub_queries, args.top_n_complex)

        log('Generating final report...')
        final_report = sum_agent.generate(prompt, merged_result)
        log(f'-> Report: {len(final_report)} chars')

        for iteration in range(args.max_judge_iters):
            log(f'Judge evaluating report (iter {iteration + 1}/{args.max_judge_iters})...')
            verdict = judge_agent.generate(prompt, final_report)
            if '0' in verdict:
                log('-> Judge: report is sufficient')
                break
            follow_up = [q.strip() for q in verdict.split('\n') if q.strip()]
            if not follow_up:
                break
            log(f'-> Judge: needs {len(follow_up)} follow-up queries:')
            for i, q in enumerate(follow_up, 1):
                log(f'  {i}. {q[:90]}')
            extra_result = scrape_queries(follow_up, args.top_n_complex)
            combined = merged_result + '\n\n### Follow-up Research\n\n' + extra_result
            log('Regenerating report with follow-up data...')
            final_report = sum_agent.generate(prompt, combined)
            log(f'-> Report: {len(final_report)} chars')
            merged_result = combined

        return final_report
    else:
        log('Simple mode: single search query')
        time.sleep(args.search_delay)
        search_results = search_engine.search(prompt, top_n=args.top_n_simple)
        log(f'-> {len(search_results)} results found')
        result_text = ''
        for idx, res in enumerate(search_results):
            url_short = res['url'][:70]
            log(f'  Visiting [{idx + 1}/{len(search_results)}]: {url_short}')
            clean_text = visit_site(res['url'], fallback_snippet=res.get('snippet', ''))
            log(f'    -> {len(clean_text)} chars')
            result_text += f'\n\nSite {idx + 1}:\n\n{clean_text}'
        log('Generating report...')
        return sum_agent.generate(prompt, result_text)


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

    print(f'Pipeline: always_complex={args.always_complex}, '
          f'top_n_complex={args.top_n_complex}, top_n_simple={args.top_n_simple}, '
          f'max_judge_iters={args.max_judge_iters}, search_delay={args.search_delay}s')
    print(f'Output: {OUTPUT_FILE}\n')

    with open(OUTPUT_FILE, mode, encoding='utf-8') as out_f:
        for item in queries:
            task_id = item['id']
            prompt = item['prompt']

            if task_id in done_ids:
                print(f'[{task_id}/100] Skipping (already done)')
                continue

            print(f'[{task_id}/100] {prompt[:100]}...')
            try:
                article = run_research(prompt)
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
