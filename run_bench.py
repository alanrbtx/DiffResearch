import os
import json
import argparse
from pathlib import Path
from src.agents.agents_collection import SummarizationAgent, DecomposeAgent, JudgeAgent, ComplexityAgent
from src.web_tools.search_engine import ArXiv, SemanticScholar
from src.web_tools.visit_site import visit_site

api_key = os.environ['API_KEY']
base_url = os.environ['BASE_URL']
model = os.environ['MODEL_NAME']

parser = argparse.ArgumentParser('Run simple_deep_research on DeepResearch Bench')
parser.add_argument('--model-name', type=str, required=True, help='Output filename (without .jsonl)')
parser.add_argument('--resume', action='store_true', help='Skip already completed task IDs')
args = parser.parse_args()

BENCH_DIR = Path(__file__).parent.parent / 'deep_research_bench'
QUERY_FILE = BENCH_DIR / 'data' / 'prompt_data' / 'query.jsonl'
OUTPUT_FILE = BENCH_DIR / 'data' / 'test_data' / 'raw_data' / f'{args.model_name}.jsonl'

sum_agent = SummarizationAgent(api_key=api_key, base_url=base_url, model=model)
comp_agent = ComplexityAgent(api_key=api_key, base_url=base_url, model=model)
judge_agent = JudgeAgent(api_key=api_key, base_url=base_url, model=model)
decompose_agent = DecomposeAgent(api_key=api_key, base_url=base_url, model=model)
arxiv = ArXiv()
s2 = SemanticScholar()


def _format_result(result: dict) -> str:
    abstract = result.get('abstract', '')
    if abstract and abstract != 'N/A':
        return (
            f"Title: {result['title']}\n"
            f"Authors: {result.get('authors', 'N/A')}\n"
            f"Year: {result.get('year', 'N/A')}\n\n"
            f"Abstract:\n{abstract}"
        )
    url = result['url']
    if 'semanticscholar.org' in url:
        return (
            f"Title: {result['title']}\n"
            f"Authors: {result.get('authors', 'N/A')}\n"
            f"Year: {result.get('year', 'N/A')}"
        )
    return visit_site(url)


def search_all(query: str, top_n: int = 3) -> list[dict]:
    """Search ArXiv and Semantic Scholar, deduplicated by title."""
    s2_results = s2.search(query, top_n=top_n) if query.isascii() else []
    results = arxiv.search(query, top_n=top_n) + s2_results
    seen, combined = set(), []
    for r in results:
        key = r['title'].lower().strip()
        if key not in seen:
            seen.add(key)
            combined.append(r)
    return combined


def run_research(prompt: str) -> str:
    is_complex = comp_agent.generate(prompt)

    if '1' in is_complex:
        queries = decompose_agent.generate(prompt)

        while True:
            sub_queries = [q for q in queries.split('\n') if q.strip()]
            merged_result = ''

            for q in sub_queries:
                search_results = search_all(q, top_n=2)
                result_text = ''.join(
                    f'\n\nPaper {idx + 1}:\n\n{_format_result(r)}'
                    for idx, r in enumerate(search_results)
                )
                partial = sum_agent.generate(q, result_text)
                merged_result += f'\n\nResult:\n{partial}'

            final_sum = sum_agent.generate(prompt, merged_result)
            is_enough = judge_agent.generate(prompt, final_sum)

            if '0' in is_enough:
                return final_sum

            queries = is_enough
    else:
        search_results = search_all(prompt, top_n=3)
        result_text = ''.join(
            f'\n\nPaper {idx + 1}:\n\n{_format_result(r)}'
            for idx, r in enumerate(search_results)
        )
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

    with open(OUTPUT_FILE, mode, encoding='utf-8') as out_f:
        for item in queries:
            task_id = item['id']
            prompt = item['prompt']

            if task_id in done_ids:
                print(f'[{task_id}/100] Skipping (already done)')
                continue

            print(f'[{task_id}/100] Processing: {prompt[:80]}...')
            try:
                article = run_research(prompt)
            except Exception as e:
                print(f'  ERROR on task {task_id}: {e}')
                article = f'ERROR: {e}'

            record = {'id': task_id, 'prompt': prompt, 'article': article}
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
            out_f.flush()
            print(f'  Done.')

    print(f'\nOutput saved to: {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
