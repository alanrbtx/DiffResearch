import os
import json
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm
from src.agents.agents_collection import (
    RelevanceAgent, ExtractionAgent, SummarizationAgent,
    QueryFormattingAgent, PlanningAgent, PlanCheckAgent, IntentAgent,
)
from src.web_tools.search_engine import ArXiv, SemanticScholar, Serper
from src.web_tools.visit_site import visit_site

api_key = os.environ['API_KEY']
base_url = os.environ['BASE_URL']
model = os.environ['MODEL_NAME']

parser = argparse.ArgumentParser('Run plan-based DiffResearch on DeepResearchBench')
parser.add_argument('--model-name', type=str, required=True, help='Output filename (without .jsonl)')
parser.add_argument('--resume', action='store_true', help='Skip already completed task IDs')
parser.add_argument('--squeeze', action='store_true', help='Use ExtractionAgent to compress paper content')
parser.add_argument('--relevance', action='store_true', help='Filter papers by relevance before fetching')
args = parser.parse_args()

BENCH_DIR = PROJECT_ROOT.parent / 'deep_research_bench'
QUERY_FILE = BENCH_DIR / 'data' / 'prompt_data' / 'query.jsonl'
OUTPUT_FILE = BENCH_DIR / 'data' / 'test_data' / 'raw_data' / f'{args.model_name}.jsonl'

rel_agent = RelevanceAgent(api_key=api_key, base_url=base_url, model=model)
ext_agent = ExtractionAgent(api_key=api_key, base_url=base_url, model=model)
sum_agent = SummarizationAgent(api_key=api_key, base_url=base_url, model=model)
query_agent = QueryFormattingAgent(api_key=api_key, base_url=base_url, model=model)
planning_agent = PlanningAgent(api_key=api_key, base_url=base_url, model=model)
plan_check_agent = PlanCheckAgent(api_key=api_key, base_url=base_url, model=model)
intent_agent = IntentAgent(api_key=api_key, base_url=base_url, model=model)
arxiv = ArXiv()
s2 = SemanticScholar()
serper = Serper()
serper_api_key = serper.api_key


def fetch_papers(search_query: str, prompt: str, paper_offset: int) -> tuple[str, list[str], int]:
    """Search ArXiv + Semantic Scholar, filter, and fetch paper content."""
#     arxiv_results = arxiv.search(search_query, top_n=5)
    s2_results = s2.search(search_query, top_n=5) if search_query.isascii() else []

    seen_titles: set[str] = set()
    search_results = []
    for result in  s2_results:
        key = result['title'].lower().strip()
        if key not in seen_titles:
            seen_titles.add(key)
            search_results.append(result)

    result_text = ''
    references = []
    paper_num = paper_offset

    for result in tqdm(search_results, desc='Fetching papers', leave=False):
        title = result['title']
        href = result['url']
        authors = result.get('authors', 'N/A')
        year = result.get('year', 'N/A')

        if args.relevance and '1' not in rel_agent.generate(prompt, title):
            continue

        if result.get('source') == 'S2':
            abstract = result.get('abstract', 'N/A')
            clean_text = f"Abstract:\n{abstract}" if abstract != 'N/A' else 'not available'
        else:
            clean_text = visit_site(href, serper_api_key=serper_api_key)

        if args.squeeze:
            clean_text = ext_agent.generate(prompt, clean_text)

        paper_num += 1
        result_text += f'\n\n[{paper_num}] {authors} ({year}). "{title}". {href}\n{clean_text}'
        references.append(f'[{paper_num}] {authors} ({year}). {title}. {href}')

    return result_text, references, paper_num


def fetch_web_results(search_query: str, prompt: str, paper_offset: int) -> tuple[str, list[str], int]:
    """Search the web via Serper and fetch page content."""
    web_results = serper.search(search_query, top_n=3)

    result_text = ''
    references = []
    item_num = paper_offset

    for result in tqdm(web_results, desc='Fetching web pages', leave=False):
        title = result['title']
        href = result['url']
        snippet = result.get('snippet', '')

        if args.relevance and '1' not in rel_agent.generate(prompt, title):
            continue

        page_text = visit_site(href, serper_api_key=serper_api_key)
        clean_text = page_text if page_text.strip() else snippet

        if args.squeeze:
            clean_text = ext_agent.generate(prompt, clean_text)

        item_num += 1
        result_text += f'\n\n[{item_num}] "{title}". {href}\n{clean_text}'
        references.append(f'[{item_num}] {title}. {href}')

    return result_text, references, item_num


def run_research(prompt: str) -> str:
    print('  [QueryFormattingAgent] Formatting search query...')
    search_query = query_agent.generate(prompt)
    print(f'  Search query: {search_query}')

    print('  [IntentAgent] Determining search strategy...')
    intent = intent_agent.generate(prompt)
    print(f'  Intent: {intent}')

    print('  [PlanningAgent] Creating literature review plan...')
    plan = planning_agent.generate(prompt)
     
    intent = 'web'
 
    if intent == 'web':
        print('  Searching the web via Serper...')
        result_text, references, paper_num = fetch_web_results(search_query, prompt, 0)
    else:
        print('  Searching academic databases (Semantic Scholar)...')
        result_text, references, paper_num = fetch_papers(search_query, prompt, 0)

    print('  [SummarizationAgent] Writing literature review...')
    review = sum_agent.generate(prompt, result_text, references='\n'.join(references), plan=plan)

#    print('  [PlanCheckAgent] Checking plan coverage...')
#    gap_queries = plan_check_agent.generate(plan, review)

#    if '0' not in gap_queries:
#        print(f'  Gaps found, fetching additional sources...')
#        for q in gap_queries.strip().splitlines():
#            q = q.strip()
#            if not q:
#                continue
#            formatted_q = query_agent.generate(q)
#            extra_text, extra_refs, paper_num = fetch_papers(formatted_q, prompt, paper_num)
#            result_text += extra_text
#            references.extend(extra_refs)

#        print('  [SummarizationAgent] Re-writing with new sources...')
#        review = sum_agent.generate(prompt, result_text, references='\n'.join(references), plan=plan)

    return review


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
