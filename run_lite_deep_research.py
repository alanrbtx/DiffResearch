import torch
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoModel

from src.agents.agents_collection import (
    RelevanceAgent, ExtractionAgent, SummarizationAgent,
    QueryFormattingAgent, PlanningAgent, PlanCheckAgent,
)
from src.web_tools.search_engine import ArXiv, SemanticScholar
from src.web_tools.visit_site import visit_site

# args
parser = argparse.ArgumentParser('Simple Deep Research')
parser.add_argument('--prompt', type=str)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--squeeze', action='store_true', help='If true then using agent from sites. It helps when context size is crucial')
parser.add_argument('--relevance', action='store_true', help='For complex prompts')


def load_llada(device):
    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-1.5', trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-1.5', trust_remote_code=True)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    assert tokenizer.pad_token_id != 126336
    return model, tokenizer


def fetch_papers(search_query, prompt, paper_offset, use_relevance, use_squeeze, rel_agent, ext_agent):
    """Search ArXiv + Semantic Scholar, filter, and fetch paper content. Returns (result_text, references, next_offset)."""
    print("  [ArXiv] searching...")
    arxiv_results = arxiv.search(search_query)
    print(f"  [ArXiv] {len(arxiv_results)} results")

    print("  [Semantic Scholar] searching...")
    s2_results = s2.search(search_query)
    print(f"  [Semantic Scholar] {len(s2_results)} results")

    # Deduplicate by title (case-insensitive)
    seen_titles = set()
    search_results = []
    for result in arxiv_results + s2_results:
        key = result['title'].lower().strip()
        if key not in seen_titles:
            seen_titles.add(key)
            search_results.append(result)

    print(f"  Combined unique results: {len(search_results)}\n")

    result_text = ''
    references = []
    paper_num = paper_offset
    for result in tqdm(search_results):
        title = result['title']
        href = result['url']
        authors = result.get('authors', 'N/A')
        year = result.get('year', 'N/A')

        if use_relevance and '1' not in rel_agent.generate(prompt, title):
            continue

        if result.get('source') == 'S2':
            abstract = result.get('abstract', 'N/A')
            clean_text = f"Abstract:\n{abstract}" if abstract != 'N/A' else 'not available'
        else:
            clean_text = visit_site(href)
        if use_squeeze:
            clean_text = ext_agent.generate(prompt, clean_text)

        paper_num += 1
        result_text += f'\n\n[{paper_num}] {authors} ({year}). "{title}". {href}\n{clean_text}'
        references.append(f'[{paper_num}] {authors} ({year}). {title}. {href}')
        print('adding result')

    return result_text, references, paper_num


def main():
    args = parser.parse_args()

    print("Loading LLaDA model...")
    model, tokenizer = load_llada(args.device)

    # agents — all share the same model and tokenizer
    rel_agent = RelevanceAgent(model=model, tokenizer=tokenizer)
    ext_agent = ExtractionAgent(model=model, tokenizer=tokenizer)
    sum_agent = SummarizationAgent(model=model, tokenizer=tokenizer)
    query_agent = QueryFormattingAgent(model=model, tokenizer=tokenizer)
    planning_agent = PlanningAgent(model=model, tokenizer=tokenizer)
    plan_check_agent = PlanCheckAgent(model=model, tokenizer=tokenizer)

    # search engines
    global arxiv, s2
    arxiv = ArXiv()
    s2 = SemanticScholar()

    print("\n\n\n||QUERY FORMATTING AGENT|| Formatting prompt for search\n\n\n")
    search_query = query_agent.generate(args.prompt)
    print(f"Search query: {search_query}")

    print("\n\n\n||PLANNING AGENT|| Creating literature review plan\n\n\n")
    plan = planning_agent.generate(args.prompt)
    print(f"Plan:\n{plan}\n")

    result_text, references, paper_num = fetch_papers(
        search_query, args.prompt, 0, args.relevance, args.squeeze, rel_agent, ext_agent
    )

    print("\n\n\n||SUMMARIZATION AGENT|| Writing literature review\n\n\n")
    review = sum_agent.generate(args.prompt, result_text, references='\n'.join(references), plan=plan)

    with open("report_2.txt", "w", encoding="utf-8") as file:
        file.write(review)


if __name__ == '__main__':
    main()
