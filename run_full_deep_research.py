from tqdm import tqdm
import argparse
from src.agents.agent_template import agent_kwargs_from_env
from src.agents.agents_collection import (
    SummarizationAgent, DecomposeAgent, JudgeAgent, ComplexityAgent,
    QueryFormattingAgent, PlanningAgent, PlanCheckAgent,
)
from src.web_tools.search_engine import ArXiv
from src.web_tools.visit_site import visit_site

agent_kwargs = agent_kwargs_from_env()

# args
parser = argparse.ArgumentParser('Simple Deep Research')
parser.add_argument('--prompt', type=str)

# agents
sum_agent = SummarizationAgent(**agent_kwargs)
comp_agent = ComplexityAgent(**agent_kwargs)
judge_agent = JudgeAgent(**agent_kwargs)
decompose_agent = DecomposeAgent(**agent_kwargs)
query_agent = QueryFormattingAgent(**agent_kwargs)
planning_agent = PlanningAgent(**agent_kwargs)
plan_check_agent = PlanCheckAgent(**agent_kwargs)

# search engine
ddg = ArXiv()

args = parser.parse_args()
prompt = args.prompt


def build_result_text(search_results, paper_offset=0):
    """Fetch and format results with numbered paper metadata. Returns (result_text, references, next_offset)."""
    result_text = ''
    references = []
    paper_num = paper_offset
    for result in search_results:
        title = result['title']
        href = result['url']
        authors = result.get('authors', 'N/A')
        year = result.get('year', 'N/A')
        clean_text = visit_site(href)
        paper_num += 1
        result_text += f'\n\n[{paper_num}] {authors} ({year}). "{title}". {href}\n{clean_text}'
        references.append(f'[{paper_num}] {authors} ({year}). {title}. {href}')
    return result_text, references, paper_num


print("\n\n\n||QUERY FORMATTING AGENT|| Formatting prompt for search\n\n\n")
search_query = query_agent.generate(prompt)
print(f"Search query: {search_query}")

print("\n\n\n||PLANNING AGENT|| Creating literature review plan\n\n\n")
plan = planning_agent.generate(prompt)
print(f"Plan:\n{plan}\n")

print("\n\n\n||COMPLEXITY AGENT|| Analysing prompt\n\n\n")
is_complex = comp_agent.generate(prompt)


if '1' in is_complex:
    print("\n\n\n||DECOMPOSING AGENT|| Prompt is complex, decomposing\n\n\n")
    new_queris = decompose_agent.generate(prompt)
    all_result_text = ''
    all_references = []
    paper_offset = 0

    while True:
        print(len(new_queris.split('\n')))
        new_queris_sep = new_queris.split('\n')
        merged_result = ''

        for q in new_queris_sep:
            print(f"\n\n\nSearching for {q}\n\n\n")
            formatted_q = query_agent.generate(q)
            search_results = ddg.search(formatted_q, top_n=2)
            result_text, references, paper_offset = build_result_text(search_results, paper_offset)
            all_result_text += result_text
            all_references.extend(references)

            print(f"\n\n\n||SUMMARIZING AGENT||: Summarizing results for query: {q}\n\n\n")
            final_result = sum_agent.generate(q, result_text, references='\n'.join(references), plan=plan)
            merged_result += f'\n\nResult for "{q}":\n' + final_result

        print("\n\n\n||SUMMARIZING AGENT||: Synthesizing final literature review\n\n\n")
        final_sum = sum_agent.generate(prompt, merged_result, references='\n'.join(all_references), plan=plan)

        print("Judging")
        is_enough = judge_agent.generate(prompt, final_sum)
        if '0' in is_enough:
            print("\n\n\n||JUDGE AGENT||: The answer is complete\n\n\n")
            break

        if '0' not in is_enough:
            new_queris = is_enough.strip().splitlines()

    # Plan check loop
    while True:
        print("\n\n\n||PLAN CHECK AGENT|| Checking plan coverage\n\n\n")
        gap_queries = plan_check_agent.generate(plan, final_sum)

        if '0' in gap_queries:
            print("\n\n\n||PLAN CHECK AGENT|| Plan fully covered — writing report\n\n\n")
            with open("report_2.txt", "w", encoding="utf-8") as file:
                file.write(final_sum)
            break

        print(f"\n\n\n||PLAN CHECK AGENT|| Gaps found, searching for missing topics:\n{gap_queries}\n\n\n")
        for q in gap_queries.strip().splitlines():
            q = q.strip()
            if not q:
                continue
            formatted_q = query_agent.generate(q)
            search_results = ddg.search(formatted_q, top_n=2)
            extra_text, extra_refs, paper_offset = build_result_text(search_results, paper_offset)
            all_result_text += extra_text
            all_references.extend(extra_refs)

        print("\n\n\n||SUMMARIZING AGENT||: Re-writing literature review with new sources\n\n\n")
        final_sum = sum_agent.generate(prompt, all_result_text, references='\n'.join(all_references), plan=plan)


elif '0' in is_complex:
    print("Prompt is simple")
    search_results = ddg.search(search_query)
    result_text, references, paper_offset = build_result_text(search_results)

    print("\n\n\n||SUMMARIZATION AGENT|| Writing literature review\n\n\n")
    review = sum_agent.generate(prompt, result_text, references='\n'.join(references), plan=plan)

    # Plan check loop
    while True:
        print("\n\n\n||PLAN CHECK AGENT|| Checking plan coverage\n\n\n")
        gap_queries = plan_check_agent.generate(plan, review)

        if '0' in gap_queries:
            print("\n\n\n||PLAN CHECK AGENT|| Plan fully covered — writing report\n\n\n")
            with open("report.txt", "w", encoding="utf-8") as file:
                file.write(review)
            break

        print(f"\n\n\n||PLAN CHECK AGENT|| Gaps found, searching for missing topics:\n{gap_queries}\n\n\n")
        for q in gap_queries.strip().splitlines():
            q = q.strip()
            if not q:
                continue
            formatted_q = query_agent.generate(q)
            search_results = ddg.search(formatted_q, top_n=2)
            extra_text, extra_refs, paper_offset = build_result_text(search_results, paper_offset)
            result_text += extra_text
            references.extend(extra_refs)

        print("\n\n\n||SUMMARIZATION AGENT|| Re-writing literature review with new sources\n\n\n")
        review = sum_agent.generate(prompt, result_text, references='\n'.join(references), plan=plan)
