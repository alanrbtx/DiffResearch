import os
from tqdm import tqdm
import argparse
from src.agents.agents_collection import SummarizationAgent, DecomposeAgent, JudgeAgent, ComplexityAgent
from src.web_tools.search_engine import DuckDuckGo
from src.web_tools.visit_site import visit_site

# vLLM / OpenAI
api_key = os.environ['API_KEY']
base_url = os.environ['BASE_URL']
model = os.environ['MODEL_NAME']

# args
parser = argparse.ArgumentParser('Simple Deep Research')

parser.add_argument('--prompt', type=str)



# agents
sum_agent = SummarizationAgent(
    api_key=api_key,
    base_url=base_url,
    model=model
)

comp_agent = ComplexityAgent(
    api_key=api_key,
    base_url=base_url,
    model=model
)

judge_agent = JudgeAgent(
    api_key=api_key,
    base_url=base_url,
    model=model
)

decompose_agent = DecomposeAgent(
    api_key=api_key,
    base_url=base_url,
    model=model
)


# search engine

ddg = DuckDuckGo(url='https://html.duckduckgo.com/html/')
    
args = parser.parse_args()

prompt = args.prompt


print("\n\n\n||COMPLEXITY AGENT|| Analysing prompt\n\n\n")
is_complex = comp_agent.generate(prompt)





if '1' in is_complex:
    print("\n\n\n||DECOMPOSING AGENT||Prompt is complex, decomposing\n\n\n")
    new_queris = decompose_agent.generate(prompt)

    while True:
        print(len(new_queris.split('\n')))
        new_queris_sep = new_queris.split('\n')
        merged_result = ''
        
        for idx, q in enumerate(new_queris_sep):
                result_text = ''
                print(f"\n\n\nSearching for {q}\n\n\n")
                search_results = ddg.search(q, top_n=2)

                for idx, result in enumerate(search_results):
                        title = result['title']
                        href = result['url']

                        clean_text = visit_site(href)
                        sum_clean_text = sum_agent.generate(q, clean_text)
                        result_text += f'\n\nSite {idx + 1}:\n\n{clean_text}'


                print(f"\n\n\n||SUMMARIZING AGENT||: I will summarize the information for this query: {q}\n\n\n")
                final_result = sum_agent.generate(q, result_text)
                merged_result += f'\n\nResult {idx}:' + final_result

        
        print("\n\n\n||SUMMARIZING AGENT||: I will summarize the final answer\n\n\n")
        final_sum = sum_agent.generate(prompt, merged_result)
        
        print("Judging")
        is_enough = judge_agent.generate(prompt, final_sum)
        if '0' in is_enough:
            print("\n\n\n||JUDGE AGENT||: The answer is complete\n\n\n")
            with open("report_2.txt", "w", encoding="utf-8") as file:
                file.write(final_sum)   
            
            break

        if '0' not in is_enough:
             new_queris = is_enough.split('\n')




elif '0' in is_complex:
    print("Prompt is simple")
    result_text = ''
    search_results = ddg.search(args.prompt)
    for idx, result in tqdm(enumerate(search_results)):
            title = result['title']
            href = result['url']

            clean_text = visit_site(href)

            result_text += f'\n\nSite {idx + 1}:\n\n{clean_text}'
            print('adding result')



    final_result = sum_agent.generate(args.prompt, result_text)
    with open("report.txt", "w", encoding="utf-8") as file:
        file.write(final_result)


