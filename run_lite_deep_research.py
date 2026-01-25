import os
from tqdm import tqdm
import argparse
from src.agents.agents_collection import RelevanceAgent, ExtractionAgent, SummarizationAgent
from src.web_tools.search_engine import DuckDuckGo
from src.web_tools.visit_site import visit_site

# vLLM / OpenAI
api_key = os.environ['API_KEY']
base_url = os.environ['BASE_URL']
model = os.environ['MODEL_NAME']

# args
parser = argparse.ArgumentParser('Simple Deep Research')

parser.add_argument('--prompt', type=str)
parser.add_argument('--squeeze', action='store_true', help='If true then using agent from sites. It helps when context size is crucial')
parser.add_argument('--relevance', action='store_true', help='For complex prompts')


# agents

rel_agent = RelevanceAgent(
    api_key=api_key, 
    base_url=base_url, 
    model=model
)


ext_agent = ExtractionAgent(
    api_key=api_key,
    base_url=base_url,
    model=model
)


sum_agent = SummarizationAgent(
    api_key=api_key,
    base_url=base_url,
    model=model
)

# search engine

ddg = DuckDuckGo(url='https://html.duckduckgo.com/html/')


def main():
    args = parser.parse_args()

    search_results = ddg.search(args.prompt)

    result_text = ''
    for idx, result in tqdm(enumerate(search_results)):
        title = result['title']
        href = result['url']

        
        relevance = rel_agent.generate(args.prompt, title)
        if args.relevance:
            if '1' in relevance:
                clean_text = visit_site(href)

                if args.squeeze:
                    clean_text = ext_agent.generate(args.prompt, clean_text)

                result_text += f'\n\nSite {idx + 1}:\n\n{clean_text}'
                print('adding result')

        else:
            clean_text = visit_site(href)

            if args.squeeze:
                clean_text = ext_agent.generate(args.prompt, clean_text)

            result_text += f'\n\nSite {idx + 1}:\n\n{clean_text}'
            print('adding result')



    result = sum_agent.generate(args.prompt, result_text)
    
    with open("report_2.txt", "w", encoding="utf-8") as file:
        file.write(result)


if __name__ == '__main__':
    main()