#!/usr/bin/env python
# coding=utf-8

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from src.agents.agent_template import OpenAIAgent

class RelevanceAgent(OpenAIAgent):

    def generate(self, prompt, search_result):        
        completion = self.client.chat.completions.create(
            model=self.model, 
            messages=[
                {"role": "user", "content": f"Act as a relevance filter. Compare the search result title with the user query. Query: {prompt} Title: {search_result}. If relevant, return only 1, else 0" },
            ],
            extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
                }
            }
            )

        return completion.choices[0].message.content


class ExtractionAgent(OpenAIAgent):

    def generate(self, prompt, results):
        completion = self.client.chat.completions.create(
            model=self.model, 
            messages=[
                {"role": "user", "content": f"Extract information relevant to tre query: {prompt}. Text for extraction: {results}"},
            ],
            extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
                }
            }
            )

        return completion.choices[0].message.content


class SummarizationAgent(OpenAIAgent):

    def generate(self, prompt, result):
        completion = self.client.chat.completions.create(
            model=self.model, 
            messages=[
                {"role": "user", "content": f"""Create a single detailed report based on multiple search snippets. \n\nUser Query: {prompt}. \n\nResults to process: {result}. If result is empty, just skip it.\n\nFinal Report:"""},
            ],
            extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
                }
            }
            )

        return completion.choices[0].message.content


class ComplexityAgent(OpenAIAgent):
    def generate(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model, 
            messages=[
                {"role": "user", "content": f'Analyze the prompt. Decide whether the question requires multiple queries or just one. If multiple, return only 1; otherwise, return 0. Prompt: {prompt}'},
            ],
            extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
                }
            }
            )

        return completion.choices[0].message.content

class DecomposeAgent(OpenAIAgent):

    def generate(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model, 
            messages=[
                {"role": "user", "content": f'Analyze the user prompt and break it down into 2 diverse search queries. Each simple query must target a different sub-part of the original prompt to ensure maximum information coverage. Example:\nUser Prompt: "How to build a local RAG system with Llama 3?"\nSearch Queries:\nLlama 3 hardware requirements and quantization for local inference\nBest vector databases and embedding models for RAG in 2026\nStep-by-step tutorial for LangChain and Llama 3 local RAG setup. \n\nPrompt: {prompt}. Return only queries separated by "\n\n"'},
            ],
            extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
                }
            }
            )

        return completion.choices[0].message.content
    

class JudgeAgent(OpenAIAgent):
    def generate(self, prompt, result):
        completion = self.client.chat.completions.create(
            model=self.model, 
            messages=[
                {"role": "user", "content": f'Analyze the result and decide whether additional queries need to be made. If so, return only the new queries, separated by newlines; otherwise, return 0. Prompt: {prompt}. Answer: {result}'},
            ],
            extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
                }
            }
            )

        return completion.choices[0].message.content