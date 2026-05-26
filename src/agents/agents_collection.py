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
            max_tokens=10,
            messages=[
                {'role': 'user', 'content': (
                    f'You are a relevance filter for a research pipeline.\n'
                    f'Research question: {prompt}\n'
                    f'Search result title/snippet: {search_result}\n\n'
                    f'Does this result likely contain useful information to answer the research question? '
                    f'Reply with only 1 (relevant) or 0 (not relevant).'
                )},
            ],
            **self._extra()
        )
        return completion.choices[0].message.content


class ExtractionAgent(OpenAIAgent):

    def generate(self, prompt, results):
        completion = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1000,
            messages=[
                {'role': 'user', 'content': (
                    f'You are an information extraction specialist.\n'
                    f'Research question: {prompt}\n\n'
                    f'Extract ALL facts, data points, statistics, expert opinions, and key claims '
                    f'that are relevant to the research question from the text below. '
                    f'Preserve specific numbers, dates, names, and citations.\n\n'
                    f'Text: {results}'
                )},
            ],
            **self._extra()
        )
        return completion.choices[0].message.content


class SummarizationAgent(OpenAIAgent):

    def generate(self, prompt, result, language: str = 'English'):
        completion = self.client.chat.completions.create(
            model=self.model,
            max_tokens=4000,
            messages=[
                {'role': 'user', 'content': (
                    f'You are a senior research analyst writing a comprehensive research report.\n\n'
                    f'IMPORTANT: Write the entire report in {language}.\n\n'
                    f'Research question: {prompt}\n\n'
                    f'Source material collected from web research:\n{result}\n\n'
                    f'Write a detailed, well-structured research report that fully answers the research question. '
                    f'The report must:\n'
                    f'- Start with a concise executive summary (2-3 sentences)\n'
                    f'- Cover background/context\n'
                    f'- Present key findings with specific facts, data, and evidence\n'
                    f'- Analyze different perspectives or subtopics\n'
                    f'- End with conclusions\n'
                    f'- Use clear section headers\n'
                    f'- Be at least 1000 words\n\n'
                    f'If the source material is empty or insufficient, write what is known from general knowledge '
                    f'and clearly note where additional sources would be needed.\n\n'
                    f'Research Report:'
                )},
            ],
            **self._extra()
        )
        return completion.choices[0].message.content


class ComplexityAgent(OpenAIAgent):

    def generate(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model,
            max_tokens=10,
            messages=[
                {'role': 'user', 'content': (
                    f'You are a research query analyzer.\n'
                    f'Determine whether the following research question requires multiple targeted search queries '
                    f'(because it has several distinct sub-topics, requires comparing different sources, or involves '
                    f'specialized knowledge areas) OR can be answered well with a single broad search.\n\n'
                    f'Research question: {prompt}\n\n'
                    f'If multiple queries are needed, reply with only: 1\n'
                    f'If a single query suffices, reply with only: 0\n\n'
                    f'Note: Prefer 1 for any question that is multifaceted, technical, comparative, or historical.'
                )},
            ],
            **self._extra()
        )
        return completion.choices[0].message.content


class DecomposeAgent(OpenAIAgent):

    def generate(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model,
            max_tokens=400,
            messages=[
                {'role': 'user', 'content': (
                    f'You are a research strategist. Decompose the following research question into '
                    f'4-5 specific, diverse search queries that together provide comprehensive coverage.\n\n'
                    f'Each query should target a DIFFERENT aspect:\n'
                    f'- Background and definitions\n'
                    f'- Key mechanisms, methods, or historical events\n'
                    f'- Quantitative data, statistics, or empirical results\n'
                    f'- Expert analysis, recent developments, or controversies\n'
                    f'- Practical implications, applications, or comparisons\n\n'
                    f'Research question: {prompt}\n\n'
                    f'Rules:\n'
                    f'- Each query must be self-contained and searchable\n'
                    f'- Queries must NOT overlap — each covers a unique sub-topic\n'
                    f'- Use precise terminology relevant to the domain\n'
                    f'- Output ONLY the raw query text, one per line\n'
                    f'- NO numbering, NO bullets, NO dashes, NO prefixes of any kind\n\n'
                    f'Search queries:'
                )},
            ],
            **self._extra()
        )
        return completion.choices[0].message.content


class JudgeAgent(OpenAIAgent):

    def generate(self, prompt, result):
        completion = self.client.chat.completions.create(
            model=self.model,
            max_tokens=200,
            messages=[
                {'role': 'user', 'content': (
                    f'You are a research quality judge evaluating whether a research report adequately answers '
                    f'a research question.\n\n'
                    f'Research question: {prompt}\n\n'
                    f'Current report:\n{result}\n\n'
                    f'Evaluate the report on these criteria:\n'
                    f'1. Does it answer all key aspects of the question?\n'
                    f'2. Does it include specific facts, data, or evidence (not just vague statements)?\n'
                    f'3. Is it comprehensive enough (covers multiple angles/perspectives)?\n'
                    f'4. Are there significant knowledge gaps that web search could fill?\n\n'
                    f'Reply format rules — follow exactly:\n'
                    f'- If the report is sufficient: reply with exactly the single word SUFFICIENT\n'
                    f'- If the report needs improvement: reply with INSUFFICIENT on the first line, '
                    f'then 2-3 specific follow-up search queries on separate lines (no numbering, no bullets)\n\n'
                    f'Your reply:'
                )},
            ],
            **self._extra()
        )
        return completion.choices[0].message.content
