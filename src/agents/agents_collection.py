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
from src.agents.agent_template import LLaDAAgent


class QueryFormattingAgent(LLaDAAgent):

    def generate(self, prompt):
        return self._call(
            f"Convert the following literature review topic into a short academic search query for ArXiv. "
            f"Use 3-6 plain keywords or short phrases separated by spaces. Do not use boolean operators, "
            f"parentheses, quotes, OR/AND, or wildcards. Return only the query string, nothing else.\n\n"
            f"Topic: {prompt}"
        )


class PlanningAgent(LLaDAAgent):

    def generate(self, prompt):
        return self._call(
            f"Create a detailed literature review plan for the following topic.\n\n"
            f"Topic: {prompt}\n\n"
            f"The plan must include:\n"
            f"1. A list of main sections (e.g., Introduction, Background, Thematic Areas, Methodology Comparison, Key Findings, Research Gaps, Conclusion)\n"
            f"2. For each section: 2-4 specific sub-questions or points that must be addressed\n"
            f"3. A list of key concepts, methods, or datasets that must be covered\n\n"
            f"Return the plan as a structured, numbered outline. Be specific and thorough — this plan will be used to verify the completeness of the final literature review.\n\n"
            f"Literature Review Plan:"
        )


class PlanCheckAgent(LLaDAAgent):

    def generate(self, plan, review):
        return self._call(
            f"You are a literature review auditor. Check whether the literature review fully satisfies every point in the plan.\n\n"
            f"Literature Review Plan:\n{plan}\n\n"
            f"Written Literature Review:\n{review}\n\n"
            f"Instructions:\n"
            f"- Go through each section and sub-point in the plan.\n"
            f"- Identify any points that are missing, superficial, or inadequately covered.\n"
            f"- If the review fully satisfies the plan, return only: 0\n"
            f"- If points are missing, return only the academic search queries needed to find papers that would cover the gaps — one query per line, nothing else.\n\n"
            f"Response:"
        )


class RelevanceAgent(LLaDAAgent):

    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, gen_length=32, steps=32, block_length=32, **kwargs)

    def generate(self, prompt, search_result):
        return self._call(
            f"Act as a relevance filter for an academic literature review. Compare the paper title with the literature review topic. "
            f"Topic: {prompt} Paper Title: {search_result}. "
            f"If the paper is relevant to this literature review topic, return only 1, else 0"
        )


class ExtractionAgent(LLaDAAgent):

    def generate(self, prompt, results):
        return self._call(
            f"Extract information relevant to the literature review topic: {prompt}. "
            f"Focus on: research objectives, methods, datasets, key findings, limitations, and contributions. "
            f"Text for extraction: {results}"
        )


class SummarizationAgent(LLaDAAgent):

    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, gen_length=512, steps=128, block_length=32, **kwargs)

    def generate(self, prompt, result, references=None, plan=None):
        ref_block = f"\n\nReferences list for citation:\n{references}" if references else ""
        plan_block = f"\n\nLiterature Review Plan to follow:\n{plan}" if plan else ""
        return self._call(
            f"Write a comprehensive, in-depth academic literature review based on the provided papers and excerpts. "
            f"The review must be long and thorough — aim for a graduate-level survey paper in scope and depth.\n\n"
            f"Literature Review Topic: {prompt}{plan_block}\n\n"
            f"Source Material (each paper is numbered [N]):\n{result}{ref_block}\n\n"
            f"Instructions:\n"
            f"- Follow every section and sub-point in the plan above (if provided).\n"
            f"- Each section must be substantial: write multiple paragraphs per section, not bullet points.\n"
            f"- For each thematic area or sub-topic, dedicate a full paragraph (5–8 sentences minimum) that discusses the relevant papers in depth.\n"
            f"- Synthesize and compare findings across papers — explain how they agree, contradict, or build on each other.\n"
            f"- Cite papers inline using [N] notation (e.g., \"Smith et al. [3] demonstrated that...\").\n"
            f"- For every cited paper, describe: (a) the research question or objective, (b) the methodology or model used, (c) the key findings or contributions, and (d) limitations or open questions raised.\n"
            f"- Discuss historical context and progression of ideas where relevant.\n"
            f"- Highlight quantitative results, benchmark comparisons, and dataset details when available in the source material.\n"
            f"- Identify and discuss conflicting findings between papers with analytical depth.\n"
            f"- Dedicate a full section to research gaps and future directions, with specific actionable suggestions.\n"
            f"- Use formal, precise academic language throughout.\n"
            f"- Skip any sources that are empty or clearly irrelevant.\n"
            f"- End with a \"References\" section listing all cited papers in the format: [N] Authors (Year). Title. URL\n\n"
            f"Literature Review:"
        )


class ComplexityAgent(LLaDAAgent):

    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, gen_length=32, steps=32, block_length=32, **kwargs)

    def generate(self, prompt):
        return self._call(
            f"Analyze the literature review topic. Decide whether it covers multiple sub-topics or themes that require "
            f"separate searches, or whether a single search is sufficient. If multiple searches are needed, return only 1; "
            f"otherwise, return 0. Topic: {prompt}"
        )


class DecomposeAgent(LLaDAAgent):

    def generate(self, prompt):
        return self._call(
            f"Analyze the literature review topic and break it down into 2 diverse academic search queries. "
            f"Each query must target a different sub-theme or aspect of the topic to maximize coverage of the relevant literature. "
            f"Example:\nLiterature Review Topic: \"Retrieval-Augmented Generation for open-domain QA\"\n"
            f"Search Queries:\nRetrieval-Augmented Generation dense retrieval methods open-domain question answering\n"
            f"Knowledge grounding hallucination reduction large language models RAG\n\n"
            f"Topic: {prompt}. Return only the search queries separated by \"\\n\\n\""
        )


class JudgeAgent(LLaDAAgent):

    def generate(self, prompt, result):
        return self._call(
            f"Evaluate whether the literature review adequately covers the topic. Check for missing sub-themes, "
            f"underrepresented methodologies, or important gaps in coverage. If additional searches are needed, "
            f"return only the new academic search queries separated by newlines; otherwise, return 0. "
            f"Topic: {prompt}. Current Literature Review: {result}"
        )
