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


class IntentAgent(OpenAIAgent):
    """Decides whether a query should be answered via web search or academic databases."""

    def generate(self, prompt: str) -> str:
        """Returns 'web' or 'academic'."""
        result = self._complete(
            (
                "Analyze the following research query and decide whether it is best answered by:\n"
                "- \"web\": searching the Internet for recent news, current events, company/product info, "
                "non-academic practical topics, or very recent developments (post-2024)\n"
                "- \"academic\": searching academic paper databases (ArXiv, Semantic Scholar) for scientific "
                "research, peer-reviewed studies, technical papers, or established academic topics\n\n"
                "Return only one word: \"web\" or \"academic\".\n\n"
                f"Query: {prompt}"
            ),
            max_new_tokens=16,
        ).strip().lower()
        return "web" if "web" in result else "academic"


class QueryFormattingAgent(OpenAIAgent):

    def generate(self, prompt):
        return self._complete(
            (
                "Convert the following literature review topic into a short academic search query for ArXiv. "
                "Use 3-6 plain keywords or short phrases separated by spaces. Do not use boolean operators, "
                "parentheses, quotes, OR/AND, or wildcards. Return only the query string, nothing else.\n\n"
                f"Topic: {prompt}"
            ),
            max_new_tokens=80,
        ).strip()


class PlanningAgent(OpenAIAgent):

    def generate(self, prompt):
        return self._complete(
            f"""Create a detailed literature review plan for the following topic.

Topic: {prompt}

The plan must include:
1. A list of main sections (e.g., Introduction, Background, Thematic Areas, Methodology Comparison, Key Findings, Research Gaps, Conclusion)
2. For each section: 2-4 specific sub-questions or points that must be addressed
3. A list of key concepts, methods, or datasets that must be covered

Return the plan as a structured, numbered outline. Be specific and thorough; this plan will be used to verify the completeness of the final literature review.

Literature Review Plan:""",
            max_new_tokens=1024,
        ).strip()


class PlanCheckAgent(OpenAIAgent):

    def generate(self, plan, review):
        return self._complete(
            f"""You are a literature review auditor. Check whether the literature review fully satisfies every point in the plan.

Literature Review Plan:
{plan}

Written Literature Review:
{review}

Instructions:
- Go through each section and sub-point in the plan.
- Identify any points that are missing, superficial, or inadequately covered.
- If the review fully satisfies the plan, return only: 0
- If points are missing, return only the academic search queries needed to find papers that would cover the gaps, one query per line, nothing else.

Response:""",
            max_new_tokens=512,
        ).strip()


class RelevanceAgent(OpenAIAgent):

    def generate(self, prompt, search_result):
        return self._complete(
            (
                "Act as a relevance filter for an academic literature review. Compare the paper title with "
                f"the literature review topic. Topic: {prompt} Paper Title: {search_result}. "
                "If the paper is relevant to this literature review topic, return only 1, else 0"
            ),
            max_new_tokens=16,
        )


class ExtractionAgent(OpenAIAgent):

    def generate(self, prompt, results):
        return self._complete(
            (
                f"Extract information relevant to the literature review topic: {prompt}. "
                "Focus on: research objectives, methods, datasets, key findings, limitations, and contributions. "
                f"Text for extraction: {results}"
            ),
            max_new_tokens=1024,
        )


class SummarizationAgent(OpenAIAgent):

    def generate(self, prompt, result, references=None, plan=None):
        ref_block = f"\n\nReferences list for citation:\n{references}" if references else ""
        plan_block = f"\n\nLiterature Review Plan to follow:\n{plan}" if plan else ""
        return self._complete(
            f"""Write a comprehensive, in-depth academic literature review based on the provided papers and excerpts. The review must be long and thorough; aim for a graduate-level survey paper in scope and depth.

Literature Review Topic: {prompt}{plan_block}

Source Material (each paper is numbered [N]):
{result}{ref_block}

Instructions:
- Follow every section and sub-point in the plan above (if provided).
- Each section must be substantial: write multiple paragraphs per section, not bullet points.
- For each thematic area or sub-topic, dedicate a full paragraph (5-8 sentences minimum) that discusses the relevant papers in depth.
- Synthesize and compare findings across papers: explain how they agree, contradict, or build on each other.
- Cite papers inline using [N] notation (e.g., "Smith et al. [3] demonstrated that...").
- For every cited paper, describe: (a) the research question or objective, (b) the methodology or model used, (c) the key findings or contributions, and (d) limitations or open questions raised.
- Discuss historical context and progression of ideas where relevant.
- Highlight quantitative results, benchmark comparisons, and dataset details when available in the source material.
- Identify and discuss conflicting findings between papers with analytical depth.
- Dedicate a full section to research gaps and future directions, with specific actionable suggestions.
- Use formal, precise academic language throughout.
- Skip any sources that are empty or clearly irrelevant.
- End with a "References" section listing all cited papers in the format: [N] Authors (Year). Title. URL

Literature Review:""",
            max_new_tokens=4096,
        )


class ComplexityAgent(OpenAIAgent):
    def generate(self, prompt):
        return self._complete(
            (
                "Analyze the literature review topic. Decide whether it covers multiple sub-topics or themes "
                "that require separate searches, or whether a single search is sufficient. If multiple searches "
                f"are needed, return only 1; otherwise, return 0. Topic: {prompt}"
            ),
            max_new_tokens=16,
        )


class DecomposeAgent(OpenAIAgent):

    def generate(self, prompt):
        return self._complete(
            (
                "Analyze the literature review topic and break it down into 2 diverse academic search queries. "
                "Each query must target a different sub-theme or aspect of the topic to maximize coverage of "
                "the relevant literature. Example:\n"
                "Literature Review Topic: \"Retrieval-Augmented Generation for open-domain QA\"\n"
                "Search Queries:\n"
                "Retrieval-Augmented Generation dense retrieval methods open-domain question answering\n"
                "Knowledge grounding hallucination reduction large language models RAG\n\n"
                f"Topic: {prompt}. Return only the search queries separated by \"\\n\\n\""
            ),
            max_new_tokens=512,
        )


class JudgeAgent(OpenAIAgent):
    def generate(self, prompt, result):
        return self._complete(
            (
                "Evaluate whether the literature review adequately covers the topic. Check for missing sub-themes, "
                "underrepresented methodologies, or important gaps in coverage. If additional searches are needed, "
                "return only the new academic search queries separated by newlines; otherwise, return 0. "
                f"Topic: {prompt}. Current Literature Review: {result}"
            ),
            max_new_tokens=512,
        )
