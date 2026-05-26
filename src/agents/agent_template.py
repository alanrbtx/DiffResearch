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
import os
from openai import OpenAI

# Set DISABLE_THINKING=1 when using vLLM-served models that have thinking enabled
# by default (e.g. Qwen3). Leave unset for standard OpenAI-compatible APIs.
_DISABLE_THINKING = os.environ.get('DISABLE_THINKING', '0') == '1'


class OpenAIAgent:
    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def _extra(self):
        if _DISABLE_THINKING:
            return {'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}
        return {}

    def generate(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'user', 'content': prompt},
            ],
            **self._extra()
        )
        return completion.choices[0].message.content
