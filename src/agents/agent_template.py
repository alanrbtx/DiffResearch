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
from openai import OpenAI

class OpenAIAgent:
    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )


    def generate(self, prompt):
        completion = self.client.chat.completions.create(
        model=self.model, 
        messages=[
            {"role": "user", "content": prompt},
        ],
        extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
            }
        }
        )

        return completion.choices[0].message.content
    