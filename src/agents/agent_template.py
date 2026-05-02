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
import torch
from llada_inference import generate as llada_generate


class LLaDAAgent:
    def __init__(self, model, tokenizer, gen_length=128, steps=128, block_length=32, temperature=0.):
        self.model = model
        self.tokenizer = tokenizer
        self.gen_length = gen_length
        self.steps = steps
        self.block_length = block_length
        self.temperature = temperature

    def _call(self, user_content):
        messages = [{"role": "user", "content": user_content}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        encoded = self.tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded['input_ids'].to(self.model.device)
        attention_mask = encoded['attention_mask'].to(self.model.device)

        out = llada_generate(
            self.model, input_ids, attention_mask,
            steps=self.steps,
            gen_length=self.gen_length,
            block_length=self.block_length,
            temperature=self.temperature,
        )
        return self.tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

    def generate(self, prompt):
        return self._call(prompt)
    