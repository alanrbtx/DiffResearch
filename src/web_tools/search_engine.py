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
import requests
from bs4 import BeautifulSoup

class DuckDuckGo:
    def __init__(self, url):
        self.url = url


        self.headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }


    def search(self, query, top_n=5):
        payload = {'q': query}
        
        try:
            
            response = requests.get(self.url, data=payload, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []


            for i, result in enumerate(soup.select('.result'), start=1):
                if i > top_n:
                    break
                    
                title_tag = result.select_one('.result__a')
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    link = title_tag.get('href')
                    results.append({'title': title, 'url': link})

            return results

        except Exception as e:
            print(f"An error occurred: {e}")
            return []
        
