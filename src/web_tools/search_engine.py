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
import time
import requests
from bs4 import BeautifulSoup


class SerperSearch:
    """Google search via Serper.dev API.

    Requires SERPER_API_KEY env var.
    Each result includes 'title', 'url', and 'snippet' fields.
    Snippets can be used as lightweight fallback when visit_site fails.
    """

    _ENDPOINT = 'https://google.serper.dev/search'

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ['SERPER_API_KEY']

    def search(self, query: str, top_n: int = 5) -> list[dict]:
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json',
        }
        payload = {'q': query, 'num': min(top_n, 10)}

        try:
            response = requests.post(
                self._ENDPOINT, json=payload, headers=headers, timeout=15
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get('organic', [])[:top_n]:
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                })
            return results

        except Exception as e:
            print(f'Serper search error: {e}')
            return []


class DuckDuckGo:
    """Fallback search engine using DuckDuckGo HTML scraping.

    Prefer SerperSearch for production runs — DDG is rate-limited aggressively.
    """

    _RETRY_DELAYS = [2, 5, 10]

    def __init__(self, url: str = 'https://html.duckduckgo.com/html/'):
        self.url = url
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        }

    def search(self, query: str, top_n: int = 5) -> list[dict]:
        payload = {'q': query}

        for attempt, delay in enumerate([0] + self._RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                response = requests.get(
                    self.url, data=payload, headers=self.headers, timeout=15
                )
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                for i, result in enumerate(soup.select('.result'), start=1):
                    if i > top_n:
                        break
                    title_tag = result.select_one('.result__a')
                    snippet_tag = result.select_one('.result__snippet')
                    if title_tag:
                        results.append({
                            'title': title_tag.get_text(strip=True),
                            'url': title_tag.get('href', ''),
                            'snippet': snippet_tag.get_text(strip=True) if snippet_tag else '',
                        })
                return results

            except Exception as e:
                print(f'DuckDuckGo attempt {attempt + 1} error: {e}')

        return []


def make_search_engine(top_n_default: int = 5):
    """Return SerperSearch if SERPER_API_KEY is set, otherwise DuckDuckGo."""
    if os.environ.get('SERPER_API_KEY'):
        print('Search engine: Serper (Google)')
        return SerperSearch()
    print('Search engine: DuckDuckGo (fallback — set SERPER_API_KEY for Google results)')
    return DuckDuckGo()
