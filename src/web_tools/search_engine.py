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
import time
import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = {'atom': 'http://www.w3.org/2005/Atom'}


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


class ArXiv:
    """Search the ArXiv API and return paper metadata."""

    def __init__(self, max_results=10, sort_by='relevance', sort_order='descending', request_delay=3.0):
        self.max_results = max_results
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.request_delay = request_delay  # seconds between requests (ArXiv asks for ≥3s)
        self._last_request_time = 0.0

    def search(self, query, top_n=None):
        """
        Search ArXiv for papers matching *query*.

        Returns a list of dicts with keys:
            title, url, abstract, authors, year
        Compatible with the DuckDuckGo result format (title + url always present).
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)

        n = top_n if top_n is not None else self.max_results
        params = {
            'search_query': query,
            'start': 0,
            'max_results': n,
            'sortBy': self.sort_by,
            'sortOrder': self.sort_order,
        }

        for attempt in range(5):
            try:
                response = requests.get(ARXIV_API_URL, params=params, timeout=15)
                if response.status_code == 429:
                    wait = 2 ** attempt * 10
                    print(f"ArXiv rate limit hit, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                self._last_request_time = time.time()
                break
            except requests.RequestException as e:
                if attempt == 4:
                    print(f"ArXiv search error: {e}")
                    return []
                time.sleep(2 ** attempt * 5)
        else:
            print("ArXiv search error: max retries exceeded")
            return []

        try:
            root = ET.fromstring(response.text)
            results = []

            for entry in root.findall('atom:entry', ARXIV_NS):
                title_el = entry.find('atom:title', ARXIV_NS)
                summary_el = entry.find('atom:summary', ARXIV_NS)
                id_el = entry.find('atom:id', ARXIV_NS)
                published_el = entry.find('atom:published', ARXIV_NS)
                author_els = entry.findall('atom:author/atom:name', ARXIV_NS)

                title = title_el.text.strip().replace('\n', ' ') if title_el is not None else 'N/A'
                abstract = summary_el.text.strip().replace('\n', ' ') if summary_el is not None else 'N/A'
                url = id_el.text.strip() if id_el is not None else 'N/A'
                published = published_el.text.strip() if published_el is not None else ''
                year = published[:4] if published else 'N/A'
                authors = ', '.join(el.text for el in author_els) if author_els else 'N/A'

                results.append({
                    'title': title,
                    'url': url,
                    'abstract': abstract,
                    'authors': authors,
                    'year': year,
                })

            return results

        except Exception as e:
            print(f"ArXiv search error: {e}")
            return []

