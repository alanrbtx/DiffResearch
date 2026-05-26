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
import trafilatura
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote

# Hard cap on chars sent to LLM per page — keeps context manageable.
# ~10k chars ≈ 2500 tokens, enough for a dense article but not a full book.
_MAX_CHARS = 10_000

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

# Tags that contain navigation/boilerplate rather than article content
_BOILERPLATE_TAGS = ['script', 'style', 'nav', 'footer', 'header', 'aside',
                     'form', 'noscript', 'iframe']


def _resolve_url(url: str) -> str:
    """Unwrap DuckDuckGo redirect URLs and fix protocol-relative URLs."""
    if url.startswith('//'):
        url = 'https:' + url
    if 'duckduckgo.com/l/' in url:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if 'uddg' in params:
            url = unquote(params['uddg'][0])
    return url


def _extract_with_trafilatura(html: str) -> str | None:
    """Extract main article text using trafilatura."""
    return trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )


def _extract_with_bs4(html: str) -> str:
    """Fallback: strip boilerplate tags and return remaining text."""
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(_BOILERPLATE_TAGS):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)


def visit_site(url: str, fallback_snippet: str = '') -> str:
    """Fetch a URL and return clean article text, capped at _MAX_CHARS.

    Extraction order:
    1. trafilatura  (best quality — strips boilerplate, keeps prose)
    2. BeautifulSoup fallback  (if trafilatura returns nothing)
    3. search snippet  (if the page is blocked/unreachable)
    """
    url = _resolve_url(url)

    try:
        response = requests.get(url, timeout=12, headers=_HEADERS)
        response.raise_for_status()
        html = response.text

        text = _extract_with_trafilatura(html)
        if not text or len(text.strip()) < 100:
            text = _extract_with_bs4(html)

        text = text.strip()
        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS] + '\n[truncated]'

        return text if text else (fallback_snippet or 'not available')

    except Exception as e:
        print(f'Error visiting {url}: {e}')
        return fallback_snippet or 'not available'
