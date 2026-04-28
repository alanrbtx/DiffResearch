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
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse, parse_qs, unquote

import requests
import trafilatura
from bs4 import BeautifulSoup

SERPER_SCRAPE_URL = "https://scrape.serper.dev"

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_NS = {'atom': 'http://www.w3.org/2005/Atom'}

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

MAX_CHARS = 12_000


def _resolve_url(url: str) -> str:
    """Unwrap redirect wrappers (DuckDuckGo, etc.) and normalise the URL."""
    if url.startswith('//'):
        url = 'https:' + url

    if 'duckduckgo.com/l/' in url:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if 'uddg' in params:
            url = unquote(params['uddg'][0])

    return url


def _arxiv_id_from_url(url: str):
    """Return the bare arXiv ID from an arxiv.org URL, or None."""
    m = re.search(r'arxiv\.org/(?:abs|pdf|html)/([^\s/?#]+)', url)
    return m.group(1).replace('.pdf', '') if m else None


def _fetch_arxiv_metadata(arxiv_id: str) -> str:
    """Use the ArXiv API to get a clean title + abstract for a paper."""
    try:
        response = requests.get(
            ARXIV_API_URL,
            params={'id_list': arxiv_id},
            timeout=10,
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        entry = root.find('atom:entry', ARXIV_NS)
        if entry is None:
            return 'not available'

        title_el = entry.find('atom:title', ARXIV_NS)
        summary_el = entry.find('atom:summary', ARXIV_NS)
        authors_els = entry.findall('atom:author/atom:name', ARXIV_NS)
        published_el = entry.find('atom:published', ARXIV_NS)

        title = title_el.text.strip().replace('\n', ' ') if title_el is not None else ''
        abstract = summary_el.text.strip().replace('\n', ' ') if summary_el is not None else ''
        authors = ', '.join(el.text for el in authors_els) if authors_els else ''
        year = published_el.text[:4] if published_el is not None else ''

        return (
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Year: {year}\n\n"
            f"Abstract:\n{abstract}"
        )
    except Exception as e:
        print(f"ArXiv metadata fetch error for {arxiv_id}: {e}")
        return 'not available'


def _extract_with_trafilatura(html: str) -> str | None:
    """Extract main content using trafilatura; return None if extraction fails."""
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )
    return text or None


def _extract_with_bs4(html: str) -> str:
    """Fallback: strip scripts/styles and return all visible text."""
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)


def _fetch_with_serper_scrape(url: str, api_key: str) -> str | None:
    """Use the Serper Scraping API to extract page text. Returns None on failure."""
    try:
        response = requests.post(
            SERPER_SCRAPE_URL,
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"url": url},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("text") or data.get("content") or ""
        return text.strip() or None
    except Exception as e:
        print(f"Serper scrape error for {url}: {e}")
        return None


def visit_site(url: str, serper_api_key: str = "") -> str:
    """
    Fetch *url* and return its main textual content.

    Extraction priority:
    1. ArXiv URLs → ArXiv API (clean metadata, no scraping).
    2. Serper Scraping API (if SERPER_API_KEY env var or serper_api_key arg is set).
    3. Direct HTTP fetch → trafilatura → BeautifulSoup fallback.

    Output is capped at MAX_CHARS characters.
    """
    url = _resolve_url(url)

    arxiv_id = _arxiv_id_from_url(url)
    if arxiv_id:
        return _fetch_arxiv_metadata(arxiv_id)

    api_key = serper_api_key or os.environ.get("SERPER_API_KEY", "")
    if api_key:
        text = _fetch_with_serper_scrape(url, api_key)
        if text:
            return text[:MAX_CHARS]

    try:
        response = requests.get(url, timeout=15, headers=_HEADERS, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        if 'text' not in content_type and 'html' not in content_type:
            return 'not available'

        html = response.text
        text = _extract_with_trafilatura(html) or _extract_with_bs4(html)

        if not text or not text.strip():
            return 'not available'

        return text[:MAX_CHARS]

    except Exception as e:
        print(f"Error visiting {url}: {e}")
        return 'not available'
