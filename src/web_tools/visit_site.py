# #!/usr/bin/env python
# # coding=utf-8

# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote

def visit_site(url):
    if url.startswith('//'):
        url = 'https:' + url
    
    if 'duckduckgo.com/l/' in url:
        parsed_url = urlparse(url)
        params = parse_qs(parsed_url.query)
        if 'uddg' in params:
            url = unquote(params['uddg'][0]) 

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status() 
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        for script in soup(["script", "style"]):
            script.decompose()
            
        clean_text = soup.get_text(separator=' ', strip=True)
        return clean_text
        
    except Exception as e:
        print(f"Error visiting {url}: {e}")
        return 'not available'
