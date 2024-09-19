from typing import List, Dict, Any
import aiohttp
from bs4 import BeautifulSoup
import logging
import asyncio
from langchain_openai import ChatOpenAI
from src.config import settings
import json

logger = logging.getLogger(__name__)

class WebSearcher:
    def __init__(self):
        self.search_url = "https://www.google.com/search"
        self.llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
        )

    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        logger.debug(f"Performing web search for query: {query}")
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            params = {"q": query, "num": num_results}
            async with aiohttp.ClientSession() as session:
                async with session.get(self.search_url, headers=headers, params=params) as response:
                    html = await response.text()

            soup = BeautifulSoup(html, 'html.parser')
            search_results = []

            for result in soup.find_all('div', class_='g'):
                title = result.find('h3', class_='r')
                link = result.find('a')
                snippet = result.find('div', class_='s')

                if title and link and snippet:
                    search_results.append({
                        'title': title.text,
                        'link': link['href'],
                        'snippet': snippet.text
                    })
            
            if not search_results:
                return []

            logger.debug(f"Search results: {search_results}")
            return await self.filter_relevant_results(query, search_results[:num_results])

        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            return []

    async def filter_relevant_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        relevance_checks = await self.check_relevance(query, results)
        return [result for result, is_relevant in zip(results, relevance_checks) if is_relevant]

    async def check_relevance(self, query: str, results: List[Dict[str, Any]]) -> List[bool]:
        prompt = f"""
        Query: {query}

        Search Results:
        {json.dumps(results, indent=2)}

        For each search result, determine if it is relevant to the query.
        Respond with a JSON array of boolean values (true for relevant, false for not relevant).
        The array should have the same length as the number of search results.
        """
        response = await self.llm.ainvoke(prompt)
        try:
            relevance_checks = json.loads(response)
            if not isinstance(relevance_checks, list) or len(relevance_checks) != len(results):
                raise ValueError("Invalid response format")
            return relevance_checks
        except json.JSONDecodeError:
            logger.error(f"Error decoding relevance check response: {response}")
            return [True] * len(results)

    async def parse_page(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        try:
            async with session.get(url) as response:
                html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            text_content = main_content.get_text(separator='\n', strip=True) if main_content else ''
            links = [{'text': a.text, 'href': a['href']} for a in soup.find_all('a', href=True)]
            code_snippets = [code.text for code in soup.find_all('code')]
            
            return {
                'url': url,
                'text_content': text_content[:1000],
                'links': links[:10],
                'code_snippets': code_snippets[:5]
            }
        except Exception as e:
            logger.error(f"Error parsing page {url}: {str(e)}")
            return {'url': url, 'error': str(e)}

    async def parse_pages(self, urls: List[str]) -> List[Dict[str, Any]]:
        tasks = []
        async with aiohttp.ClientSession() as session:
            for url in urls:
                task = asyncio.ensure_future(self.parse_page(session, url))
                tasks.append(task)
            parsed_results = await asyncio.gather(*tasks)
        return parsed_results