import asyncio
import json
import logging
import random
import re
import time
import urllib.parse
from typing import Any, Dict, List

import aiohttp
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI

from src.config import settings

logger = logging.getLogger(__name__)

class WebSearchError(Exception):
    """Base class for WebSearcher exceptions."""
    pass

class SearchQueryError(WebSearchError):
    """Raised when there's an issue with the search query."""
    pass

class NetworkError(WebSearchError):
    """Raised when there's a network-related issue."""
    pass

class ParseError(WebSearchError):
    """Raised when there's an issue parsing the search results or web pages."""
    pass

class RelevanceCheckError(WebSearchError):
    """Raised when there's an issue checking the relevance of search results."""
    pass

class WebSearcher:
    def __init__(self):
        self.search_url = "https://html.duckduckgo.com/html/"
        self.llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
        )
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
        ]
        self.html_fetcher = HTMLFetcher(self.user_agents)
        self.result_parser = SearchResultParser()
        self.relevance_checker = RelevanceChecker(self.llm)
        self.text_processor = TextProcessor()

    async def search(self, query: str, num_results: int = 10) -> List[str]:
        logger.debug(f"Performing web search for query: {query}")
        try:
            if not query.strip():
                raise SearchQueryError("Empty search query")

            encoded_query = urllib.parse.quote(query)
            url = f"{self.search_url}?q={encoded_query}"
            headers = {"User-Agent": random.choice(self.user_agents)}
            
            html = await self.html_fetcher.fetch(url, headers)
            search_results = self.result_parser.parse(html, num_results)
            
            if not search_results:
                logger.warning("No valid search results found")
                return []

            logger.debug(f"Final search results:\n{json.dumps(search_results, indent=2, ensure_ascii=False)}")
            time.sleep(random.uniform(1, 3))
            
            relevant_results = await self.filter_relevant_results(query, search_results)
            return [result['link'] for result in relevant_results]

        except aiohttp.ClientError as e:
            logger.error(f"Network error during web search: {str(e)}")
            raise NetworkError(f"Failed to perform web search due to network error: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response: {str(e)}")
            raise ParseError(f"Failed to parse search results: {str(e)}")
        except SearchQueryError as e:
            logger.error(f"Search query error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during web search: {str(e)}")
            raise WebSearchError(f"Unexpected error during web search: {str(e)}")

    async def filter_relevant_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            relevance_checks = await self.relevance_checker.check_relevance(query, results)
            return [result for result, is_relevant in zip(results, relevance_checks) if is_relevant]
        except Exception as e:
            logger.error(f"Error during relevance filtering: {str(e)}")
            raise RelevanceCheckError(f"Failed to filter relevant results: {str(e)}")

    async def parse_pages(self, urls: List[str], summarize: bool = False) -> List[Dict[str, Any]]:
        tasks = []
        for url in urls:
            task = asyncio.ensure_future(self.parse_page(url, summarize))
            tasks.append(task)
        try:
            parsed_results = await asyncio.gather(*tasks, return_exceptions=True)
            return [result if not isinstance(result, Exception) else {'url': urls[i], 'error': str(result)} 
                    for i, result in enumerate(parsed_results)]
        except Exception as e:
            logger.error(f"Error parsing pages: {str(e)}")
            raise ParseError(f"Failed to parse pages: {str(e)}")

    async def parse_page(self, url: str, summarize: bool = False) -> Dict[str, Any]:
        try:
            headers = {
                "User-Agent": random.choice(self.user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.google.com/",
                "DNT": "1",
                "Accept-Encoding": "gzip, deflate, br"
            }
            html = await self.html_fetcher.fetch(url, headers)
            text = self._extract_main_content(html)
            text = self.text_processor.clean_text(text)
            content = self.text_processor.summarize_text(text, max_length=5000 if summarize else 1000)
            return {'url': url, 'content': content}
        except aiohttp.ClientError as e:
            logger.error(f"Network error when parsing page {url}: {str(e)}")
            return {'url': url, 'error': "Network error occurred"}
        except Exception as e:
            logger.error(f"Error parsing page {url}: {str(e)}")
            return {'url': url, 'error': "Failed to parse page"}

    def _extract_main_content(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
            element.decompose()

        main_content = None
        content_candidates = [
            soup.find('main'),
            soup.find('article'),
            soup.find('div', class_=re.compile('content|main|article', re.I)),
            soup.find('div', id=re.compile('content|main|article', re.I)),
            soup.find('div', class_=re.compile('post|body', re.I)),
            soup.find('div', id=re.compile('post|body', re.I))
        ]
        main_content = next((content for content in content_candidates if content), None)

        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.body.get_text(separator='\n', strip=True) if soup.body else ''

        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return re.sub(r'\n\s*\n', '\n\n', text)

class SearchResultParser:
    @staticmethod
    def parse(html: str, num_results: int) -> List[Dict[str, str]]:
        soup = BeautifulSoup(html, 'html.parser')
        search_results = []

        results = soup.find_all('div', class_='result')
        logger.debug(f"Found {len(results)} raw results")

        for index, result in enumerate(results[:num_results], 1):
            try:
                title_elem = result.find('h2', class_='result__title')
                snippet_elem = result.find('a', class_='result__snippet')

                if title_elem is None or snippet_elem is None:
                    logger.warning(f"Skipping result {index} due to missing elements")
                    continue

                title = title_elem.text.strip()
                link = title_elem.find('a')['href'] if title_elem.find('a') else None
                snippet = snippet_elem.text.strip()

                if not link:
                    logger.warning(f"Skipping result {index} due to missing link")
                    continue

                parsed_link = urllib.parse.urlparse(link)
                query_params = urllib.parse.parse_qs(parsed_link.query)
                fixed_link = query_params.get('uddg', [link])[0]
                
                search_results.append({
                    'title': title,
                    'link': fixed_link,
                    'snippet': snippet
                })
                
                logger.debug(f"Result {index}:\n"
                             f"  Title: {title}\n"
                             f"  Link: {fixed_link}\n"
                             f"  Snippet: {snippet}")
            except Exception as e:
                logger.warning(f"Error processing result {index}: {str(e)}")
        
        logger.debug(f"Processed {len(search_results)} valid results")
        return search_results

class RelevanceChecker:
    def __init__(self, llm):
        self.llm = llm

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
            response_content = response.content if hasattr(response, 'content') else str(response)
            relevance_checks = json.loads(response_content)
            if not isinstance(relevance_checks, list) or len(relevance_checks) != len(results):
                raise ValueError("Invalid response format")
            return relevance_checks
        except json.JSONDecodeError:
            logger.error(f"Error decoding relevance check response: {response_content}")
            return [True] * len(results)
        except AttributeError:
            logger.error(f"Unexpected response format: {response}")
            return [True] * len(results)

class HTMLFetcher:
    def __init__(self, user_agents):
        self.user_agents = user_agents

    async def fetch(self, url: str, headers: Dict[str, str]) -> str:
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    logger.debug(f"Attempt {attempt + 1}: Sending request to {url}")
                    async with session.get(url, headers=headers, timeout=10) as response:
                        logger.debug(f"Attempt {attempt + 1}: Received response with status code {response.status}")
                        if response.status == 200:
                            html = await response.text()
                            logger.debug(f"Attempt {attempt + 1}: Successfully retrieved HTML content")
                            return html
                        else:
                            logger.warning(f"Attempt {attempt + 1} failed. Status code: {response.status}")
                            await asyncio.sleep(2 ** attempt)
            except aiohttp.ClientError as e:
                logger.warning(f"Network error on attempt {attempt + 1}: {str(e)}")
                if attempt == 2:
                    raise NetworkError(f"Failed to fetch search results after multiple attempts: {str(e)}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt == 2:
                    raise NetworkError("Request timed out after multiple attempts")
        
        logger.error("All attempts to fetch search results failed")
        raise NetworkError("Failed to fetch search results after multiple attempts")

class TextProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace('\x00', '').replace('\xa0', ' ').replace('\t', '\t').replace('\n', '\n')
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    @staticmethod
    def summarize_text(text: str, max_length: int = 5000) -> str:
        text = TextProcessor.clean_text(text)
        
        if len(text) <= max_length:
            return text
        
        sentences = text.split('.')
        summary = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > max_length:
                break
            summary.append(sentence)
            current_length += len(sentence) + 1
        
        return '. '.join(summary) + '.'
