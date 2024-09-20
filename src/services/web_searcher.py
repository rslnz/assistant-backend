from typing import List, Dict, Any
import aiohttp
import re
from bs4 import BeautifulSoup
import logging
import asyncio
from langchain_openai import ChatOpenAI
import json
import urllib.parse
import random
import time

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

    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        logger.debug(f"Performing web search for query: {query}")
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"{self.search_url}?q={encoded_query}"
            headers = {"User-Agent": random.choice(self.user_agents)}
            
            html = await self._fetch_html(url, headers)
            
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
            
            if not search_results:
                logger.warning("No valid search results found")
                return []

            logger.debug(f"Final search results:\n{json.dumps(search_results, indent=2, ensure_ascii=False)}")
            time.sleep(random.uniform(1, 3))
            return await self.filter_relevant_results(query, search_results)

        except aiohttp.ClientError as e:
            logger.error(f"Network error during web search: {str(e)}")
            raise NetworkError(f"Failed to perform web search due to network error: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response: {str(e)}")
            raise ParseError(f"Failed to parse search results: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during web search: {str(e)}")
            raise WebSearchError(f"Unexpected error during web search: {str(e)}")

    async def _fetch_html(self, url: str, headers: Dict[str, str]) -> str:
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

    async def filter_relevant_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            relevance_checks = await self.check_relevance(query, results)
            return [result for result, is_relevant in zip(results, relevance_checks) if is_relevant]
        except Exception as e:
            logger.error(f"Error during relevance filtering: {str(e)}")
            raise RelevanceCheckError(f"Failed to filter relevant results: {str(e)}")

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
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                    else:
                        return {'url': url, 'error': f"Error loading page. Status code: {response.status}"}

            soup = BeautifulSoup(html, 'html.parser')

            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()

            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main|article'))
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.body.get_text(separator='\n', strip=True) if soup.body else ''

            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            text = re.sub(r'\n\s*\n', '\n\n', text)

            if summarize:
                summary = text[:5000] + "..." if len(text) > 5000 else text
                return {'url': url, 'content': summary}
            else:
                return {'url': url, 'content': text}

        except aiohttp.ClientError as e:
            logger.error(f"Network error parsing page {url}: {str(e)}")
            raise NetworkError(f"Failed to fetch page content due to network error: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing page {url}: {str(e)}")
            raise ParseError(f"Failed to parse page content: {str(e)}")
