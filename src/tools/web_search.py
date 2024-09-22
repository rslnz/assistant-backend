import logging
from typing import ClassVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.services.web_searcher import WebSearcher, WebSearchError, SearchQueryError, NetworkError, ParseError, RelevanceCheckError

logger = logging.getLogger(__name__)

class WebSearchArgs(BaseModel):
    query: str = Field(..., description="Search query in the user's language")
    num_results: int = Field(default=5, description="Number of results to return (1-10)")

class WebSearchTool(BaseTool):
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Performs a web search based on the given query and returns a list of relevant URLs. "
        "Use this tool to find up-to-date information on the internet. "
        "The search is performed in the same language as the user's input."
    )
    args_schema: ClassVar[type[WebSearchArgs]] = WebSearchArgs

    def _run(self, query: str, num_results: int = 5) -> str:
        raise NotImplementedError("WebSearchTool does not support synchronous execution")

    async def _arun(self, query: str, num_results: int = 5) -> str:
        logger.debug(f"WebSearchTool._arun called with arguments: query={query}, num_results={num_results}")
        searcher = WebSearcher()
        try:
            search_results = await searcher.search(query, num_results=min(max(num_results, 1), 10))
            if not search_results:
                return "No search results found. The query might be too specific. Consider rephrasing the query or using more general terms."
            return str({"search_results": search_results})
        except SearchQueryError as e:
            logger.error(f"Search query error: {str(e)}")
            return f"There was an issue with the search query: {str(e)}. Please try rephrasing your query."
        except NetworkError as e:
            logger.error(f"Network error during web search: {str(e)}")
            return f"A network error occurred during the web search: {str(e)}. Please check your internet connection and try again."
        except ParseError as e:
            logger.error(f"Parse error during web search: {str(e)}")
            return f"There was an error parsing the search results: {str(e)}. Please try again or use a different search query."
        except RelevanceCheckError as e:
            logger.error(f"Relevance check error: {str(e)}")
            return f"There was an error checking the relevance of search results: {str(e)}. The results might not be fully filtered for relevance."
        except WebSearchError as e:
            logger.error(f"General web search error: {str(e)}")
            return f"An error occurred during the web search: {str(e)}. Please try again or rephrase your query."
        except Exception as e:
            logger.error(f"Unexpected error in web search: {str(e)}")
            return f"An unexpected error occurred: {str(e)}. Please try again or contact support."