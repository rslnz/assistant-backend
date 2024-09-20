from langchain_core.tools import BaseTool
from typing import ClassVar, Dict, Any
from pydantic import BaseModel, Field
from src.services.web_searcher import WebSearcher, WebSearchError, SearchQueryError, NetworkError, ParseError, RelevanceCheckError
import logging

logger = logging.getLogger(__name__)

class WebSearchArgs(BaseModel):
    query: str = Field(..., description="Search query")
    num_results: int = Field(default=10, description="Number of results to return")

class WebSearchTool(BaseTool):
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = "Performs a web search based on the given query and returns a list of search results."
    args_schema: ClassVar[type[WebSearchArgs]] = WebSearchArgs

    def _run(self, query: str, num_results: int = 10) -> str:
        raise NotImplementedError("WebSearchTool does not support synchronous execution")

    async def _arun(self, query: str, num_results: int = 10) -> str:
        logger.debug(f"WebSearchTool._arun called with arguments: query={query}, num_results={num_results}")

        searcher = WebSearcher()
        try:
            search_results = await searcher.search(query, num_results=num_results)
            if not search_results:
                return ("No search results found. The query might be too specific. "
                        "Consider rephrasing the query or using more general terms.")
            return str({"search_results": search_results})
        except SearchQueryError as e:
            return f"Error with the search query: {str(e)}. Please try rephrasing your query."
        except NetworkError as e:
            return f"Network error occurred: {str(e)}. Please try again later or check your internet connection."
        except ParseError as e:
            return f"Error parsing search results: {str(e)}. The search service might be experiencing issues."
        except RelevanceCheckError as e:
            return f"Error checking result relevance: {str(e)}. Returning all results without relevance filtering."
        except WebSearchError as e:
            return f"An unexpected error occurred during the web search: {str(e)}. Please try again or use a different approach."
        except Exception as e:
            logger.error(f"Unexpected error in web_search: {str(e)}")
            return f"An unexpected error occurred: {str(e)}. Please try again or contact support."