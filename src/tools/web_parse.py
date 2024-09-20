from langchain_core.tools import BaseTool
from typing import List, ClassVar, Dict, Any
from pydantic import BaseModel, Field
from src.services.web_searcher import WebSearcher, WebSearchError, NetworkError, ParseError
import logging

logger = logging.getLogger(__name__)

class WebParseArgs(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to parse")
    summarize: bool = Field(default=False, description="Whether to summarize the content")

class WebParseTool(BaseTool):
    name: ClassVar[str] = "web_parse"
    description: ClassVar[str] = "Parses and analyzes the content of web pages from a given list of URLs, extracting main text and optionally summarizing it."
    args_schema: ClassVar[type[WebParseArgs]] = WebParseArgs

    def _run(self, urls: List[str], summarize: bool = False) -> str:
        raise NotImplementedError("WebParseTool does not support synchronous execution")

    async def _arun(self, urls: List[str], summarize: bool = False) -> str:
        searcher = WebSearcher()
        try:
            parsed_results = await searcher.parse_pages(urls, summarize=summarize)
            if not parsed_results:
                return ("No content could be parsed from the provided URLs. "
                        "Please verify the URLs and try again, or consider using a different source of information.")
            return str({"parsed_results": parsed_results})
        except NetworkError as e:
            return f"Network error occurred during web parsing: {str(e)}. Please check your internet connection and try again."
        except ParseError as e:
            return f"Error parsing web pages: {str(e)}. The content might be in an unsupported format or the website might be blocking automated access."
        except WebSearchError as e:
            return f"An unexpected error occurred during web parsing: {str(e)}. Please try again or use a different approach."
        except Exception as e:
            logger.error(f"Unexpected error in web_parse: {str(e)}")
            return f"An unexpected error occurred: {str(e)}. Please try again or contact support."