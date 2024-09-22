import logging
from typing import ClassVar, List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.services.web_searcher import WebSearcher, ParseError

logger = logging.getLogger(__name__)

class WebParseArgs(BaseModel):
    urls: List[str] = Field(..., description="List of URLs to parse")
    summarize: bool = Field(default=True, description="Whether to summarize the content")

class WebParseTool(BaseTool):
    name: ClassVar[str] = "web_parse"
    description: ClassVar[str] = (
        "Parses the content of given web pages and optionally summarizes it. "
        "Use this tool to extract information from specific web pages. "
        "Provide a list of URLs to parse."
    )
    args_schema: ClassVar[type[WebParseArgs]] = WebParseArgs

    def _run(self, urls: List[str], summarize: bool = True) -> str:
        raise NotImplementedError("WebParseTool does not support synchronous execution")

    async def _arun(self, urls: List[str], summarize: bool = True) -> str:
        logger.debug(f"WebParseTool._arun called with arguments: urls={urls}, summarize={summarize}")
        searcher = WebSearcher()
        try:
            parsed_results = await searcher.parse_pages(urls, summarize=summarize)
            return str({"parsed_results": parsed_results})
        except ParseError as e:
            logger.error(f"Error parsing web pages: {str(e)}")
            return f"An error occurred while parsing web pages: {str(e)}. Please check the provided URLs and try again."