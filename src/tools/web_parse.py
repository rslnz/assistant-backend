import logging
from typing import List

from langchain_core.tools import tool

from src.services.web_searcher import ParseError, WebSearcher

logger = logging.getLogger(__name__)

@tool
async def web_parse(urls: List[str], summarize: bool = True):
    """
    Parses the content of given web pages and optionally summarizes it. 
    Use this tool to extract information from specific web pages. 
    Provide a list of URLs to parse.
    """
    logger.debug(f"web_parse called with arguments: urls={urls}, summarize={summarize}")
    searcher = WebSearcher()
    try:
        parsed_results = await searcher.parse_pages(urls, summarize=summarize)
        return str({"parsed_results": parsed_results})
    except ParseError as e:
        logger.error(f"Error parsing web pages: {str(e)}")
        return f"An error occurred while parsing web pages: {str(e)}. Please check the provided URLs and try again."