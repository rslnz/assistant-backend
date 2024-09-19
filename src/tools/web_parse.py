from typing import ClassVar, Dict
from langchain.tools import BaseTool
from pydantic import Field
from src.services.web_searcher import WebSearcher

class WebParseTool(BaseTool):
    name: ClassVar[str] = "web_parse"
    description: ClassVar[str] = "Parses and analyzes the content of web pages from a given list of URLs, extracting main text, links, and code snippets."
    args: ClassVar[Dict[str, str]] = {
        "urls": "List of URLs to parse (comma-separated)"
    }
    return_direct: ClassVar[bool] = True

    searcher: WebSearcher = Field(default_factory=WebSearcher)

    def _run(self, urls: str) -> str:
        raise NotImplementedError("WebParseTool does not support synchronous execution")

    async def _arun(self, urls: str) -> str:
        url_list = [url.strip() for url in urls.split(',')]
        parsed_results = await self.searcher.parse_pages(url_list)
        if not parsed_results:
            return "No content could be parsed from the provided URLs. There might be an issue with the parsing service or the URLs might be invalid."
        return str({"parsed_results": parsed_results})

    def invoke(self, arguments: Dict[str, str]) -> str:
        raise NotImplementedError("WebParseTool does not support synchronous execution")

    async def ainvoke(self, arguments: Dict[str, str]) -> str:
        return await self._arun(arguments["urls"])