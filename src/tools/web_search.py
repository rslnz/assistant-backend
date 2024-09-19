from typing import ClassVar, Dict
from langchain.tools import BaseTool
from pydantic import Field
from src.services.web_searcher import WebSearcher

class WebSearchTool(BaseTool):
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = "Searches the internet for relevant information based on the given query and returns a list of search results."
    args: ClassVar[Dict[str, str]] = {
        "query": "Search query"
    }
    return_direct: ClassVar[bool] = True

    searcher: WebSearcher = Field(default_factory=WebSearcher)

    def _run(self, query: str) -> str:
        raise NotImplementedError("WebSearchTool does not support synchronous execution")

    async def _arun(self, query: str) -> str:
        search_results = await self.searcher.search(query, num_results=5)
        if not search_results:
            return "No search results found. The search query might be too specific or there might be an issue with the search service."
        return str({"search_results": search_results})

    def invoke(self, arguments: Dict[str, str]) -> str:
        raise NotImplementedError("WebSearchTool does not support synchronous execution")

    async def ainvoke(self, arguments: Dict[str, str]) -> str:
        return await self._arun(arguments["query"])