from typing import ClassVar, Dict
from langchain.tools import BaseTool
import math

class MathTool(BaseTool):
    name: ClassVar[str] = "math"
    description: ClassVar[str] = "Performs mathematical calculations. Input should be a mathematical expression as a string."
    args: ClassVar[Dict[str, str]] = {
        "expression": "Mathematical expression to evaluate"
    }
    return_direct: ClassVar[bool] = True

    def _run(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": None}, {"math": math})
            return f"Result of {expression} = {result}"
        except Exception as e:
            return f"Error evaluating expression: {str(e)}"

    async def _arun(self, expression: str) -> str:
        return self._run(expression)

    def invoke(self, arguments: Dict[str, str]) -> str:
        return self._run(arguments["expression"])

    async def ainvoke(self, arguments: Dict[str, str]) -> str:
        return await self._arun(arguments["expression"])