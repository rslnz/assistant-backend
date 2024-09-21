import math
from typing import ClassVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class MathArgs(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate")

class MathTool(BaseTool):
    name: ClassVar[str] = "math_eval"
    description: ClassVar[str] = "Performs mathematical calculations. Input should be a mathematical expression as a string."
    args_schema: ClassVar[type[MathArgs]] = MathArgs

    def _run(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": None}, {"math": math})
            return f"Result of {expression} = {result}"
        except Exception as e:
            return (f"Error evaluating expression: {str(e)}. "
                    f"Please check the syntax of the expression and ensure all functions used are from the 'math' module. "
                    f"Complex or nested expressions may need to be simplified.")

    async def _arun(self, expression: str) -> str:
        return self._run(expression)