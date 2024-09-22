import logging
from typing import ClassVar
import math

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class MathArgs(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate")

class MathTool(BaseTool):
    name: ClassVar[str] = "math_evaluation"
    description: ClassVar[str] = (
        "Evaluates mathematical expressions. "
        "Use this tool for calculations, including basic arithmetic, "
        "trigonometric functions, logarithms, and more. "
        "Provide the expression as a string."
    )
    args_schema: ClassVar[type[MathArgs]] = MathArgs

    def _run(self, expression: str) -> str:
        logger.debug(f"MathTool._run called with expression: {expression}")
        try:
            safe_dict = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow, "math": math
            }
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return str(result)
        except Exception as e:
            logger.error(f"Error evaluating math expression: {str(e)}")
            return f"An error occurred while evaluating the expression: {str(e)}. Please check the syntax and try again."

    async def _arun(self, expression: str) -> str:
        return self._run(expression)