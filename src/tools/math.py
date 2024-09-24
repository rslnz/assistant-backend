import logging
import math

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool
async def math_evaluation(expression: str) -> str:
    """
    Evaluates mathematical expressions. Use this tool for calculations, 
    including basic arithmetic, trigonometric functions, logarithms, and more. 
    Provide the expression as a string.
    """
    logger.debug(f"math_evaluation called with expression: {expression}")
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