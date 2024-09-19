import logging
from functools import wraps
from fastapi import HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

def handle_exceptions(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException as http_exc:
            raise http_exc
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}", exc_info=True)

            if isinstance(e, ValueError):
                status_code = 400  # Bad Request
            elif isinstance(e, NotImplementedError):
                status_code = 501  # Not Implemented
            else:
                status_code = 500  # Internal Server Error
            
            error_message = str(e) if status_code != 500 else "An unexpected error occurred"
            
            return JSONResponse(
                status_code=status_code,
                content={"error": error_message, "type": type(e).__name__}
            )
    return wrapper