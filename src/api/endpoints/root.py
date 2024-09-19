from fastapi.responses import JSONResponse
from src.api.utils import handle_exceptions

class RootEndpoints:
    @staticmethod
    @handle_exceptions
    async def root():
        """
        Root endpoint of the API.
        
        Returns:
        - message: a welcome message
        """
        return JSONResponse(content={"message": "Hello! Welcome to the LLM Backend Service API."})

    @staticmethod
    @handle_exceptions
    async def health_check():
        """
        Endpoint for checking the service health.
        
        Returns:
        - status: the health status of the service
        """
        return JSONResponse(content={"status": "healthy"})