from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1: str
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    OPENAI_API_KEY: str
    OPENAI_API_BASE: str
    OPENAI_MODEL: str

    MAX_HISTORY_MESSAGES: int = 10

    class Config:
        env_file = ".env"

settings = Settings()