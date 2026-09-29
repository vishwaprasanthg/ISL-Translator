from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    port: int = 8000
    host: str = "0.0.0.0"
    cors_origins: str = "http://localhost:5173"
    database_url: str = "sqlite+aiosqlite:///data/isl_translator.db"

    class Config:
        env_file = ".env"

settings = Settings()
