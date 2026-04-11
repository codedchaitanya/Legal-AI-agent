# core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

    MONGO_URI: str = "mongodb://localhost:27017/legalai"
    REDIS_URL: str = "redis://localhost:6379/0"
    POSTGRES_URL: str = "postgresql+asyncpg://legalai:changeme@localhost:5432/legalai"
    POSTGRES_PASSWORD: str  # no default — must be set in .env

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str  # no default — must be set in .env
    MINIO_BUCKET_DOCS: str = "legal-docs"
    MINIO_BUCKET_ADAPTERS: str = "lora-adapters"

    BASE_MODEL_ID: str = "Qwen/Qwen2.5-7B-Instruct"
    ADAPTERS_DIR: str = "./adapters"

    CRITIC_THRESHOLD: float = 0.72
    ARGUMENT_K: int = 5
    SECONDARY_ADAPTER_CONFIDENCE: float = 0.40
    API_KEY: str  # no default — must be set in .env


settings = Settings()
