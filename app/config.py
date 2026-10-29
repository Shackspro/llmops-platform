
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_name: str = "llama3.2:1b"
    ollama_url: str = "http://127.0.0.1:11434/api/generate"
    ollama_tags_url: str = "http://127.0.0.1:11434/api/tags"
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl_seconds: int = 3600
    request_timeout_seconds: float = 120.0
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 60

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
