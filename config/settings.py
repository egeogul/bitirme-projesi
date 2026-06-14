from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "changeme"

    database_url: str = "sqlite:///./supply_chain.db"
    redis_url: str = "redis://localhost:6379"

    anthropic_api_key: str = ""
    openai_api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
