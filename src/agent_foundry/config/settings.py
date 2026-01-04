"""Configuration management for agent_foundry."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from agent_foundry.constants import ModelNames

load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Model Mapping (pro/flash -> actual model names)
    pro_model: str = ModelNames.DEFAULT_PRO_MODEL
    flash_model: str = ModelNames.DEFAULT_FLASH_MODEL

    openai_api_key: str = ""
    openai_base_url: str = ""


settings = Settings()
