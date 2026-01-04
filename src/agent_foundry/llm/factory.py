"""LLM factory for creating GPT instances."""

import logging

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from agent_foundry.config import settings

logger = logging.getLogger(__name__)


def get_llm(model: str, temperature: float = 0.5, **kwargs) -> ChatOpenAI:
    """Get a GPT model instance.

    Args:
        model: Model name (e.g., "gpt-5.2", "gpt-5-mini")
        temperature: Sampling temperature
        **kwargs: Additional model parameters

    Returns:
        ChatOpenAI instance
    """
    api_key = settings.openai_api_key
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in settings")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=SecretStr(api_key) if api_key else None,
        base_url=settings.openai_base_url,
        **kwargs,
    )
