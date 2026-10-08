"""Factory del chat model según LLM_PROVIDER (openai | anthropic | gemini | openrouter)."""
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

from app.config import OPENROUTER_BASE_URL, LLMProvider, LLMSettings


def build_rate_limiter(requests_per_minute: Optional[int]) -> Optional[BaseRateLimiter]:
    """Token bucket compartido por todos los agentes y todos los jobs del worker (mismo modelo, misma cuota).

    Con varias ejecuciones concurrentes, las llamadas al LLM se espacian en vez de recibir 429 del proveedor.
    """
    if not requests_per_minute:
        return None
    return InMemoryRateLimiter(requests_per_second=requests_per_minute / 60, check_every_n_seconds=0.1,
                               max_bucket_size=1)


def build_chat_model(settings: LLMSettings) -> BaseChatModel:
    # max_tokens explícito: sin él algunos proveedores reservan el máximo del modelo en cada request.
    common = {"model": settings.model, "temperature": settings.temperature, "max_tokens": settings.max_tokens,
              "rate_limiter": build_rate_limiter(settings.requests_per_minute)}
    api_key = settings.api_key.get_secret_value()
    if settings.provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=api_key, **common)
    if settings.provider == LLMProvider.OPENROUTER:
        # OpenRouter expone una API compatible con OpenAI: mismo cliente, otra base_url.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL, **common)
    if settings.provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(api_key=api_key, **common)
    if settings.provider == LLMProvider.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(google_api_key=api_key, **common)
    raise ValueError(f"LLM provider no soportado: {settings.provider}")
