"""Punto único de configuración: rutas, límites del orquestador, LLM, Redis, worker y observabilidad (desde .env)."""
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, SecretStr

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "docs"

# --- Orquestador multi-agente (heredado de la Pre-entrega 6) ---
# Condición de parada dura contra el "supervisor infinito": cantidad máxima de delegaciones a especialistas.
MAX_DELEGATIONS = 6
# Máximo de veces que se puede delegar en un mismo especialista (1 intento + refinamientos).
MAX_CALLS_PER_AGENT = 3
# Techo de seguridad de LangGraph (super-steps por invocación). Cada delegación consume 2 pasos
# (supervisor + especialista); el margen cubre síntesis, aprobación humana y ejecución de la acción.
RECURSION_LIMIT = 2 * MAX_DELEGATIONS + 8
# Techo de pasos del loop ReAct interno de cada especialista (modelo <-> tools).
SPECIALIST_RECURSION_LIMIT = 12
# Largo máximo del resultado de cada tool call que se guarda en el estado como evidencia.
TOOL_RESULT_MAX_CHARS = 300

# --- Búsqueda en la base de conocimiento (BM25 sobre data/docs) ---
TOP_K = 4
CHUNK_MAX_CHARS = 1200

# --- Human-in-the-loop ---
# Destinatario por defecto cuando la consulta pide enviar el informe pero no indica un email.
DEFAULT_REPORT_RECIPIENT = "direccion@genpet.com.ar"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"


DEFAULT_MODEL_POR_PROVIDER = {
    LLMProvider.OPENAI: "gpt-4.1-mini",
    LLMProvider.ANTHROPIC: "claude-haiku-4-5",
    LLMProvider.GEMINI: "gemini-3.5-flash-lite",
    LLMProvider.OPENROUTER: "openai/gpt-4.1-mini",
}

ENV_VAR_POR_LLM_PROVIDER = {
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProvider.GEMINI: "GEMINI_API_KEY",
    LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _require_env(nombre: str) -> str:
    valor = os.getenv(nombre)
    if not valor:
        raise ValueError(f"Falta la variable de entorno {nombre} (ver .env.example)")
    return valor


def _env_bool(nombre: str, default: bool) -> bool:
    valor = os.getenv(nombre)
    return default if valor in (None, "") else valor.strip().lower() in ("1", "true", "yes", "si", "sí")


class LLMSettings(BaseModel):
    provider: LLMProvider
    model: str
    api_key: SecretStr
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=2048, gt=0)
    # Tope de requests por minuto al proveedor (None = sin límite). Ej.: el free tier de Gemini permite 15 RPM.
    requests_per_minute: Optional[int] = Field(default=None, gt=0)

    @property
    def label(self) -> str:
        return f"{self.provider.value}:{self.model}"


def get_llm_settings(provider: Optional[str] = None) -> LLMSettings:
    """Settings del LLM (llamar después de load_dotenv). LLM_MODEL pisa el modelo por defecto del provider."""
    proveedor = LLMProvider((provider or os.getenv("LLM_PROVIDER", "openrouter")).lower())
    return LLMSettings(
        provider=proveedor,
        model=os.getenv("LLM_MODEL") or DEFAULT_MODEL_POR_PROVIDER[proveedor],
        api_key=SecretStr(_require_env(ENV_VAR_POR_LLM_PROVIDER[proveedor])),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS") or 2048),
        requests_per_minute=int(os.getenv("LLM_REQUESTS_PER_MINUTE") or 0) or None,
    )


def get_tavily_api_key() -> Optional[str]:
    """La búsqueda web es opcional: sin TAVILY_API_KEY el investigador usa solo la base de conocimiento local."""
    return os.getenv("TAVILY_API_KEY") or None


class AppSettings(BaseModel):
    """Infraestructura de la API: Redis (estado + checkpoints + cola), worker y observabilidad."""

    redis_url: str = "redis://localhost:6379/0"
    # Prefijo de todas las claves propias (jobs, cola, outbox). Los checkpoints usan las claves de RedisSaver.
    key_prefix: str = "genpet"
    # Los jobs y los checkpoints expiran a los 7 días (evita que Redis crezca sin límite).
    job_ttl_seconds: int = Field(default=7 * 24 * 3600, gt=0)
    # Worker: True = corre dentro del proceso de la API (lifespan); False = proceso aparte (python -m app.worker).
    run_worker_in_api: bool = True
    worker_concurrency: int = Field(default=5, gt=0)
    # Tiempo máximo de una ejecución del grafo antes de marcar el job como FAILED.
    job_timeout_seconds: int = Field(default=300, gt=0)
    # Observabilidad: Arize Phoenix vía OpenTelemetry/OpenInference.
    phoenix_enabled: bool = True
    phoenix_collector_endpoint: str = "http://localhost:6006"
    phoenix_project_name: str = "genpet-multiagente"


def get_app_settings() -> AppSettings:
    return AppSettings(
        redis_url=os.getenv("REDIS_URL") or AppSettings.model_fields["redis_url"].default,
        run_worker_in_api=_env_bool("RUN_WORKER_IN_API", True),
        worker_concurrency=int(os.getenv("WORKER_CONCURRENCY") or 5),
        job_timeout_seconds=int(os.getenv("JOB_TIMEOUT_SECONDS") or 300),
        phoenix_enabled=_env_bool("PHOENIX_ENABLED", True),
        phoenix_collector_endpoint=(os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
                                    or AppSettings.model_fields["phoenix_collector_endpoint"].default),
        phoenix_project_name=(os.getenv("PHOENIX_PROJECT_NAME")
                              or AppSettings.model_fields["phoenix_project_name"].default),
    )
