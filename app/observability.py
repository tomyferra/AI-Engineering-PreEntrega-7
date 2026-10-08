"""Observabilidad con Arize Phoenix (OpenTelemetry + OpenInference).

- `init_observability()` registra el TracerProvider que exporta a Phoenix e instrumenta LangChain/LangGraph:
  cada nodo del grafo, cada llamada al LLM (con tokens de entrada/salida y modelo, de donde Phoenix calcula el
  costo) y cada tool call quedan como spans.
- `traced_job` es el decorador del span raíz de cada ejecución (una corrida del worker = una traza).
- El provider NO se registra como global: FastAPI trae telemetría nativa que usaría el provider global y
  mandaría a Phoenix un span por cada GET de polling, ensuciando las trazas y la latencia p95 del proyecto.
- `job_context()` etiqueta todos los spans de una ejecución con `session.id = job_id`: en la vista Sessions de
  Phoenix la corrida inicial y la reanudación después de la aprobación humana aparecen juntas.
"""
import logging
from contextlib import contextmanager
from typing import Awaitable, Callable, Iterator, Optional, TypeVar

from openinference.instrumentation import OITracer, TraceConfig, using_attributes
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

from app.config import AppSettings

logger = logging.getLogger(__name__)

_provider: Optional[TracerProvider] = None

T = TypeVar("T")


def get_tracer() -> OITracer:
    """Tracer de OpenInference sobre el provider de Phoenix (no-op si la observabilidad está apagada)."""
    provider = _provider or trace_api.NoOpTracerProvider()
    return OITracer(provider.get_tracer("genpet.api"), config=TraceConfig())


def traced_job(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Decorador del span raíz de cada ejecución del grafo (span kind CHAIN, con input/output del job).

    Se aplica en tiempo de ejecución (al crear el Worker, después de init_observability), no al importar.
    """
    return get_tracer().chain(name="genpet.job")(fn)


def init_observability(settings: AppSettings) -> Optional[TracerProvider]:
    """Conecta el exportador OTLP de Phoenix e instrumenta LangChain. Idempotente."""
    global _provider
    if not settings.phoenix_enabled:
        logger.info("Observabilidad deshabilitada (PHOENIX_ENABLED=false)")
        return None
    if _provider is not None:
        return _provider

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    _provider = register(
        endpoint=f"{settings.phoenix_collector_endpoint.rstrip('/')}/v1/traces",
        project_name=settings.phoenix_project_name,
        batch=True,                       # exporta en segundo plano: no agrega latencia a los requests
        set_global_tracer_provider=False,  # solo exportan los spans del agente (ver docstring del módulo)
        verbose=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=_provider)
    logger.info("Trazas -> Phoenix %s (proyecto %s)", settings.phoenix_collector_endpoint,
                settings.phoenix_project_name)
    return _provider


def shutdown_observability() -> None:
    """Vacía el buffer de spans pendientes antes de terminar el proceso."""
    global _provider
    if _provider is not None:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().uninstrument()
        _provider.force_flush()
        _provider.shutdown()
        _provider = None


@contextmanager
def job_context(job_id: str, kind: str, query: str) -> Iterator[None]:
    """Atributos OpenInference que heredan todos los spans de la ejecución (sesión, metadata y tags)."""
    with using_attributes(session_id=job_id, metadata={"job_id": job_id, "run_kind": kind, "query": query[:200]},
                          tags=["genpet", "api", kind]):
        yield
