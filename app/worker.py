"""Worker: consume la cola de Redis, ejecuta el grafo en segundo plano y actualiza el estado del job.

Corre de dos formas (mismo código):
- dentro de la API (RUN_WORKER_IN_API=true): N corrutinas lanzadas en el lifespan de FastAPI;
- como proceso aparte: `python -m app.worker` (RUN_WORKER_IN_API=false en la API; ver docker-compose.yml).

Todo es async (Redis, LLM, checkpointer), así que las N ejecuciones concurrentes comparten un solo event loop
sin bloquearlo. Cualquier excepción del grafo (error del proveedor LLM, timeout, límite de pasos) se captura y
deja el job en FAILED con el motivo: el cliente que hace polling nunca queda esperando para siempre.
"""
import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from redis.asyncio import Redis

from app.config import RECURSION_LIMIT, AppSettings, get_app_settings
from app.graph import create_orchestrator, redis_checkpointer
from app.jobs import JobStatus, JobStore, QueueItem, utc_now
from app.observability import init_observability, job_context, shutdown_observability, traced_job
from app.orchestrator.state import initial_state
from app.orchestrator.tools.report_sender import RedisOutbox

logger = logging.getLogger(__name__)


def summarize_flow(values: dict) -> List[str]:
    """Secuencia de destinos que eligió el supervisor (ej. investigador -> analista -> sintetizador)."""
    return [d["next"] for d in values.get("decisions", [])]


class Worker:
    def __init__(self, store: JobStore, graph: CompiledStateGraph, concurrency: int = 5,
                 job_timeout_seconds: float = 300, poll_timeout: float = 1.0):
        self.store = store
        self.graph = graph
        self.concurrency = concurrency
        self.job_timeout_seconds = job_timeout_seconds
        self.poll_timeout = poll_timeout
        self._tasks: List[asyncio.Task] = []
        self._stopping = asyncio.Event()
        # Span raíz de cada ejecución en Phoenix (decorador de OpenInference sobre el método ya ligado).
        self._traced_run = traced_job(self._run_graph)

    # --- ciclo de vida -------------------------------------------------------------------------------------

    def start(self) -> None:
        self._stopping.clear()
        self._tasks = [asyncio.create_task(self._consume(i), name=f"worker-{i}") for i in range(self.concurrency)]
        logger.info("Worker iniciado con %d consumidores", self.concurrency)

    async def stop(self) -> None:
        self._stopping.set()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._tasks = []

    async def _consume(self, index: int) -> None:
        while not self._stopping.is_set():
            try:
                item = await self.store.dequeue(timeout=self.poll_timeout)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - Redis caído un instante: se reintenta sin matar al consumidor
                logger.exception("worker-%d: error leyendo la cola; reintento en 1s", index)
                await asyncio.sleep(1)
                continue
            if item is not None:
                await self.process(item)

    # --- ejecución de un job -------------------------------------------------------------------------------

    def _config(self, job_id: str) -> dict:
        # thread_id = job_id: el checkpoint del job en Redis es el que se reanuda después de la aprobación.
        return {"configurable": {"thread_id": job_id}, "recursion_limit": RECURSION_LIMIT}

    async def _run_graph(self, run: dict) -> dict:
        """Ejecuta (o reanuda) el grafo y devuelve el resumen que queda como output del span raíz."""
        job_id, kind = run["job_id"], run["kind"]
        if kind == "resume":
            graph_input = Command(resume=run["resume"])
        else:
            graph_input = initial_state(run["query"], run.get("pending_action"))
        await self.graph.ainvoke(graph_input, config=self._config(job_id))
        snapshot = await self.graph.aget_state(self._config(job_id))
        values = snapshot.values
        if snapshot.interrupts:
            return {"status": JobStatus.WAITING_APPROVAL.value, "approval_request": snapshot.interrupts[0].value,
                    "final_answer": values.get("final_answer"), "delegations": values.get("delegations"),
                    "flow": summarize_flow(values)}
        return {"status": JobStatus.DONE.value, "final_answer": values.get("final_answer"),
                "action_result": values.get("action_result"), "approval": values.get("approval"),
                "delegations": values.get("delegations"), "flow": summarize_flow(values)}

    async def process(self, item: QueueItem) -> None:
        job_id, kind = item["job_id"], item.get("kind", "start")
        job = await self.store.get(job_id)
        if job is None:
            logger.warning("Job %s no existe (¿expiró?): se descarta", job_id)
            return
        await self.store.set_status(job_id, JobStatus.RUNNING, started_at=utc_now(), error=None)
        run = {"job_id": job_id, "kind": kind, "query": job["query"], "pending_action": job.get("pending_action"),
               "resume": item.get("resume")}
        try:
            with job_context(job_id, kind, job["query"]):
                result = await asyncio.wait_for(self._traced_run(run), timeout=self.job_timeout_seconds)
        except asyncio.TimeoutError:
            await self._fail(job_id, f"Timeout: la ejecución superó {self.job_timeout_seconds}s")
            return
        except asyncio.CancelledError:
            await self._fail(job_id, "El worker se detuvo durante la ejecución")
            raise
        except Exception as exc:  # noqa: BLE001 - cualquier falla del grafo termina en FAILED, nunca en un loop
            logger.exception("Job %s falló", job_id)
            await self._fail(job_id, f"{type(exc).__name__}: {exc}")
            return

        status = JobStatus(result.pop("status"))
        if status == JobStatus.DONE:
            result["finished_at"] = utc_now()
        await self.store.set_status(job_id, status, **result)
        logger.info("Job %s -> %s", job_id, status.value)

    async def _fail(self, job_id: str, error: str) -> None:
        with suppress(Exception):  # si Redis tampoco responde, al menos queda en el log
            await self.store.set_status(job_id, JobStatus.FAILED, error=error[:2000], finished_at=utc_now())
        logger.error("Job %s -> FAILED: %s", job_id, error)


@dataclass
class Runtime:
    redis: Redis
    store: JobStore
    graph: Optional[CompiledStateGraph]
    worker: Optional[Worker]


@asynccontextmanager
async def build_runtime(settings: Optional[AppSettings] = None, with_worker: bool = True) -> AsyncIterator[Runtime]:
    """Conexiones compartidas por la API y el worker.

    Siempre: Redis (jobs + cola). Con worker: además el RedisSaver, el grafo compilado y los consumidores.
    Una API con el worker en otro proceso no necesita ni el grafo ni la API key del LLM.
    """
    settings = settings or get_app_settings()
    async with AsyncExitStack() as stack:
        init_observability(settings)
        stack.callback(shutdown_observability)
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        stack.push_async_callback(redis.aclose)
        await redis.ping()
        store = JobStore(redis, prefix=settings.key_prefix, ttl_seconds=settings.job_ttl_seconds)
        graph, worker = None, None
        if with_worker:
            checkpointer = await stack.enter_async_context(
                redis_checkpointer(settings.redis_url, settings.job_ttl_seconds))
            graph = create_orchestrator(checkpointer, RedisOutbox(redis, store.outbox_key))
            worker = Worker(store, graph, settings.worker_concurrency, settings.job_timeout_seconds)
            worker.start()
            stack.push_async_callback(worker.stop)
        yield Runtime(redis=redis, store=store, graph=graph, worker=worker)


async def _run_standalone() -> None:
    async with build_runtime(with_worker=True):
        logger.info("Worker standalone escuchando la cola de Redis (Ctrl+C para salir)")
        await asyncio.Event().wait()


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with suppress(KeyboardInterrupt):
        asyncio.run(_run_standalone())
