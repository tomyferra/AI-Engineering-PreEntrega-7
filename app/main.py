"""API REST del orquestador multi-agente de GenPet.

    POST /tasks                 -> clasifica la tarea, la encola en Redis y devuelve job_id (202, no espera al LLM)
    GET  /tasks/{job_id}        -> estado del job leído de Redis (PENDING/RUNNING/WAITING_APPROVAL/DONE/FAILED)
    POST /tasks/{job_id}/approve -> decisión humana para las tareas críticas (ver app/hitl.py)
    GET  /health                -> ping a Redis

Levantar:  uvicorn app.main:app --port 8000
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncContextManager, AsyncIterator, Callable

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Request, status

from app.config import get_app_settings
from app.hitl import classify_task
from app.hitl import router as hitl_router
from app.jobs import JobStatus, JobStore
from app.schemas import TaskAccepted, TaskCreate, TaskView
from app.worker import Runtime, build_runtime

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


tasks_router = APIRouter()


def _links(job_id: str) -> dict:
    return {"self": f"/tasks/{job_id}", "approve": f"/tasks/{job_id}/approve"}


@tasks_router.post("/tasks", response_model=TaskAccepted, status_code=status.HTTP_202_ACCEPTED, tags=["tasks"])
async def create_task(body: TaskCreate, request: Request) -> TaskAccepted:
    """Encola la tarea y responde enseguida: el grafo corre en el worker, no en este request."""
    store: JobStore = request.app.state.store
    pending_action = classify_task(body.query, body.send_report_to)
    job = await store.create(body.query, pending_action)
    return TaskAccepted(job_id=job["job_id"], status=JobStatus.PENDING, critical=pending_action is not None,
                        links=_links(job["job_id"]))


@tasks_router.get("/tasks/{job_id}", response_model=TaskView, tags=["tasks"])
async def get_task(job_id: str, request: Request) -> TaskView:
    """Estado actual del job (una lectura de Redis: pensado para polling)."""
    job = await request.app.state.store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
    return TaskView(**job)


@tasks_router.get("/health", tags=["ops"])
async def health(request: Request) -> dict:
    try:
        await request.app.state.store.redis.ping()
    except Exception as exc:  # noqa: BLE001 - el health check informa, no explota
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Redis no responde: {exc}") from exc
    runtime = request.app.state.runtime
    return {"status": "ok", "redis": "ok", "worker": "in-process" if runtime.worker else "external"}


RuntimeFactory = Callable[[], AsyncContextManager[Runtime]]


def _default_runtime() -> AsyncContextManager[Runtime]:
    settings = get_app_settings()
    return build_runtime(settings, with_worker=settings.run_worker_in_api)


def create_app(runtime_factory: RuntimeFactory = _default_runtime) -> FastAPI:
    """Crea la app. `runtime_factory` se inyecta en los tests (Redis falso + LLMs guionados)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with runtime_factory() as runtime:
            app.state.store = runtime.store
            app.state.runtime = runtime
            logger.info("API lista (worker %s)", "en proceso" if runtime.worker else "externo: python -m app.worker")
            yield

    app = FastAPI(
        title="GenPet · API del orquestador multi-agente",
        version="1.0.0",
        description="Supervisor + investigador + analista + sintetizador (LangGraph) con estado en Redis, "
                    "trazas en Arize Phoenix y aprobación humana para las acciones críticas.",
        lifespan=lifespan,
    )
    app.include_router(tasks_router)
    app.include_router(hitl_router)
    return app


app = create_app()
