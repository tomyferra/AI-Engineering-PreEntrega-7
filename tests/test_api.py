"""API end-to-end con Redis falso (fakeredis) y LLMs guionados: encolado, polling, FAILED y aprobación humana."""
import asyncio
from contextlib import asynccontextmanager

import fakeredis
import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from app.jobs import JobStatus, JobStore
from app.main import create_app
from app.orchestrator.agents.supervisor import FINISH, SupervisorDecision
from app.orchestrator.state import RESEARCHER
from app.orchestrator.tools.report_sender import InMemoryOutbox
from app.worker import Runtime, Worker
from tests.conftest import fake_model, scripted_router
from tests.test_graph import build_test_graph

TERMINAL = {JobStatus.DONE.value, JobStatus.FAILED.value, JobStatus.WAITING_APPROVAL.value}


def ok_graph(kb, outbox):
    router = scripted_router([
        SupervisorDecision(reasoning="faltan datos", next=RESEARCHER, instruction="buscar ingresos Q1"),
        SupervisorDecision(reasoning="ok", next=FINISH),
    ])
    return build_test_graph(kb, router, fake_model([AIMessage("[REAL] Q1: $4.427.580")]), fake_model([]),
                            fake_model([AIMessage("Q1: $4.427.580")]), outbox=outbox, checkpointer=InMemorySaver())


def broken_graph(kb):
    def sin_creditos(_messages):
        raise ConnectionError("402 Payment Required")

    return build_test_graph(kb, RunnableLambda(sin_creditos), fake_model([]), fake_model([]), fake_model([]),
                            checkpointer=InMemorySaver())


@asynccontextmanager
async def client_for(graph, job_timeout_seconds: float = 30):
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    store = JobStore(redis, prefix="test")

    @asynccontextmanager
    async def runtime_factory():
        worker = Worker(store, graph, concurrency=2, job_timeout_seconds=job_timeout_seconds, poll_timeout=0.05)
        worker.start()
        try:
            yield Runtime(redis=redis, store=store, graph=graph, worker=worker)
        finally:
            await worker.stop()

    app = create_app(runtime_factory)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client, store


async def poll(client, job_id: str, statuses=TERMINAL, timeout: float = 10) -> dict:
    async def _wait():
        while True:
            body = (await client.get(f"/tasks/{job_id}")).json()
            if body["status"] in statuses:
                return body
            await asyncio.sleep(0.02)

    return await asyncio.wait_for(_wait(), timeout)


async def test_post_encola_y_devuelve_job_id_sin_esperar_al_grafo(kb):
    async with client_for(ok_graph(kb, InMemoryOutbox())) as (client, store):
        response = await client.post("/tasks", json={"query": "¿Cuánto facturamos en el Q1?"})

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "PENDING" and body["critical"] is False and len(body["job_id"]) == 32

        done = await poll(client, body["job_id"])
        assert done["status"] == "DONE" and done["final_answer"] == "Q1: $4.427.580"
        assert done["flow"] == [RESEARCHER, "sintetizador"] and done["finished_at"]


async def test_excepcion_del_agente_deja_el_job_en_failed(kb):
    async with client_for(broken_graph(kb)) as (client, _store):
        job_id = (await client.post("/tasks", json={"query": "¿Cuánto facturamos?"})).json()["job_id"]

        failed = await poll(client, job_id)

        assert failed["status"] == "FAILED"
        assert "ConnectionError" in failed["error"] and "402" in failed["error"]


async def test_timeout_deja_el_job_en_failed(kb):
    async def lento(_messages):
        await asyncio.sleep(5)

    graph = build_test_graph(kb, RunnableLambda(lento), fake_model([]), fake_model([]), fake_model([]),
                             checkpointer=InMemorySaver())
    async with client_for(graph, job_timeout_seconds=0.2) as (client, _store):
        job_id = (await client.post("/tasks", json={"query": "consulta lenta"})).json()["job_id"]

        failed = await poll(client, job_id)

        assert failed["status"] == "FAILED" and failed["error"].startswith("Timeout")


async def test_flujo_hitl_completo_por_api(kb):
    outbox = InMemoryOutbox()
    async with client_for(ok_graph(kb, outbox)) as (client, _store):
        created = (await client.post("/tasks", json={"query": "Ingresos del Q1",
                                                     "send_report_to": "ceo@genpet.com.ar"})).json()
        assert created["critical"] is True

        waiting = await poll(client, created["job_id"])
        assert waiting["status"] == "WAITING_APPROVAL"
        assert waiting["approval_request"]["action"]["recipient"] == "ceo@genpet.com.ar"
        assert outbox.sent == []

        approve = await client.post(f"/tasks/{created['job_id']}/approve", json={"approved": True, "reviewer": "ana"})
        assert approve.status_code == 202

        done = await poll(client, created["job_id"], statuses={"DONE", "FAILED"})
        assert done["status"] == "DONE" and done["action_result"]["executed"] is True
        assert done["approval"]["reviewer"] == "ana"
        assert [m["recipient"] for m in outbox.sent] == ["ceo@genpet.com.ar"]

        # Una segunda aprobación ya no aplica: el job no está esperando.
        again = await client.post(f"/tasks/{created['job_id']}/approve", json={"approved": True})
        assert again.status_code == 409


async def test_aprobaciones_simultaneas_reanudan_una_sola_vez(kb):
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    store = JobStore(redis, prefix="test")
    job = await store.create("consulta")
    await redis.delete(store.queue_key)
    await store.set_status(job["job_id"], JobStatus.WAITING_APPROVAL)

    results = await asyncio.gather(*(store.request_resume(job["job_id"], {"approved": True}) for _ in range(5)))

    assert results.count(True) == 1
    assert await redis.llen(store.queue_key) == 1


@pytest.mark.parametrize("payload", [{"query": "hey"}, {"query": "consulta válida", "send_report_to": "no-es-email"}])
async def test_validacion_de_entrada(kb, payload):
    async with client_for(ok_graph(kb, InMemoryOutbox())) as (client, _store):
        assert (await client.post("/tasks", json=payload)).status_code == 422


async def test_job_inexistente_da_404(kb):
    async with client_for(ok_graph(kb, InMemoryOutbox())) as (client, _store):
        assert (await client.get("/tasks/no-existe")).status_code == 404
        assert (await client.post("/tasks/no-existe/approve", json={})).status_code == 404
        assert (await client.get("/health")).json()["status"] == "ok"
