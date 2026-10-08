"""Human-in-the-loop: clasificación de tareas críticas y pausa/reanudación del grafo con interrupt()."""
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.config import DEFAULT_REPORT_RECIPIENT, RECURSION_LIMIT
from app.hitl import classify_task
from app.orchestrator.agents.supervisor import FINISH, SupervisorDecision
from app.orchestrator.state import APPROVAL, RESEARCHER
from app.orchestrator.tools.report_sender import InMemoryOutbox
from tests.conftest import fake_model, run_query, scripted_router
from tests.test_graph import build_test_graph

CONFIG = {"configurable": {"thread_id": "t1"}, "recursion_limit": RECURSION_LIMIT}


@pytest.mark.parametrize("query", [
    "Enviá el informe de ventas del Q1 a gerencia@genpet.com.ar",
    "Calculá el crecimiento y mandale el resumen al equipo comercial",
    "¿Cuánto facturamos? Notificá el resultado a ventas@genpet.com.ar",
])
def test_consultas_que_piden_enviar_son_criticas(query):
    action = classify_task(query)
    assert action is not None and action["type"] == "enviar_informe"


@pytest.mark.parametrize("query", [
    "¿Cuáles fueron los ingresos netos del Q1?",
    "¿Cuánto cuesta el envío del pack x50?",  # "envío" es un sustantivo: no pide ejecutar ninguna acción
    "¿Qué canal de ventas concentra más ingresos?",
])
def test_consultas_de_solo_lectura_no_son_criticas(query):
    assert classify_task(query) is None


def test_destinatario_explicito_o_por_defecto():
    assert classify_task("ingresos Q1", send_report_to="a@b.com")["recipient"] == "a@b.com"
    assert classify_task("Mandá el informe de ventas")["recipient"] == DEFAULT_REPORT_RECIPIENT
    assert classify_task("Enviá el informe a ceo@genpet.com.ar")["recipient"] == "ceo@genpet.com.ar"


def _graph(kb, outbox):
    router = scripted_router([
        SupervisorDecision(reasoning="faltan datos", next=RESEARCHER, instruction="buscar ingresos Q1"),
        SupervisorDecision(reasoning="ok", next=FINISH),
    ])
    return build_test_graph(kb, router, fake_model([AIMessage("[REAL] Q1: $4.427.580")]), fake_model([]),
                            fake_model([AIMessage("Informe: Q1 $4.427.580")]), outbox=outbox,
                            checkpointer=InMemorySaver())


async def test_tarea_critica_se_pausa_hasta_la_aprobacion(kb):
    outbox = InMemoryOutbox()
    graph = _graph(kb, outbox)
    action = classify_task("Enviá el informe a ceo@genpet.com.ar")

    await run_query(graph, "Enviá el informe a ceo@genpet.com.ar", pending_action=action)

    snapshot = await graph.aget_state(CONFIG)
    assert snapshot.next == (APPROVAL,)  # detenido en el nodo de aprobación
    assert snapshot.interrupts[0].value["action"]["recipient"] == "ceo@genpet.com.ar"
    assert snapshot.interrupts[0].value["draft"] == "Informe: Q1 $4.427.580"
    assert outbox.sent == []  # la acción con efectos secundarios todavía no se ejecutó

    final = await graph.ainvoke(Command(resume={"approved": True, "reviewer": "ana"}), config=CONFIG)

    assert final["approval"] == {"approved": True, "reviewer": "ana", "comment": ""}
    assert final["action_result"]["executed"] is True
    assert [m["recipient"] for m in outbox.sent] == ["ceo@genpet.com.ar"]
    assert outbox.sent[0]["body"] == "Informe: Q1 $4.427.580"


async def test_rechazo_no_ejecuta_la_accion(kb):
    outbox = InMemoryOutbox()
    graph = _graph(kb, outbox)
    await run_query(graph, "Enviá el informe", pending_action=classify_task("Enviá el informe"))

    final = await graph.ainvoke(Command(resume={"approved": False, "comment": "no todavía"}), config=CONFIG)

    assert final["approval"]["approved"] is False and final["action_result"] is None
    assert outbox.sent == []
    assert (await graph.aget_state(CONFIG)).next == ()  # terminó


async def test_resume_mal_formado_cuenta_como_rechazo(kb):
    outbox = InMemoryOutbox()
    graph = _graph(kb, outbox)
    await run_query(graph, "Enviá el informe", pending_action=classify_task("Enviá el informe"))

    final = await graph.ainvoke(Command(resume="sí, dale"), config=CONFIG)

    assert final["approval"]["approved"] is False and outbox.sent == []


async def test_tarea_no_critica_termina_sin_pausa(kb):
    outbox = InMemoryOutbox()
    graph = _graph(kb, outbox)

    final = await run_query(graph, "¿Cuánto facturamos en el Q1?")

    assert final["final_answer"] == "Informe: Q1 $4.427.580"
    assert (await graph.aget_state(CONFIG)).next == () and outbox.sent == []
