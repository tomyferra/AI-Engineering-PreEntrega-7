"""Flujo completo del grafo con LLMs falsos: supervisor -> investigador -> analista -> sintetizador."""
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from app.orchestrator.agents import make_analyst_node, make_research_node, make_supervisor_node, make_synthesizer_node
from app.orchestrator.agents.supervisor import FINISH, SupervisorDecision
from app.orchestrator.agents.validation import build_grounding_validator
from app.config import MAX_CALLS_PER_AGENT
from app.graph import OrchestratorNodes, build_graph
from app.hitl import make_execute_action_node
from app.orchestrator.state import ANALYST, RESEARCHER, SYNTHESIZER
from app.orchestrator.tools import build_math_tools, build_research_tools
from app.orchestrator.tools.report_sender import InMemoryOutbox, build_report_sender_tool
from tests.conftest import fake_model, run_query, scripted_router, tool_call


def build_test_graph(kb, router, research_model, analyst_model, synth_model, outbox=None, checkpointer=None):
    return build_graph(OrchestratorNodes(
        supervisor=make_supervisor_node(router),
        researcher=make_research_node(research_model, build_research_tools(kb),
                                      validator=build_grounding_validator(kb.documents.values())),
        analyst=make_analyst_node(analyst_model, build_math_tools()),
        synthesizer=make_synthesizer_node(synth_model),
        execute_action=make_execute_action_node(build_report_sender_tool(outbox or InMemoryOutbox())),
    ), checkpointer=checkpointer)


async def test_delegacion_investigador_analista_sintesis(kb):
    router = scripted_router([
        SupervisorDecision(reasoning="faltan datos", next=RESEARCHER, instruction="buscar ingresos Q1 y Q2"),
        SupervisorDecision(reasoning="faltan cálculos", next=ANALYST, instruction="variación Q1->Q2"),
        SupervisorDecision(reasoning="rúbrica ok", next=FINISH),
    ])
    research_model = fake_model([
        tool_call("buscar_base_conocimiento", {"consulta": "ingresos netos por trimestre"}, "r1"),
        AIMessage("Q1: $4.427.580, Q2: $7.931.680 [fuente: historial_ventas_2026.md]"),
    ])
    analyst_model = fake_model([
        tool_call("variacion_porcentual", {"valor_inicial": 4427580, "valor_final": 7931680}, "a1"),
        AIMessage("Crecimiento Q1->Q2: 79.14%"),
    ])
    synth_model = fake_model([AIMessage("Los ingresos crecieron 79.14% del Q1 al Q2.")])
    graph = build_test_graph(kb, router, research_model, analyst_model, synth_model)

    final = await run_query(graph, "¿Cuánto crecieron los ingresos del Q1 al Q2?")

    assert [d["next"] for d in final["decisions"]] == [RESEARCHER, ANALYST, SYNTHESIZER]
    assert final["task_completed"] and final["final_answer"] == "Los ingresos crecieron 79.14% del Q1 al Q2."
    assert final["delegations"] == 2
    # Trazabilidad: cada aporte registra agente, instrucción y las tool calls reales con su resultado.
    investigacion, analisis = final["contributions"]
    assert [tc["tool"] for tc in investigacion["tool_calls"]] == ["buscar_base_conocimiento"]
    assert analisis["tool_calls"][0]["args"] == {"valor_inicial": 4427580, "valor_final": 7931680}
    assert '"variacion_pct": 79.14' in analisis["tool_calls"][0]["result"]
    # Los tool calls internos de los especialistas no contaminan el historial compartido.
    assert not any(isinstance(m, ToolMessage) for m in final["messages"])
    assert [getattr(m, "name", None) for m in final["messages"]] == [None, RESEARCHER, ANALYST, SYNTHESIZER]


async def test_contexto_acotado_por_especialista(kb):
    """El analista recibe los datos del investigador; el investigador no recibe aportes del analista."""
    router = scripted_router([
        SupervisorDecision(reasoning="r", next=RESEARCHER, instruction="buscar metas"),
        SupervisorDecision(reasoning="r", next=ANALYST, instruction="calcular cumplimiento"),
        SupervisorDecision(reasoning="falta Q3", next=RESEARCHER, instruction="buscar meta Q3"),
        SupervisorDecision(reasoning="ok", next=FINISH),  # la guarda de consistencia fuerza al analista
        SupervisorDecision(reasoning="ok", next=FINISH),
    ])
    research_model = fake_model([AIMessage("DATO-INVESTIGADOR"), AIMessage("DATO-Q3")])
    analyst_model = fake_model([AIMessage("RESULTADO-ANALISTA"), AIMessage("RESULTADO-RECALCULADO")])
    graph = build_test_graph(kb, router, research_model, analyst_model, fake_model([AIMessage("fin")]))

    await run_query(graph, "consulta")

    tarea_analista = analyst_model.received[0][-1].text
    assert "DATO-INVESTIGADOR" in tarea_analista and "calcular cumplimiento" in tarea_analista
    refinamiento_investigador = research_model.received[1][-1].text
    assert "DATO-INVESTIGADOR" in refinamiento_investigador  # ve su propio aporte previo
    assert "RESULTADO-ANALISTA" not in refinamiento_investigador
    recalculo = analyst_model.received[1][-1].text
    assert "DATO-Q3" in recalculo and "RESULTADO-ANALISTA" in recalculo  # ve datos nuevos y su aporte previo


async def test_supervisor_infinito_se_corta_con_las_guardas(kb):
    """Un supervisor que siempre pide más investigación termina igual: analiza lo disponible y sintetiza."""
    router = scripted_router(
        SupervisorDecision(reasoning="quiero más", next=RESEARCHER, instruction="otra vez") for _ in range(50))
    research_model = fake_model([AIMessage("dato")])
    analyst_model = fake_model([AIMessage("cálculo")])
    graph = build_test_graph(kb, router, research_model, analyst_model, fake_model([AIMessage("parcial")]))

    final = await run_query(graph, "consulta")

    assert final["task_completed"] and final["final_answer"] == "parcial"
    assert [c["agent"] for c in final["contributions"]] == [RESEARCHER] * MAX_CALLS_PER_AGENT + [ANALYST]
    assert [d["next"] for d in final["decisions"] if d["forced"]] == [ANALYST, SYNTHESIZER]


async def test_falla_del_router_no_rompe_el_grafo(kb):
    def router_roto(_messages):
        raise OutputParserException("respuesta no parseable")

    graph = build_test_graph(kb, RunnableLambda(router_roto), fake_model([]), fake_model([]),
                             fake_model([AIMessage("No hay datos suficientes.")]))
    final = await run_query(graph, "consulta")

    assert final["task_completed"] and final["decisions"][0]["next"] == SYNTHESIZER
    assert isinstance(final["messages"][0], HumanMessage)


async def test_error_de_infraestructura_se_propaga(kb):
    """Un error de la API (credenciales, red, créditos) no se disfraza de "tarea terminada"."""
    def router_sin_creditos(_messages):
        raise ConnectionError("402 sin créditos")

    graph = build_test_graph(kb, RunnableLambda(router_sin_creditos), fake_model([]), fake_model([]),
                             fake_model([]))
    with pytest.raises(ConnectionError):
        await run_query(graph, "consulta")


async def test_monto_inventado_por_el_investigador_se_corrige_antes_de_analizar(kb):
    """El supervisor quiere pasar al analista, pero el aporte tiene un monto que no está en las fuentes."""
    router = scripted_router([
        SupervisorDecision(reasoning="r", next=RESEARCHER, instruction="buscar objetivo anual"),
        SupervisorDecision(reasoning="r", next=ANALYST, instruction="calcular"),  # la guarda lo pisa
        SupervisorDecision(reasoning="r", next=ANALYST, instruction="calcular"),
        SupervisorDecision(reasoning="ok", next=FINISH),
    ])
    research_model = fake_model([AIMessage("Objetivo anual total: $36.500.000"),
                                 AIMessage("[META] Superar los $36.000.000 en el año")])
    graph = build_test_graph(kb, router, research_model, fake_model([AIMessage("faltan $9.238.558")]),
                             fake_model([AIMessage("fin")]))

    final = await run_query(graph, "objetivo anual")

    primero, segundo = final["contributions"][0], final["contributions"][1]
    assert primero["issues"] == ["El monto 36.500.000 no aparece en ninguna fuente."]
    assert segundo["agent"] == RESEARCHER and segundo["issues"] == []
    assert "36.500.000" in segundo["instruction"]
    assert final["contributions"][2]["agent"] == ANALYST
