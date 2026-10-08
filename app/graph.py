"""Grafo del orquestador multi-agente (Pre-entrega 6) + Human-in-the-loop + checkpointer de Redis.

    START -> supervisor --(route_from_supervisor)--> investigador -> supervisor   (ciclo de delegación)
                                                \\-> analista     -> supervisor
                                                \\-> sintetizador --(route_after_synthesis)--> END
                                                                   \\-> aprobacion_humana  [interrupt]

    aprobacion_humana --(route_after_approval)--> ejecutar_accion -> END   (aprobada)
                                             \\-> END                       (rechazada)

El checkpointer (AsyncRedisSaver) guarda el estado completo después de cada super-step con `thread_id = job_id`:
es lo que permite pausar en `aprobacion_humana` y reanudar más tarde, desde cualquier worker.
"""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import LLMSettings, get_llm_settings, get_tavily_api_key
from app.hitl import approval_node, make_execute_action_node, route_after_approval, route_after_synthesis
from app.orchestrator.agents import (
    build_supervisor_router,
    make_analyst_node,
    make_research_node,
    make_supervisor_node,
    make_synthesizer_node,
    route_from_supervisor,
)
from app.orchestrator.agents.specialist import SpecialistNode
from app.orchestrator.agents.supervisor import SupervisorNode
from app.orchestrator.agents.synthesizer import SynthesizerNode
from app.orchestrator.agents.validation import build_grounding_validator
from app.orchestrator.llm import build_chat_model
from app.orchestrator.state import (
    ANALYST,
    APPROVAL,
    EXECUTE_ACTION,
    RESEARCHER,
    SUPERVISOR,
    SYNTHESIZER,
    OrchestratorState,
)
from app.orchestrator.tools import KnowledgeBase, build_math_tools, build_research_tools
from app.orchestrator.tools.report_sender import Outbox, build_report_sender_tool

Node = Callable[[OrchestratorState], Awaitable[dict]]


@dataclass(frozen=True)
class OrchestratorNodes:
    supervisor: SupervisorNode
    researcher: SpecialistNode
    analyst: SpecialistNode
    synthesizer: SynthesizerNode
    execute_action: Node


def build_graph(nodes: OrchestratorNodes, checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    """Une los nodos. Recibe nodos y checkpointer ya construidos (inyección de dependencias): testeable sin APIs."""
    workflow = StateGraph(OrchestratorState)

    workflow.add_node(SUPERVISOR, nodes.supervisor)
    workflow.add_node(RESEARCHER, nodes.researcher)
    workflow.add_node(ANALYST, nodes.analyst)
    workflow.add_node(SYNTHESIZER, nodes.synthesizer)
    workflow.add_node(APPROVAL, approval_node)
    workflow.add_node(EXECUTE_ACTION, nodes.execute_action)

    workflow.add_edge(START, SUPERVISOR)
    workflow.add_conditional_edges(SUPERVISOR, route_from_supervisor,
                                   {RESEARCHER: RESEARCHER, ANALYST: ANALYST, SYNTHESIZER: SYNTHESIZER})
    # Topología jerárquica: los especialistas nunca hablan entre sí, siempre reportan al supervisor.
    workflow.add_edge(RESEARCHER, SUPERVISOR)
    workflow.add_edge(ANALYST, SUPERVISOR)
    # Human-in-the-loop: solo las tareas críticas pasan por la aprobación antes de ejecutar la acción.
    workflow.add_conditional_edges(SYNTHESIZER, route_after_synthesis, {APPROVAL: APPROVAL, END: END})
    workflow.add_conditional_edges(APPROVAL, route_after_approval, {EXECUTE_ACTION: EXECUTE_ACTION, END: END})
    workflow.add_edge(EXECUTE_ACTION, END)

    return workflow.compile(checkpointer=checkpointer)


def build_nodes(model: BaseChatModel, kb: KnowledgeBase, outbox: Outbox,
                tavily_api_key: Optional[str] = None) -> OrchestratorNodes:
    return OrchestratorNodes(
        supervisor=make_supervisor_node(build_supervisor_router(model)),
        researcher=make_research_node(model, build_research_tools(kb, tavily_api_key),
                                      validator=build_grounding_validator(kb.documents.values())),
        analyst=make_analyst_node(model, build_math_tools()),
        synthesizer=make_synthesizer_node(model),
        execute_action=make_execute_action_node(build_report_sender_tool(outbox)),
    )


def create_orchestrator(checkpointer: BaseCheckpointSaver, outbox: Outbox,
                        settings: Optional[LLMSettings] = None) -> CompiledStateGraph:
    """Orquestador listo para usar con el LLM de .env (llamar después de load_dotenv)."""
    model = build_chat_model(settings or get_llm_settings())
    return build_graph(build_nodes(model, KnowledgeBase(), outbox, get_tavily_api_key()), checkpointer)


@asynccontextmanager
async def redis_checkpointer(redis_url: str, ttl_seconds: int) -> AsyncIterator[AsyncRedisSaver]:
    """RedisSaver asíncrono (requiere Redis Stack: RedisJSON + RediSearch). Los checkpoints expiran con el job."""
    ttl = {"default_ttl": max(1, ttl_seconds // 60), "refresh_on_read": True}  # default_ttl va en minutos
    async with AsyncRedisSaver.from_conn_string(redis_url, ttl=ttl) as saver:  # __aenter__ crea los índices
        yield saver
