from typing import Any, Iterable, List

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.orchestrator.agents.supervisor import SupervisorDecision
from app.config import RECURSION_LIMIT
from app.orchestrator.state import initial_state
from app.orchestrator.tools import KnowledgeBase


class FakeToolCallingModel(FakeMessagesListChatModel):
    """LLM falso con respuestas guionadas: permite testear el grafo sin API keys ni costos.

    Guarda los mensajes recibidos en cada llamada para verificar qué contexto vio cada agente.
    """

    received: List[List[BaseMessage]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeToolCallingModel":
        return self

    def _generate(self, messages: List[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))
        resultado = super()._generate(messages, *args, **kwargs)
        # Copia por llamada: si se devolviera siempre el mismo objeto, add_messages lo deduplicaría por id.
        mensaje = resultado.generations[0].message.model_copy(deep=True)
        mensaje.id = None
        return ChatResult(generations=[ChatGeneration(message=mensaje)])


def fake_model(responses: Iterable[BaseMessage]) -> FakeToolCallingModel:
    return FakeToolCallingModel(responses=list(responses), received=[])


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def scripted_router(decisions: Iterable[SupervisorDecision]) -> RunnableLambda:
    """Router del supervisor con decisiones guionadas (reemplaza a model.with_structured_output)."""
    pendientes = iter(decisions)
    return RunnableLambda(lambda _messages: next(pendientes))


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase()


async def run_query(graph, question: str, pending_action=None, thread_id: str = "t1") -> dict:
    """Ejecuta el grafo como lo hace el worker (thread_id + recursion_limit) y devuelve el estado final."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    return await graph.ainvoke(initial_state(question, pending_action), config=config)
