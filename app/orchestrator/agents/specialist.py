"""Nodo genérico de especialista: envuelve un agente ReAct (create_agent) y registra su aporte en el estado."""
import logging
from typing import Awaitable, Callable, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langgraph.errors import GraphRecursionError

from app.orchestrator.agents.context import filter_contributions, format_contributions, last_ai_text, user_request
from app.orchestrator.agents.validation import Validator
from app.config import SPECIALIST_RECURSION_LIMIT, TOOL_RESULT_MAX_CHARS
from app.orchestrator.state import Contribution, OrchestratorState, SpecialistName, ToolCallRecord

logger = logging.getLogger(__name__)

SpecialistNode = Callable[[OrchestratorState], Awaitable[dict]]


def extract_tool_calls(messages: Sequence[BaseMessage]) -> List[ToolCallRecord]:
    """Empareja cada tool call del sub-agente con su ToolMessage (resultado), recortando resultados largos."""
    results = {m.tool_call_id: m.text for m in messages if isinstance(m, ToolMessage)}
    return [
        ToolCallRecord(tool=tc["name"], args=tc["args"], result=results.get(tc["id"], "")[:TOOL_RESULT_MAX_CHARS])
        for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls
    ]


def build_task_message(objective: str, instruction: str, context: str) -> str:
    return (f"Objetivo general del usuario (solo como referencia):\n{objective}\n\n"
            f"Tu tarea concreta (asignada por el supervisor):\n{instruction}\n\n"
            f"Contexto disponible de aportes previos:\n{context}")


def make_specialist_node(name: SpecialistName, agent: Runnable, visible_agents: Sequence[SpecialistName],
                         validator: Optional[Validator] = None) -> SpecialistNode:
    """Crea el nodo del grafo para un especialista.

    - `agent`: grafo ReAct propio (modelo + tools acotadas) que se invoca con un historial NUEVO por tarea.
    - `visible_agents`: de qué agentes puede ver aportes previos (contexto mínimo necesario).
    - `validator`: chequeo determinista opcional del resultado; sus hallazgos quedan en `issues`.
    Los tool calls internos quedan dentro del sub-agente; al estado compartido solo sube el resultado final.
    """

    async def specialist(state: OrchestratorState) -> dict:
        instruction = state["current_instruction"]
        context = format_contributions(filter_contributions(state["contributions"], visible_agents))
        task = build_task_message(user_request(state), instruction, context)
        try:
            result = await agent.ainvoke({"messages": [HumanMessage(task)]},
                                         config={"recursion_limit": SPECIALIST_RECURSION_LIMIT})
            content = last_ai_text(result["messages"])
            tool_calls = extract_tool_calls(result["messages"])
        except GraphRecursionError:
            logger.warning("El especialista %s superó su límite de pasos", name)
            content = (f"No pude completar la tarea en {SPECIALIST_RECURSION_LIMIT} pasos. "
                       "La tarea debería acotarse más.")
            tool_calls = []

        attempt = 1 + sum(1 for c in state["contributions"] if c["agent"] == name)
        issues = validator(content) if validator else []
        contribution = Contribution(agent=name, attempt=attempt, instruction=instruction,
                                    content=content, tool_calls=tool_calls, issues=issues)
        return {"contributions": [contribution], "messages": [AIMessage(content=content, name=name)]}

    specialist.__name__ = name
    return specialist
