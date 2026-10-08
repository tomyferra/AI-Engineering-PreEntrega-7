"""Estado compartido del orquestador multi-agente.

Hereda de MessagesState (conversación con el usuario) y agrega campos estructurados para saber
qué agente aportó qué información, qué decidió el supervisor en cada paso y cuándo cortar el flujo.
En la Pre-entrega 7 suma los campos del flujo Human-in-the-loop (acción crítica pendiente, decisión humana y
resultado de la acción). Todo el estado se persiste en Redis con el checkpointer (AsyncRedisSaver).
"""
import operator
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState

# Nombres de los nodos del grafo: son el "vocabulario" con el que el supervisor enruta.
SUPERVISOR = "supervisor"
RESEARCHER = "investigador"
ANALYST = "analista"
SYNTHESIZER = "sintetizador"
# Nodos del flujo Human-in-the-loop (después de la síntesis).
APPROVAL = "aprobacion_humana"
EXECUTE_ACTION = "ejecutar_accion"

SpecialistName = Literal["investigador", "analista"]
# Destinos posibles de la arista condicional del supervisor.
NextNode = Literal["investigador", "analista", "sintetizador"]


class ToolCallRecord(TypedDict):
    """Una llamada real a una herramienta dentro del especialista (evidencia verificable, no autodeclarada)."""

    tool: str
    args: dict
    result: str


class Contribution(TypedDict):
    """Aporte de un especialista: trazabilidad de quién produjo qué, con qué instrucción y con qué tools."""

    agent: SpecialistName
    attempt: int  # 1 = primer intento; >1 = refinamiento pedido por el supervisor
    instruction: str
    content: str
    tool_calls: List[ToolCallRecord]
    issues: List[str]  # problemas detectados por la validación determinista (vacío = aporte validado)


class RoutingDecision(TypedDict):
    """Decisión del supervisor en un paso. `forced` indica que una guarda de seguridad pisó al LLM."""

    step: int
    next: NextNode
    instruction: str
    reasoning: str
    forced: bool


class PendingAction(TypedDict):
    """Acción con efectos secundarios que la consulta pide ejecutar (crítica: requiere aprobación humana)."""

    type: Literal["enviar_informe"]
    recipient: str
    reason: str  # por qué el sistema la clasificó como crítica


class ApprovalDecision(TypedDict):
    """Decisión humana recibida por POST /tasks/{id}/approve (se inyecta al grafo con Command(resume=...))."""

    approved: bool
    reviewer: str
    comment: str


class OrchestratorState(MessagesState):
    """Estado del grafo.

    - messages: pregunta del usuario y respuesta final (reducer add_messages). Los mensajes internos de los
      especialistas (tool calls, observaciones) NO se vuelcan acá: se resumen en `contributions`.
    - contributions / decisions: listas acumulativas (reducer operator.add) -> nunca se pierde un aporte.
    - next_agent / current_instruction: el "despacho" del supervisor hacia el próximo nodo.
    - delegations: contador de delegaciones, base de la condición de parada.
    - task_completed / final_answer: cierre del flujo por el sintetizador.
    - pending_action / approval / action_result: flujo Human-in-the-loop de las acciones críticas.
    """

    next_agent: NextNode
    current_instruction: str
    contributions: Annotated[List[Contribution], operator.add]
    decisions: Annotated[List[RoutingDecision], operator.add]
    delegations: int
    task_completed: bool
    final_answer: str
    pending_action: Optional[PendingAction]
    approval: Optional[ApprovalDecision]
    action_result: Optional[dict]


def initial_state(question: str, pending_action: Optional[PendingAction] = None) -> dict:
    """Estado inicial con todos los campos que leen los nodos."""
    return {
        "messages": [HumanMessage(question)],
        "next_agent": SYNTHESIZER,
        "current_instruction": "",
        "contributions": [],
        "decisions": [],
        "delegations": 0,
        "task_completed": False,
        "final_answer": "",
        "pending_action": pending_action,
        "approval": None,
        "action_result": None,
    }
