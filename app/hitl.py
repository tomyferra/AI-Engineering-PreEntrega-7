"""Human-in-the-loop: clasificación de tareas críticas, nodo de aprobación (interrupt) y endpoint de aprobación.

Una tarea es CRÍTICA cuando pide una acción con efectos secundarios irreversibles: enviar el informe a un
destinatario (por el campo `send_report_to` o porque la consulta lo pide, ej. "enviá el informe a ..."). En ese
caso el grafo, después de la síntesis, entra a `aprobacion_humana`, que llama a `interrupt()`: la ejecución se
detiene, el checkpoint queda en Redis y el job pasa a WAITING_APPROVAL. Solo POST /tasks/{id}/approve la
reanuda (`Command(resume=decision)`), aun desde otro proceso o después de reiniciar la API.

    sintetizador --(hay acción pendiente)--> aprobacion_humana --(aprobada)--> ejecutar_accion -> END
                 \\-> END                                      \\-> END (rechazada: no se ejecuta nada)
"""
import re
import unicodedata
from typing import Awaitable, Callable, Literal, Optional

from fastapi import APIRouter, HTTPException, Request, status
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.types import interrupt

from app.config import DEFAULT_REPORT_RECIPIENT
from app.jobs import JobStatus, JobStore
from app.orchestrator.agents.context import user_request
from app.orchestrator.state import APPROVAL, EXECUTE_ACTION, ApprovalDecision, OrchestratorState, PendingAction
from app.schemas import ApprovalAccepted, ApprovalRequest

# --- Clasificación de criticidad -------------------------------------------------------------------------------

# Verbos (en imperativo/infinitivo, sin acentos) que piden enviar algo. No matchea sustantivos como "envío".
_SEND_VERBS = re.compile(
    r"\b(envia|enviar|enviame|enviale|envialo|enviales|manda|mandar|mandale|mandalo|mandame|"
    r"notifica|notificar|notificale|compartilo|compartir|comparti|remiti|remitir)\b")
_REPORT_WORDS = re.compile(r"\b(informe|reporte|resumen|respuesta|resultado|resultados|analisis)\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PREVIEW_CHARS = 1500


def _normalize(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii").lower()


def classify_task(query: str, send_report_to: Optional[str] = None) -> Optional[PendingAction]:
    """Devuelve la acción crítica que pide la tarea, o None si la tarea es de solo lectura."""
    if send_report_to:
        return PendingAction(type="enviar_informe", recipient=send_report_to,
                             reason="La tarea indica un destinatario: enviar el informe es irreversible.")
    normalizada = _normalize(query)
    if _SEND_VERBS.search(normalizada) and _REPORT_WORDS.search(normalizada):
        email = _EMAIL.search(query)
        return PendingAction(type="enviar_informe", recipient=email.group(0) if email else DEFAULT_REPORT_RECIPIENT,
                             reason="La consulta pide enviar el informe a un tercero: es una acción irreversible.")
    return None


# --- Nodos del grafo -------------------------------------------------------------------------------------------

def _parse_decision(value: object) -> ApprovalDecision:
    """Normaliza lo que llega por Command(resume=...). Cualquier cosa que no sea una aprobación explícita rechaza."""
    data = value if isinstance(value, dict) else {}
    return ApprovalDecision(approved=data.get("approved") is True, reviewer=str(data.get("reviewer", "anonimo")),
                            comment=str(data.get("comment", "")))


async def approval_node(state: OrchestratorState) -> dict:
    """Pausa obligatoria: interrupt() corta la ejecución hasta recibir la decisión humana.

    Al reanudar, LangGraph vuelve a ejecutar este nodo desde el principio y `interrupt()` devuelve el valor de
    `Command(resume=...)`; por eso el nodo no tiene efectos secundarios antes del interrupt.
    """
    action = state["pending_action"]
    decision = interrupt({
        "type": "aprobacion_accion_critica",
        "question": f"¿Aprobás {action['type'].replace('_', ' ')} a {action['recipient']}?",
        "action": action,
        "draft": state["final_answer"][:_PREVIEW_CHARS],
        "how_to_resume": "POST /tasks/{job_id}/approve  {\"approved\": true|false, \"reviewer\": \"...\"}",
    })
    return {"approval": _parse_decision(decision)}


def make_execute_action_node(sender: BaseTool) -> Callable[[OrchestratorState], Awaitable[dict]]:
    """Ejecuta la acción crítica ya aprobada con la herramienta de efectos secundarios."""

    async def execute_action(state: OrchestratorState) -> dict:
        action = state["pending_action"]
        result = await sender.ainvoke({
            "destinatario": action["recipient"],
            "asunto": f"Informe GenPet: {user_request(state)[:80]}",
            "cuerpo": state["final_answer"],
        })
        return {"action_result": {"executed": True, "type": action["type"], **result}}

    return execute_action


def route_after_synthesis(state: OrchestratorState) -> Literal["aprobacion_humana", "__end__"]:
    return APPROVAL if state.get("pending_action") else END


def route_after_approval(state: OrchestratorState) -> Literal["ejecutar_accion", "__end__"]:
    approval = state.get("approval")
    return EXECUTE_ACTION if approval and approval["approved"] else END


# --- Endpoint de aprobación ------------------------------------------------------------------------------------

router = APIRouter(tags=["human-in-the-loop"])


@router.post("/tasks/{job_id}/approve", response_model=ApprovalAccepted, status_code=status.HTTP_202_ACCEPTED)
async def approve_task(job_id: str, body: ApprovalRequest, request: Request) -> ApprovalAccepted:
    """Aprueba (o rechaza) la acción crítica de un job en WAITING_APPROVAL y encola su reanudación."""
    store: JobStore = request.app.state.store
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No existe el job {job_id}")
    if not await store.request_resume(job_id, body.model_dump()):
        raise HTTPException(status.HTTP_409_CONFLICT,
                            f"El job {job_id} no está esperando aprobación (estado actual: {job['status']})")
    return ApprovalAccepted(job_id=job_id, status=JobStatus.PENDING, approval=body)
