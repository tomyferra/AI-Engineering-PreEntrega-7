"""Modelos Pydantic de entrada y salida de la API."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.jobs import JobStatus

EMAIL_PATTERN = r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$"


class TaskCreate(BaseModel):
    query: str = Field(min_length=5, max_length=2000, description="Consulta para el orquestador multi-agente",
                       examples=["¿Cuáles fueron los ingresos netos de cada trimestre de 2026 y cuánto crecieron?"])
    send_report_to: Optional[str] = Field(
        default=None, pattern=EMAIL_PATTERN,
        description="Si se indica, el informe final se envía a este email (acción crítica: requiere aprobación)")


class TaskAccepted(BaseModel):
    job_id: str
    status: JobStatus
    critical: bool = Field(description="True si la tarea incluye una acción que va a requerir aprobación humana")
    links: Dict[str, str]


class ApprovalRequest(BaseModel):
    approved: bool = Field(default=True, description="True aprueba la acción crítica; False la rechaza")
    reviewer: str = Field(default="anonimo", max_length=100)
    comment: str = Field(default="", max_length=1000)


class ApprovalAccepted(BaseModel):
    job_id: str
    status: JobStatus
    approval: ApprovalRequest


class TaskView(BaseModel):
    """Estado del job tal como está en Redis."""

    job_id: str
    status: JobStatus
    query: str
    created_at: str
    updated_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    pending_action: Optional[Dict[str, Any]] = None
    approval_request: Optional[Dict[str, Any]] = Field(
        default=None, description="Payload del interrupt mientras el job espera aprobación")
    approval: Optional[Dict[str, Any]] = None
    action_result: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    delegations: Optional[int] = None
    flow: Optional[List[str]] = Field(default=None, description="Secuencia de nodos que decidió el supervisor")
    error: Optional[str] = None
