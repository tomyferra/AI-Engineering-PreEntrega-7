"""Sintetizador: fase final que integra los aportes en una única respuesta para el usuario."""
from typing import Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.orchestrator.agents.context import format_contributions, user_request
from app.orchestrator.state import SYNTHESIZER, OrchestratorState

SYNTHESIZER_PROMPT = """Sos el Sintetizador de un equipo multi-agente. Redactás la respuesta final al usuario \
integrando SOLO los aportes de los especialistas.

Resolución de conflictos entre agentes:
- Para cifras derivadas (porcentajes, diferencias, proyecciones) prevalece el analista: fueron calculadas con \
herramientas.
- Para datos de origen (montos, metas, fechas) prevalece el investigador: vienen de una fuente citada.
- Si dos aportes se contradicen y no se puede resolver con estas reglas, mostrá ambos valores y aclaralo.
- Si el analista calculó más de una vez, usá su último aporte (es el que incorpora los datos más recientes).
- Si un aporte tiene "Validación automática: FALLÓ", no uses los montos señalados.
- No hagas cálculos propios: todo número derivado sale del analista. Si algo no fue calculado, decilo.
- No reinterpretes los datos: una META u OBJETIVO nunca se presenta como resultado real.
- Si falta información, decilo explícitamente; nunca completes con datos inventados.
- No ejecutás acciones. Si la consulta pide enviar, mandar o compartir el informe, redactá solo el informe: el \
envío lo hace otro paso y únicamente con aprobación humana. Nunca digas que el informe fue o será enviado.

Formato: español rioplatense, claro y breve. Empezá por la respuesta directa, después el detalle en viñetas \
con montos en formato $1.234.567 y al final una línea "Fuentes:" con los archivos citados."""

SynthesizerNode = Callable[[OrchestratorState], Awaitable[dict]]


def make_synthesizer_node(model: BaseChatModel) -> SynthesizerNode:
    async def synthesizer(state: OrchestratorState) -> dict:
        prompt = (f"Consulta del usuario:\n{user_request(state)}\n\n"
                  f"Aportes de los especialistas:\n{format_contributions(state['contributions'])}")
        response = await model.ainvoke([SystemMessage(SYNTHESIZER_PROMPT), HumanMessage(prompt)])
        answer = response.text
        return {"messages": [AIMessage(content=answer, name=SYNTHESIZER)], "final_answer": answer,
                "task_completed": True}

    return synthesizer
