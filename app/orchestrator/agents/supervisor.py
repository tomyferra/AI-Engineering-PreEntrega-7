"""Supervisor: router inteligente y controlador de flujo (con rúbrica de validación y guardas de parada)."""
import logging
from typing import Awaitable, Callable, List, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from app.orchestrator.agents.context import format_contributions, user_request
from app.config import MAX_CALLS_PER_AGENT, MAX_DELEGATIONS
from app.orchestrator.state import ANALYST, RESEARCHER, SYNTHESIZER, NextNode, OrchestratorState, RoutingDecision

logger = logging.getLogger(__name__)

FINISH = "FINALIZAR"

SUPERVISOR_PROMPT = f"""Sos el Supervisor de un equipo multi-agente de análisis e investigación sobre GenPet.
Dada la consulta del usuario y los aportes actuales, decidí quién debe intervenir ahora o si es momento de \
finalizar.

Especialistas disponibles:
- "{RESEARCHER}": busca datos en los documentos internos de GenPet (ventas, finanzas, plan comercial, \
políticas, productos). No calcula.
- "{ANALYST}": hace cálculos sobre los datos que YA obtuvo el investigador (variaciones %, cumplimiento de \
metas, sumas, estadísticas). No busca datos.
- "{FINISH}": los aportes son suficientes; pasa al sintetizador que redacta la respuesta final.

Flujo esperado: primero "{RESEARCHER}" consigue los datos, después "{ANALYST}" los procesa (si hacen falta \
cálculos) y por último "{FINISH}". Nunca mandes al analista sin datos del investigador.

Rúbrica de validación (antes de elegir "{FINISH}" TODOS los puntos deben cumplirse):
1. Cobertura: cada parte de la consulta tiene datos que la respaldan.
2. Trazabilidad: los datos del investigador citan su fuente y su "Validación automática" es OK (si FALLÓ, \
reportó montos que no están en las fuentes: hay que pedirle que los corrija).
3. Cómputo verificado: cada número derivado (porcentajes, diferencias, totales) fue calculado por el analista \
con herramientas. Revisá su "Registro de herramientas" (llamadas reales con argumentos y resultados): cada \
cálculo tiene que usar los valores correctos y completos (por ejemplo, un total anual suma TODOS los trimestres) \
y el texto del analista tiene que coincidir con esos resultados.
4. Consistencia: los valores que usó el analista coinciden con los que reportó el investigador, y ningún dato \
marcado como META se usa como si fuera un resultado REAL. Si el investigador aportó datos nuevos después del \
último aporte del analista, el analista tiene que recalcular con ellos.
Si un punto falla, delegá un refinamiento al especialista responsable con una instrucción que diga \
exactamente qué corregir o completar.

Reglas para las instrucciones:
- La primera instrucción al investigador enumera TODOS los datos que la consulta necesita (valores reales, \
metas, objetivos anuales, etc.).
- Nunca supongas ni inventes valores o definiciones (por ejemplo "asumí que el objetivo anual es la suma de..."): \
si falta un dato, pedíselo al investigador.
- Al analista pasale los valores exactos a usar y qué calcular con cada uno.

Criterio de suficiencia (evita bucles):
- Si un dato figura en "Datos no encontrados" de un aporte del investigador, NO existe en las fuentes: no lo \
vuelvas a pedir. Seguí con lo disponible (el sintetizador aclarará el faltante).
- Los períodos que todavía no terminaron (por ejemplo, trimestres futuros) no tienen resultados reales, solo \
metas: no los pidas como datos reales.
- No pidas refinamientos por estilo o formato.
- Si la consulta pide cálculos y ya hay datos del investigador, el próximo paso es el analista.
- Si la consulta no requiere cálculos, no llames al analista.

La instrucción para el especialista debe ser autocontenida y concreta (qué datos buscar o qué cálculos hacer, \
con los valores a usar). Si elegís "{FINISH}", dejá la instrucción vacía."""


class SupervisorDecision(BaseModel):
    """Salida estructurada del supervisor: los valores de `next` son los nombres de los nodos del grafo."""

    reasoning: str = Field(description="Evaluación breve de los aportes contra la rúbrica")
    next: Literal["investigador", "analista", "FINALIZAR"] = Field(description="Próximo nodo a ejecutar")
    instruction: str = Field(default="", description="Instrucción concreta para el especialista elegido")


SupervisorRouter = Runnable  # Runnable[list[BaseMessage], SupervisorDecision]
SupervisorNode = Callable[[OrchestratorState], Awaitable[dict]]


def build_supervisor_router(model: BaseChatModel) -> SupervisorRouter:
    return model.with_structured_output(SupervisorDecision)


def build_supervisor_messages(state: OrchestratorState) -> list:
    """El supervisor ve la consulta, los aportes con su evidencia (tool calls) y el presupuesto restante.

    No recibe el historial crudo de mensajes de los especialistas (razonamientos intermedios, prompts internos).
    """
    restantes = MAX_DELEGATIONS - state["delegations"]
    resumen = (f"Consulta del usuario:\n{user_request(state)}\n\n"
               f"Aportes de los especialistas hasta ahora:\n"
               f"{format_contributions(state['contributions'], with_evidence=True)}\n\n"
               f"Delegaciones restantes: {restantes} de {MAX_DELEGATIONS}.")
    return [SystemMessage(SUPERVISOR_PROMPT), HumanMessage(resumen)]


ANALYZE_AVAILABLE_INSTRUCTION = ("Con los datos disponibles del investigador, calculá todo lo que pide la consulta "
                                 "del usuario. Lo que no se pueda calcular por falta de datos, listalo como faltante.")
FIX_GROUNDING_INSTRUCTION = ("Corregí tu aporte anterior: {issues} Reportá solo valores que figuren literalmente "
                             "en las fuentes; lo que no figure (por ejemplo, un total que habría que calcular) "
                             "va en \"Datos no encontrados\".")
RECOMPUTE_INSTRUCTION = ("Recalculá todos los resultados que pide la consulta del usuario usando los datos más "
                         "recientes del investigador (su último aporte corrige o completa los anteriores).")


def _calls(state: OrchestratorState, agent: str) -> int:
    return sum(1 for c in state["contributions"] if c["agent"] == agent)


def _has_budget(state: OrchestratorState, agent: str) -> bool:
    return state["delegations"] < MAX_DELEGATIONS and _calls(state, agent) < MAX_CALLS_PER_AGENT


def _unvalidated_research(state: OrchestratorState) -> List[str]:
    """Problemas del último aporte, si es del investigador y no pasó la validación (aún no fue corregido)."""
    contributions = state["contributions"]
    if contributions and contributions[-1]["agent"] == RESEARCHER:
        return contributions[-1]["issues"]
    return []


def _analysis_is_stale(state: OrchestratorState) -> bool:
    """El analista ya calculó, pero el investigador aportó datos nuevos después: los cálculos pueden no valer."""
    agents = [c["agent"] for c in state["contributions"]]
    return ANALYST in agents and agents[-1] == RESEARCHER


def apply_guards(decision: SupervisorDecision, state: OrchestratorState) -> RoutingDecision:
    """Guardas deterministas sobre la decisión del LLM: el corte y la consistencia no dependen solo del modelo."""
    step = len(state["decisions"]) + 1
    target: NextNode = SYNTHESIZER if decision.next == FINISH else decision.next
    instruction, reasoning, forced = decision.instruction, decision.reasoning, False
    issues = _unvalidated_research(state)

    if issues and target != RESEARCHER and _has_budget(state, RESEARCHER):
        # Validación: no se avanza con montos que no salen de las fuentes.
        target, forced = RESEARCHER, True
        reasoning = "El último aporte del investigador no pasó la validación automática: debe corregirlo."
        instruction = FIX_GROUNDING_INSTRUCTION.format(issues=" ".join(issues))
    elif target == SYNTHESIZER:
        if _analysis_is_stale(state) and _has_budget(state, ANALYST):
            target, instruction, forced = ANALYST, RECOMPUTE_INSTRUCTION, True
            reasoning = "Hay datos nuevos del investigador posteriores al análisis: el analista debe recalcular."
    elif state["delegations"] >= MAX_DELEGATIONS:
        reasoning = f"Límite de {MAX_DELEGATIONS} delegaciones alcanzado: se sintetiza lo disponible."
        target, forced = SYNTHESIZER, True
    elif _calls(state, target) >= MAX_CALLS_PER_AGENT:
        reasoning = f"{target} ya intervino {_calls(state, target)} veces (máximo {MAX_CALLS_PER_AGENT})"
        forced = True
        if target == RESEARCHER and _calls(state, ANALYST) == 0:
            # Los datos ya están, pero nadie los procesó: se analiza lo disponible antes de sintetizar.
            target, instruction = ANALYST, ANALYZE_AVAILABLE_INSTRUCTION
            reasoning += ": se analizan los datos disponibles."
        else:
            target = SYNTHESIZER
            reasoning += ": se sintetiza lo disponible."
    elif target == ANALYST and _calls(state, RESEARCHER) == 0:
        # El analista sin datos solo podría inventar: primero tiene que investigar alguien.
        target, reasoning, forced = RESEARCHER, "El analista no tiene datos todavía: se investiga primero.", True

    instruction = "" if target == SYNTHESIZER else (instruction or "Completá la consulta del usuario.")
    return RoutingDecision(step=step, next=target, instruction=instruction, reasoning=reasoning, forced=forced)


def make_supervisor_node(router: SupervisorRouter) -> SupervisorNode:
    async def supervisor(state: OrchestratorState) -> dict:
        try:
            decision = await router.ainvoke(build_supervisor_messages(state))
            if decision is None:
                raise ValueError("el modelo no devolvió una decisión estructurada")
        except ValueError as exc:
            # Salida estructurada inválida (OutputParserException y ValidationError heredan de ValueError):
            # se cierra con lo disponible. Los errores de infraestructura (API, red, créditos) se propagan.
            logger.warning("Decisión inválida del supervisor (%s): se pasa a la síntesis", exc)
            decision = SupervisorDecision(reasoning=f"Decisión inválida del supervisor: {exc}", next=FINISH)

        routing = apply_guards(decision, state)
        update = {"next_agent": routing["next"], "current_instruction": routing["instruction"],
                  "decisions": [routing]}
        if routing["next"] != SYNTHESIZER:
            update["delegations"] = state["delegations"] + 1
        return update

    return supervisor


def route_from_supervisor(state: OrchestratorState) -> NextNode:
    """Función de la arista condicional: el retorno tipado con Literal mapea 1:1 a los nodos del grafo."""
    return state["next_agent"]
