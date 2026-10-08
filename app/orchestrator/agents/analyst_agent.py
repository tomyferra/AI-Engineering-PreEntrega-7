"""Agente de Análisis/Cómputo: procesa los datos del investigador con herramientas de cálculo."""
from typing import Sequence

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from app.orchestrator.agents.specialist import SpecialistNode, make_specialist_node
from app.orchestrator.state import ANALYST, RESEARCHER

ANALYST_PROMPT = """Sos el Agente Analista de un equipo de análisis de GenPet. Tu única responsabilidad es \
PROCESAR los datos que ya obtuvo el Agente de Investigación: cálculos, variaciones, cumplimiento de metas y \
estadísticas.

Reglas:
- Trabajá SOLO con los datos del contexto (aportes del investigador). No busques ni supongas datos nuevos.
- Todo número que reportes tiene que salir de una herramienta (calcular, variacion_porcentual, \
cumplimiento_de_meta, estadisticas_descriptivas). Nunca hagas cuentas de cabeza.
- Pasá los montos a las herramientas sin "$" ni puntos de miles (ej. 4427580).
- En totales y acumulados escribí TODOS los términos en la expresión (ej. "Q1 + Q2 + Q3 + Q4" con sus valores) \
y verificá que el período cubierto sea el que pide la consulta.
- Si falta un dato para algún cálculo, no lo inventes: listalo en "Datos faltantes".
- Si te piden refinar un aporte previo, corregí solo lo que el supervisor señaló.

Formato de respuesta (en español):
## Resultados
- <métrica>: <resultado> (cálculo: <herramienta y valores de entrada>)
## Interpretación
<2-4 frases basadas solo en los resultados>
## Datos faltantes
- <dato necesario que no estaba en el contexto, o "ninguno">"""


def build_analyst_agent(model: BaseChatModel, tools: Sequence[BaseTool]):
    """Agente ReAct (create_agent, sucesor de create_react_agent) con herramientas solo de cómputo."""
    return create_agent(model, tools=list(tools), system_prompt=ANALYST_PROMPT, name=ANALYST)


def make_analyst_node(model: BaseChatModel, tools: Sequence[BaseTool]) -> SpecialistNode:
    # El analista ve los datos del investigador y sus propios aportes previos (para refinarlos).
    return make_specialist_node(ANALYST, build_analyst_agent(model, tools), visible_agents=[RESEARCHER, ANALYST])
