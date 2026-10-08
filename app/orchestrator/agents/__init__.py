"""Nodos del orquestador: supervisor, especialistas (investigación y análisis) y sintetizador."""
from app.orchestrator.agents.analyst_agent import make_analyst_node
from app.orchestrator.agents.research_agent import make_research_node
from app.orchestrator.agents.supervisor import build_supervisor_router, make_supervisor_node, route_from_supervisor
from app.orchestrator.agents.synthesizer import make_synthesizer_node

__all__ = [
    "build_supervisor_router",
    "make_analyst_node",
    "make_research_node",
    "make_supervisor_node",
    "make_synthesizer_node",
    "route_from_supervisor",
]
