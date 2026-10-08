"""Herramientas acotadas por especialista: el investigador solo busca y el analista solo calcula."""
from typing import List, Optional

from langchain_core.tools import BaseTool

from app.orchestrator.tools.knowledge_base import KnowledgeBase, build_knowledge_tools
from app.orchestrator.tools.math_tools import build_math_tools
from app.orchestrator.tools.web_search import build_web_search_tool


def build_research_tools(kb: KnowledgeBase, tavily_api_key: Optional[str] = None) -> List[BaseTool]:
    tools = build_knowledge_tools(kb)
    web = build_web_search_tool(tavily_api_key)
    if web is not None:
        tools.append(web)
    return tools


__all__ = ["KnowledgeBase", "build_math_tools", "build_research_tools"]
