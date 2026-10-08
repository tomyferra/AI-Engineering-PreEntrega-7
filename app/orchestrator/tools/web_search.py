"""Búsqueda web opcional con Tavily (solo se habilita si hay TAVILY_API_KEY)."""
from typing import Optional

from langchain_core.tools import BaseTool


def build_web_search_tool(api_key: Optional[str], max_results: int = 3) -> Optional[BaseTool]:
    if not api_key:
        return None
    from langchain_tavily import TavilySearch

    return TavilySearch(max_results=max_results, tavily_api_key=api_key,
                        description="Busca en la web información pública y actual (mercado, inflación, "
                                    "competencia). Usar solo si la base de conocimiento interna no alcanza.")
