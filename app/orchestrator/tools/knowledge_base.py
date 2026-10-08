"""Base de conocimiento local: búsqueda léxica BM25 sobre los documentos de GenPet (corpus de la Pre-entrega 5).

Simula la "Vector DB de pre-entregas anteriores" sin depender de servicios externos: los documentos se
cortan por sección Markdown (## ...) y se indexan en memoria con BM25.
"""
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from langchain_core.tools import BaseTool, tool
from rank_bm25 import BM25Okapi

from app.config import CHUNK_MAX_CHARS, DOCS_DIR, TOP_K

_STOPWORDS = {
    "a", "al", "con", "cual", "cuales", "cuanto", "de", "del", "el", "en", "es", "esta", "la", "las", "lo",
    "los", "para", "por", "que", "se", "su", "sus", "un", "una", "y", "o", "como", "fue", "son",
}


# Stemming liviano por prefijo: "objetivo/objetivos" -> "objet", "anual/anuales" -> "anual".
STEM_LENGTH = 5


def _stem(token: str) -> str:
    return token[:STEM_LENGTH] if token.isalpha() else token


def tokenize(texto: str) -> List[str]:
    """Minúsculas, sin acentos, sin stopwords y con stemming: "Objetivos anuales" matchea "objetivo anual"."""
    sin_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return [_stem(t) for t in re.findall(r"\w+", sin_acentos.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    source: str
    title: str
    section: str
    content: str


def split_markdown(source: str, text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[Chunk]:
    """Un chunk por sección "## ..."; el título del documento (# ...) viaja en cada chunk como contexto."""
    title_match = re.search(r"^# (.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else source
    chunks: List[Chunk] = []
    for bloque in re.split(r"(?=^## )", text, flags=re.MULTILINE):
        bloque = bloque.strip()
        if not bloque:
            continue
        primera_linea, *cuerpo = bloque.splitlines()
        if primera_linea.startswith("# ") and not "".join(cuerpo).strip():
            continue  # bloque que solo tiene el título del documento: no aporta contenido
        section = primera_linea[3:].strip() if primera_linea.startswith("## ") else "Introducción"
        chunks.append(Chunk(source=source, title=title, section=section, content=bloque[:max_chars]))
    return chunks


class KnowledgeBase:
    """Índice BM25 en memoria sobre los .md de un directorio."""

    def __init__(self, docs_dir: Path = DOCS_DIR):
        paths = sorted(docs_dir.glob("*.md"))
        if not paths:
            raise FileNotFoundError(f"No hay documentos .md en {docs_dir}")
        self.documents: Dict[str, str] = {path.name: path.read_text(encoding="utf-8") for path in paths}
        self.chunks: List[Chunk] = [
            chunk for source, text in self.documents.items() for chunk in split_markdown(source, text)
        ]
        self._bm25 = BM25Okapi([tokenize(f"{c.title} {c.section} {c.content}") for c in self.chunks])

    @property
    def sources(self) -> List[str]:
        return sorted({c.source for c in self.chunks})

    def search(self, query: str, k: int = TOP_K) -> List[dict]:
        scores = self._bm25.get_scores(tokenize(query))
        ranking = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        return [
            {"fuente": self.chunks[i].source, "seccion": self.chunks[i].section,
             "score": round(float(scores[i]), 3), "contenido": self.chunks[i].content}
            for i in ranking[:k] if scores[i] > 0
        ]


def build_knowledge_tools(kb: KnowledgeBase) -> List[BaseTool]:
    """Tools del investigador sobre la base de conocimiento (la KB se inyecta, no vive en el State)."""

    @tool
    def buscar_base_conocimiento(consulta: str) -> dict:
        """Busca en los documentos internos de GenPet (ventas, resultados financieros, plan comercial, políticas,
        productos y datos de la empresa) y devuelve los fragmentos más relevantes con su fuente.

        Usar consultas cortas y específicas, por ejemplo "ingresos netos por trimestre" o "metas trimestrales
        plan comercial". Si los resultados no alcanzan, reformular la consulta con otras palabras clave.
        """
        resultados = kb.search(consulta)
        if not resultados:
            return {"error": "Sin resultados para la consulta.",
                    "accion_sugerida": "Reformular con otras palabras clave o usar listar_documentos.",
                    "documentos_disponibles": kb.sources}
        return {"consulta": consulta, "resultados": resultados}

    @tool
    def listar_documentos() -> dict:
        """Lista los documentos disponibles en la base de conocimiento y sus secciones."""
        secciones: dict = {}
        for chunk in kb.chunks:
            secciones.setdefault(chunk.source, []).append(chunk.section)
        return {"documentos": secciones}

    return [buscar_base_conocimiento, listar_documentos]
