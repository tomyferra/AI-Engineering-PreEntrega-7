"""Exporta el diagrama del grafo a docs/graph.mmd (Mermaid) y docs/graph.png (requiere internet: mermaid.ink).

Uso:  python scripts/export_diagram.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from app.config import BASE_DIR  # noqa: E402
from app.graph import create_orchestrator  # noqa: E402
from app.orchestrator.tools.report_sender import InMemoryOutbox  # noqa: E402


def main() -> None:
    load_dotenv(BASE_DIR / ".env")
    out_dir = BASE_DIR / "docs"
    out_dir.mkdir(exist_ok=True)
    drawable = create_orchestrator(InMemorySaver(), InMemoryOutbox()).get_graph()
    (out_dir / "graph.mmd").write_text(drawable.draw_mermaid(), encoding="utf-8")
    print(f"Diagrama exportado: {out_dir / 'graph.mmd'}")
    try:
        (out_dir / "graph.png").write_bytes(drawable.draw_mermaid_png())
        print(f"Diagrama exportado: {out_dir / 'graph.png'}")
    except Exception as exc:  # noqa: BLE001 - el PNG es opcional; el .mmd ya documenta el grafo
        logging.warning("No se pudo generar el PNG del grafo: %s", exc)


if __name__ == "__main__":
    main()
