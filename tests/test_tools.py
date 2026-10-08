import pytest

from app.orchestrator.agents.validation import build_grounding_validator, extract_amounts
from app.orchestrator.tools.knowledge_base import build_knowledge_tools, split_markdown, tokenize
from app.orchestrator.tools.math_tools import (
    calcular,
    cumplimiento_de_meta,
    estadisticas_descriptivas,
    safe_eval,
    variacion_porcentual,
)


def test_safe_eval_resuelve_aritmetica():
    assert safe_eval("(7931680 - 4427580) / 4427580 * 100") == pytest.approx(79.1415, rel=1e-4)


@pytest.mark.parametrize("expresion", ["__import__('os').system('dir')", "abs(-1)", "x + 1", "2 ** 1000"])
def test_safe_eval_rechaza_codigo(expresion):
    with pytest.raises(ValueError):
        safe_eval(expresion)


def test_calcular_devuelve_error_accionable():
    resultado = calcular.invoke({"expresion": "$4.427.580 + 1"})
    assert "error" in resultado and "accion_sugerida" in resultado


def test_variacion_y_cumplimiento():
    assert variacion_porcentual.invoke({"valor_inicial": 100, "valor_final": 150})["variacion_pct"] == 50.0
    assert "error" in variacion_porcentual.invoke({"valor_inicial": 0, "valor_final": 10})
    meta = cumplimiento_de_meta.invoke({"valor_real": 4427580, "meta": 4000000})
    assert meta["cumplimiento_pct"] == 110.69 and meta["meta_alcanzada"] is True


def test_estadisticas_descriptivas():
    resultado = estadisticas_descriptivas.invoke({"valores": [1, 2, 3, 4]})
    assert resultado["suma"] == 10 and resultado["promedio"] == 2.5 and resultado["mediana"] == 2.5


def test_tokenize_normaliza_acentos_y_stopwords():
    assert tokenize("La Facturación del Trimestre") == ["factu", "trime"]
    assert tokenize("Objetivos anuales") == tokenize("objetivo anual")


def test_split_markdown_por_seccion():
    chunks = split_markdown("x.md", "# Título\nIntro\n## Ventas\nA\n## Metas\nB")
    assert [c.section for c in chunks] == ["Introducción", "Ventas", "Metas"]
    assert all(c.title == "Título" for c in chunks)


def test_busqueda_encuentra_metas_del_plan(kb):
    resultados = kb.search("metas trimestrales de ingresos netos plan comercial")
    assert resultados[0]["fuente"] == "plan_comercial_2026.md"
    assert "$12.000.000" in resultados[0]["contenido"]


def test_tool_de_busqueda_sin_resultados_sugiere_accion(kb):
    buscar = {t.name: t for t in build_knowledge_tools(kb)}["buscar_base_conocimiento"]
    resultado = buscar.invoke({"consulta": "zzzz qqqq"})
    assert "error" in resultado and resultado["documentos_disponibles"]


def test_extract_amounts_ignora_anios_y_porcentajes():
    assert extract_amounts("Q1 2026: $4.427.580 (79.14%) y 4427580") == ["4.427.580", "4427580"]


def test_validador_de_grounding(kb):
    validar = build_grounding_validator(kb.documents.values())
    assert validar("Ingresos Q3: $14.402.182; pack x20: 1.371 packs; meta: $36.000.000") == []
    assert validar("Objetivo anual: $36.500.000") == ["El monto 36.500.000 no aparece en ninguna fuente."]


def test_busqueda_encuentra_objetivo_anual_pese_a_plural(kb):
    """Regresión: "objetivo anual" debe encontrar la sección "Objetivos anuales" (stemming)."""
    primero = kb.search("objetivo anual de ingresos")[0]
    assert primero["seccion"] == "Objetivos anuales" and "$36.000.000" in primero["contenido"]
