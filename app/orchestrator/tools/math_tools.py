"""Herramientas de cómputo del analista: aritmética segura, variaciones, cumplimiento de metas y estadísticas.

Los LLMs se equivocan con cuentas largas: el analista nunca calcula "de cabeza", delega en estas funciones.
"""
import ast
import operator
import statistics
from typing import Callable, Dict, List, Union

from langchain_core.tools import BaseTool, tool

Number = Union[int, float]

_BIN_OPS: Dict[type, Callable[[Number, Number], Number]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
}
_UNARY_OPS: Dict[type, Callable[[Number], Number]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPONENT = 100


def safe_eval(expression: str) -> float:
    """Evalúa una expresión aritmética recorriendo el AST: sin eval(), sin nombres ni llamadas a funciones."""

    def _eval(node: ast.AST) -> Number:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
                raise ValueError("Exponente demasiado grande")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Elemento no permitido en la expresión: {ast.dump(node)[:60]}")

    return float(_eval(ast.parse(expression, mode="eval").body))


def _round(value: float) -> float:
    return round(value, 2)


@tool
def calcular(expresion: str) -> dict:
    """Evalúa una expresión aritmética (+, -, *, /, **, %, paréntesis) y devuelve el resultado exacto.

    Escribir los números sin separadores de miles ni símbolos: "4427580 + 7931680", no "$4.427.580".
    """
    try:
        return {"expresion": expresion, "resultado": _round(safe_eval(expresion))}
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return {"error": f"Expresión inválida: {exc}",
                "accion_sugerida": "Usar solo números y operadores, sin $ ni puntos de miles."}


@tool
def variacion_porcentual(valor_inicial: float, valor_final: float) -> dict:
    """Variación porcentual entre dos valores: (final - inicial) / inicial * 100. Útil para crecimientos
    entre períodos (mes contra mes, trimestre contra trimestre)."""
    if valor_inicial == 0:
        return {"error": "El valor inicial es 0: la variación porcentual no está definida."}
    variacion = (valor_final - valor_inicial) / abs(valor_inicial) * 100
    return {"valor_inicial": valor_inicial, "valor_final": valor_final,
            "diferencia": _round(valor_final - valor_inicial), "variacion_pct": _round(variacion)}


@tool
def cumplimiento_de_meta(valor_real: float, meta: float) -> dict:
    """Compara un valor real contra una meta: porcentaje de cumplimiento, diferencia y si se alcanzó."""
    if meta == 0:
        return {"error": "La meta es 0: el cumplimiento no está definido."}
    return {"valor_real": valor_real, "meta": meta, "cumplimiento_pct": _round(valor_real / meta * 100),
            "diferencia": _round(valor_real - meta), "meta_alcanzada": valor_real >= meta}


@tool
def estadisticas_descriptivas(valores: List[float]) -> dict:
    """Estadísticas de una serie de números: cantidad, suma, promedio, mediana, mínimo, máximo y desvío
    estándar. Útil para series mensuales o trimestrales."""
    if not valores:
        return {"error": "La lista de valores está vacía."}
    return {
        "n": len(valores), "suma": _round(sum(valores)), "promedio": _round(statistics.fmean(valores)),
        "mediana": _round(statistics.median(valores)), "minimo": min(valores), "maximo": max(valores),
        "desvio_estandar": _round(statistics.stdev(valores)) if len(valores) > 1 else 0.0,
    }


def build_math_tools() -> List[BaseTool]:
    return [calcular, variacion_porcentual, cumplimiento_de_meta, estadisticas_descriptivas]
