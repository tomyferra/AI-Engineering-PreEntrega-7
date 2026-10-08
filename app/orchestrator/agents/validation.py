"""Validación determinista de los aportes del investigador: cada monto reportado debe existir en las fuentes.

Complementa la rúbrica del supervisor (que es un LLM y puede pasar por alto un número inventado o una suma
que el investigador hizo por su cuenta).
"""
import re
from typing import Callable, Iterable, List, Set

# Montos con separadores de miles (4.427.580 / 4,427,580) o números de 5+ dígitos (4427580).
_AMOUNT = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:[.,]\d{3})+|\d{5,})(?![\d]|[.,]\d)")
_ANY_NUMBER = re.compile(r"\d[\d.,]*\d")

Validator = Callable[[str], List[str]]


def _normalize(amount: str) -> str:
    return re.sub(r"[.,]", "", amount)


def extract_amounts(text: str) -> List[str]:
    return _AMOUNT.findall(text)


def build_grounding_validator(source_texts: Iterable[str]) -> Validator:
    """Devuelve una función que lista los montos del texto que NO aparecen en ninguna fuente."""
    # En las fuentes se indexa cualquier secuencia numérica (incluye "1371" sin separador), en forma normalizada.
    known: Set[str] = {_normalize(n) for text in source_texts for n in _ANY_NUMBER.findall(text)}

    def validate(text: str) -> List[str]:
        faltantes = []
        for amount in extract_amounts(text):
            if _normalize(amount) not in known and amount not in faltantes:
                faltantes.append(amount)
        return [f"El monto {a} no aparece en ninguna fuente." for a in faltantes]

    return validate
