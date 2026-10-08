"""Herramienta con efectos secundarios: envía el informe final a un destinatario (bandeja de salida en Redis).

Es la acción "crítica" del sistema: una vez enviado, el informe no se puede retirar. Por eso solo la ejecuta el
nodo `ejecutar_accion`, que el grafo alcanza únicamente después de la aprobación humana (ver app/hitl.py).
El envío es simulado: el mensaje queda en una lista de Redis (`<prefijo>:outbox`) que haría las veces de cola
de un servicio de email real.
"""
import json
import uuid
from datetime import datetime, timezone
from typing import List, Protocol

from langchain_core.tools import BaseTool, tool
from redis.asyncio import Redis


class Outbox(Protocol):
    async def send(self, recipient: str, subject: str, body: str) -> dict: ...


def _message(recipient: str, subject: str, body: str) -> dict:
    return {"message_id": uuid.uuid4().hex, "recipient": recipient, "subject": subject, "body": body,
            "sent_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


class RedisOutbox:
    """Bandeja de salida persistida en Redis: cada envío es un RPUSH (efecto secundario observable)."""

    def __init__(self, redis: Redis, key: str):
        self._redis = redis
        self.key = key

    async def send(self, recipient: str, subject: str, body: str) -> dict:
        message = _message(recipient, subject, body)
        await self._redis.rpush(self.key, json.dumps(message, ensure_ascii=False))
        return {k: v for k, v in message.items() if k != "body"}


class InMemoryOutbox:
    """Bandeja en memoria para tests."""

    def __init__(self) -> None:
        self.sent: List[dict] = []

    async def send(self, recipient: str, subject: str, body: str) -> dict:
        message = _message(recipient, subject, body)
        self.sent.append(message)
        return {k: v for k, v in message.items() if k != "body"}


def build_report_sender_tool(outbox: Outbox) -> BaseTool:
    @tool
    async def enviar_informe(destinatario: str, asunto: str, cuerpo: str) -> dict:
        """Envía el informe final por email al destinatario. ACCIÓN IRREVERSIBLE: requiere aprobación humana."""
        return await outbox.send(destinatario, asunto, cuerpo)

    return enviar_informe
