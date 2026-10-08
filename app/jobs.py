"""Estado de los jobs y cola de trabajo, ambos en Redis (cliente asíncrono: nunca bloquea el event loop).

- Job: hash `<prefijo>:job:<id>` con cada campo serializado en JSON. Expira a los `job_ttl_seconds`.
- Cola: lista `<prefijo>:queue` (RPUSH para encolar, BLPOP para consumir). Cada item es
  `{"job_id": ..., "kind": "start" | "resume", "resume": {...}}`: el mismo worker arranca un grafo nuevo o lo
  reanuda después de la aprobación humana. Como la cola vive en Redis, el worker puede correr dentro de la API
  o en un proceso aparte (`python -m app.worker`).

Ciclo de vida: PENDING -> RUNNING -> (WAITING_APPROVAL -> PENDING -> RUNNING) -> DONE | FAILED
"""
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional, TypedDict

from redis.asyncio import Redis
from redis.exceptions import WatchError


class JobStatus(str, Enum):
    PENDING = "PENDING"                    # encolado, esperando un worker libre
    RUNNING = "RUNNING"                    # el grafo se está ejecutando
    WAITING_APPROVAL = "WAITING_APPROVAL"  # pausado en el nodo Human-in-the-loop
    DONE = "DONE"                          # terminó (con o sin acción ejecutada)
    FAILED = "FAILED"                      # excepción, timeout o límite de pasos


TERMINAL_STATUSES = {JobStatus.DONE, JobStatus.FAILED}


class QueueItem(TypedDict, total=False):
    job_id: str
    kind: Literal["start", "resume"]
    resume: dict


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class JobStore:
    def __init__(self, redis: Redis, prefix: str = "genpet", ttl_seconds: int = 7 * 24 * 3600):
        self.redis = redis
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds

    @property
    def queue_key(self) -> str:
        return f"{self.prefix}:queue"

    @property
    def outbox_key(self) -> str:
        return f"{self.prefix}:outbox"

    def job_key(self, job_id: str) -> str:
        return f"{self.prefix}:job:{job_id}"

    @staticmethod
    def _encode(fields: dict) -> dict:
        return {k: json.dumps(v, ensure_ascii=False) for k, v in fields.items()}

    async def create(self, query: str, pending_action: Optional[dict] = None) -> dict:
        """Crea el job en PENDING y lo encola en una sola transacción (no queda un job sin encolar)."""
        job_id = uuid.uuid4().hex
        now = utc_now()
        job = {"job_id": job_id, "status": JobStatus.PENDING.value, "query": query, "created_at": now,
               "updated_at": now, "pending_action": pending_action}
        item: QueueItem = {"job_id": job_id, "kind": "start"}
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hset(self.job_key(job_id), mapping=self._encode(job))
            pipe.expire(self.job_key(job_id), self.ttl_seconds)
            pipe.rpush(self.queue_key, json.dumps(item))
            await pipe.execute()
        return job

    async def get(self, job_id: str) -> Optional[dict]:
        raw = await self.redis.hgetall(self.job_key(job_id))
        if not raw:
            return None
        return {k: json.loads(v) for k, v in raw.items()}

    async def update(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = utc_now()
        await self.redis.hset(self.job_key(job_id), mapping=self._encode(fields))

    async def set_status(self, job_id: str, status: JobStatus, **fields: Any) -> None:
        await self.update(job_id, status=status.value, **fields)

    async def dequeue(self, timeout: float = 1.0) -> Optional[QueueItem]:
        """Espera hasta `timeout` segundos un item de la cola (BLPOP no bloquea el loop: es I/O async)."""
        popped = await self.redis.blpop([self.queue_key], timeout=timeout)
        if popped is None:
            return None
        _key, payload = popped
        return json.loads(payload)

    async def request_resume(self, job_id: str, decision: dict) -> bool:
        """Registra la decisión humana y encola la reanudación, solo si el job está en WAITING_APPROVAL.

        Compare-and-set con WATCH/MULTI: dos aprobaciones simultáneas no reanudan el grafo dos veces.
        Devuelve False si el job no estaba esperando aprobación.
        """
        key = self.job_key(job_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            while True:
                try:
                    await pipe.watch(key)
                    status = await pipe.hget(key, "status")
                    if status is None or json.loads(status) != JobStatus.WAITING_APPROVAL.value:
                        await pipe.unwatch()
                        return False
                    item: QueueItem = {"job_id": job_id, "kind": "resume", "resume": decision}
                    pipe.multi()
                    pipe.hset(key, mapping=self._encode({"status": JobStatus.PENDING.value, "approval": decision,
                                                         "updated_at": utc_now()}))
                    pipe.rpush(self.queue_key, json.dumps(item, ensure_ascii=False))
                    await pipe.execute()
                    return True
                except WatchError:
                    continue  # otro cliente modificó el job entre el WATCH y el EXEC: se reintenta
