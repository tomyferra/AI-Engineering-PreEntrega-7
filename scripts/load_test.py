"""Prueba de carga: N peticiones concurrentes contra la API, con aprobación humana automática de las críticas.

Uso (con la API y Redis levantados):
    python scripts/load_test.py                    # 5 peticiones concurrentes (2 críticas -> se aprueban)
    python scripts/load_test.py --reject           # las críticas se rechazan (no se envía nada)
    python scripts/load_test.py -n 1 --query "..." # una sola consulta propia

Cada job se sigue por polling (GET /tasks/{id}); si queda en WAITING_APPROVAL el script actúa como el humano y
llama a POST /tasks/{id}/approve. Al final imprime la tabla de resultados, la latencia p50/p95 medida del lado
del cliente y guarda el detalle en results/load_test.json. El costo por ejecución y la latencia p95 de cada traza
se leen en el dashboard de Phoenix (http://localhost:6006).
"""
import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional

import httpx

QUERIES = [
    "¿Cuáles fueron los ingresos netos de cada trimestre de 2026 (Q1 a Q3)? Calculá el crecimiento porcentual "
    "entre trimestres consecutivos.",
    "¿Qué canal de ventas concentra más ingresos en lo que va de 2026 y qué porcentaje del total representa?",
    "Compará los ingresos reales de Q1, Q2 y Q3 con las metas del plan comercial y calculá el porcentaje de "
    "cumplimiento de cada meta. Enviá el informe a gerencia@genpet.com.ar",
    "¿Cuál fue la ganancia neta de cada trimestre y cuál fue el margen neto (ganancia neta / ingresos netos) "
    "de cada uno?",
    "¿Cuánto falta facturar en el Q4 para superar el objetivo anual de ingresos? Mandá el resumen a "
    "direccion@genpet.com.ar",
]
POLL_SECONDS = 2.0
WAITING, TERMINAL = "WAITING_APPROVAL", {"DONE", "FAILED"}


def percentile(values: List[float], pct: float) -> float:
    """Percentil por el método nearest-rank (el mismo criterio que usan los dashboards de trazas)."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


async def run_job(client: httpx.AsyncClient, index: int, query: str, approve: bool, timeout: float) -> dict:
    start = time.perf_counter()
    created = (await client.post("/tasks", json={"query": query})).raise_for_status().json()
    job_id = created["job_id"]
    accepted_ms = (time.perf_counter() - start) * 1000
    print(f"[{index}] encolado {job_id} en {accepted_ms:.0f} ms (crítica: {created['critical']})")

    approval_wait = 0.0
    while True:
        if time.perf_counter() - start > timeout:
            return {"index": index, "job_id": job_id, "status": "CLIENT_TIMEOUT", "query": query}
        job = (await client.get(f"/tasks/{job_id}")).raise_for_status().json()
        if job["status"] == WAITING:
            print(f"[{index}] WAITING_APPROVAL -> {'aprueba' if approve else 'rechaza'} el humano "
                  f"({job['approval_request']['question']})")
            await asyncio.sleep(1)  # el "humano" revisa el borrador
            approval_wait += 1
            (await client.post(f"/tasks/{job_id}/approve",
                               json={"approved": approve, "reviewer": "load-test",
                                     "comment": "aprobado en la prueba de carga" if approve else "rechazado"})
             ).raise_for_status()
        elif job["status"] in TERMINAL:
            total = time.perf_counter() - start
            print(f"[{index}] {job['status']} en {total:.1f}s · flujo: {job.get('flow')}")
            return {"index": index, "job_id": job_id, "status": job["status"], "critical": created["critical"],
                    "accepted_ms": round(accepted_ms, 1), "total_s": round(total, 2),
                    "processing_s": round(total - approval_wait, 2), "flow": job.get("flow"),
                    "action_result": job.get("action_result"), "error": job.get("error"), "query": query,
                    "final_answer": job.get("final_answer")}
        await asyncio.sleep(POLL_SECONDS)


async def main(args: argparse.Namespace) -> int:
    queries = [args.query] * args.n if args.query else [QUERIES[i % len(QUERIES)] for i in range(args.n)]
    async with httpx.AsyncClient(base_url=args.base_url, timeout=30) as client:
        (await client.get("/health")).raise_for_status()
        start = time.perf_counter()
        results = await asyncio.gather(*(run_job(client, i + 1, q, not args.reject, args.timeout)
                                         for i, q in enumerate(queries)))
        wall = time.perf_counter() - start

    print("\n" + "=" * 100)
    print(f"{'#':<3}{'estado':<16}{'crítica':<9}{'encolado (ms)':<15}{'total (s)':<11}{'flujo'}")
    for r in results:
        print(f"{r['index']:<3}{r['status']:<16}{str(r.get('critical', '-')):<9}{r.get('accepted_ms', '-'):<15}"
              f"{r.get('total_s', '-'):<11}{' -> '.join(r.get('flow') or [])}")
        if r.get("error"):
            print(f"   error: {r['error'][:200]}")
    finished = [r["processing_s"] for r in results if r["status"] == "DONE"]
    accepted = [r["accepted_ms"] for r in results if "accepted_ms" in r]
    summary = {"requests": len(results), "done": len(finished),
               "failed": sum(r["status"] != "DONE" for r in results), "wall_clock_s": round(wall, 2),
               "post_latency_ms_max": max(accepted) if accepted else None}
    if finished:
        summary.update(latency_p50_s=percentile(finished, 50), latency_p95_s=percentile(finished, 95))
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"Detalle guardado en {out}")
    return 0 if summary["failed"] == 0 else 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueba de carga concurrente de la API")
    parser.add_argument("-n", type=int, default=5, help="Cantidad de peticiones concurrentes (default 5)")
    parser.add_argument("--query", help="Consulta a repetir (default: 5 consultas distintas, 2 críticas)")
    parser.add_argument("--reject", action="store_true", help="Rechazar las acciones críticas en vez de aprobarlas")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=600, help="Timeout por job del lado del cliente (s)")
    parser.add_argument("--output", default="results/load_test.json")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(asyncio.run(main(parse_args())))
