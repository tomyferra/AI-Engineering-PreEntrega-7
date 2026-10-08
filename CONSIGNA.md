# Pre-entrega 7: API de producción y monitoreo activo

## ¿Qué construir?

Debes entregar el código fuente de una **API REST (FastAPI)** que exponga tu **sistema multi-agente desarrollado en
el Módulo 6**, con las siguientes capacidades nuevas:

1. **Endpoints asíncronos:** la API no debe bloquearse. Debe recibir la tarea, encolarla y devolver un ID.
2. **Gestión de estado con Redis:** uso de Redis para persistir el estado de los trabajos y los checkpoints de
   LangGraph.
3. **Capa de observabilidad:** integración activa de instrumentación para enviar trazas de cada ejecución a una
   plataforma de monitoreo.
4. **Flujo Human-in-the-loop (HITL):** implementación de una pausa obligatoria para aprobación humana en tareas que
   el sistema identifique como **"críticas"** (por ejemplo, la ejecución de una herramienta con efectos secundarios o
   de costo muy alto).
5. **Prueba de carga y evidencia de costos:** enviá **5 peticiones concurrentes** contra la API y capturá del
   dashboard de monitoreo (Phoenix o LangSmith) dos métricas de esa corrida:
   - el **costo por ejecución** (que la plataforma calcula sola a partir de los tokens de entrada y salida), y
   - la **latencia p95**.

   Las capturas van en `/screenshots`, junto con las de las trazas.

## Pasos sugeridos

1. **Refactorización del grafo:** tomá tu orquestador multi-agente del Módulo 6 y asegurate de que use un
   `RedisSaver` o similar para persistir el estado. Definí un punto de interrupción en una arista o nodo crítico.
2. **Wrappers de observabilidad:** configurá las variables de entorno para LangSmith o inicializá el colector de
   Arize Phoenix mediante OpenInference. Asegurate de decorar tus funciones de llamada al LLM.
3. **Arquitectura de la API:** creá la aplicación FastAPI. Implementá el patrón de **"Worker"** (puede ser un thread
   asíncrono simple en esta etapa, o un proceso separado con Celery/Arq si buscás excelencia).
4. **Lectura del dashboard:** con las 5 peticiones concurrentes ya lanzadas (punto 5 de "¿Qué construir?"), revisá en
   Phoenix o LangSmith cómo se comportan las trazas, **dónde se concentra la latencia** y **qué nodo del grafo consume
   más tokens**. Esa lectura es la que sostiene tus capturas.

## Errores comunes a evitar

| Error | Descripción | Cómo evitarlo |
|---|---|---|
| **Bloquear el event loop** | Hacer llamadas síncronas a bases de datos o LLMs dentro de los endpoints de FastAPI mata el rendimiento. | Usar `run_in_threadpool` o clientes `async` nativos. |
| **Ignorar el manejo de errores en background tasks** | Si el agente falla en segundo plano y nadie actualiza el estado, el cliente queda esperando en un loop infinito de polling. | Capturar la excepción y actualizar el estado en Redis a `FAILED`. |

## 📦 Qué se entrega y en qué formato

- **Tipo:** 💻 Código — un repositorio de GitHub.
- **Artefacto concreto:** repo con la API FastAPI asíncrona, Redis para el estado, observabilidad (LangSmith o Arize
  Phoenix), el nodo Human-in-the-loop, el archivo de requerimientos, Docker Compose (opcional) y capturas del
  dashboard de trazas.
- **Qué NO hace falta:** no hay documento; las capturas del dashboard son evidencia de las trazas, no un informe.

## 🗂️ Estructura de referencia del repositorio

Tomala de molde y adaptala a tu implementación. Esta estructura cubre 1:1 lo que evalúa la rúbrica:

```text
mi-api-agente/
├── app/
│   ├── main.py            # FastAPI: POST /tasks → encola y devuelve job_id · GET /tasks/{id} → estado (no bloquea)
│   ├── graph.py           # tu orquestador multi-agente del M6 + checkpointer de Redis (RedisSaver)
│   ├── worker.py          # corre la tarea en segundo plano y actualiza estado en Redis (PENDING/RUNNING/FAILED/DONE)
│   ├── observability.py   # init de Arize Phoenix o LangSmith (decoradores de trazas)
│   └── hitl.py            # nodo de aprobación humana: interrupt + endpoint POST /tasks/{id}/approve
├── requirements.txt
├── docker-compose.yml     # (opcional) app + redis
├── .env.example           # REDIS_URL y claves (sin secretos reales)
├── screenshots/           # capturas del dashboard de trazas (evidencia de observabilidad)
└── README.md              # cómo levantar Redis + la API y cómo probar las 5 peticiones concurrentes
```

## ✅ Checklist de entrega (revisá antes de subir)

| Requisito | Criterio de la rúbrica |
|---|---|
| Endpoint asíncrono que encola y devuelve un `job_id` sin bloquear el event loop. | Orquestación asíncrona y API (30%) |
| Estado del job persistido en Redis, incluyendo el paso a `FAILED` ante una excepción. | Persistencia con Redis (25%) |
| Trazas visibles en el dashboard (Phoenix/LangSmith), con captura en `/screenshots`. | Observabilidad (25%) |
| Capturas del costo por ejecución y la latencia p95 de las 5 peticiones concurrentes, en `/screenshots`. | Observabilidad (25%) |
| Nodo Human-in-the-loop que pausa la ejecución hasta recibir aprobación externa. | HITL (15%) |
| README con los pasos para inicializar Redis y la API, cómo lanzar las 5 peticiones concurrentes, y `requirements.txt`. | Documentación y estructura (5%) |
| Repo público: probá un `git clone` en limpio antes de entregar. | — |

## Entregable

Repositorio de GitHub con el código de la API, archivo de requerimientos, configuración de Docker Compose (opcional
pero recomendado) y capturas de pantalla del dashboard de observabilidad con trazas activas.

1. Creá una estructura de proyecto **Python 3.12+**.
2. Implementá tu sistema multi-agente del Módulo 6 dentro de un orquestador que use **Redis** como almacenamiento de
   estado.
3. Desarrollá la API con **FastAPI** utilizando los endpoints asíncronos mencionados en el briefing.
4. Configurá **Arize Phoenix o LangSmith** para capturar trazas de cada nodo del agente.
5. Implementá un nodo de **"Aprobación Humana"** que detenga la ejecución del grafo hasta recibir un input externo.
6. Subí el código a un repositorio y documentá en el `README.md` cómo inicializar Redis y la API.
