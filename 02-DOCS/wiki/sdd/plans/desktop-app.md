---
type: plan
title: Plan — App de escritorio (sustituye al dashboard Streamlit)
description: Plan a nivel de estructura de la app de escritorio — componentes, contratos, flujos, tests, orden y riesgos.
tags: [sdd, plan]
timestamp: 2026-11-03T01:10:00Z
topic: sdd
slug: desktop-app
status: approved
---

# Plan — App de escritorio (sustituye al dashboard Streamlit)

> Spec: [../specs/desktop-app.md](../specs/desktop-app.md) (clarified) · Constitution: [../constitution.md](../constitution.md) v1.1.0 · Status: approved
> Last updated: 2026-11-02

## 0. Global Constraints

- **Python:** 3.13; dependencias en `requirements.txt` (pip). Lógica de negocio solo en `src/` (constitución §11).
- **Base de datos:** MariaDB en `localhost:3306` (servicio de Windows `MariaDB`), credenciales por `DB_*`. Cambios de esquema solo en `src/database.py`.
- **Llamadas externas:** siempre con los decoradores de `src/retry_config.py`.
- **Calidad Python:** `ruff check` + `ruff format --check` limpios en los ficheros tocados; `pytest` verde entero; cada lógica nueva o bug trae su test; sin red en los tests.
- **Idioma:** UI, mensajes y comentarios en español; identificadores en inglés.
- **Zona horaria:** America/Panama. Nada de `datetime.now()` sin zona en código nuevo o tocado.
- **Logging:** `src/logger.py`; sin `print` en código nuevo de `src/`.
- **Commits:** `tipo(ámbito): asunto`, sin gitmoji, sin `Co-Authored-By` ni pie de IA. Directo a `main`, solo con la suite verde.
- **Secretos:** nunca en pantalla, logs de UI, notificaciones ni respuestas de la API. `.env`, `credentials.json`, `token.json` y `token_integrations.json` fuera de git.
- **Contenido de terceros** (títulos, comentarios, descripciones): se pinta como texto, nunca como HTML sin escapar.
- **Acceso:** la API local escucha solo en `127.0.0.1` y exige el token de sesión en cada petición.
- **Horario por defecto:** una ejecución diaria a las 15:00 America/Panama.
- **Umbral de salud:** caída ≥ 10 puntos respecto al informe anterior del mismo canal (ajustable).
- **Notificaciones:** como mucho una por tipo de evento y ejecución.
- **Primera ejecución de detectores:** siembra NotifiedEvent en silencio; no notifica nada existente (D2).
- **Front-end:** Tauri 2 + React + TypeScript strict + Vite (pnpm), Tailwind, Apache ECharts; Biome + tsc strict + vitest; clippy + fmt (constitución §20-21, v1.1.0). Motor: FastAPI + uvicorn.

## 1. Context & constraints

- **Criterios que dan forma al diseño:** la paridad con cifras idénticas (spec §Acceptance, paridad y cifras) obliga a que Streamlit y la app usen *las mismas funciones*. La cancelación entre pasos y el bloqueo de doble ejecución obligan a reescribir `main.py` como una lista de pasos con un runner. Las notificaciones de canibalización, despegue tardío y salud obligan a que esos detectores entren en el pipeline: hoy solo se ejecutan desde la UI (`page_modules/ml_predictions.py`, `page_modules/health.py`).
- **Hechos del repo:** `main.py::main()` se traga las excepciones y no informa de los pasos. `scheduler.py` usa APScheduler con `SCHEDULE_HOUR`/`SCHEDULE_MINUTE`. Las páginas suman 5.515 líneas, con cálculo mezclado con `st.*`. Los tests cubren `src/` y nada de la UI.
- **Toolchain de la máquina:** Node 22, pnpm 9.15 y WebView2 156 sí están. **Falta Rust/cargo**, necesario para compilar Tauri.
- **Constitución en juego:** §1.4 (front-end abierto → enmienda MINOR), §5-7 (calidad), §11 (sin lógica en la UI), §15-16 (secretos, contenido de terceros), §17-18 (estados vacío/error, temas).
- **Fuera de alcance (no derivar hacia ello):** distribución a terceros, empaquetar Python o MariaDB, cambiar la lógica de análisis, Telegram, macOS/Linux, acceso remoto, presupuesto de rendimiento, recuperar ejecuciones perdidas.

## 2. Architecture

```text
┌──────────────────────── App de escritorio (Tauri, proceso Rust) ────────────────────────┐
│  Shell: bandeja · instancia única · autoarranque · cierre→bandeja · salir con confirm.   │
│  Supervisor del motor: lanza/vigila/mata el proceso Python; le pasa puerto+token         │
│  Notificador: toasts de Windows con activación → abre la vista destino                   │
│           ▲ eventos (IPC Tauri)                         │ deep-link (área/vista)          │
│  ┌────────┴──────────────── WebView (UI) ───────────────▼─────────────────────────┐       │
│  │ 6 áreas · selector de canal · tema · modo presentación · estados vacío/error    │       │
│  └──────────────────────────────┬─────────────────────────────────────────────────┘       │
└─────────────────────────────────┼─────────────────────────────────────────────────────────┘
                                  │ HTTP + stream de eventos, 127.0.0.1:<puerto>, token
┌─────────────────────────────────▼──────────── Motor Python (proceso hijo) ───────────────┐
│  API local ── vistas (view models) ── jobs (acciones IA largas) ── ajustes                 │
│  Scheduler diario ──► Runner del pipeline ──► subproceso `main.py --run-id`               │
│                                   ▲ eventos de paso/hallazgos (stdout JSON por línea)     │
│  src/views/*  (lógica extraída de page_modules, compartida con Streamlit hasta retirarlo) │
│  src/*        (extractores, analizadores, detectores, database)                           │
└───────────────┬──────────────────────────────────────────────┬────────────────────────────┘
                ▼                                              ▼
        MariaDB (externo, local)                Anthropic / Google APIs (externos)
```

- **Shell de escritorio** (interno): ciclo de vida de la app en Windows (bandeja, instancia única, autoarranque minimizado, cerrar→ocultar, salir confirmando si hay pipeline).
- **Supervisor del motor** (interno): arranca el motor Python con el Python del sistema y la ruta del proyecto, espera su handshake (puerto + listo), lo reinicia bajo demanda y lo mata al salir sin dejar huérfanos.
- **Notificador** (interno): convierte eventos del motor en toasts agrupados y, al pulsarlos, abre la ventana en la vista destino.
- **UI** (interna): presenta los view models; no calcula nada de negocio.
- **API local** (interna): contrato único entre la UI y Python; autentica con el token y expone vistas, jobs, pipeline, ajustes y salud.
- **Jobs** (interno): ejecución única por clave (`pipeline`, `recommendation:<canal>`…), progreso y cancelación cooperativa.
- **Scheduler** (interno): dispara el pipeline a la hora configurada mientras vive el motor.
- **Runner del pipeline** (interno): ejecuta `main.py` como subproceso (un fallo del pipeline no tumba la API) y traduce su salida a eventos.
- **`src/views/*`** (interno, nuevo): una función pura por vista que devuelve datos listos para pintar. La usan Streamlit y la API, así que la paridad de cifras se cumple por construcción.
- **MariaDB, Anthropic, Google** (externos).

**Decisión de arquitectura principal:** *estrangular* la UI extrayendo primero la lógica de cada página a `src/views/`, de forma que Streamlit pase a consumirla, y después servirla por la API. La alternativa, reescribir cada vista directamente en la nueva UI, obliga a reimplementar los cálculos y a demostrar luego que coinciden. Con la extracción, la cifra de Streamlit y la de la app salen de la misma función. Eso cumple la constitución §11, mantiene Streamlit usable hasta la paridad (suposición de la spec) y convierte el criterio "las cifras coinciden" en un test de contrato en lugar de una comparación manual.

**Segunda decisión:** el pipeline corre en un **subproceso** controlado por el motor, no dentro de él. El pipeline ya funciona como script (`python main.py` debe seguir siendo CLI), puede tardar minutos o colgarse en una llamada externa, y aislarlo permite cancelar y recuperarse sin reiniciar la API.

## 3. Interfaces & contracts

```text
Supervisor → Motor (arranque)
  spawn(python, project_dir, token) -> Ready{port} | EngineMissing{what} | EngineCrashed{log_tail}
  - el motor escribe una línea de handshake {"ready": true, "port": N} en stdout
  - EngineMissing: Python no encontrado o falta un import (nombra el paquete)
  - postcondición al salir la app: no queda ningún proceso Python del motor ni del pipeline

API (todas las peticiones llevan el token; sin token → 401; host distinto de 127.0.0.1 → 403)
  health() -> {engine: ok, db: ok | DbDown{reason}, version}
  settings.get() -> Settings ; settings.update(patch: Partial<Settings>) -> Settings | Invalid{field}
  channels.list() -> [Channel{id, title, is_competitor}]
  views.get(view_id, channel_id, params) -> ViewModel | Empty{hint} | DbDown | NotFound
  jobs.start(kind, args) -> JobId | AlreadyRunning{job_id}
  jobs.get(job_id) -> Job{kind, status, progress, result | error}
  jobs.cancel(job_id) -> Cancelling | NotRunning
  pipeline.start(trigger=manual) -> RunId | AlreadyRunning{run_id}
  pipeline.cancel(run_id) -> Cancelling | NotRunning
  pipeline.runs(limit) -> [PipelineRun] ; pipeline.next_scheduled() -> datetime(America/Panama)
  events.stream() -> secuencia de Event{type, run_id?, payload}

Runner ↔ main.py (subproceso)
  main.py --run-id R [--events-jsonl]
    - emite una línea JSON por evento: step_started{step}, step_done{step, summary}, finding{type, count, keys},
      run_done{status: success|failed|cancelled, last_step, error?}
    - cancelación: el runner marca la petición; main.py la consulta ENTRE pasos y termina con status=cancelled
    - exclusión: main.py adquiere el lock global `pipeline` de MariaDB al empezar;
      si no lo obtiene → run_done{status: skipped, reason: already_running} (cubre CLI y app a la vez)
    - sin --events-jsonl el comportamiento CLI se mantiene (logs legibles)

Detectores en el pipeline (pasos nuevos, misma lógica que hoy en la UI)
  health_step(channel) -> HealthReport{score} ; drop = previous.score - score ≥ threshold → finding(health_drop)
  late_bloomer_step(channel) -> [video_id] ; cannibalization_step(channel) -> [(video_a, video_b)]
  evergreen_step (existente) -> transiciones no-evergreen → evergreen → finding(evergreen)
  - "nuevo" = clave no registrada antes en NotifiedEvent para ese tipo

Notificador (Rust) ← Event
  finding{type, count} -> un toast por tipo y ejecución; texto en español con el recuento
  run_done{status} -> toast terminado/fallido/cancelado (skipped no notifica, solo historial)
  activate(toast) -> muestra la ventana y navega a la vista destino del tipo

src/views/<view>.build(db, channel_id, params) -> dict serializable
  - pura respecto a la UI: sin st.*, sin estado de sesión
  - Streamlit page = build() + pintar ; API = build() + serializar
```

## 4. Data model & flow

### Entities

- **PipelineRun** (tabla nueva): id, trigger (manual | scheduled | cli), status (running | success | failed | cancelled | skipped), started_at, finished_at (America/Panama), last_completed_step, error_message (sin secretos), findings_summary (recuento por tipo).
- **NotifiedEvent** (tabla nueva): event_type, entity_key, first_seen_at, con clave única (event_type, entity_key). Distingue "nuevo" de "ya visto".
- **HealthReport** (tabla existente `channel_health_reports`): se escribe ahora desde el paso de pipeline, no solo desde la página.
- **Settings** (fichero `config/app_settings.json`, gestionado por el motor): schedule_hour, schedule_minute, autostart, theme, widget_order, last_channel_id, last_area, health_drop_threshold. En el primer arranque hereda tema y orden de widgets de `config/dashboard_prefs.json`, y la hora de `SCHEDULE_HOUR`/`SCHEDULE_MINUTE`.
- **Job** (en memoria del motor): kind, key, status, progress, result/error. No sobrevive a un reinicio, y no hace falta.

### Primary flow (ejecución programada con alerta de competencia)

1. Scheduler del motor dispara a las 15:00 → `pipeline.start(trigger=scheduled)`.
2. El runner crea un PipelineRun (running) y lanza `main.py --run-id R --events-jsonl`.
3. `main.py` toma el lock `pipeline`, ejecuta los pasos y emite `step_started` y `step_done`. El paso de competidores emite `finding{competitor_alert, count: 3}`.
4. El runner actualiza el PipelineRun y reenvía cada evento al stream.
5. La UI, aunque esté oculta, recibe el stream y lo pasa al notificador → un toast "3 alertas de competencia".
6. `run_done{success}` → PipelineRun success + toast "Pipeline terminado".
7. El usuario pulsa el toast de competencia → se muestra la ventana en *Audiencia y competencia › Competencia*.

- **Consistencia:** cada paso confirma sus datos en la BD antes de emitir `step_done`, así que una cancelación entre pasos nunca deja un paso a medias. El historial es consistente al final del paso.
- **Migración:** dos tablas nuevas (aditivas). Sin backfill. Nada destructivo hasta la retirada final (borrar Streamlit, `scheduler.py` y el servicio `dashboard`).

## 5. Testing strategy

| Criterio de la spec | Nivel | Qué afirma | Qué se simula |
| --- | --- | --- | --- |
| Abrir sin terminal ni navegador; datos del último canal | e2e (smoke) | la ventana muestra Inicio con el canal guardado | BD de prueba con datos semilla |
| Cerrar → bandeja; Salir termina | manual guiado + unit del shell | el evento de cierre oculta; Salir mata el motor (sin procesos huérfanos) | — |
| Autoarranque minimizado | manual guiado | tras iniciar sesión, icono en bandeja sin ventana | — |
| Instancia única | e2e (smoke) | la segunda apertura no crea proceso y enfoca la ventana | — |
| Tabla de paridad completa | revisión de artefacto | las 24 filas con ubicación y estado ✓ | — |
| Cifras idénticas Streamlit/app | contract (pytest) | `views.<v>.build()` = lo que pinta Streamlit (misma función) + la API serializa lo mismo | BD en memoria o fixtures de DataFrame |
| Lanzar pipeline, ver paso, notificación al final | integration (pytest) | el runner traduce los eventos de `main.py` a PipelineRun y al stream | pasos falsos que emiten eventos |
| No doble ejecución (manual, programada, CLI) | integration | segunda llamada → AlreadyRunning; lock de BD → skipped | lock simulado |
| Ejecución programada en bandeja | unit (scheduler) | con reloj falso, a la hora dispara `pipeline.start(scheduled)` | reloj |
| Notificación agrupada por tipo | unit | 3 findings del mismo tipo → 1 toast con "3" | — |
| Notificación pulsada abre la vista | unit (mapa tipo→vista) + manual guiado del toast | cada tipo tiene destino válido | — |
| MariaDB parado → error + reintento | integration (API) + e2e | `health()` → DbDown; la vista muestra el error; el reintento recupera | conexión que falla |
| Estado vacío por vista | unit (UI, por vista) | con Empty → mensaje en español | API simulada |
| Tema claro/oscuro legible y recordado | unit (UI) + revisión visual | tokens aplicados; persiste en Settings | — |
| Retirada de Streamlit | integration | sin `streamlit` en requirements ni imports; `python main.py` corre; pytest verde | — |
| Cancelar entre pasos | integration | con cancelación pedida a mitad del paso 3, el paso 3 termina, el 4 no empieza, status=cancelled | pasos falsos lentos |
| Salir con pipeline en curso | unit del shell + manual guiado | pide confirmación; confirmar → cancela y sale | — |
| Caída de salud / canibalización / evergreen nuevos | unit (detección de "nuevo") | solo claves no vistas generan finding; la caída ≥ umbral genera health_drop | NotifiedEvent en memoria |
| Modo presentación + Esc | unit (UI) | oculta navegación y controles; Esc restaura | — |
| Acción IA larga sin relanzar | integration (jobs) | segundo start de la misma clave → AlreadyRunning | Anthropic simulado |
| Motor no arranca → mensaje | unit del supervisor | import ausente → EngineMissing{paquete}; la UI lo nombra | proceso falso |
| Otro equipo de la red no accede | integration | la API rechaza un host distinto de 127.0.0.1 y las peticiones sin token | — |
| Hora 09:30 persiste | integration | settings.update → next_scheduled = 09:30 tras reiniciar el motor | reloj |

- **Línea e2e:** solo smoke (arranque, instancia única, una vista, BD caída), si el spike S2 confirma que es viable automatizar la app en Windows. Bandeja, autoarranque y toasts se verifican con una **lista manual guiada** en `verify`, porque dependen del shell de Windows.
- **Siempre real:** MariaDB en los tests de integración de NotifiedEvent/PipelineRun y del lock (BD de pruebas aparte; nunca la de producción). Las APIs de Anthropic y Google siempre simuladas (constitución §7).
- **Antes de extraer cada vista:** un test de caracterización fija las cifras actuales de su cálculo, y después se mueve el código.

## 6. Sequencing & dependencies

1. **Prerrequisitos** — Rust (rustup + MSVC Build Tools), Ruff en el entorno, BD de pruebas. Enmienda de la constitución con el stack de front-end y su barra de calidad (D1). — depende de: nada — serial
2. **Spikes** — S1 toast con activación en Windows; S2 e2e automatizable en Windows; S3 ciclo de vida del motor (handshake, matar sin huérfanos, tiempo de arranque). — depende de: #1 — paralelizables entre sí
3. **Motor: base** — API con token y solo localhost, `health`, Settings (incluida la herencia de prefs), `channels.list`. — depende de: #1 — serial
4. **Pipeline como pasos** — refactor de `main.py` en lista de pasos + eventos JSONL + cancelación entre pasos + lock de BD + PipelineRun; el CLI se mantiene. — depende de: #1 — paralelizable con #3
5. **Jobs + runner + scheduler + stream de eventos** en el motor. — depende de: #3, #4 — serial
6. **Detectores al pipeline** — extraer el cálculo de salud de `page_modules/health.py` a `src/`; pasos de salud, despegue tardío y canibalización; NotifiedEvent y findings (incluida la transición a evergreen). — depende de: #4 — paralelizable con #5
7. **Shell de escritorio** — Tauri, supervisor, bandeja, instancia única, autoarranque, cerrar→bandeja, salir confirmando, notificador + deep-link, pantallas de motor caído/BD caída. — depende de: #2, #3 — paralelizable con #5-#6
8. **Sistema de diseño** — dirección visual (fase `design`), tokens claro/oscuro, layout de 6 áreas, selector de canal, modo presentación, componentes de vacío/error/progreso. — depende de: #1 (D1) — paralelizable con #3-#7
9. **Área Sistema** — pipeline (lanzar, progreso, cancelar, historial, próxima ejecución), horario, ajustes. — depende de: #5, #7, #8
10. **Migración por áreas** (cada vista: caracterización → `src/views/` → Streamlit consume → API → vista nueva → fila de paridad ✓):
    10a Inicio (resumen con widgets, salud) · 10b Rendimiento (6 vistas) · 10c Contenido (5) · 10d Planificar y crear (8, con jobs de IA, what-if, guiones, SEO, exportación Calendar/Sheets, A/B de miniaturas) · 10e Audiencia y competencia (3).
    — depende de: #3, #5 (para 10d), #8 — **las áreas son paralelizables entre sí** salvo por tres ficheros que comparten varias vistas (`analytics.py`: Analytics + Tendencias; `content.py`: Contenido + Comparar canales; `ml_predictions.py`: Canibalización + Despegue + Retención). Esas vistas se ordenan con dependencias en §Tasks (corregido en `tasks`, 2026-11-02); dentro de un área, en serie. Empezar por **10a como piloto** para calibrar el esfuerzo antes de abrir las demás.
11. **Verificación de paridad** — tabla completa, contratos de cifras verdes, lista manual guiada. — depende de: #9, #10
12. **Retirada** — borrar `dashboard.py`, `page_modules/`, `scheduler.py`, el servicio `dashboard` de docker-compose y `streamlit` de requirements; actualizar README/QUICKSTART. — depende de: #11 — serial, destructivo (confirmar antes)

- **Orden obligatorio:** #4 antes de #6 (los detectores son pasos), #8 antes de cualquier vista, #11 antes de #12 (no se retira nada sin paridad verificada).

### Apéndice — tabla de paridad inicial (se completa en #10/#11)

| Vista actual | Módulo | Área destino |
| --- | --- | --- |
| Resumen General (widgets personalizables) | overview | Inicio |
| Salud del Canal | health | Inicio |
| Análisis de Performance | performance | Rendimiento |
| Histórico de Métricas | history | Rendimiento |
| Analytics Avanzados | analytics | Rendimiento |
| Comparador Temporal | temporal | Rendimiento |
| Cadencia y Horarios | cadence | Rendimiento |
| Revenue y ROI | revenue | Rendimiento |
| Análisis de Contenido | content | Contenido |
| Series y Formatos | series | Contenido |
| Tendencias de Temas | analytics | Contenido |
| Canibalización | ml_predictions | Contenido |
| Despegue Tardío | ml_predictions | Contenido |
| Recomendaciones (+ guiones, SEO) | recommendations | Planificar y crear |
| Generar Nueva Recomendación | recommendations | Planificar y crear |
| Plan Semanal (+ export Calendar/Sheets) | planning | Planificar y crear |
| Predicción de Viralidad (+ what-if) | predictions | Planificar y crear |
| Predicción de Vistas | predictions | Planificar y crear |
| Predicción de Retención | ml_predictions | Planificar y crear |
| Galería de Miniaturas | thumbnails | Planificar y crear |
| Comparador A/B | thumbnails | Planificar y crear |
| Comentarios y Audiencia | comments | Audiencia y competencia |
| Análisis de Competencia | competitors | Audiencia y competencia |
| Comparar Canales | content | Audiencia y competencia |
| Transversales: tema claro/oscuro, modo presentación, selector de canal, "actualizar datos" | dashboard.py | Todas / Sistema |

## 7. Risks & open decisions

### Risks (ordenados)

| Riesgo | Disparador | Impacto | Mitigación |
| --- | --- | --- | --- |
| Esfuerzo total (24 vistas, ~5.500 líneas) y valor solo al final | la migración avanza más lento de lo previsto | meses sin usar la app | piloto 10a para medir; la extracción a `src/views/` ya mejora Streamlit (lógica testeada) aunque la app no esté lista; áreas en paralelo |
| Lógica enredada con `st.session_state`, cachés y widgets | una página no se puede extraer limpia | la extracción cuesta más; riesgo de cambiar cifras | test de caracterización antes de mover; si hace falta, extraer por partes |
| Toast con activación en Windows no soportado por el plugin estándar | S1 falla | el criterio "pulsar abre la vista" no se cumple | S1 al inicio; plan B: API nativa de toasts de Windows desde Rust con callback de activación |
| Procesos huérfanos / arranque lento del motor (pandas, sklearn) | la app se cierra mal o el arranque tarda segundos | Python vivo en segundo plano; UI en blanco | S3: handshake, Job Object de Windows para matar el árbol, pantalla de "arrancando motor" |
| Pasos largos hacen lenta la cancelación (p. ej. comentarios con muchas llamadas a Claude) | cancelar a mitad de un paso largo | "cancelando…" durante minutos | mostrar estado "cancelando"; checkpoints internos en pasos largos como mejora posterior, sin romper el contrato |
| API local alcanzable por otras páginas o procesos locales | una web abierta llama a `127.0.0.1` | lectura de datos o lanzar acciones | token por sesión, comprobación de Host/Origin, CORS solo para el origen de la app |
| Primer uso: todo es "nuevo" | primera ejecución de los detectores | una tanda de toasts (como mucho uno por tipo) | sembrar NotifiedEvent en silencio la primera vez (D2) |
| Falta Rust/MSVC | paso #1 | no compila | instalación guiada en #1 |
| `main.py` se traga excepciones | refactor #4 | fallos reportados como éxito | el runner devuelve status explícito; test de fallo |

### Open decisions

- **D1 — CERRADA (2026-11-02):** React + TS + ECharts, ratificado como constitución v1.1.0 §20-21. Texto original: **Stack de front-end y su barra de calidad** (enmienda MINOR de la constitución §1.4 y §2). Propuesta: React + TypeScript (estricto) + Vite, pnpm, componentes propios con Tailwind, gráficos con Apache ECharts (temas claro/oscuro), Biome para lint/format, Vitest + Testing Library. Motor: FastAPI + uvicorn. Se cierra con la ratificación del propietario.
- **D2 — CERRADA (2026-11-02): sí, sembrar en silencio.** La primera ejecución registra lo existente en NotifiedEvent sin notificar. Recomendado: sí. Se cierra con la respuesta del propietario.
- **D3 — Dirección visual.** Se cierra en el paso #8 con la fase `design` (propuesta visual delante del propietario).
- **Aislamiento:** la constitución (§13) y la política del proyecto mandan los cambios directos a `main`. No se abre rama ni worktree. El paso destructivo #12 se confirma antes.

## Tasks
<!-- generated by tasks on 2026-11-02; IDs are stable, do not renumber -->

**Índice de criterios de aceptación** (en el orden de `spec §Acceptance criteria`):

- AC1 abrir sin terminal · AC2 cerrar→bandeja / Salir · AC3 autoarranque · AC4 instancia única
- AC5 tabla de paridad · AC6 cifras idénticas · AC7 lanzar pipeline + notificación · AC8 sin doble ejecución
- AC9 ejecución programada · AC10 alerta de competencia + abrir vista · AC11 MariaDB parado · AC12 estado vacío
- AC13 temas · AC14 retirada de Streamlit · AC15 cancelar · AC16 salir con pipeline
- AC17 agrupación de avisos · AC18 salud/canibalización/evergreen · AC19 modo presentación · AC20 acción IA no relanzable
- AC21 motor no arranca · AC22 sin acceso desde la red · AC23 hora persistente

**Rutas fijadas para todas las tareas:**

- La app vive en `desktop/`: la UI en `desktop/src/` y el código Rust en `desktop/src-tauri/`.
- El motor vive en `src/engine/`, con la entrada `engine.py` en la raíz.
- La lógica de las vistas va en `src/views/`.
- El pipeline va en `src/pipeline.py`; `main.py` queda como CLI.
- Los tests van en `tests/engine/`, `tests/pipeline/` y `tests/views/`.
- Las notas de los spikes van en `02-DOCS/wiki/sdd/spikes/desktop-app-sN.md`.
- Los tests con BD real llevan el marcador `db` y usan `DB_NAME_TEST`, nunca la BD de producción.

| ID | [P] | Task | Done-check | Depends-on | Trace |
| --- | --- | --- | --- | --- | --- |
| T001 |  | Install Rust stable (MSVC) y comprobar Build Tools | `rustc --version` y `cargo --version` responden; un proyecto `cargo new` de prueba compila con `cargo build` | — | plan §6.1 (AC1) |
| T002 | [P] | Add ruff, fastapi, uvicorn y httpx a `requirements.txt`, y crear `ruff.toml` | `pip install -r requirements.txt` ok; `ruff check src/logger.py` y `ruff format --check src/logger.py` ejecutan; `pytest` con 97 verdes | — | constitución §5 |
| T003 | [P] | Add la fixture `test_db` y el marcador `db` en `tests/conftest.py` | `pytest -m db tests/test_db_fixture.py` verde contra `DB_NAME_TEST`; sin BD hace skip y no falla | — | plan §5 (deps reales) |
| T004 |  | Add `scripts/verify.sh`: ruff en ficheros tocados y pytest; biome/tsc/vitest y clippy/fmt si existe `desktop/` | `bash scripts/verify.sh` sale con 0 en el árbol actual | T002 | constitución DoD |
| T005 |  | Scaffold `desktop/`: Tauri 2 + React + TS strict + Vite, pnpm, Tailwind, Biome, Vitest y ECharts | `pnpm --dir desktop install`; `biome check .`, `tsc --noEmit` y `vitest run` verdes; `pnpm --dir desktop tauri build --debug` genera el .exe; `bash scripts/verify.sh` = 0 | T001, T004 | constitución §20-21 |
| T006 | [P] | Spike S1: toast de Windows con activación que abre una ruta | nota `spikes/desktop-app-s1.md` con el mecanismo elegido; manual: pulsar el toast enfoca la ventana y registra la ruta recibida | T005 | AC10, AC18 |
| T007 | [P] | Spike S2: e2e automatizado de la app en Windows | nota `spikes/desktop-app-s2.md` con un smoke verde que abre la app y lee un texto, o la decisión registrada de "solo lista manual" | T005 | AC1, AC4 |
| T008 | [P] | Spike S3: ciclo de vida del motor (handshake, matar el árbol de procesos, tiempo de arranque) | nota `spikes/desktop-app-s3.md` con el tiempo medido; manual: tras Salir, `tasklist` no muestra ningún python del motor ni del pipeline | T005 | AC2, AC21 |
| T009 |  | Write failing tests de Settings: defaults, herencia de `dashboard_prefs.json` y de `SCHEDULE_*`, validación | `pytest tests/engine/test_settings.py` falla por módulo ausente | T002 | AC23, spec §Arranque |
| T010 |  | Implement `src/engine/settings.py` | `pytest tests/engine/test_settings.py` verde | T009 | AC23 |
| T011 |  | Write failing tests de la API base: sin token → 401, Host ≠ 127.0.0.1 → 403, `health` ok/DbDown | `pytest tests/engine/test_api_base.py` falla por módulo ausente | T002 | AC11, AC22 |
| T012 |  | Implement la API base y `engine.py` (escucha en 127.0.0.1, puerto 0, con handshake) | `pytest tests/engine/test_api_base.py` verde; `python engine.py --token t` imprime `{"ready": true, "port": N}` como primera línea | T011 | AC11, AC21, AC22 |
| T013 |  | Add los endpoints `channels.list` y `settings.get/update`, tests primero | `pytest tests/engine/test_api_settings.py` pasa de rojo a verde; un update inválido devuelve `Invalid{field}` | T010, T012 | AC1, AC23 |
| T014 |  | Write failing tests del pipeline por pasos: orden, fallo → `failed`, cancelación entre pasos, JSONL | `pytest tests/pipeline/test_pipeline_steps.py` falla (hoy `main()` se traga el error) | T002 | AC7, AC15 |
| T015 |  | Implement `src/pipeline.py` y la CLI de `main.py` (`--run-id`, `--events-jsonl`) | `pytest tests/pipeline/test_pipeline_steps.py` verde; `python main.py --help` muestra los flags; sin flags, los logs no cambian (test) | T014 | AC7, AC14, AC15 |
| T016 |  | Write failing tests de PipelineRun y del lock global (db) | `pytest -m db tests/pipeline/test_pipeline_runs.py` falla por tabla ausente | T003, T015 | AC8 |
| T017 |  | Implement la tabla `pipeline_runs`, sus métodos en `src/database.py` y el lock en el pipeline | `pytest -m db tests/pipeline/test_pipeline_runs.py` verde; con dos `main.py` simultáneos, el segundo termina `skipped` (test) | T016 | AC8 |
| T018 |  | Write failing tests de jobs: una ejecución por clave, progreso, cancelación cooperativa | `pytest tests/engine/test_jobs.py` falla por módulo ausente | T012 | AC20 |
| T019 |  | Implement `src/engine/jobs.py` | `pytest tests/engine/test_jobs.py` verde | T018 | AC20 |
| T020 |  | Write failing tests del runner: subproceso falso con JSONL → PipelineRun + stream; AlreadyRunning; cancel → cancelled | `pytest tests/engine/test_runner.py` falla | T017, T019 | AC7, AC8, AC15 |
| T021 |  | Implement `src/engine/runner.py` y los endpoints `pipeline.*` y `events.stream` | `pytest tests/engine/test_runner.py tests/engine/test_api_pipeline.py` verdes | T020 | AC7, AC8, AC15 |
| T022 |  | Write failing tests del scheduler: reloj falso, reprogramación al cambiar la hora, `next_scheduled` | `pytest tests/engine/test_scheduler.py` falla | T010, T021 | AC9, AC23 |
| T023 |  | Implement `src/engine/scheduler.py` y arrancarlo con el motor | `pytest tests/engine/test_scheduler.py` verde | T022 | AC9, AC23 |
| T024 | [P] | Write el test de caracterización del cálculo de salud de `page_modules/health.py` | `pytest tests/test_health_scorer.py` fija con fixtures las cifras actuales y falla por módulo ausente | T002 | AC18, AC6 |
| T025 |  | Extract `src/health_scorer.py` y hacer que `page_modules/health.py` lo use | `pytest tests/test_health_scorer.py` verde; `grep -n "health_scorer" page_modules/health.py` encuentra el import | T024 | AC18, constitución §11 |
| T026 | [P] | Write failing tests de NotifiedEvent: `is_new` y siembra silenciosa en la primera ejecución (db) | `pytest -m db tests/pipeline/test_notified_events.py` falla por tabla ausente | T003 | AC18, D2 |
| T027 |  | Implement la tabla `notified_events` y sus métodos | `pytest -m db tests/pipeline/test_notified_events.py` verde | T026 | AC18, D2 |
| T028 |  | Write failing tests de los pasos detectores: findings solo de claves nuevas, caída de salud ≥ umbral, transición a evergreen, finding de competencia | `pytest tests/pipeline/test_detector_steps.py` falla | T015, T025, T027 | AC10, AC17, AC18 |
| T029 |  | Implement los pasos de salud, despegue tardío, canibalización y evergreen, con sus findings | `pytest tests/pipeline/test_detector_steps.py` verde; `pytest` entero verde | T028 | AC10, AC18 |
| T030 |  | Write failing Rust tests del supervisor: parseo del handshake → Ready, import ausente → EngineMissing | `cargo test --manifest-path desktop/src-tauri/Cargo.toml supervisor` falla | T008, T012 | AC21 |
| T031 |  | Implement el supervisor del motor: spawn, espera, reinicio y matar el árbol al salir | `cargo test … supervisor` verde; manual M1 | T030 | AC2, AC21 |
| T032 |  | Implement bandeja, cerrar → ocultar, instancia única y autoarranque con `--minimized` | `cargo test …` verde; `cargo clippy -- -D warnings` limpio; manual M2-M4 registrados | T031 | AC2, AC3, AC4 |
| T033 |  | Implement Salir con pipeline en curso → confirmar → cancelar y salir, tests primero | `vitest run src/shell/quit.test.ts` pasa de rojo a verde; manual M5 | T021, T032 | AC16 |
| T034 |  | Implement el notificador: agrupar por tipo y ejecución, mapa tipo → vista, activación con navegación; tests primero | `vitest run src/shell/notifications.test.ts` pasa de rojo a verde: 3 findings → 1 toast con "3", y cada tipo tiene ruta válida; manual M6 | T006, T021, T032 | AC7, AC10, AC17, AC18 |
| T035 |  | Implement las pantallas de motor caído y BD caída con reintento, tests primero | `vitest run src/shell/engine-status.test.tsx` pasa de rojo a verde para EngineMissing{paquete} y DbDown | T031, T041 | AC11, AC21 |
| T036 |  | Run la fase `design` hasta tener una dirección visual aprobada por el propietario | existe `02-DOCS/wiki/sdd/design/desktop-app.md` con la dirección y la aprobación citada | T005 | D3, spec §Problem |
| T037 |  | Implement los tokens claro/oscuro y la persistencia del tema en Settings, tests primero | `vitest run src/theme/theme.test.ts` pasa de rojo a verde; un test de tokens comprueba contraste AA del texto en ambos temas | T036, T013 | AC13 |
| T038 |  | Implement el layout: navegación de 6 áreas, router y selector de canal que restaura el último usado; tests primero | `vitest run src/layout/*.test.tsx` pasa de rojo a verde | T037 | AC1, spec §Navegación |
| T039 | [P] | Implement los componentes EmptyState, ErrorState, Progress y un gráfico ECharts con tema; tests primero | `vitest run src/components/*.test.tsx` pasa de rojo a verde | T037 | AC12, AC13 |
| T040 |  | Implement el modo presentación (atajo + Esc), tests primero | `vitest run src/layout/presentation.test.tsx` pasa de rojo a verde | T038 | AC19 |
| T041 | [P] | Implement el cliente tipado de la API (token, Empty/DbDown/NotFound, suscripción a eventos); tests primero | `vitest run src/api/*.test.ts` pasa de rojo a verde con fetch simulado | T005, T012 | AC11, AC12 |
| T042 |  | Implement el área Sistema: lanzar, progreso, cancelar, historial, próxima ejecución, hora, autoarranque, umbral | `vitest run src/areas/sistema/*.test.tsx` pasa de rojo a verde; manual M7 | T021, T023, T038, T039, T041 | AC3, AC7, AC8, AC9, AC15, AC23 |
| T043 |  | Migrate la vista piloto Resumen General, con widgets (receta R) | receta R verde para `overview` | T038, T039, T041 | AC5, AC6, AC12 |
| T044 |  | Migrate Salud del Canal (R) | R verde para `health` | T043, T025 | AC5, AC6 |
| T045 |  | Review el piloto: medir el esfuerzo, ajustar la receta R y registrarlo en decisions.md | entrada fechada en `decisions.md` con las horas por vista y los cambios en R | T044 | plan §7 (riesgo 1) |
| T046 | [P] | Migrate Análisis de Performance (R) | R verde para `performance` | T045 | AC5, AC6 |
| T047 |  | Migrate Histórico de Métricas (R) | R verde para `history` | T046 | AC5, AC6 |
| T048 |  | Migrate Analytics Avanzados (R) | R verde para `advanced_analytics` | T047 | AC5, AC6 |
| T049 |  | Migrate Comparador Temporal (R) | R verde para `temporal` | T048 | AC5, AC6 |
| T050 |  | Migrate Cadencia y Horarios (R) | R verde para `cadence` | T049 | AC5, AC6 |
| T051 |  | Migrate Revenue y ROI (R) | R verde para `revenue` | T050 | AC5, AC6 |
| T052 | [P] | Migrate Análisis de Contenido (R) | R verde para `content` | T045 | AC5, AC6 |
| T053 |  | Migrate Series y Formatos (R) | R verde para `series` | T052 | AC5, AC6 |
| T054 |  | Migrate Tendencias de Temas (R); comparte `analytics.py` con T048 | R verde para `trends` | T053, T048 | AC5, AC6 |
| T055 |  | Migrate Canibalización (R); reutiliza el paso de T029 | R verde para `cannibalization` | T054, T029 | AC5, AC6 |
| T056 |  | Migrate Despegue Tardío (R) | R verde para `late_bloomer` | T055 | AC5, AC6 |
| T057 | [P] | Migrate Recomendaciones (+ guiones, SEO, tags) con jobs de IA (R) | R verde para `recommendations`; un segundo start con la misma clave → AlreadyRunning (test) | T045, T019 | AC5, AC6, AC20 |
| T058 |  | Migrate Generar Nueva Recomendación como job (R) | R verde para `generate_recommendation` | T057 | AC5, AC6, AC20 |
| T059 |  | Migrate Plan Semanal y la exportación a Calendar/Sheets (R) | R verde para `weekly_plan`; la exportación solo abre el navegador para OAuth (test del cliente) | T058 | AC5, AC6 |
| T060 |  | Migrate Predicción de Viralidad + what-if (R) | R verde para `virality` | T059 | AC5, AC6 |
| T061 |  | Migrate Predicción de Vistas (R) | R verde para `views_prediction` | T060 | AC5, AC6 |
| T062 |  | Migrate Predicción de Retención (R); comparte `ml_predictions.py` con T055-T056 | R verde para `retention` | T061, T056 | AC5, AC6 |
| T063 |  | Migrate la Galería de Miniaturas (R) | R verde para `thumbnails` | T062 | AC5, AC6 |
| T064 |  | Migrate el Comparador A/B, con subida de imágenes como job (R) | R verde para `thumbnail_ab` | T063 | AC5, AC6, AC20 |
| T065 | [P] | Migrate Comentarios y Audiencia (R) | R verde para `comments`; el texto de los comentarios se pinta escapado (test) | T045 | AC5, AC6, constitución §16 |
| T066 |  | Migrate Análisis de Competencia (R) | R verde para `competitors` | T065 | AC5, AC6, AC10 |
| T067 |  | Migrate Comparar Canales (R); comparte `content.py` con T052 | R verde para `channel_comparison` | T066, T052 | AC5, AC6 |
| T068 |  | Implement "actualizar datos" global: invalida las cachés del motor y recarga la vista | `pytest tests/engine/test_api_views.py -k refresh` y `vitest run src/layout/refresh.test.tsx` verdes | T043 | AC5 |
| T069 |  | Complete la tabla de paridad del apéndice, con ubicación y ✓ en cada fila | las 25 filas del apéndice con ubicación y ✓, ninguna vacía | T042, T044, T051, T056, T064, T067, T068 | AC5 |
| T070 |  | Run el smoke e2e (si S2 lo permitió) y la lista manual M1-M8 | resultados fechados en la sección `## Verification` de este plan, todos OK | T069, T033, T034, T035 | AC1-AC4, AC10, AC11, AC16, AC18 |
| T071 |  | Retire Streamlit, `scheduler.py` y el servicio `dashboard`. Es destructivo: confirmar antes con el propietario | `grep -rn "streamlit" --include=*.py --include=*.txt .` sin resultados; `docker compose config` sin `dashboard`; `python main.py --help` ok; `pytest` verde | T069, T070 | AC14 |
| T072 |  | All done-checks pass + `verify.sh` green | todas las filas anteriores marcadas; `bash scripts/verify.sh` sale con 0 | all | spec §Acceptance |

### Receta R — migración de una vista (Interfaces común a T043-T067)

- **Consumes:** `page_modules/<módulo>.py::show_<vista>(…)` (el código actual), el cliente de API de T041, los componentes de T039 y el layout y las rutas de T038.
- **Pasos, en orden:**
  1. Test de caracterización que fija, con fixtures de DataFrame, las cifras que calcula hoy la vista → `tests/views/test_<vista>.py`.
  2. Extraer el cálculo a `src/views/<vista>.py::build(db, channel_id, params) -> dict`, sin `st.*`.
  3. La página Streamlit llama a `build()` y solo pinta.
  4. Endpoint `views.get("<vista>", …)` que serializa el resultado de `build()`.
  5. Vista React en `desktop/src/areas/<área>/<Vista>.tsx`, con estados vacío y de error.
  6. Fila de paridad con su ubicación y una cifra de control.
- **Produces:** `GET views/<vista>` → `ViewModel` | `Empty{hint}` | `DbDown` | `NotFound`.
- **R verde** significa que se cumple todo esto:
  - `pytest tests/views/test_<vista>.py` verde;
  - `pytest tests/engine/test_api_views.py -k <vista>` verde;
  - `pnpm --dir desktop exec vitest run src/areas/<área>/<Vista>.test.tsx` verde, incluido el estado vacío;
  - `grep -n "views.<vista>" page_modules/<módulo>.py` encuentra la llamada;
  - la fila del apéndice está marcada con ✓.

### T015 — Interfaces

- **Produces:** `pipeline.run(steps, run_id, cancel_requested: Callable[[], bool], emit: Callable[[dict], None]) -> RunResult{status: success|failed|cancelled|skipped, last_step: str|None, error: str|None}`. Eventos `{"type": "step_started"|"step_done"|"finding"|"run_done", ...}`, uno por línea en stdout con `--events-jsonl`.

### T021 — Interfaces

- **Consumes:** `RunResult` y los eventos de T015, `pipeline_runs` de T017 y `jobs` de T019.
- **Produces:** `pipeline.start(trigger) -> {run_id} | 409 {already_running: run_id}`; `pipeline.cancel(run_id) -> {cancelling: true} | 409 {not_running}`; `events.stream` → `{type, run_id?, payload}`.

### T034 — Interfaces

- **Consumes:** los eventos `finding{type, count, keys}` y `run_done{status}` de T021, y el mecanismo de activación elegido en S1 (T006).
- **Produces:** el mapa tipo → ruta:
  - `competitor_alert` → `/audiencia/competencia`
  - `late_bloomer` → `/contenido/despegue-tardio`
  - `cannibalization` → `/contenido/canibalizacion`
  - `health_drop` → `/inicio/salud`
  - `evergreen` → `/rendimiento/revenue`
  - `run_done` → `/sistema/pipeline`

### Lista manual guiada (para T031-T034, T042 y T070)

- **M1:** tras Salir no queda ningún proceso python.
- **M2:** al cerrar la ventana, la app sigue en la bandeja.
- **M3:** la app arranca al iniciar sesión, sin abrir ventana.
- **M4:** una segunda apertura enfoca la ventana existente.
- **M5:** Salir con un pipeline en curso, probando confirmar y rechazar.
- **M6:** pulsar cada tipo de toast abre su vista.
- **M7:** un pipeline real completo lanzado desde la app, con progreso y notificación.
- **M8:** con MariaDB parado aparece el mensaje y el reintento funciona.

## Review Workload Forecast

| Dimension | Forecast | Why |
| --- | --- | --- |
| Estimated changed lines | 15.000-25.000, incluidas unas 6.000 de Streamlit que se borran en T071 | 72 tareas: motor, pipeline, shell Rust, sistema de diseño, y las 24 vistas extraídas y reescritas |
| Files / areas | más de 150 ficheros en `src/engine/`, `src/views/`, `src/pipeline.py`, `src/database.py`, `page_modules/`, `desktop/` (TS + Rust), `tests/` y docs | stack cruzado Python + TypeScript + Rust |
| Review risk | high | cambio de arquitectura, procesos del SO, API local con token, esquema nuevo y una retirada destructiva |
| Suggested delivery | autochain por tarea: un commit por tarea en `main` con la suite verde, con pausas en T008, T036, T045 y T071 | supera con mucho el presupuesto por defecto de 400 líneas; la constitución manda directo a `main`; las pausas caen en los puntos de mayor incertidumbre o destructivos |
