---
type: decisions
title: SDD decisions log
description: Append-only log of significant decisions in the rsc-sdd chain.
tags: [sdd, decisions]
topic: sdd
---

# SDD decisions log (append-only)

## 2026-11-02 — Constitución v1.0.0 ratificada

- **Lint/tipos:** Ruff check + format en ficheros nuevos o tocados. *Opciones:* Ruff + mypy gradual; nada. *Por qué:* barrera baja, sin fricción sobre el legado.
- **Tests:** suite verde + test por cada lógica nueva o bug. *Opciones:* TDD + cobertura ≥80%; solo suite verde. *Por qué:* protege sin imponer un % arbitrario.
- **Commits:** Conventional Commits sin gitmoji; guarda de gitmoji desactivada con `.rsc/.no-gitmoji`. *Opciones:* gitmoji + Conventional. *Por qué:* mantener el estilo del historial.
- **Autoría:** solo humana, sin `Co-Authored-By` de IA. *Opciones:* mantener la coautoría de Claude (práctica previa). *Por qué:* elección del propietario.
- **Legado:** las reglas de TZ, `print` y Ruff aplican al código nuevo o modificado; el legado (39 `datetime.now()` sin zona, `print` en 2 módulos) se corrige al tocarlo.

## 2026-11-02 — Front-end de escritorio (Tauri): encuadre previo a la spec

- **Usuario:** solo el propietario, en su PC Windows. Python, MariaDB y Docker ya están instalados; no hace falta empaquetarlos.
- **Motivo:** integración con el escritorio + rediseño de la interfaz. *Descartado:* envolver Streamlit en una ventana Tauri (no resuelve el rediseño); "no construir" (instalar el dashboard como app desde Chrome), que tampoco lo resuelve.

## 2026-11-02 — Spec `desktop-app` redactada (draft)

- **Alcance:** paridad total con las 24 vistas antes de usar la app; Streamlit se retira después de verificarla. *Opciones:* un núcleo de vistas conviviendo con Streamlit; solo el shell con la integración. *Por qué:* elección del propietario, que conoce el coste (la app no aporta valor hasta el final).
- **Escritorio:** pipeline desde la app, bandeja, notificaciones nativas e inicio con Windows. La app es dueña de la ejecución programada mientras está viva y sustituye a `scheduler.py`.
- **Navegación:** reorganizada por tareas (6 áreas propuestas, a validar en `clarify`). *Opción descartada:* mismas páginas con otro aspecto.
- **Dirección técnica (diferida a `plan`):** Tauri + API local en Python, que también cubre la mejora 18.1. *Opciones descartadas:* sidecar por stdio (protocolo propio); Rust leyendo la BD (duplica lógica fuera de `src/`, choca con el principio 11).

## 2026-11-02 — Spec `desktop-app`: Telegram fuera de alcance

- Telegram no forma parte de esta mejora. La app no lo toca ni lo sustituye. *Por qué:* petición del propietario al revisar la spec.

## 2026-11-02 — Spec `desktop-app` aprobada y aclarada (status: clarified)

- Cancelación al terminar el paso en curso; "Salir" confirma y cancela.
- Notificaciones añadidas: canibalización, caída de salud (≥10 puntos, ajustable), evergreen. Agrupadas por tipo y ejecución.
- Al retirar Streamlit se retiran también `scheduler.py` y el servicio `dashboard` de docker-compose; `main.py` sigue como CLI.
- Modo presentación = pantalla completa limpia.

## 2026-11-02 — Plan `desktop-app` redactado (draft)

- **Estrangulamiento:** la lógica de cada página se extrae a `src/views/` y la consumen Streamlit y la API; la paridad de cifras se cumple por construcción. *Alternativa descartada:* reescribir cada vista en la nueva UI y comparar después. *Por qué:* constitución §11 y criterio de cifras idénticas.
- **Pipeline en subproceso** gestionado por el motor, con eventos JSONL, cancelación entre pasos y lock global en MariaDB (cubre CLI y app). *Por qué:* aislar fallos y cuelgues, y mantener `python main.py` como CLI.
- **Detectores al pipeline:** salud (extraída de `page_modules/health.py`), despegue tardío y canibalización pasan a ser pasos para poder notificar. Tabla NotifiedEvent para distinguir lo nuevo.
- **Sin rama ni worktree:** constitución §13 (directo a main).
- **Abiertas:** D1 stack front-end, D2 sembrar en silencio la primera ejecución, D3 dirección visual.

## 2026-11-02 — Plan `desktop-app` aprobado

- D1: React + TypeScript + Vite + Tailwind + ECharts, Biome/Vitest; motor FastAPI + uvicorn → constitución v1.1.0 (§20-21). *Opciones:* Svelte + ECharts; React + Plotly.js. *Por qué:* mayor ecosistema, plantilla oficial de Tauri, ECharts deja libertad para el rediseño.
- D2: la primera ejecución de los detectores siembra en silencio. *Por qué:* evitar avisos de todo lo ya existente.

## 2026-11-02 — Hallazgos de `analyze` resueltos (desktop-app)

- **F1:** constitución v1.2.0. El principio 7 queda superado por el 22, que permite una MariaDB local de pruebas con el marcador `db`. *Opción descartada:* simular la BD, que no prueba el lock real.
- **F4:** el pipeline guarda solo la puntuación diaria de salud en `channel_health_scores`, sin Claude; los informes con diagnóstico IA siguen bajo demanda. *Opción descartada:* informe completo diario, con coste de Claude por canal y día.
- **F5:** una fila "no ejecutada (app cerrada)" por día perdido, con un máximo de 7. *Opciones descartadas:* solo un aviso; nada.
- **F2, F3, F6, F7, F8-F13, F15-F18:** resueltos en spec, plan y tareas (T073-T080 nuevas). F14 queda en T071.
