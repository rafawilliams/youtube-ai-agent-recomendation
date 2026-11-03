---
type: analysis
title: Analysis — desktop-app (constitución ↔ spec ↔ plan ↔ tasks)
description: Informe de consistencia previo a implement para la app de escritorio (segunda pasada).
tags: [sdd, analysis]
timestamp: 2026-11-03T02:40:00Z
topic: sdd
slug: desktop-app
---

# Analysis — desktop-app (segunda pasada)

> Entradas: [constitución](../constitution.md) v1.2.0 · [spec](../specs/desktop-app.md) (clarified, 26 criterios) · [plan + tasks](../plans/desktop-app.md) (approved, T001-T080)
> Informe puntual del 2026-11-02. Sustituye a la primera pasada (1 CRITICAL, 3 HIGH, 8 MEDIUM, 5 LOW), cuyos hallazgos se resolvieron así:
> - **Constitución:** F1.
> - **Spec:** F5, y criterios para F2 y F3.
> - **Plan:** F3, F4, F6, F10, F13.
> - **Tareas:** F2, F7-F9, F11, F12, F15-F18, con las tareas nuevas T073-T080.
> - **Pendiente a propósito:** F14 queda programado en T071.

## Veredicto

**GATE: PASS** — 0 CRITICAL · 0 HIGH · 0 MEDIUM · 2 LOW

## Mapa de cobertura (spec → plan → tasks)

| REQ | Requisito de la spec (corto) | Plan | Tasks | Estado |
| --- | --- | --- | --- | --- |
| R1 | Abrir sin terminal ni navegador (AC1) | §2 Shell, Supervisor | T031, T038, T013 | covered |
| R2 | Cerrar→bandeja, Salir termina, el scheduler sigue (AC2) | §2 Shell | T032, M2 | covered |
| R3 | Autoarranque minimizado (AC3) | §2 Shell | T032, T042, M3 | covered |
| R4 | Instancia única (AC4) | §2 Shell | T032, M4 | covered |
| R5 | Tabla de paridad (AC5) | §6 apéndice | T043-T069 | covered |
| R6 | Cifras idénticas (AC6) | §2, receta R | T043-T067 | covered |
| R7 | Lanzar pipeline, ver paso, notificación (AC7) | §3 Runner | T015, T021, T034, T042 | covered |
| R8 | Sin doble ejecución (AC8) | §3 lock + jobs | T017, T020 | covered |
| R9 | Ejecución programada (AC9) | §3 Scheduler | T022-T023 | covered |
| R10 | Programada con otra en curso → `skipped` en el historial | §3 lock | T017, T022 | covered |
| R11 | Ejecuciones perdidas → filas "no ejecutada" (AC24) | §0, §4 `missed` | T079-T080, T042 | covered |
| R12 | Notificaciones por evento + agrupación (AC10, AC17, AC18) | §2-§3 Notificador | T028-T029, T034 | covered |
| R13 | MariaDB parado (AC11) | §3 health | T011, T035, M8 | covered |
| R14 | Estado vacío (AC12) | receta R | T039, T043-T067 | covered |
| R15 | Temas (AC13) | §4 Settings | T037, T039 | covered |
| R16 | Retirada de Streamlit (AC14) | §6.12 | T071 | covered |
| R17 | Cancelar entre pasos (AC15) | §3 Runner | T014-T015, T020-T021 | covered |
| R18 | Salir con pipeline (AC16) | §2 Shell | T033, M5 | covered |
| R19 | Modo presentación (AC19) | §6.8 | T040 | covered |
| R20 | Acción IA no relanzable (AC20) | §3 jobs | T018-T019, T057-T058, T064 | covered |
| R21 | Motor no arranca (AC21) | §3 Supervisor | T030-T031, T035 | covered |
| R22 | Sin acceso desde la red; token fuera de la línea de comandos (AC22) | §0, §3 | T011-T012 | covered |
| R23 | Hora persistente (AC23) | §4 Settings | T009-T013, T022 | covered |
| R24 | Credencial caducada o ausente → mensaje con renovación (AC25) | §3 CredentialError | T075-T076, T042 | covered |
| R25 | Secretos enmascarados (AC26) | §0, §3 `redact()` | T073-T074, T034 | covered |
| R26 | Recordar canal, área, tema y widgets; heredar prefs | §4 Settings | T009, T037, T038 | covered |
| R27 | Navegador solo para OAuth | — | T059 | covered |
| R28 | `python main.py` sigue como CLI | §3 | T015, T071 | covered |
| R29 | Telegram sin cambios | §1 | T014 (caracterización) | covered |
| R30 | Caída de salud entre puntuaciones diarias, sin tocar los informes IA | §0, §4 HealthScore | T077-T078, T028-T029 | covered |

Sin `GAP` ni `DRIFT`.

## Hallazgos

| # | Severidad | Tipo | Artefacto A (ubicación) | Artefacto B (ubicación) | Conflicto | Resolver en |
| --- | --- | --- | --- | --- | --- | --- |
| N1 | LOW | Cobertura parcial | constitución §17 (estado vacío y de error en cada página) | T042 (área Sistema) | El done-check de T042 no nombra los estados vacío y de error. Las vistas de la receta R sí los cubren | tasks, o aceptarlo |
| F14 | LOW | Wording (diferido) | constitución §18 "(hoy en `config/dashboard_prefs.json`)" | plan §4 Settings | Se corrige con el PATCH programado en T071 | ya programado (T071) |

## Comprobaciones sin hallazgo

- **Constitución v1.2.0:**
  - El §22 cubre la BD de pruebas.
  - El SQL nuevo vive en `src/database.py` (T017, T027, T078).
  - Los secretos están cubiertos por `redact()` y `ENGINE_TOKEN`.
  - El stack coincide con el §20 y la calidad con el §21.
  - La autoría y los commits siguen §12-14.
- **Carriers:** el §0 lleva los valores exactos (umbrales, parámetros de los detectores, variables a enmascarar, máximo de 7). Hay Interfaces en la receta R, T015, T021 y T034, y el lock tiene nombre en T017.
- **Paralelismo:**
  - Los spikes van en carpetas desechables.
  - Cada vista registra su endpoint en un módulo propio y su ruta en el fichero de su área.
  - Las cuatro tareas [P] de inicio de área tocan módulos distintos de `page_modules/`.
- **Duplicación:** ninguna.

## Enrutado recomendado

- **implement:** se puede empezar por T001.
- **N1:** se acepta como está o se añade al done-check de T042 cuando se llegue a esa tarea.
