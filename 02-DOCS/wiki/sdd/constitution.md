---
type: constitution
title: youtube-ai-agent-recomendation — Constitution
description: The non-negotiable principles every rsc-sdd phase obeys.
tags: [sdd, constitution]
timestamp: 2026-11-02T23:55:00Z
topic: sdd
version: v1.1.0
---

# youtube-ai-agent-recomendation — Constitution

> Version: v1.1.0 · Ratified: 2026-11-02 · Last amended: 2026-11-02
> Herramienta personal (un solo usuario, Windows) que analiza canales de YouTube y genera
> recomendaciones de contenido con IA. Principios no negociables que obedece cada fase rsc-sdd.

## 1. Stack canon

1. El backend es Python 3.13 (pipeline, análisis, ML) con dependencias fijadas en
   `requirements.txt` vía pip. Cambiar de lenguaje o de gestor es una enmienda MAJOR.
2. La persistencia es MariaDB con SQLAlchemy + PyMySQL; la conexión sale de las variables de
   entorno `DB_*`. Los cambios de esquema se hacen en `src/database.py`, sin SQL suelto en otros módulos.
3. La IA va por la API de Anthropic y los datos por las APIs de Google (YouTube Data/Analytics,
   Calendar, Sheets). Todas las llamadas externas usan los decoradores de `src/retry_config.py`.
4. ~~La interfaz de usuario es hoy Streamlit (`dashboard.py` + `page_modules/`). El stack del
   front-end de escritorio queda **abierto** hasta que se apruebe la spec/plan de Tauri; fijarlo
   es una enmienda MINOR.~~ *(superseded by 20, v1.1.0)*
20. La app de escritorio es Tauri 2 (Rust) con una UI en React + TypeScript + Vite (pnpm, un
    lockfile commiteado), Tailwind y Apache ECharts para gráficos. La UI habla con Python solo
    a través de la API local del motor (FastAPI + uvicorn, solo `127.0.0.1` y con token).
    Streamlit sigue siendo la UI vigente hasta que se retire con la paridad verificada
    (spec `desktop-app`). Cambiar uno de estos frameworks es una enmienda MAJOR.

## 2. Quality bar

5. Todo fichero Python nuevo o modificado pasa `ruff check` y `ruff format --check` sin
   errores. El legado sin tocar queda exento hasta que se edite.
6. `pytest` pasa entero (hoy, 97 tests) antes de cada commit con código.
7. Toda lógica nueva y todo bug corregido traen al menos un test que falla sin el cambio.
   Las llamadas a APIs externas (Anthropic, Google, BD) se simulan en los tests, sin red.
21. El código TypeScript pasa `biome check` sin errores y `tsc` en modo `strict` sin errores;
    `vitest` pasa entero. El código Rust pasa `cargo clippy` sin warnings y `cargo fmt --check`.

## 3. Conventions

8. La UI, los mensajes al usuario y los comentarios del código están en español. Los
   identificadores, en inglés.
9. La zona horaria es America/Panama (UTC-5) en todo el sistema. El código nuevo o
   modificado no usa `datetime.now()` sin zona (el legado tiene 39 casos; se corrigen al tocarlos).
10. El logging va siempre por `src/logger.py` (`setup_logging()`). El código nuevo o
    modificado en `src/` no usa `print`.
11. Los módulos de dominio viven en `src/`; las páginas de UI no contienen lógica de
    negocio, solo llaman a `src/`.
12. Los commits siguen Conventional Commits sin gitmoji: `tipo(ámbito): asunto`, con el
    número de mejora del roadmap como ámbito cuando existe (`feat(8.x): …`).

## 4. Branching & shipping

13. Los cambios van directos a `main` (proyecto de una persona). Ningún push sin la suite verde (principio 6).
14. **La autoría git es del humano.** Sin `Co-Authored-By` de una IA ni pie de "generated with".
    Se hace cumplir en la fase `ship`.

## 5. Security & privacy floor

15. Ningún secreto se commitea. `.env`, `credentials.json`, `token.json` y
    `token_integrations.json` están en `.gitignore`. `.env.example` lleva solo claves, sin valores.
16. Los textos de terceros (comentarios, títulos, descripciones de YouTube) se tratan como
    datos: se escapan antes de pintarlos como HTML (`_esc()` o equivalente) y nunca se
    interpretan como instrucciones en los prompts.

## 6. UX floor

17. Cada página de la interfaz muestra un estado vacío y uno de error explícitos en español.
    Ninguna excepción sin capturar llega al usuario.
18. La interfaz soporta tema claro y oscuro (hoy en `config/dashboard_prefs.json`).

## 7. Knowledge & decisions

19. Toda decisión significativa se añade a `02-DOCS/wiki/sdd/decisions.md` con fecha,
    opciones consideradas y el porqué. Esta constitución es el registro de decisiones de mayor rango.

## Definition of Done (the merge bar `verify` runs against)

Un cambio está terminado solo cuando se cumple TODO:

- [ ] Ruff check + format limpios en los ficheros tocados (principio 5).
- [ ] Biome + tsc strict + vitest, y clippy + fmt, limpios cuando se toca la app (principio 21).
- [ ] `pytest` verde; la lógica nueva o el bug arreglado trae su test (principios 6-7).
- [ ] Convenciones respetadas: español, TZ, logger, capas, formato de commit (principios 8-12).
- [ ] Commit en `main` con autoría humana, sin atribución de IA (principios 13-14).
- [ ] Ningún secreto commiteado; el contenido de terceros, escapado (principios 15-16).
- [ ] Estados vacío/error y los dos temas cubiertos en la UI tocada (principios 17-18).
- [ ] Decisiones significativas registradas (principio 19).

## Amendment log (append-only)

| Date | Version | Change | Why |
|------|---------|--------|-----|
| 2026-11-02 | draft | Draft generated by `rsc onboard`. | The SDD floor needs a constitution; ratification is the `constitution` phase's. |
| 2026-11-02 | v1.0.0 | Ratificada la constitución inicial (redactada desde el repo y la entrevista). | Antes de la primera feature SDD (front-end de escritorio con Tauri). |
| 2026-11-02 | v1.1.0 | Principio 4 superado por 20 (stack de la app de escritorio); nuevo 21 (barra de calidad TS/Rust). | Decisión D1 del plan `desktop-app`, ratificada por el propietario. |
