---
type: progress
title: Progress — desktop-app
description: Ledger append-only de implement para la app de escritorio.
tags: [sdd, progress]
timestamp: 2026-11-03T03:00:00Z
topic: sdd
slug: desktop-app
---

# Progress — desktop-app (append-only)

## T001 — 2026-11-02

- status: complete
- red: n/a (tarea de entorno, sin código). Estado previo: `rustc` y `cargo` no existían en PATH.
- green: `rustc 1.99.1 (58cf8f9c3 2026-10-13)`, `cargo 1.99.1`, toolchain `stable-x86_64-pc-windows-msvc (default)`. Proyecto `cargo new` de prueba compilado con `cargo build` y ejecutado ("Hello, world!").
- entorno: rustup 1.29.2 instalado con winget (`Rustlang.Rustup`). MSVC ya presente (VS 2022 Community + Build Tools 2026) y Windows SDK 10.0.26100; no hizo falta instalar Build Tools.
- files: ninguno del repo.
- nota: las terminales abiertas antes de la instalación no ven `%USERPROFILE%\.cargo\bin` hasta reabrirse.
- decision: none
- blocker: falta `02-DOCS/wiki/sdd/config.yaml`; `implement` exige `sdd-init` antes de T002.

## Desvío — 2026-11-02 (antes de T002)

- Commits previos separados por tema (trabajo anterior del propietario + harness/SDD), `sdd-init` (config.yaml) e instalación de 9 skills.
- FTD `comment-prompt-injection` (commits `3a51c3b`, `67a522a`, `c08da72`), a raíz de la revisión automática de seguridad. Documento: `02-DOCS/wiki/ftd/comment-prompt-injection.md`.
- `.claude/settings.json`: `git push` y `gh pr create` pasan a pedir confirmación (`9862c16`).

## T002 — 2026-11-02

- status: complete
- red: `import fastapi` y `import uvicorn` → ModuleNotFoundError; no existía `ruff.toml`. Al instalar con el pin antiguo, `Anthropic(api_key='x')` → `TypeError: Client.__init__() got an unexpected keyword argument 'proxies'` (anthropic 0.18.1 + httpx 0.28.1).
- green: `pip install -r requirements.txt` exit 0; fastapi 0.162.0, uvicorn 0.54.0, httpx 0.28.1, anthropic 0.84.0, ruff 0.17.3. `ruff check src/logger.py` y `ruff format --check src/logger.py` se ejecutan con `ruff.toml` (Settings path confirmado) y señalan deuda de legado exenta (§5). `pytest tests/test_dependencies.py` → 2 passed. Suite: 106 passed.
- triangulation: `tests/test_dependencies.py` construye el cliente real de Anthropic (sin red) para detectar combinaciones de versiones rotas que los mocks ocultan.
- files: requirements.txt, ruff.toml, tests/test_dependencies.py
- decision: `anthropic>=0.84.0,<1.0.0` en lugar de `==0.18.1` (ver decisions.md).
- incidente: el primer `pip install -r` bajó anthropic de 0.84.0 a 0.18.1 en el `.venv`; restaurado de inmediato a 0.84.0.
- blocker: none
