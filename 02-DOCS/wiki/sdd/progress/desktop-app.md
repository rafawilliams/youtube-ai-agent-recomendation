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
