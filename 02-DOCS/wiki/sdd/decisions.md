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
