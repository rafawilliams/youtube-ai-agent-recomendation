---
type: spec
title: Spec — App de escritorio (sustituye al dashboard Streamlit)
description: WHAT and WHY de la app de escritorio — problema, objetivos, comportamiento y criterios de aceptación.
tags: [sdd, spec]
timestamp: 2026-11-03T00:20:00Z
topic: sdd
slug: desktop-app
status: draft
---

# Spec — App de escritorio (sustituye al dashboard Streamlit)

> Slug: `desktop-app` · Status: draft · Created: 2026-11-02
> Inherits: [constitution](../constitution.md) v1.0.0

## Problem & why

Hoy, para usar la herramienta, el propietario abre una terminal, arranca el dashboard, va al
navegador y deja otra terminal bloqueada con el scheduler. No hay nada que viva en el escritorio.
Cuando el pipeline termina, falla o detecta algo (alertas de competencia, despegues tardíos), no se
entera hasta que abre el dashboard. Además, la interfaz son 24 vistas en una sola
lista larga del sidebar, ordenadas por cómo se construyeron y no por lo que él hace con ellas
(planificar, analizar, vigilar a la competencia). Y el aspecto es el genérico de Streamlit.

## Cost of not building it

Moderado, y solo lo paga el propietario. Sigue con el ritual de terminal, navegador y terminal, y
con un scheduler que muere al cerrar esa terminal, así que las ejecuciones programadas se pierden
sin aviso. Encontrar una vista entre 24 sigue costando, y no hay avisos en el escritorio. No
hay coste para terceros: es una herramienta personal.

## The cheapest alternative

1. Instalar el dashboard como app desde Chrome (ventana propia y acceso directo).
2. Registrar `scheduler.py` como tarea de Windows.
3. Reagrupar el sidebar de Streamlit.

Juntas, cubren la ventana propia, el pipeline que no muere y parte del orden. No dan bandeja del
sistema ni notificaciones nativas. Tampoco dan el rediseño de la interfaz, que es uno de los dos
motivos declarados. Por eso no bastan.

## Goals

- La herramienta se usa como una app de Windows: se abre desde el menú Inicio o la bandeja, sin
  terminal ni navegador.
- Todo lo que hoy hace el dashboard se puede hacer en la app (paridad total), reorganizado por áreas
  de tarea.
- El pipeline se lanza, se sigue y se programa desde la app, y la ejecución programada ocurre
  mientras la app está viva.
- Los eventos importantes llegan como notificación de Windows.
- Alcanzada y verificada la paridad, el dashboard Streamlit se retira y queda una sola interfaz.

## Non-goals / out of scope

- Distribuir la app a otras personas: instalador para terceros, firma de código, auto-actualización
  pública, multiusuario.
- Empaquetar o instalar Python, MariaDB ni sus dependencias. Se asume que existen en la máquina.
- Cambiar la lógica de análisis, ML o IA. Las mismas entradas producen los mismos resultados que hoy.
- Funciones nuevas de análisis que no existan hoy en el dashboard.
- macOS y Linux.
- Acceso remoto o desde el móvil.
- Integración con Notion (14.2) y autenticación del dashboard.
- Telegram. Esta mejora no lo toca, no lo sustituye ni lo configura: sigue como está hoy.

## Users & context

Un único usuario: el propietario del canal. Trabaja en su PC con Windows 11, donde ya corre MariaDB
como servicio y están instalados Python y las credenciales de Google y Anthropic. Usa la herramienta
para decidir qué publicar y cuándo, revisar el rendimiento de sus canales y vigilar a los
competidores. Lee la interfaz en español.

## Behaviour

### Arranque y ciclo de vida

- Main: al abrir la app, aparece una ventana con la navegación por áreas y los datos del canal
  seleccionado. No hace falta ningún paso manual previo.
- Al cerrar la ventana, la app sigue viva en la bandeja del sistema. Solo sale con "Salir" desde la bandeja.
- Con "Iniciar con Windows" activo, la app arranca al iniciar sesión, directamente en la bandeja.
- La app recuerda entre sesiones el último canal, el área abierta, el tema (claro/oscuro) y el orden
  de widgets del resumen.

### Navegación (reorganizada por tareas)

- Las 24 vistas actuales se agrupan en pocas áreas según la tarea. Ninguna vista se pierde: cada
  vista actual tiene una ubicación en la app.
- Un selector de canal, siempre visible, filtra todas las áreas.

### Pipeline y programación

- El usuario lanza el pipeline completo desde la app y ve en qué paso va, cuánto lleva y el resultado final.
- No se pueden lanzar dos ejecuciones a la vez: con una en curso, la acción de lanzar está bloqueada
  y se ve la que está corriendo.
- La app ejecuta el pipeline según el horario configurado mientras está viva, en ventana o en la
  bandeja. El usuario ve la próxima ejecución programada y el historial de las últimas.
- El usuario puede ver y cambiar el horario desde la app.

### Notificaciones

- La app muestra una notificación de Windows cuando el pipeline termina, cuando falla y cuando el
  pipeline genera alertas de competencia o detecta despegues tardíos.
- Al pulsar una notificación, la app se abre en la vista relacionada.

### Edge cases

- Sin datos todavía (primer uso o canal sin vídeos): cada vista muestra un estado vacío en español
  que dice qué hacer, por ejemplo ejecutar el pipeline.
- Ejecución programada que llega con otra en curso: no se lanza una segunda. La omisión queda en el historial.
- El equipo estaba apagado o la app cerrada a la hora programada: esa ejecución no ocurre. El
  historial lo refleja al volver.
- Se abre la app con otra instancia ya en marcha: se trae al frente la existente y no se crea otra.

### Error paths

- Base de datos no disponible: la app abre igual, muestra un estado de error claro en español con
  la causa y una acción de reintento, y no se cierra.
- Credenciales de Google o Anthropic ausentes o caducadas: el paso afectado del pipeline falla con
  un mensaje que dice qué credencial y cómo renovarla. El resto de la app sigue usable.
- Fallo del pipeline: notificación de fallo, y el historial guarda el paso donde falló y el mensaje.
- Ninguna excepción sin capturar llega al usuario como pantalla rota (constitución, principio 17).
- Las claves y tokens nunca aparecen en pantalla, notificaciones ni mensajes de error.

## Acceptance criteria

- Given la app instalada y MariaDB en marcha, When el usuario la abre desde el menú Inicio, Then ve
  la vista de inicio con datos del último canal usado sin abrir ninguna terminal ni navegador.
- Given la app abierta, When el usuario cierra la ventana, Then la app sigue en la bandeja y el
  scheduler sigue activo. When elige "Salir" en la bandeja, Then el proceso termina.
- Given "Iniciar con Windows" activado, When el usuario inicia sesión, Then la app está en la bandeja
  sin ventana abierta.
- Given la app ya en marcha, When el usuario la abre otra vez, Then hay una sola instancia y su ventana pasa al frente.
- Given la tabla de paridad (las 24 vistas actuales y sus funciones: widgets personalizables, modo
  presentación, comparador temporal, simulador what-if, generador de guiones, exportación a
  Calendar/Sheets, comparador A/B de miniaturas, etc.), When se recorre en la app, Then cada fila
  está marcada como presente con su ubicación, y no hay ninguna sin ubicar.
- Given los mismos datos en la BD, When se compara una cifra mostrada por Streamlit con la misma
  cifra en la app (al menos una por vista), Then coinciden.
- Given ninguna ejecución en curso, When el usuario lanza el pipeline, Then ve el paso actual
  mientras corre y, al acabar, una notificación de Windows con el resultado.
- Given una ejecución en curso, When el usuario intenta lanzar otra (manual o programada), Then no
  empieza una segunda y el intento queda reflejado.
- Given un horario configurado y la app viva en la bandeja, When llega la hora, Then el pipeline se
  ejecuta sin intervención y aparece en el historial.
- Given un pipeline que genera una alerta de competencia, When termina, Then llega una notificación.
  When el usuario la pulsa, Then la app se abre en la vista de competencia.
- Given MariaDB parado, When el usuario abre la app, Then ve un mensaje de error en español con un
  botón de reintento, y al arrancar MariaDB y reintentar se cargan los datos.
- Given cualquier vista y un canal sin datos, When se abre, Then se ve un estado vacío en español y no un error.
- Given cualquier vista, When se cambia entre tema claro y oscuro, Then todo el texto y los
  gráficos siguen legibles, y el tema se recuerda al reabrir.
- Given la paridad verificada, When se retira Streamlit, Then el repo ya no contiene el dashboard
  Streamlit ni su dependencia, y la suite de tests sigue verde.

## Points to clarify

- **suposición tomada** — Agrupación propuesta de las 24 vistas en 6 áreas.
  *Inicio*: resumen personalizable y salud del canal. *Rendimiento*: performance, histórico,
  analytics avanzados, comparador temporal, cadencia y horarios, revenue y ROI. *Contenido*:
  análisis de contenido, series y formatos, tendencias, canibalización, despegue tardío.
  *Planificar y crear*: recomendaciones, generar recomendación, plan semanal, predicciones de
  viralidad/vistas/retención, miniaturas y comparador A/B. *Audiencia y competencia*:
  comentarios, competencia, comparar canales. *Sistema*: pipeline, programación y ajustes.
  *Base:* la elección "reorganizar por tareas" y los grupos que ya existen en el sidebar.
  *Riesgo:* si el usuario piensa en otras tareas, cambia la navegación, pero no el criterio de paridad.
- **suposición tomada** — Las ejecuciones programadas perdidas (PC apagado o app cerrada) no se
  recuperan al volver. *Base:* es lo que pasa hoy con `scheduler.py`. *Riesgo:* si se quiere
  recuperar la última, cambia un criterio de aceptación.
- **suposición tomada** — Mientras no se alcance la paridad, Streamlit sigue siendo la herramienta de
  uso diario y la app se desarrolla al lado. *Base:* la elección "paridad total antes de usarla".
  *Riesgo:* el esfuerzo es grande y la app no aporta valor hasta el final. El propietario lo eligió
  sabiéndolo; queda registrado.
- **pregunta abierta** — ¿Qué otros eventos merecen notificación, además de los listados
  (pipeline terminado o fallido, alertas de competencia, despegues tardíos)? ¿Canibalización
  detectada? ¿Cambios bruscos de salud del canal?
- **pregunta abierta** — ¿El modo presentación (15.2) se mantiene tal cual, o se sustituye por
  pantalla completa o exportar?
- **decisión diferida** — Dirección técnica: Tauri como contenedor de escritorio (requisito del
  propietario), con la lógica en Python expuesta por una API local que la interfaz consume. Esto
  cubre también la mejora 18.1 (REST API). El detalle es de `plan`.
- **área no formulable** — La dirección visual del rediseño (estilo, densidad, tipo de gráficos).
  Hay una pregunta, pero aún no se puede enunciar sin una propuesta delante. La aborda `design`/`ux`
  durante `plan`.

## Revisions

- 2026-11-02 — Telegram sale del alcance (petición del propietario). Se elimina la suposición
  "Telegram sigue en paralelo" y se añade a Non-goals. Se quitan también sus menciones en el
  problema y en el coste.
