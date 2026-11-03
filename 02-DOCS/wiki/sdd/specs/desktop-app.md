---
type: spec
title: Spec — App de escritorio (sustituye al dashboard Streamlit)
description: WHAT and WHY de la app de escritorio — problema, objetivos, comportamiento y criterios de aceptación.
tags: [sdd, spec]
timestamp: 2026-11-03T00:20:00Z
topic: sdd
slug: desktop-app
status: clarified
---

# Spec — App de escritorio (sustituye al dashboard Streamlit)

> Slug: `desktop-app` · Status: clarified · Created: 2026-11-02
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
- Un presupuesto numérico de rendimiento (no era un motivo del cambio).
- Recuperar ejecuciones programadas perdidas con el PC apagado o la app cerrada.

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
- "Salir" con un pipeline en curso pide confirmación. Si se confirma, el pipeline se cancela (como en
  "Cancelar", abajo) y la app termina. Si no, la app sigue.
- Con "Iniciar con Windows" activo, la app arranca al iniciar sesión, directamente en la bandeja.
- La app recuerda entre sesiones el último canal, el área abierta, el tema (claro/oscuro) y el orden
  de widgets del resumen. En el primer arranque hereda el tema y el orden de widgets que el
  usuario tiene hoy en el dashboard.
- Los datos y acciones de la app solo son accesibles desde el propio PC. Ningún otro equipo de la
  red puede leerlos ni lanzarlos.
- La app no abre el navegador salvo para el consentimiento de Google (OAuth) que piden las
  exportaciones a Calendar/Sheets.

### Navegación (reorganizada por tareas)

- Las 24 vistas actuales se agrupan en pocas áreas según la tarea. Ninguna vista se pierde: cada
  vista actual tiene una ubicación en la app.
- Un selector de canal, siempre visible, filtra todas las áreas.
- Modo presentación (15.2): un atajo pone la vista actual a pantalla completa, sin navegación ni
  controles. El mismo atajo, o Esc, sale.
- Las acciones largas de IA (generar recomendación, guion, análisis de miniaturas, etc.) muestran
  progreso mientras corren, y no se pueden relanzar hasta que terminan.

### Pipeline y programación

- El usuario lanza el pipeline completo desde la app y ve en qué paso va, cuánto lleva y el resultado final.
- No se pueden lanzar dos ejecuciones a la vez: con una en curso, la acción de lanzar está bloqueada
  y se ve la que está corriendo.
- El usuario puede cancelar una ejecución en curso. La cancelación se hace efectiva al terminar el
  paso actual: ningún paso queda a medias, y el historial la marca como "cancelada" con el último
  paso completado.
- El horario es una ejecución diaria a una hora configurable (hoy, 15:00 America/Panama). La app
  ejecuta el pipeline a esa hora mientras está viva, en ventana o en la bandeja. El usuario ve la
  próxima ejecución programada y el historial de las últimas.
- El usuario puede ver y cambiar la hora desde la app. El cambio persiste entre sesiones.
- `python main.py` sigue funcionando como CLI para ejecutar el pipeline sin la app.

### Notificaciones

- La app muestra una notificación de Windows cuando el pipeline termina, falla o se cancela;
  genera alertas de competencia; detecta despegues tardíos; detecta canibalización nueva; la salud
  de un canal cae de forma brusca respecto al informe anterior; o un vídeo pasa a ser evergreen
  (antes no lo era).
- Cada ejecución produce como mucho una notificación por tipo de evento, que agrupa todos los
  casos de ese tipo (por ejemplo, "3 alertas de competencia").
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
- El motor de análisis no arranca (falta Python o una dependencia): la app muestra un error en
  español que dice qué falta, con una acción de reintento, y no se cierra.
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
  Streamlit ni su dependencia, ni `scheduler.py`, ni el servicio `dashboard` de docker-compose; la
  suite de tests sigue verde y `python main.py` ejecuta el pipeline completo.
- Given un pipeline en curso, When el usuario pulsa Cancelar, Then el paso actual termina, no
  empieza el siguiente, el historial lo marca "cancelada" con el último paso completado, y llega
  una notificación de cancelación.
- Given un pipeline en curso, When el usuario elige "Salir", Then se le pide confirmación. Si
  rechaza, la app y el pipeline siguen. Si confirma, el pipeline se cancela y la app termina.
- Given una ejecución que produce 3 alertas de competencia, When termina, Then llega una sola
  notificación de competencia que dice 3, y no tres separadas.
- Given un informe de salud con una caída brusca respecto al anterior, una canibalización nueva o
  un vídeo que pasa a evergreen, When termina la ejecución, Then llega la notificación de ese tipo.
  Pulsarla abre la vista correspondiente.
- Given cualquier vista, When el usuario activa el modo presentación, Then la vista ocupa toda la
  pantalla sin navegación ni controles. When pulsa Esc, Then vuelve a la vista normal.
- Given una acción larga de IA en curso, When el usuario intenta relanzarla, Then no puede, y ve el
  progreso de la que está corriendo.
- Given el motor de análisis sin poder arrancar, When el usuario abre la app, Then ve un mensaje en
  español que nombra lo que falta y un botón de reintento.
- Given otro equipo en la misma red, When intenta acceder a los datos o acciones de la app, Then no lo consigue.
- Given el usuario cambia la hora programada a las 09:30, When cierra y vuelve a abrir la app, Then
  la próxima ejecución mostrada es a las 09:30.

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
- **suposición tomada** — "Caída brusca" de salud = la puntuación de salud baja 10 puntos o más
  respecto al informe anterior del mismo canal. *Base:* la puntuación es un entero de 0 a 100.
  *Riesgo:* demasiados o muy pocos avisos. El umbral es ajustable.
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
- 2026-11-02 — Spec aprobada por el propietario ("si apruebo las spec y continua con clarify").

## Clarifications

- 2026-11-02 — ¿Control del pipeline en curso? → Se puede cancelar; la cancelación es efectiva
  al terminar el paso actual. "Salir" pide confirmación y cancela. → *Por qué:* el pipeline tarda
  minutos, y cortar a mitad de paso deja datos incompletos.
- 2026-11-02 — ¿Qué más notifica? → Canibalización nueva, caída brusca de salud y vídeo que pasa a
  evergreen, además de los de la spec; como mucho una notificación agrupada por tipo y ejecución.
  → *Por qué:* elección del propietario. Agrupar evita una ráfaga de avisos.
- 2026-11-02 — ¿`scheduler.py` y Docker al retirar Streamlit? → Se retiran `scheduler.py` y el
  servicio `dashboard`; `python main.py` sigue como CLI. → *Por qué:* la app es dueña del horario,
  y mantener dos caminos duplica el mantenimiento.
- 2026-11-02 — ¿Modo presentación? → Pantalla completa limpia de la vista actual, con un atajo y Esc.
  → *Por qué:* cubre el uso de 15.2 sin replicar su mecánica.
- 2026-11-02 — Defaults aceptados sin objeción: heredar las preferencias actuales en el primer
  arranque; progreso y bloqueo de relanzado en las acciones largas de IA; error claro si el motor
  no arranca; acceso solo local; navegador solo para OAuth; sin presupuesto de rendimiento.
- 2026-11-02 — Suposiciones validadas (se mantienen): agrupación en 6 áreas; las ejecuciones
  perdidas no se recuperan; Streamlit sigue en uso diario hasta la paridad.
- Resultado por punto: agrupación → validada · ejecuciones perdidas → validada · Streamlit hasta
  paridad → validada · otros eventos → preguntada y resuelta · modo presentación → preguntada y
  resuelta · dirección técnica → se deja diferida a `plan` · dirección visual → anotada, sigue no
  formulable (`design` en `plan`).
