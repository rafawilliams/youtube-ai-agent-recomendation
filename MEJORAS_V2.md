# Roadmap de Mejoras V2 — YouTube AI Agent

Nuevas funcionalidades propuestas para llevar el proyecto al siguiente nivel. Complementa `MEJORAS.md` (V1) con ideas que NO están duplicadas.

> **Estado actual del proyecto:** 13 páginas en dashboard, 2 modelos ML (viralidad + vistas), recomendaciones diarias + plan semanal con Claude, feedback loop, Analytics API, Google Trends, scheduler automático, persistencia de modelos.

---

## 7. Análisis de Competencia

### 7.1 Tracker de canales competidores
**Problema:** Solo se analizan los canales propios. No hay contexto de cómo se compara con la competencia directa.
**Mejora:** Nueva variable `COMPETITOR_CHANNEL_IDS` en `.env`. El pipeline extrae datos de competidores (sin Analytics API, solo Data API pública) y los guarda en las mismas tablas con un flag `is_competitor=True`.

**Funcionalidad:**
- Comparar frecuencia de publicación, engagement rate, y crecimiento de vistas
- Detectar qué temas están publicando los competidores esta semana
- Identificar "content gaps" — temas que funcionan en competidores pero no se cubren en el canal propio

**Dashboard:** Nueva página "🕵 Análisis de Competencia" con:
- Tabla comparativa: tu canal vs N competidores
- Gráfico de crecimiento de vistas relativo (normalizado al mismo punto de partida)
- Nube de palabras de títulos de competidores vs propios

### 7.2 Alertas de contenido competidor
**Mejora:** Cuando un competidor publica un video que obtiene >2x su promedio de vistas en las primeras 48h, generar una alerta con análisis de Claude sobre por qué despegó y cómo replicar el éxito.

---

## 8. Análisis de Thumbnails con IA (Visión)

### 8.1 Scoring de miniaturas con Claude Vision
**Problema:** Las miniaturas son el factor #1 de CTR pero no se analizan.
**Mejora:** Usar Claude Vision (modelo multimodal) para analizar las miniaturas de cada video:

```python
# Descargar thumbnail
thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

# Enviar a Claude Vision
response = client.messages.create(
    model="claude-sonnet-4-6-20250514",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "url", "url": thumbnail_url}},
            {"type": "text", "text": "Analiza esta miniatura de YouTube..."}
        ]
    }]
)
```

**Métricas a extraer:**
- Claridad del texto (legible en móvil?)
- Expresión facial (si hay rostro)
- Contraste y saturación de colores
- Composición visual (regla de tercios)
- Score general de "clickability" (1-10)

**Dashboard:** Galería de miniaturas con su score, ordenadas de mejor a peor. Sugerencias de mejora para las peores.

### 8.2 Comparador de thumbnails A/B
**Mejora:** Interfaz en el dashboard donde el usuario sube 2-3 opciones de miniatura y Claude Vision las compara, indicando cuál tiene mayor probabilidad de CTR y por qué.

---

## 9. Generador de Contenido con IA

### 9.1 Generador de guiones/outlines
**Problema:** La recomendación dice "haz un video sobre X" pero no ayuda con la estructura.
**Mejora:** Nueva función `generate_script_outline()` en `AIAnalyzer` que, dado un tema y tipo de video, genere:

- **Para Shorts (< 60s):**
  - Hook (primeros 3 segundos)
  - Desarrollo (punto principal)
  - CTA (call to action final)

- **Para videos largos:**
  - Hook + preview del valor
  - Estructura de secciones con timestamps estimados
  - Puntos clave por sección
  - CTA + sugerencia de video siguiente

**Dashboard:** Botón "Generar Guión" en cada recomendación que expande el outline en la misma página.

### 9.2 Generador de descripciones SEO
**Mejora:** Dado el título y tema, generar una descripción optimizada para YouTube SEO con:
- Primeras 2 líneas con palabras clave (visibles sin expandir)
- Timestamps sugeridos
- Hashtags relevantes (máx. 3, como recomienda YouTube)
- Links sugeridos a videos anteriores relacionados

### 9.3 Generador de tags optimizados
**Mejora:** Analizar los tags de los videos más exitosos del canal + tendencias actuales para sugerir una lista de 15-20 tags ordenados por relevancia. Usar la combinación de datos históricos del canal + Google Trends.

---

## 10. Sistema de Alertas Inteligentes

### 10.1 Alertas por Telegram Bot
**Mejora:** Bot de Telegram (`python-telegram-bot`) que envíe automáticamente:

| Evento | Trigger | Mensaje |
|--------|---------|---------|
| Recomendación diaria | Después de `main.py` | Resumen + título sugerido |
| Video viral detectado | Vistas > 3x promedio en 24h | Alerta con métricas |
| Caída de engagement | Engagement < 50% del promedio últimos 7 días | Advertencia |
| Competidor despega | Video competidor > 2x su promedio | Análisis rápido |
| Plan semanal | Lunes a las 8am | Calendario de la semana |

**Configuración en `.env`:**
```ini
TELEGRAM_BOT_TOKEN=123456:ABC-DEF
TELEGRAM_CHAT_ID=987654321
ALERTS_ENABLED=true
```

### 10.2 Webhook genérico
**Mejora:** Soporte para webhooks HTTP POST (compatible con Discord, Slack, Zapier, Make, n8n):
```python
WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
WEBHOOK_EVENTS=recommendation,viral_alert,weekly_plan
```

---

## 11. Análisis Avanzado de Audiencia

### 11.1 Detector de "mejores fans" via comentarios
**Problema:** No se sabe quiénes son los espectadores más activos.
**Mejora:** Extraer comentarios via YouTube API y crear ranking de comentaristas frecuentes:
- Top 20 comentaristas por frecuencia
- Sentimiento promedio de sus comentarios
- Videos donde más interactúan (detectar nichos de audiencia)

### 11.2 Análisis de sentimiento temporal
**Mejora:** Rastrear el sentimiento promedio de comentarios por video a lo largo del tiempo. Detectar si videos recientes generan más negatividad (posible señal de fatiga de audiencia o cambio de contenido).

**Implementación:** Usar Claude para clasificar comentarios en lotes:
```python
prompt = f"""Clasifica cada comentario como POSITIVO, NEGATIVO, o NEUTRO.
Comentarios:
{batch_of_comments}
Responde en formato JSON: [{{"comment_id": "x", "sentiment": "POSITIVO"}}]"""
```

### 11.3 Detector de preguntas frecuentes (FAQ)
**Mejora:** Extraer preguntas de los comentarios y agruparlas por tema. Esto alimenta directamente las recomendaciones: "Tu audiencia pregunta mucho sobre X, considera hacer un video dedicado".

---

## 12. Modelos ML Avanzados

### 12.1 Predictor de retención
**Problema:** Se predice viralidad y vistas, pero no retención (% del video visto).
**Mejora:** Nuevo modelo `RetentionPredictor` que use datos de YouTube Analytics API para predecir `avg_view_percentage`. Features adicionales:
- `duration_seconds` (crítico — videos más largos tienden a menor retención)
- `has_intro_hook` (detectar via título si promete algo al inicio)
- `is_tutorial` vs `is_entertainment` (clasificación automática por título/tags)

### 12.2 Clasificador automático de categoría de contenido
**Mejora:** Modelo que clasifique videos en categorías internas (tutorial, review, vlog, entretenimiento, noticias, etc.) basándose en título + tags + descripción. Esto permite:
- Análisis de performance por categoría
- Recomendaciones más específicas ("tus tutoriales rinden 2x mejor que tus vlogs")
- Feature adicional para los modelos de viralidad/vistas

### 12.3 Detector de "video que despega tarde"
**Mejora:** Usando el histórico de métricas (tracking diario), identificar patrones de videos que tienen un crecimiento lento las primeras 48h pero despegan después (típico de videos SEO-driven). Alertar cuando un video reciente tiene este patrón para no descartarlo prematuramente.

### 12.4 Modelo de canibalización
**Problema:** Publicar dos videos sobre el mismo tema puede dividir la audiencia.
**Mejora:** Detectar similitud semántica entre títulos/tags usando embeddings (e.g., sentence-transformers o Claude embeddings). Si dos videos recientes tienen >80% similitud, alertar de posible canibalización.

---

## 13. Optimización de Publicación

### 13.1 Simulador "What-If"
**Mejora:** Interfaz interactiva en el dashboard donde el usuario configure:
- Tipo de video (Short/Largo)
- Día y hora de publicación
- Tema / título tentativo
- Duración estimada

Y obtenga en tiempo real:
- Predicción de vistas (con rango de confianza)
- Score de viralidad
- Comparación: "Si publicaras esto el martes a las 3pm en vez del lunes a las 10am, obtendrías ~25% más vistas"

**Implementación:** Ya existen `predict_single()` en ambos modelos. Solo falta la UI que itere sobre múltiples combinaciones día/hora y muestre el delta.

### 13.2 Cadencia óptima de publicación
**Problema:** No se sabe si publicar 3 videos/semana es mejor que 5.
**Mejora:** Analizar correlación entre `days_since_last_upload` y performance del siguiente video. Generar recomendación de frecuencia óptima por tipo de video.

Ejemplo de output: "Tus Shorts rinden mejor cuando publicas 1 cada 2 días. Tus videos largos rinden mejor con al menos 4 días entre ellos."

### 13.3 Detector de saturación horaria
**Mejora:** Cruzar datos de competidores + Google Trends para detectar franjas horarias saturadas (muchos creadores publican a la misma hora). Recomendar horarios con menos competencia.

---

## 14. Integración con Plataformas Externas

### 14.1 Google Calendar sync
**Mejora:** Exportar el plan semanal como eventos de Google Calendar con:
- Título del video sugerido
- Hora óptima de publicación como hora del evento
- Descripción con el outline/guión sugerido
- Recordatorio 2h antes

**Implementación:** Usar `google-auth` (ya instalado) + Calendar API v3.

### 14.2 Notion database sync
**Mejora:** Sincronizar recomendaciones y plan semanal a una base de datos de Notion via API. Permite al creador gestionar su pipeline de contenido desde Notion con estados (Idea → Grabando → Editando → Publicado).

### 14.3 Export a Google Sheets
**Mejora:** Botón en el dashboard que exporte todos los datos a un Google Sheet compartible. Útil para equipos donde el editor/manager necesita ver métricas sin acceder al dashboard.

---

## 15. Dashboard UX Avanzado

### 15.1 Dashboard personalizable (widgets)
**Mejora:** Permitir al usuario arrastrar y reorganizar widgets en la página de Resumen General. Guardar layout en `config/dashboard_layout.json`. Streamlit no soporta drag-and-drop nativo, pero se puede simular con `st.columns` + selectbox de widgets por posición.

### 15.2 Modo "Presentación"
**Mejora:** Botón que cambie el dashboard a un modo limpio (sin sidebar, fuentes más grandes, solo gráficos clave) para compartir pantalla en reuniones con clientes o equipo.

### 15.3 Comparador temporal (periodo vs periodo)
**Mejora:** Selector de dos rangos de fecha para comparar métricas: "Este mes vs mes anterior", "Q1 vs Q4". Mostrar deltas con flechas verdes/rojas.

### 15.4 Dark/Light mode toggle
**Mejora:** Actualmente solo hay dark mode. Agregar toggle para light mode para usuarios que prefieran modo claro o para capturas de pantalla en presentaciones.

### 15.5 Página de "Salud del Canal"
**Mejora:** Dashboard tipo "semáforo" que muestre de un vistazo:

| Métrica | Estado | Detalle |
|---------|--------|---------|
| Frecuencia de publicación | 🟢 | 4 videos/semana (objetivo: 3+) |
| Engagement trend | 🟡 | -5% vs mes anterior |
| Retención promedio | 🔴 | 35% (benchmark: 50%+) |
| CTR de miniaturas | 🟢 | 8.2% (benchmark: 5%+) |
| Crecimiento de suscriptores | 🟡 | +120/mes (objetivo: +200) |

---

## 16. Revenue y Monetización

### 16.1 Estimador de ingresos
**Mejora:** Dado un CPM configurable por nicho (en `.env`: `ESTIMATED_CPM=4.50`), estimar ingresos por video y totales mensuales. Mostrar proyección a 3/6/12 meses basada en tendencia de crecimiento.

### 16.2 ROI de tipos de contenido
**Mejora:** Calcular "ingreso estimado por hora de producción" para cada tipo de video. Si un Short toma 1h de producción y genera $5, pero un video largo toma 8h y genera $50, el ROI por hora es: Short=$5/h, Largo=$6.25/h. Esto requiere que el usuario ingrese tiempo estimado de producción por tipo.

### 16.3 Detector de videos "evergreen"
**Mejora:** Identificar videos que siguen generando vistas meses después de publicados (crecimiento constante vs pico y caída). Estos videos son los más valiosos para monetización a largo plazo. Requerir histórico de métricas (tracking diario).

---

## 17. Gestión de Series y Formatos

### 17.1 Detector automático de series
**Mejora:** Usar NLP (similitud de títulos + tags comunes) para agrupar videos en "series" automáticamente. Ejemplo: detectar que videos con "Tutorial Python #1", "Tutorial Python #2" son una serie.

**Métricas por serie:**
- Retención de audiencia entre episodios (¿cuántos ven el ep. 2 después del ep. 1?)
- Tendencia de vistas por episodio (creciente = serie exitosa, decreciente = fatiga)
- Engagement promedio de la serie vs videos sueltos

### 17.2 Recomendador de formato
**Mejora:** Basado en el análisis de series, sugerir: "Tu serie de tutoriales Python pierde audiencia después del episodio 4. Considera hacer series más cortas (3 episodios) o cambiar el formato."

---

## 18. API REST para Integraciones

### 18.1 API interna con FastAPI
**Mejora:** Exponer los datos y predicciones via API REST para integraciones externas:

```
GET  /api/channels                    → Lista de canales
GET  /api/channels/{id}/metrics       → Métricas del canal
GET  /api/channels/{id}/recommendations → Últimas recomendaciones
POST /api/predict/virality            → Predicción de viralidad
POST /api/predict/views               → Predicción de vistas
GET  /api/channels/{id}/weekly-plan   → Plan semanal actual
POST /api/generate/recommendation     → Generar nueva recomendación
```

**Uso:** Permite crear apps móviles, extensiones de Chrome, o conectar con herramientas no-code (Zapier, Make).

---

## Prioridad Sugerida V2

| # | Mejora | Impacto | Esfuerzo | Dependencia |
|---|--------|---------|----------|-------------|
| 1 | 13.1 Simulador "What-If" | **Muy alto** | **Bajo** | Modelos ya existen |
| 2 | 10.1 Alertas Telegram | **Alto** | **Bajo** | Solo nuevo módulo |
| 3 | 9.1 Generador de guiones | **Alto** | **Bajo** | Claude ya integrado |
| 4 | 8.1 Scoring de thumbnails | **Muy alto** | **Medio** | Claude Vision |
| 5 | 7.1 Tracker de competidores | **Muy alto** | **Medio** | Extractor ya existe |
| 6 | 15.5 Página Salud del Canal | **Alto** | **Bajo** | Datos ya en BD |
| 7 | 9.2 Descripciones SEO | **Alto** | **Bajo** | Claude ya integrado |
| 8 | 12.2 Clasificador de categorías | **Alto** | **Medio** | NLP + Claude |
| 9 | 11.3 FAQ de audiencia | **Alto** | **Medio** | YouTube Comments API |
| 10 | 15.3 Comparador temporal | **Medio** | **Bajo** | Datos históricos |
| 11 | 14.1 Google Calendar sync | **Medio** | **Medio** | google-auth ya está |
| 12 | 16.1 Estimador de ingresos | **Medio** | **Bajo** | Solo cálculo + UI |
| 13 | 13.2 Cadencia óptima | **Alto** | **Medio** | Análisis estadístico |
| 14 | 12.3 Detector de "despegue tardío" | **Alto** | **Medio** | Histórico de métricas |
| 15 | 18.1 API REST (FastAPI) | **Alto** | **Alto** | Reestructuración |
| 16 | 17.1 Detector de series | **Medio** | **Alto** | NLP/embeddings |
| 17 | 8.2 Comparador thumbnails A/B | **Medio** | **Medio** | Claude Vision |
| 18 | 10.2 Webhooks genéricos | **Medio** | **Bajo** | HTTP POST simple |

---

## Quick Wins (implementables en < 1 hora cada uno)

1. **Simulador What-If** — La UI llama a `predict_single()` que ya existe en ambos modelos
2. **Estimador de ingresos** — `estimated_revenue = views * CPM / 1000`, solo agregar al dashboard
3. **Página Salud del Canal** — Métricas con semáforos, datos ya disponibles en BD
4. **Generador de descripciones SEO** — Nuevo prompt a Claude, infraestructura ya existe
5. **Comparador temporal** — Filtrar DataFrame por 2 rangos de fecha y mostrar deltas
