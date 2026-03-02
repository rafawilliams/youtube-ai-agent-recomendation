# Roadmap de Mejoras — YouTube AI Agent

Documento de referencia para evolucionar el proyecto. Las mejoras están ordenadas por impacto y complejidad estimada.

---

## 1. Extracción de Datos

### 1.1 Rastreo histórico de métricas
**Problema actual:** Cada ejecución de `main.py` guarda UNA snapshot de vistas/likes. No hay evolución en el tiempo por video.
**Mejora:** Agregar columna `days_since_published` en `video_metrics` y comparar la trayectoria de crecimiento de vistas de cada video a lo largo de los días. Esto permite detectar si un video "despega tarde" o "cae rápido".

### 1.2 Paginación completa de videos
**Problema actual:** `max_videos_per_channel=50` limita el historial.
**Mejora:** Exponer ese parámetro en `.env` (`MAX_VIDEOS_PER_CHANNEL`) y considerar incrementar el default a 200+ para tener datos de ML más robustos.

### 1.3 YouTube Analytics API (métricas avanzadas)
**Problema actual:** La YouTube Data API v3 solo da vistas, likes y comentarios. No da retención ni impresiones.
**Mejora:** Integrar YouTube Analytics API para obtener:
- **Retention rate** (porcentaje promedio visto)
- **Click-through rate (CTR)** de miniaturas
- **Fuentes de tráfico** (búsqueda, sugerido, externo)
- **Demografía** de la audiencia

> Requiere OAuth 2.0 en lugar de API Key simple.

### 1.4 Ejecución automática diaria
**Problema actual:** El usuario debe ejecutar `python main.py` manualmente.
**Mejora:** Agregar scheduler con `schedule` o `APScheduler` en un script `scheduler.py` que ejecute `main.py` diariamente a una hora configurable. Alternativamente, usar Task Scheduler de Windows o cron en Linux.

---

## 2. Modelos de Machine Learning

### 2.1 Persistencia del modelo entrenado
**Problema actual:** Cada vez que se abre el dashboard, el modelo se reentrena desde cero con `RandomForest`.
**Mejora:** Serializar el modelo con `joblib` (`model.pkl`) y solo reentrenar si hay nuevos videos desde el último entrenamiento. Reduce latencia del dashboard significativamente.

```python
# Ejemplo de persistencia
import joblib
joblib.dump(predictor, 'models/virality_model_UCxxxxxx.pkl')
predictor = joblib.load('models/virality_model_UCxxxxxx.pkl')
```

### 2.2 Nuevas features para los modelos
**Mejora:** Agregar features adicionales que probablemente mejoran la predicción:

| Feature | Fuente | Razón |
|---------|--------|-------|
| `title_has_number` | título | Títulos con números tienden a tener más clics |
| `title_has_question` | título | Preguntas generan curiosidad |
| `description_length` | descripción | Indica nivel de optimización SEO |
| `days_since_last_upload` | BD | Frecuencia de publicación afecta algoritmo |
| `channel_age_days` | BD | Canales mayores tienen más autoridad |
| `thumbnail_brightness` | URL miniatura | Miniaturas brillantes llaman más atención |

### 2.3 Validación cruzada robusta
**Problema actual:** Los modelos usan un split 80/20 fijo. Con pocos datos, esto puede ser ruidoso.
**Mejora:** Implementar `TimeSeriesSplit` de scikit-learn para respetar el orden temporal de los videos (no mezclar futuro con pasado en el entrenamiento).

### 2.4 Modelo de tendencias de temas
**Mejora nueva:** Conectar Google Trends API (via `pytrends`) para comparar qué temas están en auge en Panamá/LATAM. Integrar el "trend score" del tema del video como feature adicional.

---

## 3. Dashboard (Streamlit)

### 3.1 Comparación entre canales
**Problema actual:** El dashboard muestra un canal a la vez.
**Mejora:** Agregar una vista "Comparar Canales" que muestre métricas side-by-side de todos los canales configurados. Útil para benchmarking.

### 3.2 Evolución temporal de vistas
**Problema actual:** No hay gráfico de tendencia de vistas en el tiempo.
**Mejora:** Agregar gráfico de línea con vistas acumuladas por semana/mes, separado por tipo de video (Shorts vs Largos). Requiere el rastreo histórico de métricas (mejora 1.1).

### 3.3 Análisis de títulos y tags
**Mejora nueva:** Agregar sección "🔤 Análisis de Contenido" con:
- WordCloud de los tags más usados
- Análisis de palabras más frecuentes en títulos de videos con alta performance
- Comparación de longitud de título vs vistas

### 3.4 Exportación de reportes
**Mejora nueva:** Botón "📥 Exportar reporte" que genere:
- PDF con resumen del canal (usando `reportlab` o `fpdf2`)
- Excel con todos los videos y métricas (usando `openpyxl`)

### 3.5 Caché de datos del dashboard
**Problema actual:** `load_data()` consulta la BD en cada interacción del usuario.
**Mejora:** Agregar `@st.cache_data(ttl=300)` al decorador de `load_data()` para cachear la consulta por 5 minutos. Mejora el tiempo de respuesta del dashboard notablemente.

```python
@st.cache_data(ttl=300)
def load_data():
    ...
```

### 3.6 Notificaciones por correo/Telegram
**Mejora nueva:** Enviar la recomendación diaria automáticamente por email (`smtplib`) o Telegram Bot (`python-telegram-bot`) cuando se ejecuta `main.py`.

---

## 4. Recomendaciones de IA

### 4.1 Retroalimentación del ciclo de aprendizaje
**Problema actual:** No se sabe si siguiendo las recomendaciones el canal mejora.
**Mejora:** Agregar tabla `recommendation_results` que registre si el video publicado el día siguiente tuvo mejor/peor performance que el promedio. Con esto, el prompt de Claude se puede enriquecer con: "Las últimas 3 recomendaciones de Shorts resultaron en X% sobre el promedio".

### 4.2 Planificación semanal
**Problema actual:** La recomendación es solo para "mañana".
**Mejora:** Agregar una función `generate_weekly_plan()` en `AIAnalyzer` que genere un plan de contenido para los próximos 7 días, especificando tipo de video y tema para cada día, basándose en los mejores días/horas del canal.

### 4.3 Sugerencia de títulos ✅
**Mejora implementada:** El prompt de recomendación ahora solicita explícitamente 3 opciones de título para el video recomendado, con análisis de por qué cada título podría funcionar bien (palabras clave, claridad, curiosidad). Las sugerencias se almacenan en la base de datos (columna `title_suggestions` en tabla `recommendations`) y se muestran tanto en la consola (`main.py`) como en el dashboard de Streamlit.

### 4.4 Análisis de comentarios
**Mejora nueva:** Extraer comentarios de los videos más populares via YouTube API y usar Claude para:
- Identificar preguntas frecuentes de la audiencia
- Detectar sentimiento (positivo/negativo/neutral)
- Extraer ideas de contenido sugeridas por la comunidad

---

## 5. Infraestructura y Calidad de Código

### 5.1 Contenedor Docker
**Mejora nueva:** Crear `Dockerfile` y `docker-compose.yml` que levante en un solo comando:
- MariaDB
- El scheduler de `main.py`
- El dashboard de Streamlit

### 5.2 Tests unitarios
**Problema actual:** No hay tests automatizados.
**Mejora:** Agregar tests con `pytest` para:
- `ViralityPredictor.train()` y `predict()` con datos sintéticos
- `ViewPredictor.get_publishing_heatmap()`
- `YouTubeDatabase._create_tables()` (contra SQLite en memoria para tests)
- `AIAnalyzer._parse_recommendation()`

### 5.3 Logging estructurado
**Problema actual:** Se usa `print()` para todo.
**Mejora:** Reemplazar `print()` con el módulo `logging` estándar. Guardar logs en `logs/main_{fecha}.log` con rotación automática. Facilita el diagnóstico de errores en producción.

### 5.4 Retry automático en llamadas a APIs
**Problema actual:** Si la YouTube API o la API de Anthropic falla por un error transitorio (rate limit, timeout), el script falla completo.
**Mejora:** Implementar reintentos con backoff exponencial usando `tenacity`:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _call_claude(self, prompt):
    ...
```

### 5.5 Variables de entorno validadas al inicio
**Problema actual:** Los errores de configuración se descubren tarde en la ejecución.
**Mejora:** Agregar validación completa al inicio con mensajes claros de qué variable falta y cómo configurarla (con enlace a documentación).

---

## 6. Seguridad

### 6.1 Protección del dashboard con contraseña
**Mejora nueva:** Agregar autenticación básica al dashboard de Streamlit con `streamlit-authenticator`. Evita que cualquier persona en la red local acceda a los datos del canal.

### 6.2 Rotación de API Keys
**Mejora nueva:** Documentar y soportar múltiples API Keys de YouTube (cuota de 10,000 unidades/día por key) con rotación automática cuando se agota la cuota.

---

## Prioridad Sugerida

| # | Mejora | Impacto | Esfuerzo |
|---|--------|---------|---------|
| 1 | Caché del dashboard (`@st.cache_data`) | Alto | Muy bajo |
| 2 | Persistencia del modelo ML | Alto | Bajo |
| 3 | Ejecución automática diaria | Alto | Bajo |
| 4 | Rastreo histórico de métricas | Muy alto | Medio |
| 5 | Planificación semanal de contenido | Alto | Medio |
| 6 | Sugerencia de títulos en recomendación | Alto | Bajo |
| 7 | Exportación PDF/Excel | Medio | Medio |
| 8 | Análisis de comentarios | Alto | Medio |
| 9 | Tests unitarios | Medio | Medio |
| 10 | Docker | Medio | Alto |
