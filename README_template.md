# 📊 YouTube AI Agent — Recomendaciones de Contenido

> **Sistema inteligente de análisis de YouTube usando ML, LLMs y APIs** – Proyecto portfolio para demostrar habilidades en Data Engineering, Machine Learning y MLOps

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41+-FF4B4B.svg)](https://streamlit.io)
[![MariaDB](https://img.shields.io/badge/MariaDB-11-003545.svg)](https://mariadb.org)

---

## 🎯 Descripción del Proyecto

**YouTube AI Agent** es una herramienta de análisis de datos que combina la **YouTube Data API v3**, la **YouTube Analytics API** y **Claude AI** para extraer insights automatizados de canales y videos de YouTube. El sistema recolecta métricas, almacena el histórico en **MariaDB**, entrena modelos de machine learning para predecir rendimiento y genera recomendaciones de contenido accionables.

### Problema que resuelve
Los creadores de contenido necesitan entender patrones de crecimiento, engagement y rendimiento de videos, pero el análisis manual es tedioso y poco escalable. Este agente automatiza la recolección, procesamiento y análisis de datos para generar recomendaciones accionables.

### Características principales
- ✅ **Pipeline ETL**: extracción → guardado → análisis → comentarios → recomendaciones (`main.py`)
- ✅ **Data Warehouse**: 22 tablas en MariaDB con snapshots históricos de métricas
- ✅ **Análisis con IA**: Claude para recomendaciones, títulos, guiones, SEO, clasificación de contenido y sentimiento de comentarios
- ✅ **Análisis Predictivo**: modelos RandomForest para viralidad, vistas y retención
- ✅ **Computer Vision**: scoring de thumbnails y comparador A/B con Claude Vision
- ✅ **Dashboard Interactivo**: 17 páginas en Streamlit, tema claro/oscuro y widgets personalizables
- ✅ **Automatización**: scheduler diario, notificaciones por Telegram, exportación a Google Calendar y Sheets

---

## 🏗️ Arquitectura del Sistema

```
┌──────────────────────────┐
│ YouTube Data API v3      │
│ YouTube Analytics API    │
│ Google Trends            │
└────────────┬─────────────┘
             │
             ▼
┌─────────────────┐      ┌──────────────┐
│  Pipeline ETL   │─────▶│  MariaDB     │
│  (main.py)      │      │  (Histórico) │
└────────┬────────┘      └──────┬───────┘
         │                      │
         ▼                      ▼
┌─────────────────┐      ┌──────────────┐
│  Claude API     │◀────▶│  ML Models   │
│  (Texto+Visión) │      │  (Predicción)│
└────────┬────────┘      └──────┬───────┘
         │                      │
         └──────────┬───────────┘
                    ▼
     ┌───────────────────────────────┐
     │  Streamlit UI · Telegram ·    │
     │  Google Calendar / Sheets     │
     └───────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisitos

- Python 3.11 o superior
- MariaDB (local o vía Docker Compose)
- API Key de YouTube Data API v3 ([Obtener aquí](https://developers.google.com/youtube/v3/getting-started))
- API Key de Anthropic Claude ([Obtener aquí](https://console.anthropic.com/))
- Opcional: `credentials.json` de Google OAuth para YouTube Analytics, Calendar y Sheets

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/rafawilliams/youtube-ai-agent-recomendation.git
cd youtube-ai-agent-recomendation

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt   # En Windows: requirements-windows.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys y datos de MariaDB
```

### Configuración

Variables mínimas en `.env` (ver `.env.example` para todas las opciones):

```bash
YOUTUBE_API_KEY=tu_youtube_api_key_aqui
YOUTUBE_CHANNEL_IDS=UCxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=tu_claude_api_key_aqui

DB_HOST=localhost
DB_PORT=3306
DB_USER=tu_usuario
DB_PASSWORD=tu_password
DB_NAME=youtube_analytics
```

### Ejecución

```bash
# Ejecutar el pipeline completo
python main.py

# Ejecutar el dashboard
streamlit run dashboard.py

# Ejecución automática diaria (--now para correr una vez al iniciar)
python scheduler.py

# O todo con Docker Compose (MariaDB + dashboard + scheduler)
docker compose up -d
```

---

## 📁 Estructura del Proyecto

```
youtube-ai-agent-recomendation/
│
├── main.py                    # Orquestación del pipeline
├── dashboard.py               # Orquestador Streamlit (config, tema, sidebar, router)
├── scheduler.py               # Ejecución diaria automática (APScheduler)
├── requirements.txt           # Dependencias Python
├── .env.example               # Template de variables de entorno
├── Dockerfile / docker-compose.yml
│
├── src/                       # Módulos core
│   ├── youtube_extractor.py   # YouTube Data API
│   ├── analytics_extractor.py # YouTube Analytics API (OAuth)
│   ├── database.py            # Gestión de MariaDB
│   ├── ai_analyzer.py         # Recomendaciones con Claude
│   ├── virality_predictor.py  # ML: probabilidad de viralidad
│   ├── view_predictor.py      # ML: predicción de vistas
│   ├── retention_predictor.py # ML: predicción de retención
│   ├── content_classifier.py  # Clasificación de contenido con Claude
│   ├── comment_analyzer.py    # Sentimiento y FAQ de comentarios
│   ├── thumbnail_analyzer.py  # Scoring de thumbnails (Claude Vision)
│   ├── series_detector.py     # Detección de series (regex + TF-IDF)
│   ├── revenue_analyzer.py    # Ingresos estimados, ROI, evergreen
│   └── ...                    # trends, telegram, google, logger, retry
│
├── page_modules/              # 17 páginas del dashboard
│   └── shared/                # constants, theme, ui, data, predictors
│
├── models/                    # Modelos ML cacheados (joblib)
├── config/                    # Preferencias del dashboard
├── tests/                     # Suite pytest
└── 02-DOCS/                   # Wiki, specs y planes
```

---

## 🔧 Tecnologías Utilizadas

### Backend & Data
- **Python 3.11+**: Lenguaje principal
- **MariaDB** + **SQLAlchemy** + **PyMySQL**: Base de datos relacional
- **Pandas** / **NumPy**: Manipulación de datos

### APIs & Integración
- **YouTube Data API v3** y **YouTube Analytics API**: Extracción de métricas
- **Anthropic Claude API**: Análisis con LLM (texto y visión)
- **Google Trends** (pytrends), **Telegram**, **Google Calendar / Sheets**
- **Tenacity**: Reintentos con backoff exponencial

### Machine Learning
- **Scikit-learn**: RandomForest con TimeSeriesSplit, TF-IDF + similitud coseno
- **Joblib**: Persistencia de modelos

### Frontend & Visualización
- **Streamlit**: Dashboard interactivo
- **Plotly**: Gráficos dinámicos

### DevOps & MLOps
- **Git/GitHub**: Control de versiones
- **pytest** + **ruff**: Testing y calidad de código
- **Docker / Docker Compose**: Containerización
- **APScheduler**: Ejecución programada

---

## 📊 Funcionalidades Actuales

### ✅ Implementado

1. **Recolección de Datos**
   - Estadísticas de canal y métricas de videos
   - YouTube Analytics: retención, fuentes de tráfico
   - Seguimiento de canales competidores y alertas
   - Comentarios por video

2. **Almacenamiento**
   - 22 tablas en MariaDB con índices
   - Snapshots históricos de métricas (`video_metrics`)

3. **Machine Learning**
   - Predicción de viralidad, vistas y retención
   - Detección de *late bloomers*, contenido *evergreen* y canibalización
   - Detección automática de series

4. **IA Generativa**
   - Recomendaciones de contenido y plan semanal
   - Sugerencias de títulos, guiones, descripciones SEO y tags
   - Sentimiento de comentarios y detección de preguntas frecuentes
   - Scoring de thumbnails y comparador A/B

5. **Negocio**
   - Estimación de ingresos (CPM) y ROI por tipo de contenido
   - Cadencia y horarios óptimos de publicación

6. **Visualización**
   - Dashboard con 17 páginas, modo presentación y comparador temporal
   - Tema claro/oscuro y widgets personalizables

### 🚧 En Desarrollo (Roadmap)

- [ ] App de escritorio con motor local (FastAPI)
- [ ] API REST para predicciones
- [ ] Autenticación del dashboard
- [ ] Integración con Notion
- [ ] CI/CD con GitHub Actions
- [ ] Model registry y monitoreo de drift

---

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Solo tests unitarios (sin la MariaDB local de pruebas)
pytest -m "not db"

# Tests específicos
pytest tests/test_view_predictor.py -v
```

Los tests marcados `db` usan una base local de pruebas (`DB_NAME_TEST`) y se saltan si no está disponible.

---

## 📈 Métricas del Proyecto

| Métrica | Valor |
|---------|-------|
| Líneas de código Python | ~15,000 |
| Tests (pytest) | 116 |
| Tablas en MariaDB | 22 |
| Páginas del dashboard | 17 |
| Modelos ML | 3 |

---

## 🎓 Aprendizajes y Skills Demostradas

### Data Engineering
- Diseño de pipelines ETL robustos
- Manejo de APIs con OAuth, rate limiting y reintentos
- Modelado de datos relacionales e históricos

### Machine Learning
- Feature engineering para métricas de engagement
- Validación temporal (TimeSeriesSplit) para evitar fugas de datos
- NLP clásico (TF-IDF) para series y canibalización
- Integración de LLMs y visión para análisis cualitativo

### Software Engineering
- Arquitectura modular (dashboard dividido en módulos de página)
- Testing automatizado con pytest
- Logging estructurado y manejo de errores transitorios

### MLOps
- Pipeline de datos automatizado y programado
- Persistencia y caché de modelos
- Containerización con Docker Compose

---

## 🤝 Contribuciones

Este es un proyecto de portfolio educativo. Si encuentras bugs o tienes sugerencias:

1. Fork el proyecto
2. Crea un branch (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -m 'Agrega nueva feature'`)
4. Push al branch (`git push origin feature/mejora`)
5. Abre un Pull Request

---

## 📞 Contacto

**Rafael** - Specialist in Transformation Management @ Banistmo  
Transicionando a AI/ML & MLOps

- LinkedIn: [tu-perfil-linkedin](https://linkedin.com/in/tu-perfil)
- GitHub: [@rafawilliams](https://github.com/rafawilliams)

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

---

## 🙏 Agradecimientos

- [YouTube Data API Documentation](https://developers.google.com/youtube/v3)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Streamlit Community](https://streamlit.io/)
- Comunidad de ML/MLOps en LinkedIn

---

## 📚 Referencias y Recursos

- [Building ETL Pipelines with Python](https://realpython.com/python-etl/)
- [MLOps Best Practices](https://ml-ops.org/)
- [YouTube Analytics for Creators](https://creatoracademy.youtube.com/)

---

**⭐ Si este proyecto te resulta útil, dale una estrella en GitHub!**

> 💡 **Nota para reclutadores**: Este proyecto demuestra competencias end-to-end en Data Engineering, ML y MLOps, desde la recolección de datos hasta el deployment de modelos en producción.
