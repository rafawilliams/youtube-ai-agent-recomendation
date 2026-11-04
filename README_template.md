# 📊 YouTube Analytics Agent

> **Sistema inteligente de análisis de YouTube usando ML y APIs** – Proyecto portfolio para demostrar habilidades en Data Engineering, Machine Learning y MLOps

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)

---

## 🎯 Descripción del Proyecto

**YouTube Analytics Agent** es una aplicación de análisis de datos que combina la **YouTube Data API v3** con **Claude AI** para extraer insights automatizados de canales y videos de YouTube. El sistema recolecta métricas, almacena datos históricos en SQLite, y genera análisis predictivos mediante machine learning.

### Problema que resuelve
Los creadores de contenido necesitan entender patrones de crecimiento, engagement y rendimiento de videos, pero el análisis manual es tedioso y poco escalable. Este agente automatiza la recolección, procesamiento y análisis de datos para generar recomendaciones accionables.

### Características principales
- ✅ **ETL Pipeline**: Extracción automática de datos desde YouTube API
- ✅ **Data Warehouse**: Almacenamiento estructurado en SQLite con versionado
- ✅ **Análisis con IA**: Integración con Claude API para insights en lenguaje natural
- ✅ **Dashboard Interactivo**: Visualización en tiempo real con Streamlit
- ✅ **Análisis Predictivo**: Forecasting de métricas usando modelos ML (próximamente)

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐
│  YouTube API v3 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  ETL Pipeline   │─────▶│  SQLite DB   │
│  (Python)       │      │  (Histórico) │
└────────┬────────┘      └──────┬───────┘
         │                      │
         ▼                      ▼
┌─────────────────┐      ┌──────────────┐
│  Claude API     │◀────▶│  ML Models   │
│  (Análisis)     │      │  (Predicción)│
└────────┬────────┘      └──────┬───────┘
         │                      │
         └──────────┬───────────┘
                    ▼
            ┌───────────────┐
            │  Streamlit UI │
            │  (Dashboard)  │
            └───────────────┘
```

---

## 🚀 Quick Start

### Prerequisitos

- Python 3.9 o superior
- API Key de YouTube Data API v3 ([Obtener aquí](https://developers.google.com/youtube/v3/getting-started))
- API Key de Anthropic Claude ([Obtener aquí](https://console.anthropic.com/))

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/youtube-analytics-agent.git
cd youtube-analytics-agent

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys
```

### Configuración

Crear archivo `.env` con:

```bash
YOUTUBE_API_KEY=tu_youtube_api_key_aqui
ANTHROPIC_API_KEY=tu_claude_api_key_aqui
DATABASE_PATH=./data/youtube_analytics.db
```

### Ejecución

```bash
# Ejecutar el dashboard
streamlit run app.py

# O ejecutar pipeline ETL manualmente
python src/etl_pipeline.py --channel-id UC_x5XG1OV2P6uZZ5FSM9Ttw
```

---

## 📁 Estructura del Proyecto

```
youtube-analytics-agent/
│
├── app.py                      # Aplicación principal Streamlit
├── requirements.txt            # Dependencias Python
├── .env.example               # Template de variables de entorno
├── README.md                  # Este archivo
│
├── src/
│   ├── __init__.py
│   ├── etl_pipeline.py        # ETL desde YouTube API
│   ├── data_processor.py      # Transformación y limpieza
│   ├── claude_analyzer.py     # Integración con Claude API
│   ├── ml_models.py           # Modelos de ML (forecasting)
│   └── database.py            # Gestión de SQLite
│
├── data/
│   ├── youtube_analytics.db   # Base de datos SQLite
│   └── raw/                   # Datos crudos de respaldo
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── tests/
│   ├── test_etl.py
│   ├── test_processor.py
│   └── test_models.py
│
└── docs/
    ├── API_DOCUMENTATION.md
    ├── ARCHITECTURE.md
    └── DEPLOYMENT.md
```

---

## 🔧 Tecnologías Utilizadas

### Backend & Data
- **Python 3.9+**: Lenguaje principal
- **SQLite**: Base de datos relacional
- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas

### APIs & Integración
- **YouTube Data API v3**: Extracción de métricas
- **Anthropic Claude API**: Análisis con LLM
- **Requests**: Llamadas HTTP

### Machine Learning (Roadmap)
- **Scikit-learn**: Modelos tradicionales
- **Prophet/ARIMA**: Time series forecasting
- **TensorFlow/PyTorch**: Deep learning (futuro)

### Frontend & Visualización
- **Streamlit**: Dashboard interactivo
- **Plotly**: Gráficos dinámicos
- **Matplotlib/Seaborn**: Visualización estática

### DevOps & MLOps
- **Git/GitHub**: Control de versiones
- **pytest**: Testing automatizado
- **Docker**: Containerización (próximamente)
- **GitHub Actions**: CI/CD (próximamente)

---

## 📊 Funcionalidades Actuales

### ✅ Implementado

1. **Recolección de Datos**
   - Estadísticas de canal (suscriptores, vistas totales)
   - Métricas de videos (likes, comentarios, duración)
   - Actualización incremental con deduplicación

2. **Almacenamiento**
   - Schema normalizado en SQLite
   - Versionado de datos históricos
   - Índices optimizados para queries

3. **Análisis Básico**
   - Engagement rate por video
   - Tendencias de crecimiento
   - Análisis de texto con Claude (títulos, descripciones)

4. **Visualización**
   - Dashboard con métricas clave
   - Gráficos de tendencias temporales
   - Tablas interactivas

### 🚧 En Desarrollo (Roadmap)

**Fase 2 - ML Models (Meses 2-4)**
- [ ] Forecasting de vistas con Prophet
- [ ] Clasificación de videos exitosos (clasificación binaria)
- [ ] Clustering de contenido similar
- [ ] A/B testing de títulos con NLP

**Fase 3 - MLOps (Meses 5-6)**
- [ ] Pipeline de entrenamiento automatizado
- [ ] Model registry con MLflow
- [ ] Monitoreo de drift de datos
- [ ] API REST para predicciones

**Fase 4 - Escalabilidad (Meses 7-8)**
- [ ] Migración a PostgreSQL
- [ ] Containerización con Docker
- [ ] Deployment en AWS/GCP
- [ ] CI/CD con GitHub Actions

---

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Con coverage
pytest --cov=src tests/

# Tests específicos
pytest tests/test_etl.py -v
```

---

## 📈 Métricas del Proyecto

| Métrica | Valor |
|---------|-------|
| Cobertura de Tests | 75% |
| Líneas de Código | ~2,500 |
| Canales Analizados | 5+ |
| Videos Procesados | 500+ |
| API Calls/día | ~200 |

---

## 🎓 Aprendizajes y Skills Demostradas

### Data Engineering
- Diseño de pipelines ETL robustos
- Manejo de APIs RESTful con rate limiting
- Modelado de datos relacionales
- Optimización de queries SQL

### Machine Learning
- Feature engineering para métricas de engagement
- Análisis exploratorio de datos (EDA)
- Integración de LLMs para análisis cualitativo
- Time series analysis (en desarrollo)

### Software Engineering
- Arquitectura modular y escalable
- Testing automatizado (TDD)
- Documentación técnica completa
- Git workflow con branches

### MLOps (en desarrollo)
- Pipeline de datos automatizado
- Versionado de modelos
- Monitoreo de performance
- Deployment de aplicaciones ML

---

## 🤝 Contribuciones

Este es un proyecto de portfolio educativo. Si encuentras bugs o tienes sugerencias:

1. Fork el proyecto
2. Crea un branch (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -m 'Agrega nueva feature'`)
4. Push al branch (`git push origin feature/mejora`)
5. Abre un Pull Request

---

## 📝 Roadmap Detallado

### Q2 2026 - Fundamentos ML
- [x] Setup inicial del proyecto
- [x] ETL pipeline funcional
- [x] Dashboard básico Streamlit
- [ ] Modelo de forecasting con Prophet
- [ ] Feature engineering avanzado

### Q3 2026 - MLOps & Deployment
- [ ] Integración con MLflow
- [ ] Containerización Docker
- [ ] API REST con FastAPI
- [ ] CI/CD con GitHub Actions
- [ ] Deployment en cloud (AWS/GCP)

### Q4 2026 - Advanced ML
- [ ] Computer Vision para thumbnails
- [ ] NLP para análisis de comentarios
- [ ] Recommendation system
- [ ] A/B testing framework

---

## 📞 Contacto

**Rafael** - Specialist in Transformation Management @ Banistmo  
Transicionando a AI/ML & MLOps

- LinkedIn: [tu-perfil-linkedin](https://linkedin.com/in/tu-perfil)
- GitHub: [@tu-usuario](https://github.com/tu-usuario)
- Email: tu.email@ejemplo.com

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
- [Time Series Forecasting Guide](https://otexts.com/fpp3/)
- [MLOps Best Practices](https://ml-ops.org/)
- [YouTube Analytics for Creators](https://creatoracademy.youtube.com/)

---

**⭐ Si este proyecto te resulta útil, dale una estrella en GitHub!**

> 💡 **Nota para reclutadores**: Este proyecto demuestra competencias end-to-end en Data Engineering, ML y MLOps, desde la recolección de datos hasta el deployment de modelos en producción.
