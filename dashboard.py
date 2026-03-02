"""
Dashboard interactivo para visualizar análisis y recomendaciones
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import html as html_mod
import os
import sys
import json
import io
import re
from collections import Counter

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database import YouTubeDatabase
from ai_analyzer import AIAnalyzer
from virality_predictor import ViralityPredictor, WEEKDAY_LABELS
from virality_predictor import MODEL_VERSION as VIRALITY_MODEL_VERSION
from view_predictor import ViewPredictor
from view_predictor import MODEL_VERSION as VIEW_MODEL_VERSION
from retention_predictor import RetentionPredictor
from retention_predictor import MODEL_VERSION as RETENTION_MODEL_VERSION
from content_classifier import ContentClassifier
from late_bloomer_detector import LateBloomerDetector
from cannibalization_detector import CannibalizationDetector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from youtube_extractor import YouTubeDataExtractor
from trends_analyzer import TrendsAnalyzer, GEO_OPTIONS, TIMEFRAME_OPTIONS
from google_integrations import GoogleCalendarExporter, GoogleSheetsExporter
from dotenv import load_dotenv
import plotly.io as pio

PAGE_HISTORICO  = "📅 Histórico de Métricas"
PAGE_ANALYTICS  = "📊 Analytics Avanzados"
PAGE_TRENDS     = "📊 Tendencias de Temas"
PAGE_CONTENT    = "🔤 Análisis de Contenido"
PAGE_COMPARE    = "🆚 Comparar Canales"
PAGE_WEEKLY     = "🗓 Plan Semanal"
PAGE_HEALTH     = "🏥 Salud del Canal"
PAGE_CADENCE    = "⏱ Cadencia y Horarios"
PAGE_COMPETITORS = "🕵 Análisis de Competencia"
PAGE_RETENTION = "📊 Predicción de Retención"
PAGE_LATE_BLOOMER = "🌱 Despegue Tardío"
PAGE_CANNIBALIZATION = "🔀 Canibalización"
PAGE_TEMPORAL = "📆 Comparador Temporal"

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# ─── Design System — Colores unificados ─────────────────────────────────────
_DARK_COLORS = {
    "primary":        "#6366F1",
    "primary_light":  "#818CF8",
    "primary_dark":   "#4F46E5",
    "secondary":      "#EC4899",
    "secondary_light":"#F472B6",
    "success":        "#10B981",
    "warning":        "#F59E0B",
    "danger":         "#EF4444",
    "info":           "#3B82F6",
    "bg_primary":     "#0F172A",
    "bg_secondary":   "#1E293B",
    "bg_tertiary":    "#334155",
    "text_primary":   "#F8FAFC",
    "text_secondary": "#94A3B8",
    "text_muted":     "#64748B",
    "border":         "#334155",
}

_LIGHT_COLORS = {
    "primary":        "#6366F1",
    "primary_light":  "#818CF8",
    "primary_dark":   "#4F46E5",
    "secondary":      "#EC4899",
    "secondary_light":"#F472B6",
    "success":        "#10B981",
    "warning":        "#F59E0B",
    "danger":         "#EF4444",
    "info":           "#3B82F6",
    "bg_primary":     "#F8FAFC",
    "bg_secondary":   "#FFFFFF",
    "bg_tertiary":    "#E2E8F0",
    "text_primary":   "#0F172A",
    "text_secondary": "#475569",
    "text_muted":     "#94A3B8",
    "border":         "#CBD5E1",
}


def _get_colors() -> dict:
    """Retorna el dict de colores según el tema activo en session_state."""
    theme = st.session_state.get('prefs', {}).get('theme', 'dark')
    return _LIGHT_COLORS if theme == 'light' else _DARK_COLORS


# COLORS se inicializa con dark y se reasigna en main() tras cargar preferencias.
COLORS = dict(_DARK_COLORS)

# Secuencia de colores para graficas Plotly (8 colores)
CHART_COLORS = [
    "#6366F1", "#EC4899", "#14B8A6", "#F59E0B",
    "#8B5CF6", "#06B6D4", "#F97316", "#84CC16",
]

# Colores por tipo de video
VIDEO_TYPE_COLORS = {"Short": "#EC4899", "Video Largo": "#6366F1"}
VIDEO_TYPE_SEQUENCE = ["#EC4899", "#6366F1"]

# Mapa de colores semánticos para performance
PERF_LABEL_COLORS = {
    "above_average": "#10B981",
    "average":       "#F59E0B",
    "below_average": "#EF4444",
}

# ─── Plotly Template Global ─────────────────────────────────────────────────
_yt_template = go.layout.Template()
_yt_template.layout = go.Layout(
    font=dict(family="Inter, -apple-system, sans-serif", color="#F8FAFC", size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(30,41,59,0.5)",
    colorway=CHART_COLORS,
    title=dict(font=dict(size=16, color="#F8FAFC"), x=0, xanchor="left"),
    xaxis=dict(
        gridcolor="rgba(51,65,85,0.5)", gridwidth=1,
        zerolinecolor="rgba(51,65,85,0.7)",
        title_font=dict(size=12, color="#94A3B8"),
        tickfont=dict(size=11, color="#94A3B8"),
    ),
    yaxis=dict(
        gridcolor="rgba(51,65,85,0.5)", gridwidth=1,
        zerolinecolor="rgba(51,65,85,0.7)",
        title_font=dict(size=12, color="#94A3B8"),
        tickfont=dict(size=11, color="#94A3B8"),
    ),
    legend=dict(
        font=dict(size=11, color="#94A3B8"),
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
    ),
    margin=dict(l=16, r=16, t=40, b=16),
    hoverlabel=dict(
        bgcolor="#1E293B",
        font_size=12,
        font_color="#F8FAFC",
        bordercolor="#334155",
    ),
    hovermode="x unified",
)
pio.templates["youtube_ai"] = _yt_template
pio.templates.default = "youtube_ai"


def _get_virality_predictor(df: pd.DataFrame, channel_id: str):
    """
    Carga el ViralityPredictor desde caché si el número de videos no cambió.
    Si cambió o no existe caché, reentrena y guarda.
    Retorna (predictor, from_cache: bool).
    """
    import joblib
    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_id = channel_id.replace('/', '_')
    pkl_path  = os.path.join(MODELS_DIR, f'virality_{safe_id}.pkl')
    meta_path = os.path.join(MODELS_DIR, f'virality_{safe_id}.meta.json')

    current_count = int(pd.to_numeric(df['view_count'], errors='coerce').gt(0).sum())

    # Intentar usar caché
    if os.path.exists(pkl_path) and os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if (meta.get('video_count') == current_count
                    and meta.get('model_version') == VIRALITY_MODEL_VERSION):
                predictor = joblib.load(pkl_path)
                return predictor, True
        except Exception:
            pass  # caché corrupto → reentrenar

    # Reentrenar
    predictor = ViralityPredictor(min_samples=10)
    result = predictor.train(df)

    if result.get('trained'):
        try:
            predictor.save(pkl_path)
            with open(meta_path, 'w') as f:
                json.dump({
                    'channel_id':    channel_id,
                    'video_count':   current_count,
                    'model_version': VIRALITY_MODEL_VERSION,
                    'trained_at':    datetime.now().isoformat(),
                }, f)
        except Exception:
            pass  # no crítico — el predictor funciona en memoria

    return predictor, False


def _get_view_predictor(df: pd.DataFrame, channel_id: str):
    """
    Carga el ViewPredictor desde caché si el número de videos no cambió.
    Si cambió o no existe caché, reentrena y guarda.
    Retorna (predictor, from_cache: bool).
    """
    import joblib
    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_id = channel_id.replace('/', '_')
    pkl_path  = os.path.join(MODELS_DIR, f'views_{safe_id}.pkl')
    meta_path = os.path.join(MODELS_DIR, f'views_{safe_id}.meta.json')

    current_count = int(pd.to_numeric(df['view_count'], errors='coerce').gt(0).sum())

    # Intentar usar caché
    if os.path.exists(pkl_path) and os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if (meta.get('video_count') == current_count
                    and meta.get('model_version') == VIEW_MODEL_VERSION):
                predictor = joblib.load(pkl_path)
                return predictor, True
        except Exception:
            pass  # caché corrupto → reentrenar

    # Reentrenar
    predictor = ViewPredictor(min_samples=10)
    result = predictor.train(df)

    if result.get('trained'):
        try:
            predictor.save(pkl_path)
            with open(meta_path, 'w') as f:
                json.dump({
                    'channel_id':    channel_id,
                    'video_count':   current_count,
                    'model_version': VIEW_MODEL_VERSION,
                    'trained_at':    datetime.now().isoformat(),
                }, f)
        except Exception:
            pass  # no crítico

    return predictor, False


def _get_retention_predictor(df: pd.DataFrame, analytics_df: pd.DataFrame, channel_id: str):
    """
    Carga el RetentionPredictor desde caché. Requiere merge de videos + analytics.
    Retorna (predictor, from_cache: bool).
    """
    import joblib
    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_id = channel_id.replace('/', '_')
    pkl_path  = os.path.join(MODELS_DIR, f'retention_{safe_id}.pkl')
    meta_path = os.path.join(MODELS_DIR, f'retention_{safe_id}.meta.json')

    merged = df.merge(
        analytics_df[['video_id', 'avg_view_percentage']],
        on='video_id', how='left',
    )
    current_count = int(
        pd.to_numeric(merged['avg_view_percentage'], errors='coerce').gt(0).sum()
    )

    if os.path.exists(pkl_path) and os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if (meta.get('video_count') == current_count
                    and meta.get('model_version') == RETENTION_MODEL_VERSION):
                predictor = joblib.load(pkl_path)
                return predictor, True
        except Exception:
            pass

    predictor = RetentionPredictor(min_samples=10)
    result = predictor.train(merged)

    if result.get('trained'):
        try:
            predictor.save(pkl_path)
            with open(meta_path, 'w') as f:
                json.dump({
                    'channel_id':    channel_id,
                    'video_count':   current_count,
                    'model_version': RETENTION_MODEL_VERSION,
                    'trained_at':    datetime.now().isoformat(),
                }, f)
        except Exception:
            pass

    return predictor, False


# Configuración de la página
st.set_page_config(
    page_title="YouTube AI Agent Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar variables de entorno
load_dotenv(override=True)


# ─── Detección de idioma del canal ──────────────────────────────────────────
def _detect_channel_language(db, channel_id: str) -> str:
    """Detecta el idioma predominante de un canal a partir de los títulos de sus videos.

    Retorna 'en' para inglés, 'es' para español (por defecto).
    """
    try:
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT title FROM videos WHERE channel_id = %s ORDER BY published_at DESC LIMIT 20",
            (channel_id,),
        )
        rows = cursor.fetchall()
        if not rows:
            return 'es'

        titles_text = ' '.join(r['title'] for r in rows if r.get('title')).lower()
        words = titles_text.split()

        # Indicadores comunes de cada idioma
        en_words = {'the', 'and', 'for', 'that', 'with', 'from', 'this', 'was',
                    'are', 'but', 'not', 'you', 'she', 'her', 'his', 'they',
                    'will', 'have', 'has', 'had', 'been', 'one', 'when', 'who',
                    'what', 'why', 'how', 'my', 'your', 'it', 'of', 'in', 'to',
                    'is', 'on', 'at', 'an', 'we', 'our', 'no', 'if', 'he'}
        es_words = {'de', 'que', 'en', 'los', 'las', 'por', 'para', 'como',
                    'con', 'del', 'una', 'uno', 'ese', 'esta', 'esto', 'mas',
                    'pero', 'sin', 'sobre', 'entre', 'hasta', 'desde', 'puede',
                    'tiene', 'fue', 'son', 'hay', 'ser', 'porque', 'cuando',
                    'todo', 'sus', 'nos', 'cual', 'donde', 'quien', 'ya'}
        pt_words = {'de', 'que', 'em', 'os', 'as', 'por', 'para', 'como',
                    'com', 'uma', 'nao', 'mais', 'muito', 'voce', 'isso',
                    'seu', 'sua', 'ele', 'ela', 'nos', 'tem', 'foi', 'sao'}

        en_count = sum(1 for w in words if w in en_words)
        es_count = sum(1 for w in words if w in es_words)
        pt_count = sum(1 for w in words if w in pt_words)

        scores = {'en': en_count, 'es': es_count, 'pt': pt_count}
        best = max(scores, key=scores.get)
        # Si hay empate o muy pocas señales, default a español
        if scores[best] < 3:
            return 'es'
        return best
    except Exception:
        return 'es'


# ─── CSS Stylesheet Profesional ─────────────────────────────────────────────

def _inject_theme_css():
    """Inyecta CSS con variables del tema actual (15.4)."""
    c = _get_colors()
    _yt_template.layout.font.color = c['text_primary']

    st.markdown(f"""
<style>
:root {{
    --primary: {c['primary']};
    --primary-light: {c['primary_light']};
    --primary-dark: {c['primary_dark']};
    --secondary: {c['secondary']};
    --success: {c['success']};
    --warning: {c['warning']};
    --danger: {c['danger']};
    --info: {c['info']};
    --bg-primary: {c['bg_primary']};
    --bg-secondary: {c['bg_secondary']};
    --bg-tertiary: {c['bg_tertiary']};
    --text-primary: {c['text_primary']};
    --text-secondary: {c['text_secondary']};
    --text-muted: {c['text_muted']};
    --border: {c['border']};
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.3), 0 2px 4px -2px rgba(0,0,0,0.3);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.4), 0 4px 6px -4px rgba(0,0,0,0.3);
    --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}}
</style>
""", unsafe_allow_html=True)

    # Presentation mode CSS (15.2)
    if st.session_state.get('presentation_mode', False):
        st.markdown("""
<style>
section[data-testid="stSidebar"] { display: none !important; }
.main .block-container { max-width: 1800px !important; padding-top: 1rem !important; }
h1 { font-size: 2.5rem !important; }
h2 { font-size: 2rem !important; }
h3 { font-size: 1.5rem !important; }
.metric-card .metric-value { font-size: 2.5rem !important; }
.metric-card .metric-label { font-size: 0.9rem !important; }
[data-testid="stMetricValue"] { font-size: 2.2rem !important; }
.page-title-bar .page-title { font-size: 2rem !important; }
.page-title-bar .page-icon { font-size: 2.5rem; width: 64px; height: 64px; }
.stDownloadButton, [data-testid="stForm"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
/* ═══ Google Font Import ═══ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ═══ Root Variables — overridden dynamically by _inject_theme_css() ═══ */
:root {
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.3), 0 2px 4px -2px rgba(0,0,0,0.3);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.4), 0 4px 6px -4px rgba(0,0,0,0.3);
    --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ═══ Global Typography ═══ */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
}
h1 { font-size: 1.875rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.25rem !important; }

/* ═══ Main Container ═══ */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ═══ Sidebar Styling ═══ */
section[data-testid="stSidebar"] {
    background-color: var(--bg-primary);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

/* Sidebar brand */
.sidebar-brand {
    text-align: center;
    padding: 1.25rem 1rem 1rem 1rem;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}
.sidebar-brand .brand-icon { font-size: 2.5rem; display: block; margin-bottom: 0.25rem; }
.sidebar-brand .brand-name { font-size: 1.1rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.02em; }
.sidebar-brand .brand-tagline { font-size: 0.75rem; color: var(--text-muted); margin-top: 2px; }

/* Navigation group labels */
.nav-group-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    padding: 0.75rem 0 0.25rem 0.5rem;
    margin-top: 0.25rem;
}

/* ═══ Section Divider ═══ */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--border) 20%, var(--border) 80%, transparent 100%);
    margin: 2rem 0;
    border: none;
}
.section-divider.with-label {
    display: flex;
    align-items: center;
    gap: 1rem;
    height: auto;
    background: none;
}
.section-divider.with-label::before,
.section-divider.with-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}
.section-divider.with-label span {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    white-space: nowrap;
}

/* ═══ Page Title Bar ═══ */
.page-title-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
    border-bottom: 2px solid var(--border);
}
.page-title-bar .page-icon {
    font-size: 1.75rem;
    width: 48px; height: 48px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(236,72,153,0.1));
    border-radius: var(--radius-md);
    flex-shrink: 0;
}
.page-title-bar .page-title { font-size: 1.5rem; font-weight: 700; color: var(--text-primary); margin: 0; }
.page-title-bar .page-description { font-size: 0.85rem; color: var(--text-secondary); margin: 0; }

/* ═══ Section Header ═══ */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid var(--primary);
}
.section-header .section-icon {
    font-size: 1.3rem;
    width: 36px; height: 36px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(99,102,241,0.05));
    border-radius: var(--radius-sm);
}
.section-header .section-title { font-size: 1.15rem; font-weight: 600; color: var(--text-primary); margin: 0; }
.section-header .section-subtitle { font-size: 0.8rem; color: var(--text-secondary); margin: 0; }

/* ═══ Metric Card (Custom HTML) ═══ */
.metric-card {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    margin: 0.5rem 0;
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    border-radius: var(--radius-md) var(--radius-md) 0 0;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
    border-color: var(--primary);
}
.metric-card .metric-icon { font-size: 1.5rem; margin-bottom: 0.5rem; display: block; }
.metric-card .metric-label {
    font-size: 0.75rem; font-weight: 500; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem;
}
.metric-card .metric-value { font-size: 1.75rem; font-weight: 700; color: var(--text-primary); line-height: 1.2; }
.metric-card .metric-delta { font-size: 0.8rem; margin-top: 0.25rem; }
.metric-card .metric-delta.positive { color: var(--success); }
.metric-card .metric-delta.negative { color: var(--danger); }
.metric-card .metric-delta.neutral  { color: var(--text-secondary); }

/* ═══ Recommendation Box ═══ */
.recommendation-box {
    background: linear-gradient(135deg, #1e1b4b 0%, var(--bg-secondary) 100%);
    padding: 1.5rem;
    border-radius: var(--radius-lg);
    border-left: 4px solid var(--primary);
    margin: 1.25rem 0;
    color: var(--text-primary) !important;
    box-shadow: var(--shadow-md);
    transition: var(--transition);
}
.recommendation-box:hover {
    border-left-color: var(--primary-light);
    box-shadow: var(--shadow-lg);
}
.recommendation-box h3,
.recommendation-box p,
.recommendation-box strong {
    color: var(--text-primary) !important;
}

/* ═══ Success Box ═══ */
.success-box {
    background: linear-gradient(135deg, #052e16 0%, var(--bg-secondary) 100%);
    padding: 1.5rem;
    border-radius: var(--radius-lg);
    border-left: 4px solid var(--success);
    margin: 1.25rem 0;
    color: #d1fae5 !important;
    box-shadow: var(--shadow-md);
}
.success-box h3, .success-box p, .success-box strong { color: #d1fae5 !important; }

/* ═══ Streamlit Metric Override ═══ */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--bg-secondary) 0%, rgba(51,65,85,0.3) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    transition: var(--transition);
}
[data-testid="stMetric"]:hover {
    border-color: rgba(99,102,241,0.4);
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-secondary) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ═══ DataFrame Styling ═══ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
}

/* ═══ Button Styling ═══ */
.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    transition: var(--transition) !important;
    border: 1px solid var(--border) !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-sm);
}

/* ═══ Expander Styling ═══ */
.streamlit-expanderHeader {
    background-color: var(--bg-secondary) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    transition: var(--transition);
}
.streamlit-expanderHeader:hover {
    color: var(--primary-light) !important;
}

/* ═══ Tab Styling ═══ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    background: var(--bg-secondary);
    border-radius: var(--radius-sm);
    padding: 0.25rem;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background-color: var(--primary) !important;
    color: white !important;
}

/* ═══ Selectbox / Input Styling ═══ */
.stSelectbox [data-baseweb="select"],
.stTextInput input,
.stNumberInput input {
    border-radius: var(--radius-sm) !important;
    transition: var(--transition);
}

/* ═══ Weekly Calendar Cards ═══ */
.calendar-card {
    border-radius: var(--radius-md);
    padding: 1.1rem 1rem;
    text-align: center;
    min-height: 220px;
    border: 1px solid var(--border);
    transition: var(--transition);
}
.calendar-card:hover {
    border-color: var(--primary);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}
.calendar-card.publish-long {
    background: linear-gradient(180deg, rgba(99,102,241,0.15) 0%, var(--bg-secondary) 100%);
    border-color: rgba(99,102,241,0.3);
}
.calendar-card.publish-short {
    background: linear-gradient(180deg, rgba(236,72,153,0.15) 0%, var(--bg-secondary) 100%);
    border-color: rgba(236,72,153,0.3);
}
.calendar-card.rest {
    background: var(--bg-secondary);
    opacity: 0.7;
}

/* ═══ Sidebar Footer ═══ */
.sidebar-footer {
    text-align: center;
    padding: 1rem;
    border-top: 1px solid var(--border);
    margin-top: 1rem;
}
.sidebar-footer .version { font-size: 0.7rem; color: var(--text-muted); }
.sidebar-footer .powered-by { font-size: 0.65rem; color: var(--text-muted); margin-top: 0.25rem; }
.sidebar-footer .powered-by span { color: var(--primary-light); }

/* ═══ Scrollbar ═══ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--bg-tertiary); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ═══ Smooth transitions ═══ */
a, button, [data-testid="stMetric"], .stButton > button,
.streamlit-expanderHeader, [data-baseweb="tab"] {
    transition: var(--transition) !important;
}

/* ═══ Hide Streamlit branding ═══ */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Sanitización HTML ──────────────────────────────────────────────────────

def _esc(value) -> str:
    """Escapa HTML en cualquier valor para prevenir XSS.

    Convierte caracteres especiales (<, >, &, ", ') en entidades HTML seguras.
    Acepta cualquier tipo y retorna string escapado.
    """
    return html_mod.escape(str(value)) if value is not None else ''


# ─── UI Component Helpers ───────────────────────────────────────────────────

def ui_section_divider(label: str = None):
    """Divisor de sección estilizado. Reemplaza st.markdown('---')."""
    if label:
        st.markdown(
            f'<div class="section-divider with-label"><span>{_esc(label)}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def ui_section_header(icon: str, title: str, subtitle: str = ""):
    """Header de sección con icono y subtítulo opcional."""
    sub_html = f'<p class="section-subtitle">{_esc(subtitle)}</p>' if subtitle else ''
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon">{_esc(icon)}</div>
        <div>
            <p class="section-title">{_esc(title)}</p>
            {sub_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def ui_page_header(icon: str, title: str, description: str = ""):
    """Header de página con icono grande y descripción."""
    desc_html = f'<p class="page-description">{_esc(description)}</p>' if description else ''
    st.markdown(f"""
    <div class="page-title-bar">
        <div class="page-icon">{_esc(icon)}</div>
        <div>
            <p class="page-title">{_esc(title)}</p>
            {desc_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def ui_metric_card(icon: str, label: str, value: str, delta: str = "", delta_type: str = "neutral"):
    """Tarjeta KPI premium con gradiente. delta_type: 'positive', 'negative', 'neutral'."""
    safe_delta_type = _esc(delta_type) if delta_type in ('positive', 'negative', 'neutral') else 'neutral'
    delta_html = f'<div class="metric-delta {safe_delta_type}">{_esc(delta)}</div>' if delta else ''
    return f"""
    <div class="metric-card">
        <span class="metric-icon">{_esc(icon)}</span>
        <div class="metric-label">{_esc(label)}</div>
        <div class="metric-value">{_esc(value)}</div>
        {delta_html}
    </div>
    """


@st.cache_data(ttl=300)
def load_data():
    """Carga los datos de la base de datos (cacheado 5 min)."""
    try:
        db = YouTubeDatabase()
        videos_df = db.get_all_videos()
        db.close()
        return videos_df
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return pd.DataFrame()


# ─── Dashboard Preferences (Mejora 15.x) ──────────────────────────────────

PREFS_PATH = os.path.join(os.path.dirname(__file__), 'config', 'dashboard_prefs.json')

_DEFAULT_PREFS = {
    'theme': 'dark',
    'widget_order': ['kpis', 'distribution', 'performance', 'export'],
}


def _load_prefs() -> dict:
    """Carga preferencias del dashboard desde config/dashboard_prefs.json."""
    try:
        if os.path.exists(PREFS_PATH):
            with open(PREFS_PATH, 'r', encoding='utf-8') as f:
                return {**_DEFAULT_PREFS, **json.load(f)}
    except Exception:
        pass
    return dict(_DEFAULT_PREFS)


def _save_prefs(prefs: dict):
    """Guarda preferencias del dashboard a config/dashboard_prefs.json."""
    os.makedirs(os.path.dirname(PREFS_PATH), exist_ok=True)
    with open(PREFS_PATH, 'w', encoding='utf-8') as f:
        json.dump(prefs, f, indent=2, ensure_ascii=False)


# ─── Overview Widgets (Mejora 15.1) ────────────────────────────────────────

_WIDGET_REGISTRY = {}


def _register_widget(key, label):
    """Decorator para registrar widgets del overview."""
    def decorator(fn):
        _WIDGET_REGISTRY[key] = {'fn': fn, 'label': label}
        return fn
    return decorator


@_register_widget('kpis', '📊 KPIs principales')
def _widget_kpis(df, channel_name, subscriber_count):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        subs_display = f"{subscriber_count:,}" if subscriber_count else "No disponible"
        st.metric("👥 Suscriptores", subs_display)
    with col2:
        st.metric("Total Videos", f"{len(df)}")
    with col3:
        st.metric("Vistas Totales", f"{df['view_count'].sum():,.0f}")
    with col4:
        st.metric("Promedio por Video", f"{df['view_count'].mean():,.0f}")
    with col5:
        st.metric("Engagement Rate", f"{df['engagement_rate'].mean():.2f}%")


@_register_widget('distribution', '🎬 Distribución de Contenido')
def _widget_distribution(df, channel_name, subscriber_count):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎬 Distribución de Contenido")
        type_counts = df['video_type'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values, names=type_counts.index,
            title="Shorts vs Videos Largos",
            color_discrete_sequence=VIDEO_TYPE_SEQUENCE,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.subheader("📈 Performance por Tipo")
        perf = df.groupby('video_type').agg(
            {'view_count': 'mean', 'engagement_rate': 'mean'}
        ).reset_index()
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='Vistas Promedio',
            x=perf['video_type'], y=perf['view_count'],
            marker_color=COLORS['primary'],
        ))
        fig_bar.update_layout(title="Vistas Promedio por Tipo", yaxis_title="Vistas", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)


@_register_widget('performance', '🏆 Top Videos')
def _widget_performance(df, channel_name, subscriber_count):
    st.subheader("🏆 Top 10 Videos con Mejor Performance")
    top10 = df.nlargest(10, 'view_count')[
        ['title', 'video_type', 'view_count', 'engagement_rate', 'published_at']
    ].copy()
    top10['published_at'] = top10['published_at'].dt.strftime('%Y-%m-%d')
    st.dataframe(
        top10.rename(columns={
            'title': 'Título', 'video_type': 'Tipo', 'view_count': 'Vistas',
            'engagement_rate': 'Engagement %', 'published_at': 'Publicado',
        }),
        use_container_width=True, hide_index=True,
    )


@_register_widget('export', '📥 Exportar datos')
def _widget_export(df, channel_name, subscriber_count):
    st.subheader("📥 Exportar datos")
    export_cols = ['title', 'video_type', 'published_at', 'view_count',
                   'like_count', 'comment_count', 'engagement_rate', 'duration_seconds',
                   'tags', 'description']
    export_df = df[[c for c in export_cols if c in df.columns]].copy()
    if 'published_at' in export_df.columns:
        export_df['published_at'] = export_df['published_at'].dt.strftime('%Y-%m-%d %H:%M')

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name='Videos', index=False)
        summary = pd.DataFrame({
            'Métrica': ['Canal', 'Total videos', 'Vistas totales', 'Vistas promedio', 'Engagement %'],
            'Valor': [channel_name, len(df), int(df['view_count'].sum()),
                      int(df['view_count'].mean()), round(float(df['engagement_rate'].mean()), 2)]
        })
        summary.to_excel(writer, sheet_name='Resumen', index=False)

    st.download_button(
        label="📥 Descargar Excel (.xlsx)", data=buffer.getvalue(),
        file_name=f"{channel_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if st.button("📊 Exportar a Google Sheets", key="btn_gsheets_metrics"):
        try:
            with st.spinner("Creando Google Sheet..."):
                gsheets = GoogleSheetsExporter()
                result = gsheets.export_video_metrics(df, channel_name=channel_name)
            st.success("✅ Google Sheet creado")
            st.markdown(f"[📊 Abrir Google Sheet]({result['spreadsheet_url']})")
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Error: {e}")


def show_overview(df):
    """Muestra el resumen general del canal con widgets personalizables (15.1)."""
    ui_page_header("📊", "Resumen General", "Vista rápida de las métricas principales del canal")

    if df.empty:
        st.warning("No hay datos disponibles. Ejecuta primero el script main.py para extraer datos.")
        return

    channel_name = df['channel_title'].iloc[0] if pd.notna(df['channel_title'].iloc[0]) else "Canal"
    raw_subs = df['subscriber_count'].iloc[0] if 'subscriber_count' in df.columns else None
    subscriber_count = int(raw_subs) if raw_subs is not None and pd.notna(raw_subs) else None

    st.subheader(f"📺 {channel_name}")

    # Widget order from prefs
    prefs = st.session_state.get('prefs', _DEFAULT_PREFS)
    widget_order = prefs.get('widget_order', _DEFAULT_PREFS['widget_order'])
    all_keys = list(_WIDGET_REGISTRY.keys())
    widget_order = [k for k in widget_order if k in _WIDGET_REGISTRY] + \
                   [k for k in all_keys if k not in widget_order]

    # Layout customizer (collapsed expander)
    with st.expander("⚙ Personalizar layout", expanded=False):
        st.caption("Selecciona el orden de las secciones del resumen")
        new_order = []
        for i, key in enumerate(widget_order):
            options = [_WIDGET_REGISTRY[k]['label'] for k in widget_order]
            selected = st.selectbox(
                f"Posición {i + 1}", options, index=i,
                key=f"widget_pos_{i}",
            )
            sel_key = next(k for k, v in _WIDGET_REGISTRY.items() if v['label'] == selected)
            new_order.append(sel_key)

        if st.button("💾 Guardar layout", key="btn_save_layout"):
            seen = set()
            deduped = []
            for k in new_order:
                if k not in seen:
                    deduped.append(k)
                    seen.add(k)
            st.session_state.prefs['widget_order'] = deduped
            _save_prefs(st.session_state.prefs)
            st.success("Layout guardado")
            st.rerun()

    ui_section_divider()

    # Render widgets in order
    for key in widget_order:
        if key in _WIDGET_REGISTRY:
            _WIDGET_REGISTRY[key]['fn'](df, channel_name, subscriber_count)
            ui_section_divider()


def show_performance_analysis(df):
    """Muestra análisis detallado de performance"""
    ui_page_header("📈", "Análisis de Performance", "Tendencias, ranking y evolución de tus videos")
    
    if df.empty:
        st.warning("No hay datos disponibles.")
        return
    
    # Timeline de publicaciones
    st.subheader("📅 Timeline de Publicaciones y Vistas")
    
    df_sorted = df.sort_values('published_at')
    
    fig_timeline = px.scatter(
        df_sorted,
        x='published_at',
        y='view_count',
        color='video_type',
        size='engagement_rate',
        hover_data=['title', 'view_count', 'engagement_rate'],
        title="Evolución de Vistas en el Tiempo",
        color_discrete_sequence=VIDEO_TYPE_SEQUENCE
    )
    
    fig_timeline.update_layout(
        xaxis_title="Fecha de Publicación",
        yaxis_title="Vistas",
        height=500
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Top 10 videos
    st.subheader("🏆 Top 10 Videos con Mejor Performance")
    
    top_10 = df.nlargest(10, 'view_count')[['title', 'video_type', 'view_count', 'engagement_rate', 'published_at']]
    top_10['published_at'] = top_10['published_at'].dt.strftime('%Y-%m-%d')
    
    st.dataframe(
        top_10.rename(columns={
            'title': 'Título',
            'video_type': 'Tipo',
            'view_count': 'Vistas',
            'engagement_rate': 'Engagement %',
            'published_at': 'Publicado'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Análisis de engagement
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💬 Engagement por Tipo")
        
        engagement_by_type = df.groupby('video_type')['engagement_rate'].mean().reset_index()
        
        fig_engagement = px.bar(
            engagement_by_type,
            x='video_type',
            y='engagement_rate',
            title="Engagement Rate Promedio",
            color='video_type',
            color_discrete_sequence=VIDEO_TYPE_SEQUENCE
        )
        
        fig_engagement.update_layout(
            xaxis_title="Tipo de Video",
            yaxis_title="Engagement Rate (%)",
            showlegend=False
        )
        
        st.plotly_chart(fig_engagement, use_container_width=True)
    
    with col2:
        st.subheader("📊 Distribución de Vistas")
        
        fig_box = px.box(
            df,
            x='video_type',
            y='view_count',
            color='video_type',
            title="Distribución de Vistas por Tipo",
            color_discrete_sequence=VIDEO_TYPE_SEQUENCE
        )
        
        fig_box.update_layout(
            xaxis_title="Tipo de Video",
            yaxis_title="Vistas",
            showlegend=False
        )

        st.plotly_chart(fig_box, use_container_width=True)

    # ── Evolución semanal y mensual ───────────────────────────────────
    ui_section_divider()
    st.subheader("📆 Evolución de Vistas en el Tiempo")

    df_time = df.copy()
    df_time['published_at'] = pd.to_datetime(df_time['published_at'], utc=True)

    tab_week, tab_month = st.tabs(["📅 Por semana", "🗓 Por mes"])

    for tab, freq, label in [
        (tab_week,  'W', 'Semana'),
        (tab_month, 'ME', 'Mes'),
    ]:
        with tab:
            grouped = (
                df_time
                .groupby([pd.Grouper(key='published_at', freq=freq), 'video_type'])['view_count']
                .sum()
                .reset_index()
            )
            grouped.columns = [label, 'Tipo', 'Vistas']
            fig_evo = px.line(
                grouped,
                x=label, y='Vistas', color='Tipo',
                markers=True,
                color_discrete_sequence=VIDEO_TYPE_SEQUENCE,
                labels={label: label, 'Vistas': 'Vistas acumuladas'},
            )
            fig_evo.update_layout(
                height=360, margin={'l': 0, 'r': 0, 't': 20, 'b': 0},
                hovermode='x unified',
            )
            st.plotly_chart(fig_evo, use_container_width=True)


def show_virality_prediction(df: pd.DataFrame, channel_id: str):
    """Predicción de viralidad con ML"""
    ui_page_header("🔮", "Predicción de Viralidad", "Modelo ML para estimar potencial viral")

    if df.empty or not channel_id:
        st.warning("No hay datos disponibles para este canal.")
        return

    predictor, from_cache = _get_virality_predictor(df, channel_id)

    # ── Sección A: Puntuar videos existentes ──────────────────────────
    st.subheader("🧠 Scores de Viralidad — Videos del Canal")

    if not predictor.is_trained():
        train_result = predictor.get_train_metrics()
        st.warning(f"⚠ {train_result.get('reason', 'No se pudo entrenar el modelo.')}")
        st.info("Ejecuta `python main.py` para obtener más datos de videos.")
        return

    train_result = predictor.get_train_metrics()
    n_splits = train_result.get('cv_splits', '?')
    if from_cache:
        st.info(f"Modelo cargado desde caché — {train_result.get('samples', '?')} videos.")
    else:
        st.success(f"Modelo reentrenado con **{train_result.get('samples', '?')} videos** · {n_splits}-fold CV temporal.")

    cv_mae_p = train_result.get('cv_mae_percentile')
    if cv_mae_p is not None:
        st.metric(
            "Error CV (MAE percentil)",
            f"±{cv_mae_p:.1f} pts",
            help="Error absoluto medio de validación cruzada temporal sobre el percentil (0–100). Menor = más preciso.",
        )

    scored_df = predictor.predict(df)

    # Guardar en DB
    try:
        with YouTubeDatabase() as db:
            db.save_virality_predictions(
                scored_df[['video_id', 'channel_id', 'virality_score']].copy()
            )
    except Exception:
        pass  # No bloquear si la DB falla

    # Tabla top 20
    display_cols = ['title', 'video_type', 'view_count', 'engagement_rate', 'virality_score']
    top_df = scored_df[display_cols].sort_values('virality_score', ascending=False).head(20).copy()
    top_df['virality_score'] = top_df['virality_score'].apply(lambda s: f"{s:.1f} / 10")

    st.dataframe(
        top_df.rename(columns={
            'title': 'Título',
            'video_type': 'Tipo',
            'view_count': 'Vistas',
            'engagement_rate': 'Engagement %',
            'virality_score': '🔥 Score Viral',
        }),
        use_container_width=True,
        hide_index=True,
    )

    ui_section_divider()
    col1, col2 = st.columns(2)

    # Importancia de features
    with col1:
        st.subheader("📊 Factores que más influyen")
        fi = predictor.get_feature_importance()
        labels_map = {
            'hour': 'Hora de publicación (Panamá)',
            'weekday_num': 'Día de la semana',
            'is_short': 'Formato (Short/Largo)',
            'duration_seconds': 'Duración (seg)',
            'tags_count': 'Cantidad de tags',
            'title_length': 'Longitud del título',
            'title_has_number': 'Número en el título',
            'title_has_question': 'Pregunta en el título (?)',
            'description_length': 'Longitud de descripción',
            'days_since_last_upload': 'Días desde última subida',
            'channel_age_days': 'Antigüedad del canal (días)',
        }
        fi_df = pd.DataFrame({
            'Factor': [labels_map[k] for k in fi],
            'Importancia': list(fi.values()),
        }).sort_values('Importancia', ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi_df['Importancia'],
            y=fi_df['Factor'],
            orientation='h',
            marker_color=COLORS["primary"],
        ))
        fig_fi.update_layout(
            xaxis_title='Importancia relativa',
            height=420,
            margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Scatter score vs vistas reales
    with col2:
        st.subheader("📈 Score vs Vistas reales")
        fig_sc = px.scatter(
            scored_df,
            x='virality_score',
            y='view_count',
            color='video_type',
            hover_data=['title'],
            labels={'virality_score': 'Score Viral', 'view_count': 'Vistas'},
            color_discrete_sequence=VIDEO_TYPE_SEQUENCE,
        )
        fig_sc.update_layout(height=300, margin={'l': 0, 'r': 0, 't': 10, 'b': 0})
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Sección B: Predictor interactivo ──────────────────────────────
    ui_section_divider()
    st.subheader("🎯 ¿Qué tan viral sería tu próximo video?")

    col1, col2, col3 = st.columns(3)
    with col1:
        dia = st.selectbox("📅 Día de publicación", WEEKDAY_LABELS)
        hora = st.slider("🕐 Hora de publicación (Panamá UTC-5)", 0, 23, 18)
        tiene_numero = st.checkbox("🔢 El título contiene números", value=False)
        tiene_pregunta = st.checkbox("❓ El título es una pregunta (?)", value=False)
    with col2:
        formato = st.radio("🎬 Formato", ["Short", "Video Largo"])
        duracion = st.number_input(
            "⏱ Duración (segundos)",
            min_value=1, max_value=3600,
            value=60 if formato == "Short" else 600,
        )
        len_descripcion = st.number_input(
            "📝 Longitud de descripción (caracteres)",
            min_value=0, max_value=5000, value=500,
        )
    with col3:
        num_tags = st.number_input("🏷 Cantidad de tags", min_value=0, max_value=50, value=10)
        len_titulo = st.number_input("✍ Longitud del título (caracteres)", min_value=5, max_value=200, value=60)
        dias_desde_ultimo = st.number_input(
            "📆 Días desde última subida",
            min_value=0, max_value=365, value=7,
        )

    weekday_num = WEEKDAY_LABELS.index(dia)
    is_short = formato == "Short"

    # Edad del canal: calculada desde los datos reales del canal
    dt_utc = pd.to_datetime(df['published_at'], utc=True)
    canal_age_days = int((dt_utc.max() - dt_utc.min()).days) if len(df) > 1 else 365

    score = predictor.predict_single(
        hour=hora,
        weekday=weekday_num,
        is_short=is_short,
        duration_seconds=int(duracion),
        tags_count=int(num_tags),
        title_length=int(len_titulo),
        title_has_number=int(tiene_numero),
        title_has_question=int(tiene_pregunta),
        description_length=int(len_descripcion),
        days_since_last_upload=int(dias_desde_ultimo),
        channel_age_days=canal_age_days,
    )

    color = predictor.score_color(score)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Potencial Viral", 'font': {'size': 20}},
        number={'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [1, 10], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [1, 4], 'color': COLORS["bg_secondary"]},
                {'range': [4, 7], 'color': 'rgba(245,158,11,0.1)'},
                {'range': [7, 10], 'color': 'rgba(16,185,129,0.1)'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 3},
                'thickness': 0.75,
                'value': score,
            },
        }
    ))
    fig_gauge.update_layout(height=320, margin={'l': 20, 'r': 20, 't': 40, 'b': 0})
    st.plotly_chart(fig_gauge, use_container_width=True)

    if score >= 8:
        st.success(f"🚀 **Score {score}/10** — Alto potencial viral. ¡Excelente combinación!")
    elif score >= 5:
        st.warning(f"📊 **Score {score}/10** — Potencial moderado. Ajusta hora o formato.")
    else:
        st.error(f"⚠ **Score {score}/10** — Potencial bajo. Considera publicar en otro momento.")

    # ── What-If: Predicción cruzada de vistas ──────────────────────────
    view_pred, _ = _get_view_predictor(df, channel_id)
    if view_pred.is_trained():
        st.markdown("---")
        st.markdown("##### 👁 Estimación de vistas para este escenario")
        vp_result = view_pred.predict_single(
            hour=hora,
            weekday=weekday_num,
            is_short=is_short,
            duration_seconds=int(duracion),
            tags_count=int(num_tags),
            title_length=int(len_titulo),
            title_has_number=int(tiene_numero),
            title_has_question=int(tiene_pregunta),
            description_length=int(len_descripcion),
            days_since_last_upload=int(dias_desde_ultimo),
            channel_age_days=canal_age_days,
        )
        vp_avg = int(view_pred.get_train_metrics().get('channel_avg_views', 0))
        vp_pred = vp_result['predicted']
        vp_delta = vp_pred - vp_avg

        col_wif1, col_wif2 = st.columns(2)
        with col_wif1:
            st.metric(
                label="Vistas estimadas",
                value=f"~{vp_pred:,}",
                delta=f"{vp_delta:+,} vs promedio del canal",
            )
            st.caption(f"Rango de confianza: **{vp_result['low']:,} — {vp_result['high']:,}** vistas")
        with col_wif2:
            vp_ratio = vp_pred / vp_avg if vp_avg > 0 else 1.0
            if vp_ratio >= 1.5:
                st.success(f"🚀 **{vp_ratio:.1f}x** el promedio del canal — Alto potencial")
            elif vp_ratio >= 0.8:
                st.warning(f"📊 **{vp_ratio:.1f}x** el promedio del canal — Rendimiento moderado")
            else:
                st.error(f"⚠ **{vp_ratio:.1f}x** el promedio del canal — Bajo el promedio")


def show_view_prediction(df: pd.DataFrame, channel_id: str):
    """Predicción de vistas con ML — número esperado + rango de confianza + heatmap."""
    ui_page_header("👁", "Predicción de Vistas", "Estimación de vistas con rango de confianza")

    if df.empty or not channel_id:
        st.warning("No hay datos disponibles para este canal.")
        return

    predictor, from_cache = _get_view_predictor(df, channel_id)

    if not predictor.is_trained():
        reason = predictor.get_train_metrics().get('reason', 'No se pudo entrenar el modelo.')
        st.warning(f"⚠ {reason}")
        st.info("Ejecuta `python main.py` para obtener más datos de videos.")
        return

    # ── Sección A: Métricas de precisión + scatter ─────────────────────
    st.subheader("🎯 Precisión del modelo en datos históricos")

    metrics = predictor.get_train_metrics()
    n_splits = metrics.get('cv_splits', '?')
    if from_cache:
        st.info(f"Modelo cargado desde caché — {metrics.get('samples', '?')} videos.")
    else:
        st.success(f"Modelo reentrenado con **{metrics.get('samples', '?')} videos** · {n_splits}-fold CV temporal.")
    col1, col2, col3 = st.columns(3)
    channel_avg = int(metrics.get('channel_avg_views', 0))
    with col1:
        st.metric(
            "Videos usados para entrenar",
            f"{metrics['samples']}",
            help=f"Validación cruzada temporal con {n_splits} folds (TimeSeriesSplit).",
        )
    with col2:
        cv_mae = metrics.get('cv_mae', metrics.get('mae', 0))
        st.metric(
            "MAE — Validación cruzada",
            f"{int(cv_mae):,} vistas",
            help="Error absoluto medio promedio de los folds de CV temporal. Mide cuántas vistas se desvía la predicción en promedio.",
        )
    with col3:
        cv_mape = metrics.get('cv_mape', metrics.get('mape', 0))
        st.metric(
            "MAPE — Validación cruzada",
            f"{cv_mape:.1f}%",
            help="Error porcentual absoluto medio promedio de los folds. Indica el % de error relativo al valor real.",
        )

    # Predecir sobre todos los videos y guardar
    scored_df = predictor.predict(df)

    try:
        with YouTubeDatabase() as db:
            db.save_view_predictions(
                scored_df[['video_id', 'channel_id', 'predicted_views', 'predicted_low', 'predicted_high']].copy()
            )
    except Exception:
        pass

    # Scatter: Vistas reales vs predichas
    fig_scatter = px.scatter(
        scored_df,
        x='view_count',
        y='predicted_views',
        color='video_type',
        hover_data=['title'],
        labels={'view_count': 'Vistas Reales', 'predicted_views': 'Vistas Predichas'},
        color_discrete_sequence=VIDEO_TYPE_SEQUENCE,
    )
    # Línea de referencia 45° (predicción perfecta)
    max_val = max(scored_df['view_count'].max(), scored_df['predicted_views'].max())
    fig_scatter.add_shape(
        type='line', x0=0, y0=0, x1=max_val, y1=max_val,
        line={'color': '#888', 'dash': 'dash', 'width': 1}
    )
    fig_scatter.update_layout(height=380, margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Sección B: Predictor interactivo ───────────────────────────────
    ui_section_divider()
    st.subheader("🔢 ¿Cuántas vistas tendría tu próximo video?")

    col1, col2, col3 = st.columns(3)
    with col1:
        dia = st.selectbox("📅 Día de publicación", WEEKDAY_LABELS, key="vp_dia")
        hora = st.slider("🕐 Hora de publicación (Panamá UTC-5)", 0, 23, 18, key="vp_hora")
        tiene_numero = st.checkbox("🔢 El título contiene números", value=False, key="vp_num")
        tiene_pregunta = st.checkbox("❓ El título es una pregunta (?)", value=False, key="vp_q")
    with col2:
        formato = st.radio("🎬 Formato", ["Short", "Video Largo"], key="vp_formato")
        duracion = st.number_input(
            "⏱ Duración (segundos)",
            min_value=1, max_value=3600,
            value=60 if formato == "Short" else 600,
            key="vp_duracion",
        )
        len_descripcion = st.number_input(
            "📝 Longitud de descripción (caracteres)",
            min_value=0, max_value=5000, value=500, key="vp_desc",
        )
    with col3:
        num_tags = st.number_input("🏷 Cantidad de tags", min_value=0, max_value=50, value=10, key="vp_tags")
        len_titulo = st.number_input("✍ Longitud del título (caracteres)", min_value=5, max_value=200, value=60, key="vp_titulo")
        dias_desde_ultimo = st.number_input(
            "📆 Días desde última subida",
            min_value=0, max_value=365, value=7, key="vp_dias",
        )

    weekday_num = WEEKDAY_LABELS.index(dia)
    is_short = formato == "Short"

    # Edad del canal: calculada desde los datos reales del canal
    dt_utc_vp = pd.to_datetime(df['published_at'], utc=True)
    canal_age_days_vp = int((dt_utc_vp.max() - dt_utc_vp.min()).days) if len(df) > 1 else 365

    result = predictor.predict_single(
        hour=hora,
        weekday=weekday_num,
        is_short=is_short,
        duration_seconds=int(duracion),
        tags_count=int(num_tags),
        title_length=int(len_titulo),
        title_has_number=int(tiene_numero),
        title_has_question=int(tiene_pregunta),
        description_length=int(len_descripcion),
        days_since_last_upload=int(dias_desde_ultimo),
        channel_age_days=canal_age_days_vp,
    )

    pred = result['predicted']
    low = result['low']
    high = result['high']
    delta_vs_avg = pred - channel_avg

    col_res1, col_res2 = st.columns([1, 1])
    with col_res1:
        st.metric(
            label="Vistas estimadas",
            value=f"~{pred:,}",
            delta=f"{delta_vs_avg:+,} vs promedio del canal",
        )
        st.caption(f"Rango de confianza: **{low:,} — {high:,}** vistas")
    with col_res2:
        ratio = pred / channel_avg if channel_avg > 0 else 1.0
        if ratio >= 1.5:
            st.success(f"🚀 **{ratio:.1f}x** el promedio del canal — Alto potencial")
        elif ratio >= 0.8:
            st.warning(f"📊 **{ratio:.1f}x** el promedio del canal — Rendimiento moderado")
        else:
            st.error(f"⚠ **{ratio:.1f}x** el promedio del canal — Bajo el promedio")

    # ── What-If: Predicción cruzada de viralidad ───────────────────────
    vir_pred, _ = _get_virality_predictor(df, channel_id)
    if vir_pred.is_trained():
        st.markdown("---")
        st.markdown("##### 🔮 Score de viralidad para este escenario")
        vir_score = vir_pred.predict_single(
            hour=hora,
            weekday=weekday_num,
            is_short=is_short,
            duration_seconds=int(duracion),
            tags_count=int(num_tags),
            title_length=int(len_titulo),
            title_has_number=int(tiene_numero),
            title_has_question=int(tiene_pregunta),
            description_length=int(len_descripcion),
            days_since_last_upload=int(dias_desde_ultimo),
            channel_age_days=canal_age_days_vp,
        )
        vir_color = vir_pred.score_color(vir_score)

        fig_gauge_vp = go.Figure(go.Indicator(
            mode="gauge+number",
            value=vir_score,
            title={'text': "Potencial Viral", 'font': {'size': 18}},
            number={'font': {'size': 40, 'color': vir_color}},
            gauge={
                'axis': {'range': [1, 10], 'tickwidth': 1},
                'bar': {'color': vir_color},
                'steps': [
                    {'range': [1, 4], 'color': COLORS["bg_secondary"]},
                    {'range': [4, 7], 'color': 'rgba(245,158,11,0.1)'},
                    {'range': [7, 10], 'color': 'rgba(16,185,129,0.1)'},
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 3},
                    'thickness': 0.75,
                    'value': vir_score,
                },
            }
        ))
        fig_gauge_vp.update_layout(height=260, margin={'l': 20, 'r': 20, 't': 30, 'b': 0})
        st.plotly_chart(fig_gauge_vp, use_container_width=True)

        if vir_score >= 8:
            st.success(f"🚀 **Score {vir_score}/10** — Alto potencial viral")
        elif vir_score >= 5:
            st.warning(f"📊 **Score {vir_score}/10** — Potencial moderado")
        else:
            st.error(f"⚠ **Score {vir_score}/10** — Potencial bajo")

    # ── Sección C: Heatmap ─────────────────────────────────────────────
    ui_section_divider()
    st.subheader("🗓 Mejor momento para publicar")
    st.caption("Vistas estimadas según día y hora (timezone Panamá UTC-5)")

    tab_short, tab_largo = st.tabs(["📱 Shorts", "🎬 Videos Largos"])

    for tab, fmt_is_short, fmt_dur in [
        (tab_short, True, 45),
        (tab_largo, False, 600),
    ]:
        with tab:
            heatmap_df = predictor.get_publishing_heatmap(
                is_short=fmt_is_short,
                duration_seconds=fmt_dur,
                tags_count=10,
                title_length=60,
            )
            fig_hm = px.imshow(
                heatmap_df,
                labels={'x': 'Hora (Panamá)', 'y': 'Día', 'color': 'Vistas'},
                x=[str(h) for h in range(24)],
                y=WEEKDAY_LABELS,
                color_continuous_scale='YlOrRd',
                aspect='auto',
            )
            fig_hm.update_layout(height=300, margin={'l': 0, 'r': 0, 't': 10, 'b': 0})
            st.plotly_chart(fig_hm, use_container_width=True)


def show_metrics_history(channel_id: str):
    """Evolución histórica de métricas del canal a lo largo de las ejecuciones."""
    ui_page_header("📅", "Histórico de Métricas", "Evolución del canal a lo largo del tiempo")

    if not channel_id:
        st.warning("Selecciona un canal en el panel lateral.")
        return

    try:
        with YouTubeDatabase() as db:
            history_df = db.get_channel_metrics_history(channel_id)
            growth_df = db.get_top_videos_growth(channel_id, top_n=10)
    except Exception as e:
        st.error(f"Error cargando histórico: {e}")
        return

    if history_df.empty:
        st.info("No hay datos históricos. Ejecuta `python main.py` al menos una vez.")
        return

    n_snapshots = len(history_df)

    if n_snapshots < 2:
        st.info(
            "Solo hay **1 snapshot** registrada. "
            "Ejecuta `python main.py` en otro momento para comenzar a ver la evolución del canal."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total vistas (actual)", f"{int(history_df['total_views'].iloc[0]):,}")
        with col2:
            st.metric("Engagement promedio", f"{history_df['avg_engagement'].iloc[0]:.2f}%")
        with col3:
            st.metric("Videos rastreados", f"{int(history_df['videos_tracked'].iloc[0])}")
        return

    # ── Sección A: Evolución total del canal ───────────────────────────
    st.subheader("📈 Vistas totales del canal en el tiempo")

    fig_canal = px.line(
        history_df,
        x='snapshot_date',
        y='total_views',
        markers=True,
        labels={'snapshot_date': 'Fecha', 'total_views': 'Vistas Totales'},
        color_discrete_sequence=[COLORS["primary"]],
    )
    fig_canal.update_traces(marker_size=8)
    fig_canal.update_layout(height=320, margin={'l': 0, 'r': 0, 't': 10, 'b': 0})
    st.plotly_chart(fig_canal, use_container_width=True)

    # ── Sección B: Nuevas vistas entre ejecuciones ─────────────────────
    ui_section_divider()
    st.subheader("📊 Nuevas vistas entre ejecuciones")
    st.caption("Incremento de vistas entre cada snapshot consecutivo")

    delta_df = history_df.copy()
    delta_df['new_views'] = delta_df['total_views'].diff().fillna(0).astype(int)
    delta_df = delta_df[delta_df['new_views'] >= 0]  # descartar si BD fue limpiada

    colors = [COLORS["success"] if v >= 0 else COLORS["danger"] for v in delta_df['new_views']]
    fig_delta = go.Figure(go.Bar(
        x=delta_df['snapshot_date'],
        y=delta_df['new_views'],
        marker_color=colors,
        text=delta_df['new_views'].apply(lambda v: f"+{v:,}"),
        textposition='outside',
    ))
    fig_delta.update_layout(
        xaxis_title='Fecha',
        yaxis_title='Nuevas vistas',
        height=300,
        margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    # KPI del período
    total_growth = int(history_df['total_views'].iloc[-1] - history_df['total_views'].iloc[0])
    days_span = (history_df['snapshot_date'].iloc[-1] - history_df['snapshot_date'].iloc[0]).days
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Crecimiento total de vistas", f"+{total_growth:,}")
    with col2:
        st.metric("Período rastreado", f"{days_span} días" if days_span > 0 else "Mismo día")
    with col3:
        daily_avg = total_growth / days_span if days_span > 0 else total_growth
        st.metric("Promedio diario de nuevas vistas", f"{daily_avg:,.0f}")

    # ── Sección C: Evolución de los top 10 videos ──────────────────────
    ui_section_divider()
    st.subheader("🎬 Evolución de los videos más vistos")

    if growth_df.empty or len(growth_df['snapshot_date'].unique()) < 2:
        st.info("Se necesitan al menos 2 ejecuciones para ver la evolución por video.")
    else:
        # Selector de videos
        all_titles = growth_df['title'].unique().tolist()
        selected_titles = st.multiselect(
            "Selecciona videos a comparar",
            options=all_titles,
            default=all_titles[:5],
        )

        if selected_titles:
            filtered = growth_df[growth_df['title'].isin(selected_titles)]

            fig_videos = px.line(
                filtered,
                x='snapshot_date',
                y='view_count',
                color='title',
                line_dash='video_type',
                markers=True,
                labels={
                    'snapshot_date': 'Fecha',
                    'view_count': 'Vistas',
                    'title': 'Video',
                    'video_type': 'Tipo',
                },
            )
            fig_videos.update_traces(marker_size=6)
            fig_videos.update_layout(
                height=420,
                margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
                legend={
                    'orientation': 'h',
                    'yanchor': 'top',
                    'y': -0.25,
                    'xanchor': 'left',
                    'x': 0,
                    'font': {'size': 10},
                },
            )
            st.plotly_chart(fig_videos, use_container_width=True)

    # ── Sección D: Tabla de snapshots ─────────────────────────────────
    with st.expander("📋 Ver tabla de snapshots"):
        display = history_df.copy()
        display['snapshot_date'] = display['snapshot_date'].dt.strftime('%Y-%m-%d')
        display['total_views'] = display['total_views'].apply(lambda v: f"{int(v):,}")
        display['total_likes'] = display['total_likes'].apply(lambda v: f"{int(v):,}")
        display['avg_engagement'] = display['avg_engagement'].apply(lambda v: f"{v:.2f}%")
        st.dataframe(
            display.rename(columns={
                'snapshot_date': 'Fecha',
                'total_views': 'Vistas Totales',
                'total_likes': 'Likes Totales',
                'avg_engagement': 'Engagement Prom.',
                'videos_tracked': 'Videos',
            }),
            use_container_width=True,
            hide_index=True,
        )


def show_recommendations(channel_id=None):
    """Muestra las recomendaciones generadas por IA"""
    ui_page_header("🎯", "Recomendaciones de IA", "Sugerencias generadas por Claude AI")

    try:
        db = YouTubeDatabase()

        if not channel_id:
            st.warning("Selecciona un canal en el panel lateral.")
            db.close()
            return

        recommendations_df = db.get_recommendations(channel_id, days=30)

        if recommendations_df.empty:
            st.info("No hay recomendaciones disponibles. Ejecuta el script main.py para generar una.")
        else:
            latest = recommendations_df.iloc[0]

            topic    = str(latest['recommended_topic'] or '').strip()
            reasoning = str(latest['reasoning'] or '').strip()

            # ── Cabecera con métricas clave ──────────────────────────────
            st.markdown(f"### 💡 Recomendación para: **{latest['recommendation_date']}**")
            col1, col2 = st.columns(2)
            col1.metric("🎬 Formato recomendado", latest['recommended_type'] or '—')
            col2.metric("📊 Performance esperado", latest['predicted_performance'] or '—')

            if topic:
                st.info(f"**📝 Tema sugerido:** {topic}")

            # ── Sugerencias de título ─────────────────────────────────────
            title_suggestions_raw = latest.get('title_suggestions') if hasattr(latest, 'get') else None
            if title_suggestions_raw is None and 'title_suggestions' in latest.index:
                title_suggestions_raw = latest['title_suggestions']

            title_suggestions = []
            if title_suggestions_raw and str(title_suggestions_raw).strip():
                try:
                    title_suggestions = json.loads(title_suggestions_raw)
                except (json.JSONDecodeError, TypeError):
                    pass

            if title_suggestions:
                ui_section_divider()
                st.markdown("**✍ Sugerencias de Título:**")
                for i, sug in enumerate(title_suggestions, 1):
                    title_text = sug.get('title', '')
                    analysis_text = sug.get('analysis', '')
                    st.markdown(f"**{i}.** {title_text}")
                    if analysis_text:
                        st.caption(f"→ {analysis_text}")

            # ── Generador de guion ─────────────────────────────────────
            ui_section_divider()
            rec_id = int(latest['id']) if 'id' in latest.index and pd.notna(latest['id']) else None

            if rec_id:
                existing_outline = db.get_script_outline(rec_id)
            else:
                existing_outline = None

            if existing_outline:
                st.markdown("**📝 Guion / Outline:**")
                st.markdown(existing_outline['outline_text'])
            else:
                first_title = title_suggestions[0]['title'] if title_suggestions else (topic or '—')
                anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')

                if anthropic_key and rec_id:
                    # Detectar idioma del canal y obtener nombre real
                    ch_language = _detect_channel_language(db, channel_id)
                    try:
                        cursor_ch = db.conn.cursor()
                        cursor_ch.execute(
                            "SELECT channel_name FROM channels WHERE channel_id = %s",
                            (channel_id,),
                        )
                        ch_row = cursor_ch.fetchone()
                        ch_display_name = ch_row['channel_name'] if ch_row else channel_id
                    except Exception:
                        ch_display_name = channel_id

                    lang_label = {'en': '🇺🇸 EN', 'es': '🇪🇸 ES', 'pt': '🇧🇷 PT'}.get(ch_language, ch_language)
                    if st.button(f"📝 Generar Guion ({lang_label})", type="secondary", key="btn_script"):
                        with st.spinner("Generando guion con IA..."):
                            analyzer = AIAnalyzer(anthropic_key)
                            outline = analyzer.generate_script_outline(
                                video_type=latest['recommended_type'] or 'Short',
                                topic=topic or '—',
                                title=first_title,
                                channel_name=ch_display_name,
                                language=ch_language,
                            )
                            if 'error' not in outline:
                                db.save_script_outline(channel_id, rec_id, outline)
                                st.markdown("**📝 Guion / Outline:**")
                                st.markdown(outline['outline_text'])
                                st.success("Guion guardado")
                            else:
                                st.error(f"Error: {outline['error']}")

            # ── Generador de SEO Content (Mejora 9.2 + 9.3) ───────────
            ui_section_divider()

            if rec_id:
                existing_seo = db.get_seo_content(rec_id)
            else:
                existing_seo = None

            if existing_seo:
                st.markdown("**📋 Descripción SEO Optimizada:**")
                st.code(existing_seo['seo_description'], language=None)

                if existing_seo['tags']:
                    st.markdown("**🏷 Tags Optimizados:**")
                    tags_html = ' '.join(
                        f'<span style="display:inline-block; background:{COLORS["bg_tertiary"]}; '
                        f'border:1px solid {COLORS["border"]}; border-radius:12px; '
                        f'padding:2px 10px; margin:2px; font-size:0.85rem; '
                        f'color:{COLORS["text_secondary"]};">{_esc(tag)}</span>'
                        for tag in existing_seo['tags']
                    )
                    st.markdown(tags_html, unsafe_allow_html=True)

                if existing_seo['hashtags']:
                    st.markdown(
                        "**#️⃣ Hashtags:** "
                        + '  '.join(existing_seo['hashtags'])
                    )

                if existing_seo['related_videos']:
                    st.markdown("**🔗 Videos Relacionados Sugeridos:**")
                    for rv in existing_seo['related_videos']:
                        st.markdown(f"- [{_esc(rv.get('title', ''))}]({rv.get('url', '')})")

                # Áreas de texto para copiar fácilmente
                with st.expander("📎 Copiar descripción y tags"):
                    st.text_area(
                        "Descripción SEO (selecciona y copia)",
                        value=existing_seo['seo_description'],
                        height=200,
                        key="seo_desc_copy",
                    )
                    if existing_seo['tags']:
                        st.text_area(
                            "Tags (separados por coma)",
                            value=', '.join(existing_seo['tags']),
                            height=80,
                            key="seo_tags_copy",
                        )
            else:
                seo_first_title = title_suggestions[0]['title'] if title_suggestions else (topic or '—')
                seo_api_key = os.getenv('ANTHROPIC_API_KEY', '')

                if seo_api_key and rec_id:
                    if st.button("📋 Generar SEO (Descripción + Tags)", type="secondary", key="btn_seo"):
                        with st.spinner("Generando descripción SEO y tags optimizados..."):
                            seo_analyzer = AIAnalyzer(seo_api_key)

                            # Detectar idioma y nombre del canal
                            seo_language = _detect_channel_language(db, channel_id)
                            try:
                                cursor_seo = db.conn.cursor()
                                cursor_seo.execute(
                                    "SELECT channel_name FROM channels WHERE channel_id = %s",
                                    (channel_id,),
                                )
                                seo_ch_row = cursor_seo.fetchone()
                                seo_ch_name = seo_ch_row['channel_name'] if seo_ch_row else channel_id
                            except Exception:
                                seo_ch_name = channel_id

                            # 1. Tags más frecuentes del canal
                            channel_tags = db.get_top_tags_from_channel(channel_id)

                            # 2. Extraer keywords del título/tema
                            raw_keywords = re.findall(
                                r'\b[a-záéíóúñü]{4,}\b',
                                (seo_first_title + ' ' + (topic or '')).lower(),
                            )
                            _stopwords = {
                                'para', 'como', 'este', 'esta', 'estos', 'estas',
                                'sobre', 'cuando', 'donde', 'porque', 'pero',
                                'with', 'from', 'that', 'this', 'your', 'have',
                            }
                            seo_keywords = [w for w in dict.fromkeys(raw_keywords)
                                            if w not in _stopwords][:5]

                            # 3. Videos relacionados del canal
                            related_videos = db.get_related_videos_by_keywords(
                                channel_id, seo_keywords, limit=5,
                            )

                            # 4. Google Trends (opcional, no bloquear si falla)
                            trend_scores = None
                            rising_queries = None
                            if seo_keywords:
                                try:
                                    if TrendsAnalyzer.is_available():
                                        ta = TrendsAnalyzer()
                                        trend_scores = ta.get_trend_scores(
                                            seo_keywords[:5], geo='PA',
                                        )
                                        related_q = ta.get_related_queries(
                                            seo_keywords[0], geo='PA',
                                        )
                                        rising_df = related_q.get('rising', pd.DataFrame())
                                        if not rising_df.empty and 'query' in rising_df.columns:
                                            rising_queries = rising_df['query'].tolist()[:10]
                                except Exception:
                                    pass  # Trends no disponible, continuar sin datos

                            # 5. Generar con Claude
                            seo_result = seo_analyzer.generate_seo_content(
                                video_type=latest['recommended_type'] or 'Short',
                                topic=topic or '—',
                                title=seo_first_title,
                                channel_name=seo_ch_name,
                                channel_tags=channel_tags,
                                related_videos=related_videos,
                                trend_scores=trend_scores,
                                rising_queries=rising_queries,
                                language=seo_language,
                            )

                            if 'error' not in seo_result:
                                db.save_seo_content(channel_id, rec_id, seo_result)
                                st.success("✅ SEO content generado y guardado")
                                st.rerun()
                            else:
                                st.error(f"Error: {seo_result['error']}")

            # ── Análisis completo sin truncar ────────────────────────────
            ui_section_divider()
            st.markdown("**🧠 Análisis Detallado:**")
            st.markdown(reasoning if reasoning else '—')

            with st.expander("Ver historial de recomendaciones"):
                history_df = recommendations_df[['recommendation_date', 'recommended_type', 'predicted_performance', 'created_at']].copy()
                history_df['created_at'] = pd.to_datetime(history_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')

                st.dataframe(
                    history_df.rename(columns={
                        'recommendation_date': 'Fecha Objetivo',
                        'recommended_type': 'Tipo',
                        'predicted_performance': 'Performance Esperado',
                        'created_at': 'Generada el'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

        # ── Sección de retroalimentación del ciclo de aprendizaje ────────────
        ui_section_divider()
        st.subheader("🔄 Ciclo de Retroalimentación")
        st.caption(
            "Seguimiento de si las recomendaciones anteriores fueron aplicadas "
            "y cómo rindieron los videos publicados después."
        )

        results_df = db.get_recommendation_results(channel_id, limit=10)

        if results_df.empty:
            st.info(
                "Aún no hay resultados de retroalimentación. "
                "Se generan automáticamente cuando ejecutas **main.py** "
                "y se detectan videos publicados después de una recomendación."
            )
        else:
            # ── KPIs de resumen ───────────────────────────────────────────
            resolved = results_df[results_df['performance_label'].notna()]
            if not resolved.empty:
                above = (resolved['performance_label'] == 'above_average').sum()
                avg   = (resolved['performance_label'] == 'average').sum()
                below = (resolved['performance_label'] == 'below_average').sum()
                followed = resolved['followed_recommendation'].sum() if 'followed_recommendation' in resolved else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("✅ Sobre promedio",  int(above))
                c2.metric("🟡 En el promedio",  int(avg))
                c3.metric("❌ Bajo promedio",   int(below))
                c4.metric("🎯 Siguieron la rec.", int(followed))

            # ── Gráfico de performance_ratio ─────────────────────────────
            chart_df = resolved[resolved['performance_ratio'].notna()].copy()
            if not chart_df.empty:
                chart_df['fecha'] = pd.to_datetime(chart_df['recommendation_date']).dt.strftime('%Y-%m-%d')
                chart_df['color'] = chart_df['performance_label'].map(PERF_LABEL_COLORS).fillna(COLORS["text_muted"])

                fig_ratio = px.bar(
                    chart_df,
                    x='fecha',
                    y='performance_ratio',
                    color='performance_label',
                    color_discrete_map=PERF_LABEL_COLORS,
                    labels={
                        'fecha': 'Recomendación',
                        'performance_ratio': 'Ratio vs Promedio del Canal',
                        'performance_label': 'Resultado',
                    },
                    title='Performance del video publicado vs promedio del canal (por recomendación)',
                )
                fig_ratio.add_hline(y=1.0, line_dash='dash', line_color='white',
                                    annotation_text='Promedio del canal')
                fig_ratio.update_layout(showlegend=True)
                st.plotly_chart(fig_ratio, use_container_width=True)

            # ── Tabla detallada ───────────────────────────────────────────
            with st.expander("Ver tabla de resultados"):
                display_cols = {
                    'recommendation_date': 'Fecha Rec.',
                    'recommended_type':    'Tipo Rec.',
                    'video_type':          'Tipo Publicado',
                    'followed_recommendation': 'Siguió Rec.',
                    'performance_ratio':   'Ratio',
                    'performance_label':   'Resultado',
                    'title':               'Video Publicado',
                }
                show_df = results_df[[c for c in display_cols if c in results_df.columns]].rename(columns=display_cols).copy()
                # Formatear booleano
                if 'Siguió Rec.' in show_df.columns:
                    show_df['Siguió Rec.'] = show_df['Siguió Rec.'].map(
                        {1: '✅', 0: '❌'}
                    ).fillna('⏳')
                if 'Ratio' in show_df.columns:
                    show_df['Ratio'] = show_df['Ratio'].apply(
                        lambda v: f"{v:.2f}x" if pd.notna(v) else '—'
                    )
                label_map = {
                    'above_average':  '✅ Sobre promedio',
                    'average':        '🟡 Promedio',
                    'below_average':  '❌ Bajo promedio',
                }
                if 'Resultado' in show_df.columns:
                    show_df['Resultado'] = show_df['Resultado'].map(label_map).fillna('⏳ Pendiente')

                st.dataframe(show_df, use_container_width=True, hide_index=True)

        db.close()

    except Exception as e:
        st.error(f"Error cargando recomendaciones: {str(e)}")


def show_generate_new_recommendation():
    """Interfaz para generar nueva recomendación"""
    ui_page_header("🤖", "Generar Nueva Recomendación", "Crea una recomendación personalizada con IA")
    
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not anthropic_api_key:
        st.error("ANTHROPIC_API_KEY no configurada. Por favor configura tu API key en el archivo .env")
        return
    
    try:
        db = YouTubeDatabase()
        videos_df = db.get_all_videos()
        
        if videos_df.empty:
            st.warning("No hay datos disponibles. Ejecuta primero main.py para extraer datos de YouTube.")
            db.close()
            return
        
        channel_ids = videos_df['channel_id'].unique()
        
        # Selector de canal
        channel_options = {}
        for channel_id in channel_ids:
            channel_name = videos_df[videos_df['channel_id'] == channel_id].iloc[0]['channel_title']
            channel_options[channel_name] = channel_id
        
        selected_channel_name = st.selectbox(
            "Selecciona el canal",
            options=list(channel_options.keys())
        )
        
        selected_channel_id = channel_options[selected_channel_name]
        
        if st.button("🚀 Generar Recomendación", type="primary"):
            with st.spinner("Analizando datos y generando recomendación con IA..."):
                # Filtrar videos del canal seleccionado
                channel_videos = videos_df[videos_df['channel_id'] == selected_channel_id]

                # Actualizar subscriber_count y video_count desde YouTube API
                channel_info = {
                    'channel_id': selected_channel_id,
                    'channel_name': selected_channel_name,
                    'subscriber_count': 0
                }
                youtube_api_key = os.getenv('YOUTUBE_API_KEY')
                if youtube_api_key:
                    try:
                        extractor = YouTubeDataExtractor(youtube_api_key)
                        fresh_info = extractor.get_channel_info(selected_channel_id)
                        if fresh_info:
                            db.save_channel_data(fresh_info)
                            channel_info['subscriber_count'] = fresh_info['subscriber_count']
                    except Exception:
                        pass  # Si falla la actualización, continúa con los datos locales

                # Vincular videos recientes + recuperar historial de retroalimentación
                channel_avg_views = float(channel_videos['view_count'].mean()) if not channel_videos.empty else 0.0
                db.link_video_to_recommendation(selected_channel_id, channel_avg_views)
                past_results_df = db.get_recommendation_results(selected_channel_id, limit=5)
                past_results = past_results_df.to_dict('records') if not past_results_df.empty else []

                # Generar recomendación
                analyzer = AIAnalyzer(anthropic_api_key)
                recommendation = analyzer.generate_daily_recommendation(
                    channel_videos, channel_info, past_results=past_results
                )

                if 'error' not in recommendation:
                    # Guardar en base de datos
                    db.save_recommendation(selected_channel_id, recommendation)
                    db.save_recommendation_result(
                        selected_channel_id,
                        recommendation['recommendation_date'],
                        recommendation['recommended_type'],
                    )
                    
                    st.success("✅ Recomendación generada exitosamente!")
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>🎯 Nueva Recomendación Generada</h3>
                        <p><strong>📅 Para el día:</strong> {_esc(recommendation['recommendation_date'])}</p>
                        <p><strong>🎬 Formato:</strong> {_esc(recommendation['recommended_type'])}</p>
                        <p><strong>📊 Performance Esperado:</strong> {_esc(recommendation['predicted_performance'])}</p>
                        <hr>
                        <p><strong>💡 Análisis Completo:</strong></p>
                        <p style="white-space: pre-wrap;">{_esc(recommendation['reasoning'])}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Mostrar sugerencias de título
                    title_suggestions = recommendation.get('title_suggestions', [])
                    if title_suggestions:
                        st.markdown("### ✍ Sugerencias de Título")
                        for i, sug in enumerate(title_suggestions, 1):
                            st.markdown(f"**{i}.** {sug.get('title', '')}")
                            analysis = sug.get('analysis', '')
                            if analysis:
                                st.caption(f"→ {analysis}")
                else:
                    st.error(f"Error: {recommendation['error']}")
        
        db.close()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")


def show_advanced_analytics(channel_id: str):
    """Página de Analytics Avanzados — datos de YouTube Analytics API (OAuth)."""
    ui_page_header("📊", "Analytics Avanzados", "Datos de YouTube Analytics API (OAuth)")

    if not channel_id:
        st.warning("Selecciona un canal en el panel lateral.")
        return

    try:
        db = YouTubeDatabase()
        analytics_df = db.get_video_analytics(channel_id)
        traffic_df   = db.get_traffic_sources(channel_id)
        db.close()
    except Exception as e:
        st.error(f"Error cargando analytics: {e}")
        return

    # ── Sin datos: mostrar instrucciones de configuración ────────────────
    if analytics_df.empty and traffic_df.empty:
        st.info(
            "**No hay datos de Analytics Avanzados para este canal.**\n\n"
            "Para habilitar esta sección:\n"
            "1. En [Google Cloud Console](https://console.cloud.google.com) "
            "habilita la **YouTube Analytics API**.\n"
            "2. Crea credenciales **OAuth 2.0 → Aplicación de escritorio**.\n"
            "3. Descarga el JSON y guárdalo como **`credentials.json`** en la "
            "raíz del proyecto.\n"
            "4. Ejecuta `python main.py` — el paso 4 abrirá el navegador para "
            "autorizar el acceso y guardará los datos en la base de datos."
        )
        return

    # ── Sección A: KPIs ──────────────────────────────────────────────────
    st.subheader("📌 Métricas Clave")
    col1, col2, col3, col4 = st.columns(4)

    if not analytics_df.empty:
        avg_retention = analytics_df['avg_view_percentage'].mean()
        total_shares  = int(analytics_df['shares'].sum())
        total_subs    = int(analytics_df['subscribers_gained'].sum())

        has_ctr = analytics_df['impression_ctr'].sum() > 0
        avg_ctr = analytics_df['impression_ctr'].mean() if has_ctr else None

        col1.metric("Retención media", f"{avg_retention:.1f}%")
        col2.metric("CTR promedio", f"{avg_ctr:.2f}%" if avg_ctr else "N/D",
                    help="Porcentaje de impresiones que se convirtieron en clic")
        col3.metric("Veces compartido", f"{total_shares:,}")
        col4.metric("Suscriptores ganados", f"{total_subs:,}")
    else:
        st.info("Sin datos de métricas por video.")

    ui_section_divider()

    # ── Sección B: Retención por video (top 20) ──────────────────────────
    if not analytics_df.empty:
        st.subheader("🔁 Retención media por video (Top 20)")

        top20 = analytics_df.nlargest(20, 'avg_view_percentage').copy()
        top20['title_short'] = top20['title'].str[:55]

        fig_ret = px.bar(
            top20.sort_values('avg_view_percentage'),
            x='avg_view_percentage',
            y='title_short',
            color='video_type',
            orientation='h',
            labels={
                'avg_view_percentage': '% Promedio visto',
                'title_short': 'Video',
                'video_type': 'Tipo',
            },
            color_discrete_map=VIDEO_TYPE_COLORS,
        )
        fig_ret.update_layout(height=520, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_ret, use_container_width=True)

        ui_section_divider()

        # ── Sección C: Scatter Vistas vs Retención ───────────────────────
        st.subheader("🔭 Alcance vs Retención")
        st.caption("Cuadrante superior derecho = videos con muchas vistas Y buena retención.")

        scatter_df = analytics_df.merge(
            analytics_df[['video_id']].assign(
                view_count=analytics_df.get('views', analytics_df.get('views', 0))
            ),
            on='video_id', how='left'
        ) if 'views' not in analytics_df.columns else analytics_df.copy()

        if 'views' in analytics_df.columns:
            fig_sc = px.scatter(
                analytics_df,
                x='views',
                y='avg_view_percentage',
                color='video_type',
                hover_name='title',
                labels={
                    'views': 'Vistas reales',
                    'avg_view_percentage': '% Retencion media',
                    'video_type': 'Tipo',
                },
                color_discrete_map=VIDEO_TYPE_COLORS,
                log_x=True,
            )
            fig_sc.update_layout(height=420)
            st.plotly_chart(fig_sc, use_container_width=True)

        ui_section_divider()

        # ── Sección D: CTR por video (si hay datos) ──────────────────────
        if analytics_df['impression_ctr'].sum() > 0:
            st.subheader("🖼 CTR de miniatura (Top 20 con más impresiones)")

            top_ctr = (
                analytics_df[analytics_df['impressions'] > 0]
                .nlargest(20, 'impressions')
                .copy()
            )
            top_ctr['title_short'] = top_ctr['title'].str[:55]

            fig_ctr = px.bar(
                top_ctr.sort_values('impression_ctr'),
                x='impression_ctr',
                y='title_short',
                color='video_type',
                orientation='h',
                labels={
                    'impression_ctr': 'CTR (%)',
                    'title_short': 'Video',
                    'video_type': 'Tipo',
                },
                color_discrete_map=VIDEO_TYPE_COLORS,
            )
            fig_ctr.update_layout(height=520, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_ctr, use_container_width=True)
            ui_section_divider()

    # ── Sección E: Fuentes de tráfico ────────────────────────────────────
    if not traffic_df.empty:
        st.subheader("🚦 Fuentes de tráfico")

        col_pie, col_table = st.columns([1, 1])

        with col_pie:
            fig_pie = px.pie(
                traffic_df,
                names='source_label',
                values='views',
                hole=0.4,
                color_discrete_sequence=CHART_COLORS,
            )
            fig_pie.update_layout(height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_table:
            total_views = traffic_df['views'].sum()
            display_df = traffic_df[['source_label', 'views', 'estimated_minutes']].copy()
            display_df['% del total'] = (display_df['views'] / total_views * 100).round(1)
            display_df.columns = ['Fuente', 'Vistas', 'Minutos vistos', '% Total']
            st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_trends_analysis():
    """Análisis de tendencias de temas vía Google Trends (pytrends)."""
    ui_page_header("📊", "Tendencias de Temas", "Compara popularidad de tus ideas en Google Trends")
    st.caption("Compara la popularidad de tus ideas de contenido en Google antes de publicar.")

    if not TrendsAnalyzer.is_available():
        st.error("❌ **pytrends no está instalado.**")
        st.code("pip install pytrends>=4.9.0")
        return

    # ── Configuración ────────────────────────────────────────────────
    st.subheader("🔍 Temas a comparar")
    st.info("Ingresa entre 1 y 5 temas para comparar su popularidad en Google Trends.")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        geo_label = st.selectbox("🌍 País / Región", list(GEO_OPTIONS.keys()), index=0)
    with col_cfg2:
        tf_label = st.selectbox("📅 Período", list(TIMEFRAME_OPTIONS.keys()), index=1)

    geo = GEO_OPTIONS[geo_label]
    timeframe = TIMEFRAME_OPTIONS[tf_label]

    # Hasta 5 keyword inputs dinámicos
    if 'trends_n_kw' not in st.session_state:
        st.session_state.trends_n_kw = 2

    kw_cols = st.columns(5)
    keywords = []
    for i in range(st.session_state.trends_n_kw):
        val = kw_cols[i % 5].text_input(
            f"Tema {i + 1}", key=f"trends_kw_{i}",
            placeholder="ej: cocina panameña",
        )
        keywords.append(val)

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
    with btn_col1:
        if st.button("➕ Agregar tema") and st.session_state.trends_n_kw < 5:
            st.session_state.trends_n_kw += 1
            st.rerun()
    with btn_col2:
        if st.button("➖ Quitar") and st.session_state.trends_n_kw > 1:
            st.session_state.trends_n_kw -= 1
            keywords = keywords[:-1]
            st.rerun()

    keywords = [k.strip() for k in keywords if k.strip()]
    if not keywords:
        st.warning("Ingresa al menos un tema para analizar.")
        return

    if st.button("🔍 Analizar tendencias", type="primary"):
        st.session_state.trends_result = None   # limpiar caché previa
        with st.spinner("Consultando Google Trends… (puede tardar unos segundos)"):
            try:
                analyzer = TrendsAnalyzer()
                iot = analyzer.get_interest_over_time(keywords, geo=geo, timeframe=timeframe)
                scores = analyzer.get_trend_scores(keywords, geo=geo, timeframe=timeframe)
                related = analyzer.get_related_queries(keywords[0], geo=geo) if len(keywords) >= 1 else {}
                peak_days = TrendsAnalyzer.peak_day(iot) if not iot.empty else {}
                st.session_state.trends_result = {
                    'iot': iot,
                    'scores': scores,
                    'related': related,
                    'peak_days': peak_days,
                    'keywords': keywords,
                    'geo_label': geo_label,
                    'tf_label': tf_label,
                }
            except Exception as e:
                err = str(e)
                if '429' in err or 'Too Many Requests' in err:
                    st.error("⚠️ **Google Trends bloqueó la solicitud (demasiadas consultas).** Espera unos minutos e intenta de nuevo.")
                else:
                    st.error(f"Error al consultar Google Trends: {err}")
                return

    result = st.session_state.get('trends_result')
    if result is None:
        return

    iot        = result['iot']
    scores     = result['scores']
    related    = result['related']
    peak_days  = result['peak_days']
    kws        = result['keywords']

    ui_section_divider()

    # ── Sección A: Score promedio actual ─────────────────────────────
    st.subheader("📊 Popularidad promedio en el período")
    st.caption(f"País: **{result['geo_label']}** · Período: **{result['tf_label']}**")

    score_cols = st.columns(len(kws))
    for i, kw in enumerate(kws):
        score = scores.get(kw, 0)
        color = "normal" if score >= 50 else "off"
        score_cols[i].metric(
            label=kw,
            value=f"{score}/100",
            delta="🔥 En tendencia" if score >= 60 else ("📉 Baja demanda" if score < 25 else None),
            delta_color=color,
        )

    # ── Sección B: Interés en el tiempo ──────────────────────────────
    if not iot.empty:
        ui_section_divider()
        st.subheader("📈 Interés a lo largo del tiempo")

        fig_line = px.line(
            iot.reset_index(),
            x='date',
            y=kws,
            labels={'date': 'Fecha', 'value': 'Interés (0–100)', 'variable': 'Tema'},
            color_discrete_sequence=CHART_COLORS,
        )
        fig_line.update_layout(
            height=380,
            margin={'l': 0, 'r': 0, 't': 20, 'b': 0},
            legend={'title': 'Tema'},
            hovermode='x unified',
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Día de la semana con más interés por tema
        if peak_days:
            st.caption("**Día de la semana con mayor interés histórico:**")
            day_cols = st.columns(len(kws))
            for i, kw in enumerate(kws):
                day_cols[i].info(f"**{kw}**\n\n📅 {peak_days.get(kw, '—')}")

    # ── Sección C: Comparación en barra ─────────────────────────────
    ui_section_divider()
    st.subheader("📊 Comparación directa de popularidad")

    scores_df = pd.DataFrame({
        'Tema': list(scores.keys()),
        'Score': list(scores.values()),
    }).sort_values('Score', ascending=True)

    fig_bar = go.Figure(go.Bar(
        x=scores_df['Score'],
        y=scores_df['Tema'],
        orientation='h',
        marker_color=[
            COLORS["success"] if s >= 60 else COLORS["warning"] if s >= 30 else COLORS["danger"]
            for s in scores_df['Score']
        ],
        text=scores_df['Score'].apply(lambda s: f"{s}/100"),
        textposition='outside',
    ))
    fig_bar.update_layout(
        xaxis={'title': 'Popularidad promedio (0=nula, 100=máx)', 'range': [0, 110]},
        height=max(200, 60 * len(kws)),
        margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Sección D: Consultas relacionadas ────────────────────────────
    ui_section_divider()
    st.subheader(f"🔗 Consultas relacionadas con «{kws[0]}»")
    st.caption("Fuente: Google Trends · Período: últimos 3 meses")

    tab_top, tab_rising = st.tabs(["🏆 Más buscadas", "🚀 En ascenso"])

    with tab_top:
        top_df = related.get('top', pd.DataFrame())
        if not top_df.empty and 'query' in top_df.columns:
            top_df = top_df[['query', 'value']].head(15).copy()
            top_df.columns = ['Consulta', 'Popularidad']
            st.dataframe(top_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hay datos de consultas top disponibles.")

    with tab_rising:
        rising_df = related.get('rising', pd.DataFrame())
        if not rising_df.empty and 'query' in rising_df.columns:
            rising_df = rising_df[['query', 'value']].head(15).copy()
            rising_df.columns = ['Consulta', 'Crecimiento']
            rising_df['Crecimiento'] = rising_df['Crecimiento'].apply(
                lambda v: f"+{v}%" if isinstance(v, (int, float)) else str(v)
            )
            st.dataframe(rising_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hay datos de consultas en ascenso disponibles.")

    # ── Sección E: Recomendación de contenido ─────────────────────────
    if scores:
        best_kw = max(scores, key=lambda k: scores[k])
        best_score = scores[best_kw]
        ui_section_divider()
        st.subheader("💡 Recomendación")
        if best_score >= 60:
            st.success(
                f"🚀 **«{best_kw}»** es el tema con mayor demanda actual ({best_score}/100). "
                f"Considera publicar sobre este tema pronto para aprovechar la tendencia."
            )
        elif best_score >= 30:
            st.warning(
                f"📊 **«{best_kw}»** tiene demanda moderada ({best_score}/100). "
                f"El tema funciona, pero no está en su punto máximo."
            )
        else:
            st.error(
                f"⚠️ Ninguno de los temas tiene alta demanda en {result['geo_label']} ahora mismo. "
                f"El más popular es **«{best_kw}»** ({best_score}/100). "
                f"Considera ampliar o cambiar los temas."
            )


def _word_freq(titles: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """Frecuencia de palabras en títulos, filtrando stopwords en español."""
    stopwords = {
        'de', 'la', 'el', 'en', 'los', 'las', 'un', 'una', 'y', 'que',
        'con', 'del', 'por', 'para', 'se', 'es', 'mi', 'tu', 'su', 'al',
        'le', 'no', 'si', 'me', 'ya', 'o', 'a', 'e', 'i', 'lo', 'te',
        'nos', 'hay', 'muy', 'mas', 'pero', 'como', 'este', 'esta',
    }
    words = []
    for title in titles.dropna():
        for w in re.findall(r'[a-záéíóúüñ]+', str(title).lower()):
            if w not in stopwords and len(w) > 2:
                words.append(w)
    top = Counter(words).most_common(top_n)
    return pd.DataFrame(top, columns=['Palabra', 'Frecuencia'])


def _tag_freq(tags: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """Frecuencia de tags (separados por coma)."""
    all_tags = []
    for t in tags.dropna():
        for tag in str(t).split(','):
            tag = tag.strip().lower()
            if tag:
                all_tags.append(tag)
    top = Counter(all_tags).most_common(top_n)
    return pd.DataFrame(top, columns=['Tag', 'Frecuencia'])


def show_content_analysis(df: pd.DataFrame):
    """Análisis de palabras en títulos, tags más usados y longitud vs vistas."""
    ui_page_header("🔤", "Análisis de Contenido", "Palabras, tags y patrones de rendimiento en títulos")

    if df.empty:
        st.warning("No hay datos disponibles.")
        return

    tab_words, tab_tags, tab_length, tab_categories = st.tabs([
        "📝 Palabras en Títulos",
        "🏷️ Tags más usados",
        "📏 Longitud vs Vistas",
        "📂 Categorías",
    ])

    # ── Palabras en títulos ───────────────────────────────────────────
    with tab_words:
        st.caption("Palabras más frecuentes en títulos, separadas por rendimiento.")
        median_views = df['view_count'].median()
        df_high = df[df['view_count'] >= median_views]
        df_low  = df[df['view_count'] <  median_views]

        col_h, col_l = st.columns(2)

        with col_h:
            st.subheader("🔝 Alta performance")
            freq_h = _word_freq(df_high['title'])
            if not freq_h.empty:
                fig = px.bar(
                    freq_h, x='Frecuencia', y='Palabra', orientation='h',
                    color_discrete_sequence=[COLORS["success"]],
                )
                fig.update_layout(
                    height=420, margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
                    yaxis={'categoryorder': 'total ascending'},
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_l:
            st.subheader("📉 Baja performance")
            freq_l = _word_freq(df_low['title'])
            if not freq_l.empty:
                fig = px.bar(
                    freq_l, x='Frecuencia', y='Palabra', orientation='h',
                    color_discrete_sequence=[COLORS["danger"]],
                )
                fig.update_layout(
                    height=420, margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
                    yaxis={'categoryorder': 'total ascending'},
                )
                st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Umbral de performance: **{int(median_views):,} vistas** (mediana del canal).")

    # ── Tags más usados ───────────────────────────────────────────────
    with tab_tags:
        freq_t = _tag_freq(df['tags'])
        if freq_t.empty:
            st.info("No hay tags disponibles en los videos del canal.")
        else:
            fig = px.bar(
                freq_t, x='Frecuencia', y='Tag', orientation='h',
                color_discrete_sequence=[COLORS["primary"]],
            )
            fig.update_layout(
                height=520, margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
                yaxis={'categoryorder': 'total ascending'},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Total de tags únicos: **{len(_tag_freq(df['tags'], top_n=9999))}**")

    # ── Longitud del título vs Vistas ─────────────────────────────────
    with tab_length:
        df_len = df.copy()
        df_len['len_titulo'] = df_len['title'].apply(
            lambda t: len(str(t)) if pd.notna(t) else 0
        )
        df_len['rango_longitud'] = pd.cut(
            df_len['len_titulo'],
            bins=[0, 30, 50, 70, 100, 200],
            labels=['< 30', '30–50', '50–70', '70–100', '> 100'],
        )

        # Promedios por rango
        bucket = (
            df_len.groupby('rango_longitud', observed=True)['view_count']
            .mean().reset_index()
        )
        bucket.columns = ['Longitud', 'Vistas promedio']

        col_b, col_s = st.columns(2)
        with col_b:
            st.subheader("Vistas promedio por rango de longitud")
            fig_b = px.bar(
                bucket, x='Longitud', y='Vistas promedio',
                color_discrete_sequence=[COLORS["secondary"]],
                text='Vistas promedio',
            )
            fig_b.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_b.update_layout(height=320, margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
            st.plotly_chart(fig_b, use_container_width=True)

        with col_s:
            st.subheader("Longitud del título vs Vistas (scatter)")
            fig_s = px.scatter(
                df_len, x='len_titulo', y='view_count', color='video_type',
                hover_data=['title'],
                labels={'len_titulo': 'Longitud del título (chars)', 'view_count': 'Vistas'},
                color_discrete_sequence=VIDEO_TYPE_SEQUENCE,
            )
            fig_s.update_layout(height=320, margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
            st.plotly_chart(fig_s, use_container_width=True)

    # ── Categorías de contenido (Mejora 12.2) ─────────────────────────
    with tab_categories:
        st.caption("Clasificación automática de videos por tipo de contenido y análisis de rendimiento por categoría.")

        # Verificar si hay channel_id disponible
        channel_ids = df['channel_id'].unique()
        if len(channel_ids) == 0:
            st.warning("No hay canal seleccionado.")
        else:
            ch_id = channel_ids[0]
            try:
                db_cat = YouTubeDatabase()

                # Contar clasificados vs no clasificados
                categorized = df[df['content_category'].notna() & (df['content_category'] != '')].copy() if 'content_category' in df.columns else pd.DataFrame()
                uncategorized = db_cat.get_videos_without_category(ch_id)

                col_info1, col_info2 = st.columns(2)
                col_info1.metric("✅ Clasificados", len(categorized))
                col_info2.metric("❓ Sin clasificar", len(uncategorized))

                # Botón para clasificar
                anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')
                if not uncategorized.empty and anthropic_key:
                    if st.button("🤖 Clasificar videos con IA", type="secondary", key="btn_classify"):
                        with st.spinner(f"Clasificando {len(uncategorized)} videos con Claude AI..."):
                            classifier = ContentClassifier(anthropic_key)
                            videos_list = uncategorized.to_dict('records')

                            all_categories: dict[str, str] = {}
                            # Clasificar en lotes de 25
                            for i in range(0, len(videos_list), 25):
                                batch = videos_list[i:i+25]
                                batch_result = classifier.classify_batch(batch)
                                all_categories.update(batch_result)

                            if all_categories:
                                db_cat.save_content_categories(all_categories)
                                st.success(f"✅ {len(all_categories)} videos clasificados")
                                st.rerun()

                # Mostrar análisis por categoría
                perf_df = db_cat.get_performance_by_category(ch_id)

                if not perf_df.empty:
                    ui_section_divider()
                    st.subheader("📊 Rendimiento por Categoría")

                    # Traducir categorías
                    perf_df['categoria'] = perf_df['content_category'].apply(
                        lambda c: f"{ContentClassifier.get_category_icon(c)} {ContentClassifier.get_category_label(c)}"
                    )

                    # Bar chart: vistas promedio
                    fig_views = px.bar(
                        perf_df.sort_values('avg_views', ascending=False),
                        x='categoria', y='avg_views',
                        labels={'categoria': 'Categoría', 'avg_views': 'Vistas Promedio'},
                        color_discrete_sequence=[COLORS["primary"]],
                    )
                    fig_views.update_layout(height=350)
                    st.plotly_chart(fig_views, use_container_width=True)

                    # Bar chart: engagement promedio
                    fig_eng = px.bar(
                        perf_df.sort_values('avg_engagement', ascending=False),
                        x='categoria', y='avg_engagement',
                        labels={'categoria': 'Categoría', 'avg_engagement': 'Engagement Promedio (%)'},
                        color_discrete_sequence=[COLORS["success"]],
                    )
                    fig_eng.update_layout(height=350)
                    st.plotly_chart(fig_eng, use_container_width=True)

                    # Pie chart: distribución
                    fig_pie = px.pie(
                        perf_df, values='video_count', names='categoria',
                        color_discrete_sequence=CHART_COLORS,
                    )
                    fig_pie.update_layout(height=350)
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Insight textual
                    if len(perf_df) >= 2:
                        best = perf_df.sort_values('avg_views', ascending=False).iloc[0]
                        worst = perf_df.sort_values('avg_views').iloc[0]
                        if best['avg_views'] > 0 and worst['avg_views'] > 0:
                            ratio = best['avg_views'] / worst['avg_views']
                            best_label = ContentClassifier.get_category_label(best['content_category'])
                            worst_label = ContentClassifier.get_category_label(worst['content_category'])
                            st.info(
                                f"💡 Tus **{best_label}** obtienen **{ratio:.1f}x** más vistas "
                                f"que tus **{worst_label}** en promedio."
                            )

                elif not categorized.empty:
                    st.info("Recarga los datos del dashboard para ver las categorías recién clasificadas.")

                db_cat.close()
            except Exception as e:
                st.error(f"Error al analizar categorías: {e}")


def show_channel_comparison(df_all: pd.DataFrame):
    """Vista side-by-side de todos los canales con benchmarking de métricas."""
    ui_page_header("🆚", "Comparar Canales", "Benchmarking side-by-side de todos tus canales")

    if df_all.empty:
        st.warning("No hay datos disponibles.")
        return

    channels = df_all['channel_id'].unique()
    if len(channels) < 2:
        st.info(
            "Se necesitan **al menos 2 canales** para comparar.\n\n"
            "Agrega más canales en tu `.env` (`CHANNEL_IDS`) y ejecuta `python main.py`."
        )
        return

    # ── KPIs por canal ────────────────────────────────────────────────
    stats = (
        df_all
        .groupby('channel_title', dropna=False)
        .agg(
            videos         = ('video_id',       'count'),
            total_vistas   = ('view_count',      'sum'),
            avg_vistas     = ('view_count',      'mean'),
            avg_engagement = ('engagement_rate', 'mean'),
            shorts         = ('is_short',        'sum'),
        )
        .reset_index()
    )
    stats['channel_title'] = stats['channel_title'].fillna('Canal sin nombre')

    st.subheader("📊 Resumen por canal")
    kpi_cols = st.columns(len(stats))
    for i, row in stats.iterrows():
        with kpi_cols[i]:
            st.markdown(f"### 📺 {row['channel_title']}")
            st.metric("Videos",          f"{int(row['videos']):,}")
            st.metric("Vistas totales",  f"{int(row['total_vistas']):,}")
            st.metric("Vistas promedio", f"{int(row['avg_vistas']):,}")
            st.metric("Engagement",      f"{row['avg_engagement']:.2f}%")
            st.metric("Shorts",          f"{int(row['shorts'])}")

    # ── Gráficos comparativos ────────────────────────────────────────
    ui_section_divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vistas promedio por canal")
        fig1 = px.bar(
            stats, x='channel_title', y='avg_vistas',
            color='channel_title',
            labels={'channel_title': 'Canal', 'avg_vistas': 'Vistas promedio'},
            text='avg_vistas',
        )
        fig1.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig1.update_layout(showlegend=False, height=320,
                           margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Engagement promedio por canal")
        fig2 = px.bar(
            stats, x='channel_title', y='avg_engagement',
            color='channel_title',
            labels={'channel_title': 'Canal', 'avg_engagement': 'Engagement %'},
            text='avg_engagement',
        )
        fig2.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig2.update_layout(showlegend=False, height=320,
                           margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
        st.plotly_chart(fig2, use_container_width=True)

    # ── Distribución de vistas por canal ─────────────────────────────
    ui_section_divider()
    st.subheader("Distribución de vistas por canal")
    fig_box = px.box(
        df_all, x='channel_title', y='view_count', color='channel_title',
        labels={'channel_title': 'Canal', 'view_count': 'Vistas'},
    )
    fig_box.update_layout(showlegend=False, height=360,
                          margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Tabla resumen ─────────────────────────────────────────────────
    ui_section_divider()
    st.subheader("Tabla resumen")
    disp = stats.rename(columns={
        'channel_title':  'Canal',
        'videos':         'Videos',
        'total_vistas':   'Vistas totales',
        'avg_vistas':     'Vistas promedio',
        'avg_engagement': 'Engagement %',
        'shorts':         'Shorts',
    })
    disp['Vistas totales']  = disp['Vistas totales'].apply(lambda v: f"{int(v):,}")
    disp['Vistas promedio'] = disp['Vistas promedio'].apply(lambda v: f"{int(v):,}")
    disp['Engagement %']    = disp['Engagement %'].apply(lambda v: f"{v:.2f}%")
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Página: Plan Semanal (Mejora 4.2)
# ─────────────────────────────────────────────────────────────────────────────

def show_weekly_plan(df: pd.DataFrame, channel_id: str | None):
    """Página de planificación semanal generada por Claude."""
    ui_page_header("🗓", "Plan Semanal", "Planificación de contenido generada por Claude AI")

    if not channel_id or df.empty:
        st.warning("Selecciona un canal con datos en el panel lateral.")
        return

    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key:
        st.error("ANTHROPIC_API_KEY no configurada en el archivo .env.")
        return

    try:
        db = YouTubeDatabase()

        # ── Mostrar plan guardado más reciente ────────────────────────────
        plans_df = db.get_weekly_plans(channel_id, limit=4)

        if not plans_df.empty:
            latest = plans_df.iloc[0]
            try:
                days_data = json.loads(latest['plan_json'])
            except (json.JSONDecodeError, TypeError):
                days_data = []

            st.markdown(f"**Semana del {latest['week_start_date']}**  •  "
                        f"Generado el {str(latest['generated_at'])[:16]}")

            if latest.get('strategy'):
                st.info(f"**Estrategia:** {latest['strategy']}")

            # ── Calendario visual 7 columnas ─────────────────────────────
            _render_weekly_calendar(days_data)

            # ── Historial de planes anteriores ───────────────────────────
            if len(plans_df) > 1:
                with st.expander("📂 Ver planes anteriores"):
                    for _, row in plans_df.iloc[1:].iterrows():
                        st.markdown(f"**Semana del {row['week_start_date']}**")
                        try:
                            old_days = json.loads(row['plan_json'])
                            _render_weekly_calendar(old_days, compact=True)
                        except (json.JSONDecodeError, TypeError):
                            st.caption("(datos no disponibles)")
                        ui_section_divider()

            # ── Exportar plan actual ──────────────────────────────────────
            ui_section_divider()
            st.subheader("Exportar plan")
            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                if st.button("📅 Exportar a Google Calendar", key="btn_gcal"):
                    try:
                        with st.spinner("Exportando a Google Calendar..."):
                            gcal = GoogleCalendarExporter()
                            gcal_result = gcal.export_weekly_plan(
                                days_data,
                                strategy=latest.get('strategy', ''),
                            )
                        if gcal_result['created'] > 0:
                            st.success(f"✅ {gcal_result['created']} evento(s) creado(s) en Google Calendar")
                            for link in gcal_result['event_links']:
                                if link:
                                    st.markdown(f"[Abrir evento]({link})")
                        if gcal_result['skipped'] > 0:
                            st.info(f"ℹ️ {gcal_result['skipped']} evento(s) ya existían (omitidos)")
                        if gcal_result['errors']:
                            for err in gcal_result['errors']:
                                st.warning(f"⚠️ {err}")
                        if gcal_result['created'] == 0 and gcal_result['skipped'] == 0 and not gcal_result['errors']:
                            st.info("No hay días de publicación en este plan.")
                    except FileNotFoundError as e:
                        st.error(f"❌ {e}")
                    except Exception as e:
                        st.error(f"❌ Error al exportar a Calendar: {e}")

            with exp_col2:
                if st.button("📊 Exportar a Google Sheets", key="btn_gsheets_plan"):
                    try:
                        with st.spinner("Creando Google Sheet..."):
                            gsheets = GoogleSheetsExporter()
                            ch_name = (
                                df.iloc[0]['channel_title']
                                if not df.empty and 'channel_title' in df.columns
                                else channel_id
                            )
                            gsheets_result = gsheets.export_weekly_plan(
                                days_data,
                                strategy=latest.get('strategy', ''),
                                week_start_date=str(latest['week_start_date']),
                                channel_name=ch_name,
                            )
                        st.success("✅ Google Sheet creado exitosamente")
                        st.markdown(f"[📊 Abrir Google Sheet]({gsheets_result['spreadsheet_url']})")
                    except FileNotFoundError as e:
                        st.error(f"❌ {e}")
                    except Exception as e:
                        st.error(f"❌ Error al exportar a Sheets: {e}")

        else:
            st.info("No hay planes semanales guardados. Genera el primero con el botón de abajo.")

        # ── Botón de generación ───────────────────────────────────────────
        ui_section_divider()
        st.subheader("Generar nuevo plan")

        if st.button("🗓 Generar plan para los próximos 7 días", type="primary"):
            with st.spinner("Claude está analizando el canal y diseñando el plan semanal..."):
                past_results_df = db.get_recommendation_results(channel_id, limit=5)
                past_results    = past_results_df.to_dict('records') if not past_results_df.empty else []

                channel_info = {
                    'channel_id':       channel_id,
                    'channel_name':     df.iloc[0]['channel_title'] if 'channel_title' in df.columns else channel_id,
                    'subscriber_count': 0,
                }

                analyzer = AIAnalyzer(anthropic_api_key)
                plan     = analyzer.generate_weekly_plan(df, channel_info, past_results=past_results)

                if 'error' in plan:
                    st.error(plan['error'])
                else:
                    db.save_weekly_plan(
                        channel_id,
                        plan['week_start_date'],
                        json.dumps(plan['days'], ensure_ascii=False),
                        plan.get('strategy', ''),
                        plan['generated_at'],
                    )
                    st.success("✅ Plan semanal guardado")
                    st.rerun()

        db.close()

    except Exception as e:
        st.error(f"Error en plan semanal: {e}")


def _render_weekly_calendar(days_data: list, compact: bool = False):
    """Renderiza el calendario semanal en filas de 3 tarjetas con texto legible."""
    if not days_data:
        st.caption("(sin datos de días)")
        return

    COLS_PER_ROW = 3
    days_list = days_data[:7]

    for row_start in range(0, len(days_list), COLS_PER_ROW):
        row_days = days_list[row_start:row_start + COLS_PER_ROW]
        cols = st.columns(COLS_PER_ROW)
        for i, day in enumerate(row_days):
            col = cols[i]
            publish = day.get('publish', False)
            vtype   = day.get('type') or ''
            topic   = day.get('topic', '')
            hour    = day.get('hour')
            reason  = day.get('reason', '')

            if publish:
                css_class = 'publish-long' if vtype == 'Video Largo' else 'publish-short'
                badge = '🎬' if vtype == 'Video Largo' else '📱'
                label = vtype
            else:
                css_class = 'rest'
                badge, label = '💤', 'Descanso'

            hour_str = f"{hour}:00" if hour is not None else ''

            if compact:
                col.markdown(
                    f"<div class='calendar-card {css_class}' style='min-height:70px;font-size:13px'>"
                    f"<b>{_esc(day.get('day',''))}</b><br>{_esc(badge)} {_esc(label)}"
                    f"{'<br>' + _esc(hour_str) if hour_str else ''}"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                topic_html = (
                    f"<p style='font-size:15px;margin:8px 0 6px;color:var(--text-secondary);line-height:1.45'>"
                    f"{_esc(topic[:100])}{'…' if len(topic)>100 else ''}</p>"
                    if topic else ""
                )
                reason_html = (
                    f"<p style='font-size:13.5px;color:var(--text-muted);margin:6px 0 0;line-height:1.4'>"
                    f"{_esc(reason[:140])}{'…' if len(reason)>140 else ''}</p>"
                    if reason else ""
                )
                col.markdown(
                    f"<div class='calendar-card {css_class}'>"
                    f"<b style='font-size:18px'>{_esc(day.get('day',''))}</b>"
                    f"<p style='font-size:14px;color:var(--text-muted);margin:4px 0'>{_esc(day.get('date',''))}</p>"
                    f"<p style='font-size:30px;margin:10px 0'>{_esc(badge)}</p>"
                    f"<p style='font-size:17px;font-weight:bold;margin:2px 0'>{_esc(label)}</p>"
                    f"{'<p style=\"font-size:15px;color:var(--text-secondary);margin:5px 0\">⏰ ' + _esc(hour_str) + '</p>' if hour_str else ''}"
                    f"{topic_html}"
                    f"{reason_html}"
                    "</div>",
                    unsafe_allow_html=True,
                )

    if not compact:
        st.markdown("")
        rows = []
        for day in days_data[:7]:
            rows.append({
                'Día':     day.get('day', ''),
                'Fecha':   day.get('date', ''),
                'Publica': '✅' if day.get('publish') else '—',
                'Formato': day.get('type') or '—',
                'Tema':    day.get('topic', '') or '—',
                'Hora':    f"{day['hour']}:00" if day.get('hour') is not None else '—',
                'Razón':   day.get('reason', '') or '—',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# 🏥 Salud del Canal
# ═══════════════════════════════════════════════════════════════════════════

def _compute_health_metrics(df: pd.DataFrame, db, channel_id: str) -> list[dict]:
    """Calcula las métricas de salud del canal y retorna lista de dicts."""
    metrics: list[dict] = []
    # Usar timezone-aware si los datos lo son, sino naive
    _pub = pd.to_datetime(df['published_at'], errors='coerce') if 'published_at' in df.columns else pd.Series(dtype='datetime64[ns]')
    if _pub.dt.tz is not None:
        now = pd.Timestamp.now(tz='UTC')
    else:
        now = pd.Timestamp.now()

    # ── 1. Frecuencia de publicación (videos/semana últimos 30 días) ───
    if not df.empty and 'published_at' in df.columns:
        recent = df[_pub >= (now - pd.Timedelta(days=30))]
        weeks = max((now - _pub.min()).days / 7, 1)
        vids_per_week = round(len(recent) / min(weeks, 4.3), 1)
        if vids_per_week >= 3:
            status = 'green'
        elif vids_per_week >= 1:
            status = 'yellow'
        else:
            status = 'red'
        metrics.append({
            'name': 'Frecuencia de publicación',
            'icon': '📅',
            'value': f'{vids_per_week} vid/sem',
            'status': status,
            'detail': f'{len(recent)} videos en últimos 30 días',
            'raw': vids_per_week,
        })

    # ── 2. Frescura del canal (días desde último video) ────────────────
    if not df.empty and 'published_at' in df.columns:
        last_pub = _pub.max()
        days_ago = (now - last_pub).days if pd.notna(last_pub) else 999
        if days_ago <= 3:
            status = 'green'
        elif days_ago <= 7:
            status = 'yellow'
        else:
            status = 'red'
        metrics.append({
            'name': 'Frescura del canal',
            'icon': '🕐',
            'value': f'Hace {days_ago} días',
            'status': status,
            'detail': f'Último video: {last_pub.strftime("%Y-%m-%d")}',
            'raw': days_ago,
        })

    # ── 3. Tendencia de engagement (últimos 10 vs anteriores 10) ───────
    if not df.empty and 'engagement_rate' in df.columns:
        sorted_df = df.sort_values('published_at', ascending=False)
        eng_col = pd.to_numeric(sorted_df['engagement_rate'], errors='coerce')
        if len(eng_col.dropna()) >= 10:
            recent_eng = eng_col.iloc[:10].mean()
            older_eng = eng_col.iloc[10:20].mean() if len(eng_col) >= 20 else eng_col.iloc[10:].mean()
            if older_eng and older_eng > 0:
                change_pct = round(((recent_eng - older_eng) / older_eng) * 100, 1)
            else:
                change_pct = 0.0
            if change_pct >= 0:
                status = 'green'
            elif change_pct >= -10:
                status = 'yellow'
            else:
                status = 'red'
            sign = '+' if change_pct >= 0 else ''
            metrics.append({
                'name': 'Tendencia de engagement',
                'icon': '📈',
                'value': f'{sign}{change_pct}%',
                'status': status,
                'detail': f'Recientes: {recent_eng:.2f}% vs anteriores: {older_eng:.2f}%',
                'raw': change_pct,
            })

    # ── 4. Vistas promedio recientes vs general ────────────────────────
    if not df.empty and 'view_count' in df.columns:
        views = pd.to_numeric(df['view_count'], errors='coerce').dropna()
        sorted_df = df.sort_values('published_at', ascending=False)
        recent_views = pd.to_numeric(sorted_df['view_count'].iloc[:10], errors='coerce').dropna()
        avg_all = views.mean()
        avg_recent = recent_views.mean()
        if avg_all > 0:
            ratio = avg_recent / avg_all
            if ratio >= 1.0:
                status = 'green'
            elif ratio >= 0.7:
                status = 'yellow'
            else:
                status = 'red'
            metrics.append({
                'name': 'Vistas promedio recientes',
                'icon': '👁',
                'value': f'{int(avg_recent):,}',
                'status': status,
                'detail': f'{ratio:.0%} del promedio general ({int(avg_all):,})',
                'raw': ratio,
            })

    # ── 5. Ratio de viralidad (% videos sobre la mediana) ─────────────
    if not df.empty and 'view_count' in df.columns:
        views = pd.to_numeric(df['view_count'], errors='coerce').dropna()
        if len(views) >= 5:
            median_views = views.median()
            above_median = (views > median_views).sum()
            ratio_pct = round((above_median / len(views)) * 100, 1)
            # Por definición la mediana da ~50%, así que ajustamos thresholds
            # Miramos los últimos 10 videos vs mediana global
            recent_views = pd.to_numeric(
                df.sort_values('published_at', ascending=False)['view_count'].iloc[:10],
                errors='coerce').dropna()
            above_recent = (recent_views > median_views).sum()
            recent_pct = round((above_recent / max(len(recent_views), 1)) * 100, 1)
            if recent_pct >= 50:
                status = 'green'
            elif recent_pct >= 30:
                status = 'yellow'
            else:
                status = 'red'
            metrics.append({
                'name': 'Ratio de viralidad',
                'icon': '🔥',
                'value': f'{recent_pct}%',
                'status': status,
                'detail': f'{above_recent}/{len(recent_views)} videos recientes sobre mediana ({int(median_views):,} vistas)',
                'raw': recent_pct,
            })

    # ── 6. Consistencia de engagement ─────────────────────────────────
    if not df.empty and 'engagement_rate' in df.columns:
        eng = pd.to_numeric(df['engagement_rate'], errors='coerce').dropna()
        if len(eng) >= 5 and eng.mean() > 0:
            cv = eng.std() / eng.mean()  # Coeficiente de variación
            if cv < 0.5:
                status = 'green'
            elif cv < 1.0:
                status = 'yellow'
            else:
                status = 'red'
            metrics.append({
                'name': 'Consistencia de engagement',
                'icon': '🎯',
                'value': f'CV {cv:.2f}',
                'status': status,
                'detail': 'Baja variación = más predecible' if cv < 0.5 else 'Alta variación entre videos',
                'raw': cv,
            })

    # ── 7-8. Bonus: Retención y CTR si hay datos de Analytics API ─────
    try:
        analytics_df = db.get_video_analytics(channel_id)
        if not analytics_df.empty:
            # Retención promedio
            avg_ret = pd.to_numeric(analytics_df['avg_view_percentage'], errors='coerce').mean()
            if pd.notna(avg_ret):
                if avg_ret >= 50:
                    status = 'green'
                elif avg_ret >= 35:
                    status = 'yellow'
                else:
                    status = 'red'
                metrics.append({
                    'name': 'Retención promedio',
                    'icon': '⏱',
                    'value': f'{avg_ret:.1f}%',
                    'status': status,
                    'detail': 'Benchmark: 50%+ es excelente',
                    'raw': avg_ret,
                })
            # CTR de miniaturas
            ctr_vals = pd.to_numeric(analytics_df['impression_ctr'], errors='coerce').dropna()
            if len(ctr_vals) > 0:
                avg_ctr = ctr_vals.mean()
                if avg_ctr >= 5:
                    status = 'green'
                elif avg_ctr >= 3:
                    status = 'yellow'
                else:
                    status = 'red'
                metrics.append({
                    'name': 'CTR de miniaturas',
                    'icon': '🖼',
                    'value': f'{avg_ctr:.1f}%',
                    'status': status,
                    'detail': 'Benchmark: 5%+ es bueno',
                    'raw': avg_ctr,
                })
    except Exception:
        pass  # Analytics API no disponible — no pasa nada

    return metrics


def _health_score(metrics: list[dict]) -> int:
    """Calcula score 0-100 promediando los estados de las métricas."""
    if not metrics:
        return 0
    score_map = {'green': 100, 'yellow': 50, 'red': 0}
    total = sum(score_map.get(m['status'], 50) for m in metrics)
    return round(total / len(metrics))


def _sparkline_svg(values: list[float], color: str = '#6366F1',
                   width: int = 80, height: int = 24) -> str:
    """Genera un mini SVG sparkline inline."""
    if not values or len(values) < 2:
        return ''
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    points = []
    step = width / (len(values) - 1)
    for i, v in enumerate(values):
        x = round(i * step, 1)
        y = round(height - ((v - mn) / rng) * (height - 2) - 1, 1)
        points.append(f'{x},{y}')
    polyline = ' '.join(points)
    return (
        f'<svg width="{width}" height="{height}" style="vertical-align:middle;">'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


def show_channel_health(df: pd.DataFrame, channel_id: str):
    """Página Salud del Canal — semáforo de métricas + diagnóstico IA."""
    ui_page_header("🏥", "Salud del Canal", "Diagnóstico rápido del estado de tu canal")

    if df.empty or not channel_id:
        st.warning("No hay datos suficientes para evaluar la salud del canal.")
        return

    try:
        db = YouTubeDatabase()
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return

    # ── Calcular métricas ──────────────────────────────────────────────
    metrics = _compute_health_metrics(df, db, channel_id)
    if not metrics:
        st.warning("No se pudieron calcular métricas de salud.")
        db.close()
        return

    score = _health_score(metrics)

    # ── Score General ──────────────────────────────────────────────────
    if score >= 70:
        score_color = COLORS['success']
        score_label = 'Saludable'
    elif score >= 40:
        score_color = COLORS['warning']
        score_label = 'Necesita atención'
    else:
        score_color = COLORS['danger']
        score_label = 'Crítico'

    st.markdown(f"""
    <div style="background: {COLORS['bg_secondary']}; border-radius: 12px;
                padding: 1.5rem; margin-bottom: 1.5rem;
                border-left: 4px solid {score_color};">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div style="font-size: 0.8rem; color: {COLORS['text_muted']};
                            text-transform: uppercase; letter-spacing: 0.1em;">
                    Score General</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {score_color};
                            line-height: 1.1;">{score}<span style="font-size: 1rem; opacity: 0.6;">/100</span></div>
                <div style="font-size: 0.9rem; color: {COLORS['text_secondary']};">{score_label}</div>
            </div>
            <div style="flex: 1; min-width: 200px; max-width: 400px;">
                <div style="background: {COLORS['bg_tertiary']}; border-radius: 8px;
                            height: 16px; overflow: hidden;">
                    <div style="width: {score}%; height: 100%; border-radius: 8px;
                                background: linear-gradient(90deg, {score_color}, {score_color}dd);
                                transition: width 0.5s;"></div>
                </div>
                <div style="display: flex; justify-content: space-between;
                            font-size: 0.7rem; color: {COLORS['text_muted']}; margin-top: 4px;">
                    <span>🔴 Crítico</span><span>🟡 Atención</span><span>🟢 Saludable</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tarjetas de métricas ──────────────────────────────────────────
    status_colors = {
        'green': COLORS['success'],
        'yellow': COLORS['warning'],
        'red': COLORS['danger'],
    }
    status_emojis = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}

    # Obtener sparklines de engagement histórico
    try:
        hist_df = db.get_channel_metrics_history(channel_id)
        eng_history = hist_df['avg_engagement'].tolist() if not hist_df.empty and 'avg_engagement' in hist_df.columns else []
        views_history = hist_df['total_views'].tolist() if not hist_df.empty and 'total_views' in hist_df.columns else []
    except Exception:
        eng_history = []
        views_history = []

    # Sparkline para cada métrica según su tipo
    sparkline_map = {
        'Tendencia de engagement': eng_history[-14:] if eng_history else [],
        'Vistas promedio recientes': views_history[-14:] if views_history else [],
    }

    # Renderizar en grilla 3 columnas
    cols_per_row = 3
    for row_start in range(0, len(metrics), cols_per_row):
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx >= len(metrics):
                break
            m = metrics[idx]
            sc = status_colors.get(m['status'], COLORS['info'])
            emoji = status_emojis.get(m['status'], '⚪')
            spark_data = sparkline_map.get(m['name'], [])
            spark_html = _sparkline_svg(spark_data, sc) if spark_data else ''

            card_html = (
                f'<div style="background:{COLORS["bg_secondary"]};border-radius:10px;'
                f'padding:1rem;border-left:3px solid {sc};margin-bottom:0.5rem;">'
                f'<div style="display:flex;justify-content:space-between;align-items:start;">'
                f'<div style="font-size:0.75rem;color:{COLORS["text_muted"]};'
                f'text-transform:uppercase;letter-spacing:0.05em;">'
                f'{_esc(m["icon"])} {_esc(m["name"])}</div>'
                f'<span style="font-size:1.1rem;">{emoji}</span></div>'
                f'<div style="font-size:1.5rem;font-weight:700;color:{sc};margin:0.4rem 0;">'
                f'{_esc(m["value"])}</div>'
                f'<div style="font-size:0.7rem;color:{COLORS["text_muted"]};">'
                f'{_esc(m["detail"])}</div></div>'
            )
            col.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

    # ── Acciones sugeridas (automáticas) ──────────────────────────────
    red_metrics = [m for m in metrics if m['status'] == 'red']
    yellow_metrics = [m for m in metrics if m['status'] == 'yellow']
    actions = []

    for m in red_metrics:
        if 'Frecuencia' in m['name']:
            actions.append(f"🔴 **Publica más seguido** — solo {m['value']}. Objetivo: 3+ videos/semana.")
        elif 'Frescura' in m['name']:
            actions.append(f"🔴 **Publica un video ahora** — llevas {m['value']} sin contenido nuevo.")
        elif 'engagement' in m['name'].lower():
            actions.append(f"🔴 **Engagement cayendo** ({m['value']}) — revisa CTAs y hooks de tus últimos videos.")
        elif 'Vistas' in m['name']:
            actions.append(f"🔴 **Vistas por debajo del promedio** — experimenta con nuevos temas o formatos.")
        elif 'Retención' in m['name']:
            actions.append(f"🔴 **Retención baja** ({m['value']}) — mejora los primeros 3 segundos del video.")
        elif 'CTR' in m['name']:
            actions.append(f"🔴 **CTR bajo** ({m['value']}) — rediseña tus miniaturas y títulos.")
        else:
            actions.append(f"🔴 **{m['name']}** necesita atención: {m['value']}.")

    for m in yellow_metrics:
        if 'engagement' in m['name'].lower():
            actions.append(f"🟡 **Engagement estable pero sin crecer** ({m['value']}) — prueba preguntas en video para generar comentarios.")
        elif 'Consistencia' in m['name']:
            actions.append(f"🟡 **Resultados irregulares** — tu engagement varía mucho entre videos. Identifica qué formato funciona mejor.")
        elif 'Frescura' in m['name']:
            actions.append(f"🟡 **Han pasado {m['value']}** desde tu último video — no pierdas el ritmo.")
        else:
            actions.append(f"🟡 **{m['name']}** puede mejorar: {m['value']}.")

    if actions:
        ui_section_divider()
        st.markdown("### ⚡ Acciones sugeridas")
        for action in actions[:5]:
            st.markdown(action)
    elif all(m['status'] == 'green' for m in metrics):
        ui_section_divider()
        st.markdown("### ✅ ¡Excelente!")
        st.success("Todas las métricas están en verde. ¡Sigue así!")

    # ── Diagnóstico IA ────────────────────────────────────────────────
    ui_section_divider()
    st.markdown("### 🧠 Diagnóstico con IA")

    anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')

    # Buscar diagnóstico reciente (últimas 24h)
    existing_report = db.get_latest_health_report(channel_id)
    existing_diagnosis = None
    report_id = None
    if existing_report and existing_report.get('ai_diagnosis'):
        existing_diagnosis = existing_report['ai_diagnosis']
        report_id = existing_report['id']

    if existing_diagnosis:
        st.markdown(existing_diagnosis)
        st.caption(f"Generado: {existing_report.get('created_at', '—')}")
    else:
        if anthropic_key:
            ch_language = _detect_channel_language(db, channel_id)
            lang_label = {'en': '🇺🇸 EN', 'es': '🇪🇸 ES', 'pt': '🇧🇷 PT'}.get(ch_language, ch_language)

            # Obtener nombre real del canal
            try:
                cursor_ch = db.conn.cursor()
                cursor_ch.execute("SELECT channel_name FROM channels WHERE channel_id = %s", (channel_id,))
                ch_row = cursor_ch.fetchone()
                ch_display_name = ch_row['channel_name'] if ch_row else channel_id
            except Exception:
                ch_display_name = channel_id

            if st.button(f"🧠 Analizar con IA ({lang_label})", type="secondary", key="btn_health_ai"):
                with st.spinner("Analizando canal con IA..."):
                    # Guardar reporte base primero
                    import json as _json
                    metrics_json = _json.dumps(
                        [{k: v for k, v in m.items() if k != 'raw'} for m in metrics],
                        ensure_ascii=False,
                    )
                    report_id = db.save_health_report(channel_id, score, metrics_json)

                    # Generar diagnóstico IA
                    analyzer = AIAnalyzer(anthropic_key)
                    diagnosis = analyzer.generate_health_diagnosis(
                        channel_name=ch_display_name,
                        metrics=metrics,
                        health_score=score,
                        language=ch_language,
                    )
                    if not diagnosis.startswith("Error"):
                        db.update_health_diagnosis(report_id, diagnosis)
                        st.markdown(diagnosis)
                        st.success("Diagnóstico guardado")
                    else:
                        st.error(f"Error: {diagnosis}")
        else:
            st.info("Configura ANTHROPIC_API_KEY en .env para habilitar el diagnóstico con IA.")

    db.close()


# ═══════════════════════════════════════════════════════════════════════════
# ⏱ Cadencia y Horarios (Mejoras 13.2 / 13.3)
# ═══════════════════════════════════════════════════════════════════════════

def show_cadence_analysis(df: pd.DataFrame, channel_id: str | None):
    """Página de análisis de cadencia óptima y saturación horaria."""
    ui_page_header("⏱", "Cadencia y Horarios",
                   "Frecuencia óptima de publicación y ventanas de oportunidad horaria")

    if df.empty or not channel_id:
        st.warning("Selecciona un canal con datos en el panel lateral.")
        return

    if len(df) < 15:
        st.info("Se necesitan al menos 15 videos para un análisis de cadencia significativo. "
                f"Actualmente tienes {len(df)} videos.")
        return

    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    analyzer = AIAnalyzer(anthropic_api_key) if anthropic_api_key else None

    if not analyzer:
        st.error("ANTHROPIC_API_KEY no configurada en .env")
        return

    # ── Sección A: Cadencia Óptima ─────────────────────────────────────────
    st.subheader("📊 Cadencia Óptima de Publicación")
    st.caption("Relación entre días entre videos y rendimiento")

    cadence = analyzer.analyze_cadence(df)

    if not cadence['optimal_cadence']:
        st.info("No hay suficientes datos para calcular la cadencia óptima.")
    else:
        opt = cadence['optimal_cadence']
        cols_opt = st.columns(2)
        with cols_opt[0]:
            if 'Short' in opt:
                o = opt['Short']
                st.markdown(
                    f"<div style='background:{COLORS['bg_secondary']};border-radius:10px;"
                    f"padding:1.2rem;border-left:4px solid {COLORS['secondary']}'>"
                    f"<div style='font-size:1.5rem'>📱</div>"
                    f"<div style='font-weight:700;color:{COLORS['text_primary']};font-size:1.1rem'>"
                    f"Cadencia óptima Shorts</div>"
                    f"<div style='font-size:1.8rem;font-weight:800;color:{COLORS['secondary']}'>"
                    f"{o['bucket']}</div>"
                    f"<div style='color:{COLORS['text_secondary']};font-size:0.9rem'>"
                    f"{o['description']}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No hay suficientes Shorts para analizar.")
        with cols_opt[1]:
            if 'Video Largo' in opt:
                o = opt['Video Largo']
                st.markdown(
                    f"<div style='background:{COLORS['bg_secondary']};border-radius:10px;"
                    f"padding:1.2rem;border-left:4px solid {COLORS['primary']}'>"
                    f"<div style='font-size:1.5rem'>🎬</div>"
                    f"<div style='font-weight:700;color:{COLORS['text_primary']};font-size:1.1rem'>"
                    f"Cadencia óptima Videos Largos</div>"
                    f"<div style='font-size:1.8rem;font-weight:800;color:{COLORS['primary']}'>"
                    f"{o['bucket']}</div>"
                    f"<div style='color:{COLORS['text_secondary']};font-size:0.9rem'>"
                    f"{o['description']}</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No hay suficientes Videos Largos para analizar.")

        # Bar charts por tipo de video
        tab_short, tab_largo = st.tabs(["📱 Shorts", "🎬 Videos Largos"])
        for tab, vtype, color in [
            (tab_short, 'Short', COLORS['secondary']),
            (tab_largo, 'Video Largo', COLORS['primary']),
        ]:
            with tab:
                data = cadence['cadence_by_type'].get(vtype, [])
                if not data:
                    st.info(f"No hay suficientes datos de {vtype}s para graficar.")
                    continue
                chart_df = pd.DataFrame(data)
                fig = px.bar(
                    chart_df,
                    x='bucket_label',
                    y='avg_views',
                    text='count',
                    color_discrete_sequence=[color],
                    labels={'bucket_label': 'Días entre publicaciones',
                            'avg_views': 'Vistas promedio',
                            'count': 'Videos'},
                )
                fig.update_traces(texttemplate='%{text} videos', textposition='outside')
                fig.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=350,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=20, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Engagement overlay
                if 'avg_engagement' in chart_df.columns:
                    fig_eng = px.line(
                        chart_df,
                        x='bucket_label',
                        y='avg_engagement',
                        markers=True,
                        color_discrete_sequence=[COLORS['success']],
                        labels={'bucket_label': 'Días entre publicaciones',
                                'avg_engagement': 'Engagement promedio (%)'},
                    )
                    fig_eng.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=250,
                        margin=dict(l=0, r=0, t=20, b=0),
                    )
                    st.plotly_chart(fig_eng, use_container_width=True)

    # ── Sección B: Saturación Horaria ──────────────────────────────────────
    ui_section_divider()
    st.subheader("🕐 Saturación Horaria")
    st.caption("Dónde publicas vs dónde rinde mejor")

    hourly = analyzer.analyze_hourly_saturation(df)
    hour_labels = [str(h) for h in range(24)]

    tab_freq, tab_perf, tab_opp = st.tabs([
        "📊 Frecuencia de publicación",
        "👁 Rendimiento por horario",
        "🎯 Score de oportunidad",
    ])

    with tab_freq:
        st.caption("Cuántos videos has publicado en cada franja horaria")
        fig_freq = px.imshow(
            hourly['publishing_frequency'].values,
            labels=dict(x='Hora (Panamá)', y='Día', color='Videos'),
            x=hour_labels,
            y=WEEKDAY_LABELS,
            color_continuous_scale='Blues',
            aspect='auto',
        )
        fig_freq.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    with tab_perf:
        st.caption("Vistas promedio reales en cada franja horaria")
        fig_perf = px.imshow(
            hourly['actual_performance'].values,
            labels=dict(x='Hora (Panamá)', y='Día', color='Vistas prom.'),
            x=hour_labels,
            y=WEEKDAY_LABELS,
            color_continuous_scale='YlOrRd',
            aspect='auto',
        )
        fig_perf.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    with tab_opp:
        st.caption("Alto = alto rendimiento + baja frecuencia de publicación (oportunidad)")
        fig_opp = px.imshow(
            hourly['opportunity_score'].values,
            labels=dict(x='Hora (Panamá)', y='Día', color='Oportunidad'),
            x=hour_labels,
            y=WEEKDAY_LABELS,
            color_continuous_scale='Greens',
            aspect='auto',
        )
        fig_opp.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_opp, use_container_width=True)

    # ── Sección C: Recomendaciones ─────────────────────────────────────────
    ui_section_divider()
    col_rec, col_sat = st.columns(2)

    with col_rec:
        st.subheader("🎯 Ventanas de Oportunidad")
        if hourly['recommendations']:
            for slot in hourly['recommendations'][:5]:
                st.markdown(
                    f"<div style='background:{COLORS['bg_secondary']};border-radius:8px;"
                    f"padding:0.8rem;margin-bottom:0.5rem;"
                    f"border-left:3px solid {COLORS['success']}'>"
                    f"<div style='font-weight:600;color:{COLORS['text_primary']}'>"
                    f"{slot['day']} a las {slot['hour']}:00</div>"
                    f"<div style='font-size:0.85rem;color:{COLORS['text_secondary']}'>"
                    f"Oportunidad: {slot['opportunity_score']}/100 · "
                    f"Prom: {slot['avg_views']:,} vistas · "
                    f"Publicado {slot['times_published']}x</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No hay suficientes datos para detectar oportunidades.")

    with col_sat:
        st.subheader("⚠ Franjas Saturadas")
        if hourly['saturated_slots']:
            for slot in hourly['saturated_slots'][:5]:
                st.markdown(
                    f"<div style='background:{COLORS['bg_secondary']};border-radius:8px;"
                    f"padding:0.8rem;margin-bottom:0.5rem;"
                    f"border-left:3px solid {COLORS['warning']}'>"
                    f"<div style='font-weight:600;color:{COLORS['text_primary']}'>"
                    f"{slot['day']} a las {slot['hour']}:00</div>"
                    f"<div style='font-size:0.85rem;color:{COLORS['text_secondary']}'>"
                    f"Publicado {slot['times_published']}x · "
                    f"Prom: {slot['avg_views']:,} vistas</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No se detectaron franjas saturadas.")

    # ── Sección D: Cruce con Google Trends (opcional) ──────────────────────
    ui_section_divider()
    st.subheader("🔍 Cruce con Google Trends (opcional)")
    st.caption("Ingresa keywords de tu nicho para detectar días de alto interés de búsqueda")

    trends_kw = st.text_input(
        "Keywords (separados por coma)",
        placeholder="ej: cocina, recetas fáciles, postres",
        key="cadence_trends_kw",
    )

    if trends_kw and st.button("Analizar con Trends", key="btn_cadence_trends"):
        keywords = [k.strip() for k in trends_kw.split(",") if k.strip()][:3]
        if keywords:
            try:
                with st.spinner("Consultando Google Trends..."):
                    hourly_with_trends = analyzer.analyze_hourly_saturation(
                        df, trends_keywords=keywords, trends_geo='PA',
                    )
                st.success("Datos de Trends integrados al score de oportunidad")
                fig_opp_t = px.imshow(
                    hourly_with_trends['opportunity_score'].values,
                    labels=dict(x='Hora (Panamá)', y='Día', color='Oportunidad'),
                    x=hour_labels,
                    y=WEEKDAY_LABELS,
                    color_continuous_scale='Greens',
                    aspect='auto',
                )
                fig_opp_t.update_layout(
                    template='plotly_dark',
                    paper_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_opp_t, use_container_width=True)

                if hourly_with_trends['recommendations']:
                    st.markdown("**Top oportunidades (con Trends):**")
                    for slot in hourly_with_trends['recommendations'][:3]:
                        st.markdown(
                            f"- **{slot['day']} a las {slot['hour']}:00** — "
                            f"oportunidad {slot['opportunity_score']}/100, "
                            f"{slot['avg_views']:,} vistas prom."
                        )
            except Exception as e:
                st.warning(f"No se pudo consultar Google Trends: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Página: Análisis de Competencia (Mejora 7.1)
# ─────────────────────────────────────────────────────────────────────────────

def show_competitor_analysis(df: pd.DataFrame, channel_id: str | None):
    """Página de análisis comparativo contra canales competidores."""
    ui_page_header(
        "🕵", "Análisis de Competencia",
        "Compara tu canal contra la competencia directa — frecuencia, engagement y content gaps"
    )

    if not channel_id or df.empty:
        st.warning("Selecciona un canal con datos en el panel lateral.")
        return

    # ── Cargar datos de competidores ────────────────────────────────────
    try:
        db = YouTubeDatabase()
        competitor_channels = db.get_competitor_channels()
    except Exception as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return

    if competitor_channels.empty:
        st.info(
            "**No hay competidores configurados.**\n\n"
            "Para activar esta funcionalidad:\n"
            "1. Agrega `COMPETITOR_CHANNEL_IDS=UCxxxx,UCyyyy` en tu archivo `.env`\n"
            "2. Ejecuta `python main.py` para extraer los datos de competidores\n"
            "3. Vuelve a esta página para ver el análisis"
        )
        db.close()
        return

    # Cargar videos propios y de competidores
    own_channel_ids = [channel_id]
    comp_channel_ids = competitor_channels['channel_id'].tolist()
    all_ids = own_channel_ids + comp_channel_ids

    all_videos_df = db.get_all_videos_with_competitor_flag(all_ids)
    db.close()

    if all_videos_df.empty:
        st.warning("No hay videos disponibles para analizar.")
        return

    own_videos = all_videos_df[all_videos_df['is_competitor'] == 0]
    comp_videos = all_videos_df[all_videos_df['is_competitor'] == 1]

    own_name = (
        own_videos.iloc[0]['channel_title']
        if not own_videos.empty and pd.notna(own_videos.iloc[0].get('channel_title'))
        else "Tu canal"
    )

    # ── Sección 1: Tabla comparativa ──────────────────────────────────
    ui_section_divider("Benchmarking")
    st.subheader("📊 Tu Canal vs Competidores")

    # Construir stats por canal
    rows = []
    for cid in all_ids:
        ch_df = all_videos_df[all_videos_df['channel_id'] == cid]
        if ch_df.empty:
            continue
        ch_name = ch_df.iloc[0]['channel_title'] if pd.notna(ch_df.iloc[0].get('channel_title')) else cid
        is_comp = int(ch_df.iloc[0].get('is_competitor', 0))
        subs = int(ch_df.iloc[0].get('subscriber_count', 0)) if pd.notna(ch_df.iloc[0].get('subscriber_count')) else 0

        # Últimos 30 días
        recent = ch_df[ch_df['published_at'] >= (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=30))]

        rows.append({
            'Canal': f"📺 {ch_name}" if not is_comp else f"🕵 {ch_name}",
            'Tipo': 'Propio' if not is_comp else 'Competidor',
            'Suscriptores': subs,
            'Videos totales': len(ch_df),
            'Videos (30d)': len(recent),
            'Vistas prom.': int(ch_df['view_count'].mean()) if not ch_df['view_count'].isna().all() else 0,
            'Engagement %': round(float(ch_df['engagement_rate'].mean()), 2) if not ch_df['engagement_rate'].isna().all() else 0,
            'Shorts %': round(float(ch_df['is_short'].sum()) / max(len(ch_df), 1) * 100, 1),
            '_channel_id': cid,
            '_avg_views': float(ch_df['view_count'].mean()) if not ch_df['view_count'].isna().all() else 0,
        })

    if not rows:
        st.warning("No hay datos suficientes.")
        return

    bench_df = pd.DataFrame(rows)

    # KPI cards comparativos
    own_row = bench_df[bench_df['Tipo'] == 'Propio']
    comp_rows = bench_df[bench_df['Tipo'] == 'Competidor']

    if not own_row.empty and not comp_rows.empty:
        own_avg_views = own_row.iloc[0]['Vistas prom.']
        comp_avg_views = comp_rows['Vistas prom.'].mean()
        own_eng = own_row.iloc[0]['Engagement %']
        comp_avg_eng = comp_rows['Engagement %'].mean()
        own_freq = own_row.iloc[0]['Videos (30d)']
        comp_avg_freq = comp_rows['Videos (30d)'].mean()

        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            delta_views = ((own_avg_views / comp_avg_views - 1) * 100) if comp_avg_views > 0 else 0
            delta_str = f"{delta_views:+.1f}% vs competidores"
            st.markdown(ui_metric_card(
                "👁", "Vistas Promedio", f"{own_avg_views:,}",
                delta=delta_str,
                delta_type="positive" if delta_views >= 0 else "negative",
            ), unsafe_allow_html=True)
        with kc2:
            delta_eng = own_eng - comp_avg_eng
            st.markdown(ui_metric_card(
                "💬", "Engagement", f"{own_eng:.2f}%",
                delta=f"{delta_eng:+.2f}pp vs comp.",
                delta_type="positive" if delta_eng >= 0 else "negative",
            ), unsafe_allow_html=True)
        with kc3:
            delta_freq = own_freq - comp_avg_freq
            st.markdown(ui_metric_card(
                "📅", "Publicaciones (30d)", f"{own_freq}",
                delta=f"{delta_freq:+.0f} vs comp.",
                delta_type="positive" if delta_freq >= 0 else "negative",
            ), unsafe_allow_html=True)
        with kc4:
            st.markdown(ui_metric_card(
                "🕵", "Competidores", f"{len(comp_rows)}",
                delta="rastreados",
                delta_type="neutral",
            ), unsafe_allow_html=True)

    # Tabla
    display_df = bench_df.drop(columns=['_channel_id', '_avg_views'])
    display_df['Suscriptores'] = display_df['Suscriptores'].apply(lambda v: f"{v:,}")
    display_df['Vistas prom.'] = display_df['Vistas prom.'].apply(lambda v: f"{v:,}")
    display_df['Engagement %'] = display_df['Engagement %'].apply(lambda v: f"{v:.2f}%")
    display_df['Shorts %'] = display_df['Shorts %'].apply(lambda v: f"{v:.1f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Sección 2: Gráficos comparativos ──────────────────────────────
    ui_section_divider("Gráficos Comparativos")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Vistas promedio por canal")
        fig_views = px.bar(
            bench_df, x='Canal', y='Vistas prom.',
            color='Tipo',
            color_discrete_map={'Propio': COLORS['primary'], 'Competidor': COLORS['secondary']},
            text='Vistas prom.',
        )
        fig_views.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_views.update_layout(showlegend=True, height=360,
                                margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
        st.plotly_chart(fig_views, use_container_width=True)

    with col_g2:
        st.subheader("Engagement promedio por canal")
        fig_eng = px.bar(
            bench_df, x='Canal', y='Engagement %',
            color='Tipo',
            color_discrete_map={'Propio': COLORS['primary'], 'Competidor': COLORS['secondary']},
            text='Engagement %',
        )
        fig_eng.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_eng.update_layout(showlegend=True, height=360,
                              margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
        st.plotly_chart(fig_eng, use_container_width=True)

    # ── Sección 3: Crecimiento relativo de vistas (últimos 30 videos) ─
    ui_section_divider("Crecimiento de Vistas Relativo")
    st.subheader("📈 Crecimiento de vistas — normalizado")
    st.caption(
        "Cada canal se normaliza a 100 en su primer video del periodo, "
        "para comparar crecimiento relativo sin importar tamaño absoluto."
    )

    growth_traces = []
    for cid in all_ids:
        ch_df = all_videos_df[all_videos_df['channel_id'] == cid].sort_values('published_at')
        if ch_df.empty or ch_df['view_count'].isna().all():
            continue
        ch_name = ch_df.iloc[0]['channel_title'] if pd.notna(ch_df.iloc[0].get('channel_title')) else cid
        # Tomar últimos 30 videos
        recent_vids = ch_df.tail(30).copy()
        if recent_vids.empty:
            continue
        # Normalizar: primer video = 100
        first_views = recent_vids.iloc[0]['view_count']
        if first_views and first_views > 0:
            recent_vids['normalized'] = (recent_vids['view_count'] / first_views) * 100
        else:
            recent_vids['normalized'] = 0
        recent_vids['video_num'] = range(1, len(recent_vids) + 1)
        recent_vids['canal'] = ch_name
        growth_traces.append(recent_vids[['video_num', 'normalized', 'canal']])

    if growth_traces:
        growth_df = pd.concat(growth_traces, ignore_index=True)
        fig_growth = px.line(
            growth_df, x='video_num', y='normalized', color='canal',
            labels={'video_num': 'Video #', 'normalized': 'Vistas (normalizado a 100)', 'canal': 'Canal'},
        )
        fig_growth.update_layout(height=400, margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
        st.plotly_chart(fig_growth, use_container_width=True)
    else:
        st.caption("No hay datos suficientes para el gráfico de crecimiento.")

    # ── Sección 4: Nube de palabras (títulos) ─────────────────────────
    ui_section_divider("Análisis de Títulos")
    st.subheader("☁ Palabras más usadas en títulos")

    col_wc1, col_wc2 = st.columns(2)

    # Stopwords comunes en español para filtrar
    _stopwords = {
        'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'del', 'un', 'una',
        'con', 'por', 'para', 'que', 'es', 'se', 'no', 'su', 'al', 'lo',
        'como', 'más', 'o', 'mi', 'si', 'ya', 'te', 'me', 'esto', 'esta',
        'the', 'and', 'to', 'of', 'in', 'is', 'for', 'on', 'it', 'with',
        'this', 'that', 'are', 'was', 'you', 'your', 'my', 'i', 'we',
        'shorts', 'short', 'video', 'ft', 'vs',
    }

    def _title_word_counts(titles_series, top_n=20):
        """Extrae las top N palabras de una serie de títulos."""
        all_words = []
        for title in titles_series.dropna():
            words = re.findall(r'[a-záéíóúñü]+', str(title).lower())
            all_words.extend(w for w in words if len(w) > 2 and w not in _stopwords)
        return Counter(all_words).most_common(top_n)

    with col_wc1:
        st.markdown(f"**📺 {_esc(own_name)}**")
        own_words = _title_word_counts(own_videos['title'])
        if own_words:
            own_wc_df = pd.DataFrame(own_words, columns=['Palabra', 'Frecuencia'])
            fig_own_wc = px.bar(
                own_wc_df, x='Frecuencia', y='Palabra', orientation='h',
                color_discrete_sequence=[COLORS['primary']],
            )
            fig_own_wc.update_layout(height=420, yaxis={'categoryorder': 'total ascending'},
                                     margin={'l': 0, 'r': 0, 't': 10, 'b': 0})
            st.plotly_chart(fig_own_wc, use_container_width=True)
        else:
            st.caption("Sin datos suficientes")

    with col_wc2:
        st.markdown("**🕵 Competidores (combinados)**")
        comp_words = _title_word_counts(comp_videos['title'])
        if comp_words:
            comp_wc_df = pd.DataFrame(comp_words, columns=['Palabra', 'Frecuencia'])
            fig_comp_wc = px.bar(
                comp_wc_df, x='Frecuencia', y='Palabra', orientation='h',
                color_discrete_sequence=[COLORS['secondary']],
            )
            fig_comp_wc.update_layout(height=420, yaxis={'categoryorder': 'total ascending'},
                                      margin={'l': 0, 'r': 0, 't': 10, 'b': 0})
            st.plotly_chart(fig_comp_wc, use_container_width=True)
        else:
            st.caption("Sin datos suficientes")

    # Detectar palabras que usan los competidores pero NO el canal propio
    own_word_set = {w for w, _ in own_words} if own_words else set()
    comp_word_set = {w for w, _ in comp_words} if comp_words else set()
    gap_words = comp_word_set - own_word_set

    if gap_words:
        st.markdown(
            f"**🔍 Palabras frecuentes en competidores que NO aparecen en tu canal:** "
            f"`{'`, `'.join(sorted(gap_words)[:15])}`"
        )

    # ── Sección 5: Temas de esta semana ───────────────────────────────
    ui_section_divider("Temas Recientes de Competidores")
    st.subheader("📅 Qué están publicando los competidores esta semana")

    week_ago = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)
    comp_recent = comp_videos[comp_videos['published_at'] >= week_ago].sort_values('view_count', ascending=False)

    if not comp_recent.empty:
        for _, row in comp_recent.head(15).iterrows():
            ch_name_r = row['channel_title'] if pd.notna(row.get('channel_title')) else ''
            views_r = int(row['view_count']) if pd.notna(row['view_count']) else 0
            eng_r = float(row['engagement_rate']) if pd.notna(row['engagement_rate']) else 0
            vtype_r = row.get('video_type', '')
            vtype_color = COLORS['secondary'] if vtype_r == 'Short' else COLORS['primary']
            st.markdown(
                f"<div style='background:{COLORS['bg_secondary']};border-radius:8px;"
                f"padding:0.8rem 1rem;margin-bottom:0.5rem;"
                f"border-left:3px solid {vtype_color}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div>"
                f"<div style='font-weight:600;color:{COLORS['text_primary']};font-size:0.95rem'>"
                f"{_esc(row['title'])}</div>"
                f"<div style='font-size:0.8rem;color:{COLORS['text_secondary']};margin-top:2px'>"
                f"🕵 {_esc(ch_name_r)} · {_esc(vtype_r)}</div>"
                f"</div>"
                f"<div style='text-align:right;min-width:120px'>"
                f"<div style='font-weight:600;color:{COLORS['text_primary']}'>{views_r:,} vistas</div>"
                f"<div style='font-size:0.8rem;color:{COLORS['text_secondary']}'>"
                f"Eng: {eng_r:.2f}%</div>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Los competidores no han publicado videos en los últimos 7 días.")

    # ── Sección 6: Content Gaps con IA ─────────────────────────────────
    ui_section_divider("Content Gaps con IA")
    st.subheader("🤖 Análisis de Brechas de Contenido")
    st.caption("Claude AI analiza qué temas explotan tus competidores y cuáles son tus oportunidades")

    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key:
        st.error("ANTHROPIC_API_KEY no configurada en .env")
        return

    if st.button("🔍 Generar Análisis de Content Gaps", key="btn_content_gaps"):
        with st.spinner("Claude está analizando a tus competidores..."):
            try:
                analyzer = AIAnalyzer(anthropic_api_key)

                # Preparar datos propios
                own_titles_list = own_videos['title'].dropna().tolist()[:30]
                own_stats = {
                    'total_videos': len(own_videos),
                    'avg_views': float(own_videos['view_count'].mean()) if not own_videos.empty else 0,
                    'avg_engagement': float(own_videos['engagement_rate'].mean()) if not own_videos.empty else 0,
                }

                # Preparar datos de cada competidor
                comp_data_list = []
                for comp_cid in comp_channel_ids:
                    c_vids = comp_videos[comp_videos['channel_id'] == comp_cid]
                    if c_vids.empty:
                        continue
                    c_name = c_vids.iloc[0]['channel_title'] if pd.notna(c_vids.iloc[0].get('channel_title')) else comp_cid
                    c_subs = int(c_vids.iloc[0].get('subscriber_count', 0)) if pd.notna(c_vids.iloc[0].get('subscriber_count')) else 0
                    comp_data_list.append({
                        'name': c_name,
                        'titles': c_vids['title'].dropna().tolist()[:20],
                        'avg_views': float(c_vids['view_count'].mean()),
                        'avg_engagement': float(c_vids['engagement_rate'].mean()),
                        'total_videos': len(c_vids),
                        'subscriber_count': c_subs,
                    })

                result = analyzer.analyze_competitor_gaps(
                    own_channel_name=own_name,
                    own_titles=own_titles_list,
                    own_stats=own_stats,
                    competitor_data=comp_data_list,
                )

                if result and not result.startswith('Error'):
                    st.markdown(result)
                else:
                    st.error(f"No se pudo generar el análisis: {result}")
            except Exception as e:
                st.error(f"Error al generar análisis: {e}")

    # ── Sección 7: Alertas de Videos Virales (7.2) ────────────────────
    ui_section_divider("Alertas de Videos Virales")
    st.subheader("🚨 Videos de Competidores con Rendimiento Excepcional")
    st.caption("Videos que superaron 2x el promedio de vistas de su canal")

    try:
        db_alerts = YouTubeDatabase()
        alerts_df = db_alerts.get_competitor_alerts(limit=20)
        db_alerts.close()
    except Exception as e:
        st.error(f"Error al cargar alertas: {e}")
        alerts_df = pd.DataFrame()

    if alerts_df.empty:
        st.info(
            "**Sin alertas todavía.** Las alertas se generan automáticamente cuando "
            "`python main.py` detecta un video de competidor con >2x su promedio de vistas."
        )
    else:
        for _, alert_row in alerts_df.iterrows():
            ratio_val = float(alert_row['ratio']) if pd.notna(alert_row.get('ratio')) else 0
            views_val = int(alert_row['view_count']) if pd.notna(alert_row.get('view_count')) else 0
            avg_val = float(alert_row['competitor_avg_views']) if pd.notna(alert_row.get('competitor_avg_views')) else 0
            a_title = _esc(str(alert_row.get('video_title', ''))[:80])
            a_channel = _esc(str(alert_row.get('channel_name', '')))
            a_date = str(alert_row.get('created_at', ''))[:10]
            notified_icon = "📨" if alert_row.get('notified') else "🔕"

            # Color del badge por ratio
            if ratio_val >= 5:
                badge_color = "#ef4444"
            elif ratio_val >= 3:
                badge_color = "#f97316"
            else:
                badge_color = "#eab308"

            st.markdown(
                f"<div style='background:{COLORS['bg_secondary']};border-radius:8px;"
                f"padding:1rem;margin-bottom:0.75rem;"
                f"border-left:4px solid {badge_color}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:start'>"
                f"<div>"
                f"<div style='font-weight:600;color:{COLORS['text_primary']};font-size:0.95rem'>"
                f"{a_title}</div>"
                f"<div style='font-size:0.8rem;color:{COLORS['text_secondary']};margin-top:2px'>"
                f"🕵 {a_channel} · {a_date} {notified_icon}</div>"
                f"</div>"
                f"<div style='text-align:right;min-width:140px'>"
                f"<div style='font-weight:700;color:{badge_color};font-size:1.1rem'>"
                f"{ratio_val:.1f}x</div>"
                f"<div style='font-size:0.8rem;color:{COLORS['text_secondary']}'>"
                f"{views_val:,} vistas (prom: {avg_val:,.0f})</div>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )

            # Expandir análisis de Claude
            ai_text = alert_row.get('ai_analysis', '')
            if ai_text and str(ai_text).strip():
                with st.expander(f"🤖 Ver análisis — {a_title[:40]}..."):
                    st.markdown(str(ai_text))


# ══════════════════════════════════════════════════════════════════════
# Predicción de Retención (Mejora 12.1)
# ══════════════════════════════════════════════════════════════════════

def show_retention_prediction(df: pd.DataFrame, channel_id: str):
    """Predicción de retención de audiencia con ML."""
    ui_page_header("📊", "Predicción de Retención",
                   "Modelo ML para estimar el % de video visto por la audiencia")

    if df.empty or not channel_id:
        st.warning("No hay datos disponibles para este canal.")
        return

    # Cargar datos de analytics
    try:
        db = YouTubeDatabase()
        analytics_df = db.get_video_analytics(channel_id)
        db.close()
    except Exception:
        analytics_df = pd.DataFrame()

    if analytics_df.empty or 'avg_view_percentage' not in analytics_df.columns:
        st.warning(
            "⚠ Se requieren datos de YouTube Analytics API para entrenar este modelo. "
            "Ejecuta `python main.py` con `credentials.json` configurado."
        )
        st.info(
            "Este modelo predice la retención de audiencia (% del video visto). "
            "Necesita datos de la Analytics API que incluyan `avg_view_percentage`."
        )
        return

    predictor, from_cache = _get_retention_predictor(df, analytics_df, channel_id)

    if not predictor.is_trained():
        reason = predictor.get_train_metrics().get('reason', 'No se pudo entrenar.')
        st.warning(f"⚠ {reason}")
        return

    metrics = predictor.get_train_metrics()
    n_splits = metrics.get('cv_splits', '?')

    if from_cache:
        st.info(f"Modelo cargado desde caché — {metrics.get('samples', '?')} videos.")
    else:
        st.success(f"Modelo reentrenado con **{metrics.get('samples', '?')} videos** · {n_splits}-fold CV temporal.")

    # ── Métricas de precisión ─────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Videos usados", f"{metrics['samples']}",
                   help=f"CV temporal con {n_splits} folds.")
    with col2:
        st.metric("MAE — CV", f"{metrics.get('cv_mae', 0):.1f}%",
                   help="Error absoluto medio: cuántos puntos porcentuales se desvía en promedio.")
    with col3:
        cv_r2 = metrics.get('cv_r2', 0)
        st.metric("R² — CV", f"{cv_r2:.3f}",
                   help="Coeficiente de determinación. 1.0 = predicción perfecta, 0 = igual al promedio.")

    # ── Scores de videos ──────────────────────────────────────────────
    ui_section_divider()
    st.subheader("🎯 Retención Predicha por Video")

    merged = df.merge(
        analytics_df[['video_id', 'avg_view_percentage']],
        on='video_id', how='left',
    )
    scored_df = predictor.predict(merged)

    # Guardar en DB
    try:
        db2 = YouTubeDatabase()
        db2.save_retention_predictions(
            scored_df[['video_id', 'channel_id', 'predicted_retention']].dropna(subset=['predicted_retention'])
        )
        db2.close()
    except Exception:
        pass

    display_cols = ['title', 'video_type', 'avg_view_percentage', 'predicted_retention']
    available_cols = [c for c in display_cols if c in scored_df.columns]
    top_df = scored_df[available_cols].sort_values('predicted_retention', ascending=False).head(20).copy()
    if 'avg_view_percentage' in top_df.columns:
        top_df['avg_view_percentage'] = top_df['avg_view_percentage'].apply(
            lambda v: f"{v:.1f}%" if pd.notna(v) and v > 0 else '—'
        )
    top_df['predicted_retention'] = top_df['predicted_retention'].apply(lambda v: f"{v:.1f}%")

    st.dataframe(
        top_df.rename(columns={
            'title': 'Título', 'video_type': 'Tipo',
            'avg_view_percentage': 'Retención Real',
            'predicted_retention': 'Retención Predicha',
        }),
        use_container_width=True, hide_index=True,
    )

    # ── Feature importance ────────────────────────────────────────────
    ui_section_divider()
    importance = predictor.get_feature_importance()
    if importance:
        st.subheader("📊 Importancia de Features")
        imp_df = pd.DataFrame(
            sorted(importance.items(), key=lambda x: x[1], reverse=True),
            columns=['Feature', 'Importancia'],
        )
        fig = px.bar(imp_df, x='Importancia', y='Feature', orientation='h',
                     color_discrete_sequence=[COLORS["primary"]])
        fig.update_layout(yaxis={'autorange': 'reversed'}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ── What-If Simulator ─────────────────────────────────────────────
    ui_section_divider()
    st.subheader("🔬 Simulador What-If: Retención")

    wi_col1, wi_col2, wi_col3 = st.columns(3)
    with wi_col1:
        wi_hour = st.slider("Hora de publicación", 0, 23, 12, key="ret_hour")
        wi_weekday = st.selectbox("Día", list(range(7)),
                                  format_func=lambda d: WEEKDAY_LABELS[d], key="ret_day")
        wi_is_short = st.toggle("Es Short", key="ret_short")
        wi_duration = st.number_input("Duración (seg)", 15, 7200, 600, key="ret_dur")
        wi_tags = st.slider("Número de tags", 0, 30, 10, key="ret_tags")
    with wi_col2:
        wi_title_len = st.slider("Longitud título", 10, 100, 60, key="ret_tlen")
        wi_has_num = st.toggle("Título con número", key="ret_num")
        wi_has_q = st.toggle("Título con pregunta", key="ret_q")
        wi_desc_len = st.slider("Longitud descripción", 0, 5000, 500, key="ret_desc")
        wi_days = st.slider("Días desde última subida", 1, 60, 7, key="ret_days")
    with wi_col3:
        wi_hook = st.toggle("Título con hook (cómo, secreto, truco)", key="ret_hook")
        wi_tutorial = st.toggle("Es tutorial", key="ret_tut")
        wi_entertain = st.toggle("Es entretenimiento", key="ret_ent")

    pred_ret = predictor.predict_single(
        hour=wi_hour, weekday=wi_weekday, is_short=wi_is_short,
        duration_seconds=wi_duration, tags_count=wi_tags,
        title_length=wi_title_len, title_has_number=int(wi_has_num),
        title_has_question=int(wi_has_q), description_length=wi_desc_len,
        days_since_last_upload=wi_days, has_hook_title=int(wi_hook),
        is_tutorial=int(wi_tutorial), is_entertainment=int(wi_entertain),
    )

    color = RetentionPredictor.score_color(pred_ret)
    st.markdown(
        f"<div style='text-align:center; padding:1.5rem; background:{COLORS['bg_tertiary']}; "
        f"border-radius:12px; border:2px solid {color};'>"
        f"<div style='font-size:3rem; font-weight:bold; color:{color};'>{pred_ret:.1f}%</div>"
        f"<div style='color:{COLORS['text_secondary']};'>Retención estimada</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# Late Bloomer Detection (Mejora 12.3)
# ══════════════════════════════════════════════════════════════════════

def show_late_bloomer_analysis(df: pd.DataFrame, channel_id: str):
    """Análisis de videos con patrón de despegue tardío."""
    ui_page_header("🌱", "Despegue Tardío",
                   "Detecta videos que crecen lentamente al inicio pero despegan después")

    if df.empty or not channel_id:
        st.warning("No hay datos disponibles.")
        return

    try:
        db = YouTubeDatabase()
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return

    detector = LateBloomerDetector()
    results = detector.analyze_channel(channel_id, db)

    if not results:
        st.info(
            "No hay suficientes snapshots de métricas para analizar patrones. "
            "Ejecuta `python main.py` al menos 3 veces en días diferentes para acumular datos."
        )
        db.close()
        return

    # ── Alertas recientes ─────────────────────────────────────────────
    alerts = detector.get_recent_alerts(results, days_lookback=30)

    if alerts:
        st.subheader("🚨 Alertas — Videos Recientes con Despegue Tardío")
        for alert in alerts:
            st.success(
                f"🌱 **{_esc(alert['title'])}** — "
                f"Aceleración: **{alert['acceleration_factor']:.1f}x** · "
                f"Vistas primeras 48h: {alert['early_views']:,} → "
                f"Vistas actuales: {alert['current_views']:,}\n\n"
                f"_Este video muestra un patrón SEO-driven. No lo descartes._"
            )
        ui_section_divider()

    # ── Tabla completa ────────────────────────────────────────────────
    st.subheader("📋 Análisis de Patrones de Crecimiento")

    table_data = []
    for r in results:
        label = LateBloomerDetector.PATTERN_LABELS_ES.get(r['pattern'], r['pattern'])
        table_data.append({
            'Título': r['title'],
            'Tipo': r.get('video_type', ''),
            'Publicado': r['published_at'][:10] if r['published_at'] else '',
            'Patrón': label,
            'Aceleración': f"{r['acceleration_factor']:.1f}x",
            'Vistas 48h': f"{r['early_views']:,}",
            'Vistas Actual': f"{r['current_views']:,}",
            'Snapshots': r['total_snapshots'],
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # ── Curva de crecimiento de un video ──────────────────────────────
    ui_section_divider()
    st.subheader("📈 Curva de Crecimiento")

    video_options = {r['title'][:60]: r['video_id'] for r in results if r['total_snapshots'] >= 3}
    if video_options:
        selected_title = st.selectbox("Selecciona un video", list(video_options.keys()),
                                      key="lb_video_select")
        selected_vid = video_options[selected_title]

        hist_df = db.get_video_metrics_history(selected_vid)
        if not hist_df.empty:
            hist_df = hist_df.sort_values('recorded_at')
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df['recorded_at'], y=hist_df['view_count'],
                mode='lines+markers', name='Vistas',
                line=dict(color=COLORS["primary"], width=2),
            ))
            fig.update_layout(
                title="Evolución de vistas en el tiempo",
                xaxis_title="Fecha", yaxis_title="Vistas",
                template="youtube_ai",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Estadísticas de patrones ──────────────────────────────────────
    ui_section_divider()
    st.subheader("📊 Distribución de Patrones")

    pattern_counts = {}
    for r in results:
        label = LateBloomerDetector.PATTERN_LABELS_ES.get(r['pattern'], r['pattern'])
        pattern_counts[label] = pattern_counts.get(label, 0) + 1

    if pattern_counts:
        pc_df = pd.DataFrame(list(pattern_counts.items()), columns=['Patrón', 'Videos'])
        fig = px.pie(pc_df, values='Videos', names='Patrón',
                     color_discrete_sequence=CHART_COLORS)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        late_pct = sum(1 for r in results if r['is_late_bloomer']) / len(results) * 100
        if late_pct > 20:
            st.info(
                f"📊 **{late_pct:.0f}%** de tus videos analizados muestran despegue tardío. "
                "Esto sugiere que tu contenido tiene buena discoverability via búsqueda (SEO)."
            )

    db.close()


# ══════════════════════════════════════════════════════════════════════
# Cannibalization Detection (Mejora 12.4)
# ══════════════════════════════════════════════════════════════════════

def show_cannibalization_analysis(df: pd.DataFrame, channel_id: str):
    """Análisis de canibalización de contenido entre videos."""
    ui_page_header("🔀", "Canibalización de Contenido",
                   "Detecta videos con títulos/tags similares que compiten por la misma audiencia")

    if df.empty or not channel_id:
        st.warning("No hay datos disponibles.")
        return

    # ── Controles ─────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        threshold = st.slider(
            "Umbral de similitud", 0.50, 1.00, 0.80, 0.05,
            help="Solo se muestran pares con similitud ≥ este valor.",
            key="canib_threshold",
        )
    with col_ctrl2:
        max_days = st.slider(
            "Máx. días entre publicaciones", 1, 180, 30,
            help="Solo pares publicados dentro de este rango de días.",
            key="canib_days",
        )

    detector = CannibalizationDetector(threshold=threshold, max_days_apart=max_days)
    results = detector.detect(df)

    if not results:
        st.success(
            "✅ No se detectó canibalización con los parámetros actuales. "
            "Tu diversidad de contenido es buena."
        )
        return

    # ── Alertas recientes ─────────────────────────────────────────────
    alerts = detector.get_recent_alerts(results, days_lookback=60)
    if alerts:
        st.subheader("🚨 Alertas de Canibalización Reciente")
        for a in alerts[:5]:
            severity_color = COLORS["danger"] if a['similarity'] >= 0.90 else COLORS["warning"]
            terms = ', '.join(a['shared_terms'][:3]) if a['shared_terms'] else '—'
            st.markdown(
                f"<div style='padding:0.8rem; margin:0.5rem 0; border-left:4px solid {severity_color}; "
                f"background:{COLORS['bg_tertiary']}; border-radius:0 8px 8px 0;'>"
                f"<strong>{_esc(a['title_a'])}</strong><br>"
                f"<strong>{_esc(a['title_b'])}</strong><br>"
                f"<span style='color:{severity_color}; font-weight:bold;'>"
                f"Similitud: {a['similarity']:.0%}</span> · "
                f"{a['days_apart']} días de diferencia · "
                f"Términos: {_esc(terms)}</div>",
                unsafe_allow_html=True,
            )
        ui_section_divider()

    # ── Tabla completa ────────────────────────────────────────────────
    st.subheader(f"📋 {len(results)} pares detectados")

    table_data = []
    for r in results:
        table_data.append({
            'Video A': r['title_a'][:50],
            'Video B': r['title_b'][:50],
            'Similitud': f"{r['similarity']:.0%}",
            'Días': r['days_apart'],
            'Términos Comunes': ', '.join(r['shared_terms'][:3]),
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # ── Heatmap de similitud ──────────────────────────────────────────
    ui_section_divider()
    st.subheader("🗺 Mapa de Similitud (últimos 30 videos)")

    recent_df = df.sort_values('published_at', ascending=False).head(30).reset_index(drop=True)
    if len(recent_df) >= 2:
        corpus = CannibalizationDetector._build_text_corpus(recent_df)
        vectorizer = TfidfVectorizer(max_features=5000)
        try:
            tfidf_m = vectorizer.fit_transform(corpus)
            sim_m = cosine_similarity(tfidf_m)

            labels = [t[:30] for t in recent_df['title'].tolist()]
            fig = px.imshow(
                sim_m, x=labels, y=labels,
                color_continuous_scale='YlOrRd',
                zmin=0, zmax=1,
                labels=dict(color='Similitud'),
            )
            fig.update_layout(height=600, xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        except ValueError:
            st.info("No se pudo generar el mapa (datos insuficientes).")

    # ── Recomendaciones ───────────────────────────────────────────────
    ui_section_divider()
    if results:
        st.info(
            "💡 **Recomendación:** Considera espaciar temas similares al menos 30 días. "
            "Videos sobre el mismo tema compiten por las mismas búsquedas y pueden "
            "dividir la audiencia potencial."
        )


# ══════════════════════════════════════════════════════════════════════
# Comparador Temporal (Mejora 15.3)
# ══════════════════════════════════════════════════════════════════════

def show_temporal_comparison(df: pd.DataFrame):
    """Compara métricas entre dos rangos de fecha con deltas visuales."""
    ui_page_header(
        "📆", "Comparador Temporal",
        "Compara métricas entre dos períodos para identificar tendencias"
    )

    if df.empty:
        st.warning("No hay datos disponibles.")
        return

    # ── Presets rápidos ────────────────────────────────────────────────
    today = datetime.now().date()
    presets = {
        "Este mes vs anterior": (
            today.replace(day=1),
            today,
            (today.replace(day=1) - timedelta(days=1)).replace(day=1),
            today.replace(day=1) - timedelta(days=1),
        ),
        "Últimos 30d vs 30d anteriores": (
            today - timedelta(days=30), today,
            today - timedelta(days=60), today - timedelta(days=31),
        ),
        "Últimos 90d vs 90d anteriores": (
            today - timedelta(days=90), today,
            today - timedelta(days=180), today - timedelta(days=91),
        ),
        "Personalizado": None,
    }

    preset = st.selectbox("⏱ Período", list(presets.keys()), key="temporal_preset")

    if preset == "Personalizado":
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Período A (actual)**")
            a_start = st.date_input("Desde", today - timedelta(days=30), key="ta_start")
            a_end = st.date_input("Hasta", today, key="ta_end")
        with col_b:
            st.markdown("**Período B (comparar)**")
            b_start = st.date_input("Desde", today - timedelta(days=60), key="tb_start")
            b_end = st.date_input("Hasta", today - timedelta(days=31), key="tb_end")
    else:
        a_start, a_end, b_start, b_end = presets[preset]

    ui_section_divider()

    # ── Filtrar datos ──────────────────────────────────────────────────
    df_ts = df.copy()
    df_ts['pub_date'] = df_ts['published_at'].dt.date

    period_a = df_ts[(df_ts['pub_date'] >= a_start) & (df_ts['pub_date'] <= a_end)]
    period_b = df_ts[(df_ts['pub_date'] >= b_start) & (df_ts['pub_date'] <= b_end)]

    if period_a.empty and period_b.empty:
        st.info("No hay videos en ninguno de los dos períodos seleccionados.")
        return

    # ── KPIs comparativos ──────────────────────────────────────────────
    st.subheader("📊 Comparación de Métricas")

    def _calc_stats(period_df):
        if period_df.empty:
            return {'videos': 0, 'views': 0, 'avg_views': 0, 'engagement': 0, 'total_likes': 0}
        return {
            'videos': len(period_df),
            'views': int(period_df['view_count'].sum()),
            'avg_views': float(period_df['view_count'].mean()),
            'engagement': float(period_df['engagement_rate'].mean()),
            'total_likes': int(period_df['like_count'].sum()) if 'like_count' in period_df.columns else 0,
        }

    stats_a = _calc_stats(period_a)
    stats_b = _calc_stats(period_b)

    metrics = [
        ("🎬", "Videos publicados", stats_a['videos'], stats_b['videos'], ""),
        ("👀", "Vistas totales", stats_a['views'], stats_b['views'], ","),
        ("📊", "Vistas promedio", stats_a['avg_views'], stats_b['avg_views'], ",.0f"),
        ("💬", "Engagement %", stats_a['engagement'], stats_b['engagement'], ".2f"),
        ("👍", "Likes totales", stats_a['total_likes'], stats_b['total_likes'], ","),
    ]

    cols = st.columns(len(metrics))
    for col, (icon, label, val_a, val_b, fmt) in zip(cols, metrics):
        with col:
            if val_b and val_b != 0:
                delta_pct = ((val_a - val_b) / val_b) * 100
                delta_str = f"{'▲' if delta_pct >= 0 else '▼'} {abs(delta_pct):.1f}%"
                delta_type = "positive" if delta_pct >= 0 else "negative"
            else:
                delta_str = "—"
                delta_type = "neutral"

            display_val = f"{val_a:{fmt}}" if fmt else str(val_a)
            st.markdown(
                ui_metric_card(icon, label, display_val, delta_str, delta_type),
                unsafe_allow_html=True,
            )

    # ── Leyenda de períodos ────────────────────────────────────────────
    st.caption(
        f"**Período A:** {a_start} → {a_end} ({stats_a['videos']} videos) · "
        f"**Período B:** {b_start} → {b_end} ({stats_b['videos']} videos)"
    )

    ui_section_divider()

    # ── Gráfico comparativo ────────────────────────────────────────────
    st.subheader("📈 Evolución Comparada")

    if not period_a.empty or not period_b.empty:
        chart_data = []
        for _, row in period_a.iterrows():
            chart_data.append({
                'Fecha': row['published_at'],
                'Vistas': row['view_count'],
                'Período': f'A ({a_start} → {a_end})',
            })
        for _, row in period_b.iterrows():
            chart_data.append({
                'Fecha': row['published_at'],
                'Vistas': row['view_count'],
                'Período': f'B ({b_start} → {b_end})',
            })

        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            fig = px.scatter(
                chart_df, x='Fecha', y='Vistas', color='Período',
                color_discrete_sequence=[COLORS['primary'], COLORS['secondary']],
                title="Vistas por video — Período A vs B",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # ── Tabla detalle ──────────────────────────────────────────────────
    ui_section_divider()
    st.subheader("📋 Detalle por Tipo de Video")

    detail_rows = []
    for label_p, p_df in [("Período A", period_a), ("Período B", period_b)]:
        if p_df.empty:
            continue
        for vtype in p_df['video_type'].unique():
            vt_df = p_df[p_df['video_type'] == vtype]
            detail_rows.append({
                'Período': label_p,
                'Tipo': vtype,
                'Videos': len(vt_df),
                'Vistas prom.': int(vt_df['view_count'].mean()),
                'Engagement %': round(float(vt_df['engagement_rate'].mean()), 2),
            })

    if detail_rows:
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)


def main():
    """Función principal del dashboard"""

    # ── Sidebar: Marca ────────────────────────────────────────────────
    """Función principal del dashboard"""

    # ── Sidebar: Marca ────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <span class="brand-icon">📊</span>
        <div class="brand-name">YouTube AI Agent</div>
        <div class="brand-tagline">Analytics & Recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Cargar preferencias (15.x) ─────────────────────────────────────
    if 'prefs' not in st.session_state:
        st.session_state.prefs = _load_prefs()

    # Actualizar colores globales según tema
    global COLORS
    COLORS = _get_colors()
    _inject_theme_css()

    # ── Sidebar: Theme toggle (15.4) ───────────────────────────────────
    theme_label = "☀ Claro" if st.session_state.prefs['theme'] == 'dark' else "🌙 Oscuro"
    if st.sidebar.button(theme_label, key="btn_theme_toggle", help="Cambiar tema claro/oscuro"):
        new_theme = 'light' if st.session_state.prefs['theme'] == 'dark' else 'dark'
        st.session_state.prefs['theme'] = new_theme
        _save_prefs(st.session_state.prefs)
        st.rerun()

    # ── Sidebar: Presentation mode (15.2) ──────────────────────────────
    pres_mode = st.session_state.get('presentation_mode', False)
    pres_label = "🖥 Salir Presentación" if pres_mode else "🖥 Modo Presentación"
    if st.sidebar.button(pres_label, key="btn_presentation", help="Modo limpio para reuniones"):
        st.session_state.presentation_mode = not pres_mode
        st.rerun()

    # ── Sidebar: Navegación agrupada ──────────────────────────────────
    st.sidebar.markdown(
        '<div class="nav-group-label">Datos y Análisis</div>',
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Navegación",
        [
            # ── Datos y Análisis
            "🏠 Resumen General",
            "📈 Análisis de Performance",
            PAGE_HISTORICO,
            PAGE_ANALYTICS,
            PAGE_CONTENT,
            PAGE_COMPARE,
            PAGE_HEALTH,
            PAGE_CADENCE,
            PAGE_TEMPORAL,
            PAGE_COMPETITORS,
            # ── Inteligencia Artificial
            "🎯 Recomendaciones",
            "🤖 Generar Nueva Recomendación",
            PAGE_WEEKLY,
            # ── Predicciones
            "🔮 Predicción de Viralidad",
            "👁 Predicción de Vistas",
            PAGE_RETENTION,
            # ── Detectores ML
            PAGE_LATE_BLOOMER,
            PAGE_CANNIBALIZATION,
            PAGE_TRENDS,
        ],
        label_visibility="collapsed",
    )

    ui_section_divider()

    # ── Sidebar: Cargar datos (cacheados 5 min) ──────────────────────
    df_all = load_data()

    if st.sidebar.button("🔄 Actualizar datos", help="Fuerza recarga desde la BD ignorando caché"):
        load_data.clear()
        st.rerun()

    # ── Sidebar: Selector de canal ────────────────────────────────────
    selected_channel_id = None
    df = pd.DataFrame()
    if not df_all.empty:
        channel_map = {
            row['channel_title'] if pd.notna(row['channel_title']) else row['channel_id']: row['channel_id']
            for _, row in df_all.drop_duplicates('channel_id').iterrows()
        }
        channel_names = list(channel_map.keys())

        if len(channel_names) > 1:
            selected_name = st.sidebar.selectbox("📺 Canal", channel_names)
        else:
            selected_name = channel_names[0]
            st.sidebar.markdown(f"""
            <div style="background: var(--bg-tertiary); border-radius: var(--radius-sm);
                        padding: 0.6rem 0.8rem; margin: 0.5rem 0; border: 1px solid var(--border);">
                <div style="font-size: 0.7rem; color: var(--text-muted);
                            text-transform: uppercase; letter-spacing: 0.05em;">Canal activo</div>
                <div style="font-size: 0.95rem; font-weight: 600;
                            color: var(--text-primary); margin-top: 2px;">📺 {_esc(selected_name)}</div>
            </div>
            """, unsafe_allow_html=True)

        selected_channel_id = channel_map[selected_name]
        df = df_all[df_all['channel_id'] == selected_channel_id]

    # ── Sidebar: Footer ───────────────────────────────────────────────
    st.sidebar.markdown("""
    <div class="sidebar-footer">
        <div class="version">YouTube AI Agent v2.0</div>
        <div class="powered-by">Powered by <span>Claude AI</span> & Streamlit</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Page Router ───────────────────────────────────────────────────
    if page == "🏠 Resumen General":
        show_overview(df)

    elif page == "📈 Análisis de Performance":
        show_performance_analysis(df)

    elif page == PAGE_CONTENT:
        show_content_analysis(df)

    elif page == PAGE_COMPARE:
        show_channel_comparison(df_all)

    elif page == PAGE_HISTORICO:
        show_metrics_history(selected_channel_id)

    elif page == PAGE_ANALYTICS:
        show_advanced_analytics(selected_channel_id)

    elif page == PAGE_HEALTH:
        show_channel_health(df, selected_channel_id)

    elif page == PAGE_CADENCE:
        show_cadence_analysis(df, selected_channel_id)

    elif page == PAGE_TEMPORAL:
        show_temporal_comparison(df)

    elif page == PAGE_COMPETITORS:
        show_competitor_analysis(df, selected_channel_id)

    elif page == "🎯 Recomendaciones":
        show_recommendations(selected_channel_id)

    elif page == PAGE_WEEKLY:
        show_weekly_plan(df, selected_channel_id)

    elif page == "🔮 Predicción de Viralidad":
        show_virality_prediction(df, selected_channel_id)

    elif page == "👁 Predicción de Vistas":
        show_view_prediction(df, selected_channel_id)

    elif page == PAGE_RETENTION:
        show_retention_prediction(df, selected_channel_id)

    elif page == PAGE_LATE_BLOOMER:
        show_late_bloomer_analysis(df, selected_channel_id)

    elif page == PAGE_CANNIBALIZATION:
        show_cannibalization_analysis(df, selected_channel_id)

    elif page == PAGE_TRENDS:
        show_trends_analysis()

    elif page == "🤖 Generar Nueva Recomendación":
        show_generate_new_recommendation()


if __name__ == "__main__":
    main()
