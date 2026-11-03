"""
Constantes del dashboard: nombres de página, colores, template Plotly.
"""
import os
import plotly.graph_objects as go
import plotly.io as pio

# ─── Nombres de página ──────────────────────────────────────────────────────
PAGE_HISTORICO          = "📅 Histórico de Métricas"
PAGE_ANALYTICS          = "📊 Analytics Avanzados"
PAGE_TRENDS             = "📊 Tendencias de Temas"
PAGE_CONTENT            = "🔤 Análisis de Contenido"
PAGE_COMPARE            = "🆚 Comparar Canales"
PAGE_WEEKLY             = "🗓 Plan Semanal"
PAGE_HEALTH             = "🏥 Salud del Canal"
PAGE_CADENCE            = "⏱ Cadencia y Horarios"
PAGE_COMPETITORS        = "🕵 Análisis de Competencia"
PAGE_RETENTION          = "📊 Predicción de Retención"
PAGE_LATE_BLOOMER       = "🌱 Despegue Tardío"
PAGE_CANNIBALIZATION    = "🔀 Canibalización"
PAGE_TEMPORAL           = "📆 Comparador Temporal"
PAGE_SERIES             = "📚 Series y Formatos"
PAGE_REVENUE            = "💰 Revenue y ROI"
PAGE_THUMBNAILS         = "🖼 Galería de Miniaturas"
PAGE_THUMBNAIL_AB       = "🔀 Comparador A/B"
PAGE_COMMENTS           = "💬 Comentarios y Audiencia"

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')

# ─── Colores para gráficas ──────────────────────────────────────────────────
CHART_COLORS = [
    "#6366F1", "#EC4899", "#14B8A6", "#F59E0B",
    "#8B5CF6", "#06B6D4", "#F97316", "#84CC16",
]

VIDEO_TYPE_COLORS = {"Short": "#EC4899", "Video Largo": "#6366F1"}
VIDEO_TYPE_SEQUENCE = ["#EC4899", "#6366F1"]

PERF_LABEL_COLORS = {
    "above_average": "#10B981",
    "average":       "#F59E0B",
    "below_average": "#EF4444",
}

# ─── Plotly Template Global ──────────────────────────────────────────────────
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
