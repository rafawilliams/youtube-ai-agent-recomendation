"""
Dashboard interactivo para visualizar análisis y recomendaciones.

Orchestrador delgado: la lógica de cada página vive en pages/*.py,
las utilidades compartidas en pages/shared/.
"""
import streamlit as st
import pandas as pd
import os
import sys

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv

# Shared utilities
from page_modules.shared.constants import (
    PAGE_HISTORICO, PAGE_ANALYTICS, PAGE_TRENDS, PAGE_CONTENT, PAGE_COMPARE,
    PAGE_WEEKLY, PAGE_HEALTH, PAGE_CADENCE, PAGE_COMPETITORS, PAGE_RETENTION,
    PAGE_LATE_BLOOMER, PAGE_CANNIBALIZATION, PAGE_TEMPORAL, PAGE_SERIES,
    PAGE_REVENUE, PAGE_THUMBNAILS, PAGE_THUMBNAIL_AB, PAGE_COMMENTS,
)
from page_modules.shared.theme import _get_colors, _inject_theme_css
from page_modules.shared.ui import _esc, ui_section_divider
from page_modules.shared.data import load_data, _load_prefs, _save_prefs

# Page functions
from page_modules import (
    show_overview, show_performance_analysis,
    show_virality_prediction, show_view_prediction,
    show_metrics_history,
    show_recommendations, show_generate_new_recommendation,
    show_advanced_analytics, show_trends_analysis,
    show_content_analysis, show_channel_comparison,
    show_weekly_plan,
    show_channel_health,
    show_cadence_analysis,
    show_competitor_analysis,
    show_retention_prediction, show_late_bloomer_analysis,
    show_cannibalization_analysis,
    show_temporal_comparison,
    show_series_analysis,
    show_revenue_analysis,
    show_thumbnail_gallery, show_thumbnail_comparator,
    show_comments_analysis,
)

# ─── Configuración de la página ──────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube AI Agent Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv(override=True)

# ─── CSS Stylesheet Profesional ──────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Función principal del dashboard."""

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
            PAGE_SERIES,
            PAGE_REVENUE,
            # ── Thumbnails / Vision
            PAGE_THUMBNAILS,
            PAGE_THUMBNAIL_AB,
            # ── Audiencia
            PAGE_COMMENTS,
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
    elif page == PAGE_SERIES:
        show_series_analysis(df, selected_channel_id)
    elif page == PAGE_REVENUE:
        show_revenue_analysis(df, selected_channel_id)
    elif page == PAGE_THUMBNAILS:
        show_thumbnail_gallery(df, selected_channel_id)
    elif page == PAGE_THUMBNAIL_AB:
        show_thumbnail_comparator(df, selected_channel_id)
    elif page == PAGE_COMMENTS:
        show_comments_analysis(df, selected_channel_id)
    elif page == PAGE_TRENDS:
        show_trends_analysis()
    elif page == "🤖 Generar Nueva Recomendación":
        show_generate_new_recommendation()


if __name__ == "__main__":
    main()
