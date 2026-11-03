"""
Sistema de temas (dark/light) del dashboard (Mejora 15.4).
"""
import streamlit as st
from page_modules.shared.constants import _yt_template

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
