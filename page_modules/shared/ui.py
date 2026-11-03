"""
Componentes UI reutilizables del dashboard.
"""
import html as html_mod
import streamlit as st


def _esc(value) -> str:
    """Escapa HTML en cualquier valor para prevenir XSS."""
    return html_mod.escape(str(value)) if value is not None else ''


def ui_section_divider(label: str = None):
    """Divisor de sección estilizado."""
    if label:
        st.markdown(
            f'<div class="section-divider with-label"><span>{_esc(label)}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def ui_section_header(icon: str, title: str, subtitle: str = ""):
    """Encabezado de sección con icono."""
    sub_html = f'<div class="section-subtitle">{_esc(subtitle)}</div>' if subtitle else ''
    st.markdown(f"""
    <div class="section-header">
        <div class="section-icon">{icon}</div>
        <div>
            <div class="section-title">{_esc(title)}</div>
            {sub_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def ui_page_header(icon: str, title: str, description: str = ""):
    """Encabezado principal de página."""
    desc_html = f'<div class="page-description">{_esc(description)}</div>' if description else ''
    st.markdown(f"""
    <div class="page-title-bar">
        <div class="page-icon">{icon}</div>
        <div>
            <div class="page-title">{_esc(title)}</div>
            {desc_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def ui_metric_card(icon: str, label: str, value: str, delta: str = "", delta_type: str = "neutral"):
    """Tarjeta métrica HTML personalizada."""
    delta_html = f'<div class="metric-delta {delta_type}">{_esc(delta)}</div>' if delta else ''
    return f"""
    <div class="metric-card">
        <span class="metric-icon">{icon}</span>
        <div class="metric-label">{_esc(label)}</div>
        <div class="metric-value">{_esc(value)}</div>
        {delta_html}
    </div>
    """
