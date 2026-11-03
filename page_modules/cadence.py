"""
Página: Cadencia y Horarios (Mejoras 13.2 / 13.3)
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from ai_analyzer import AIAnalyzer
from page_modules.shared import (
    _get_colors, ui_page_header, ui_section_divider,
    CHART_COLORS,
)
from virality_predictor import WEEKDAY_LABELS


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

    COLORS = _get_colors()

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
