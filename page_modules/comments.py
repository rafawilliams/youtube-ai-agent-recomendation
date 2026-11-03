"""
Página: Comentarios y Audiencia (Mejora 11.x)
"""
import plotly.graph_objects as go
import streamlit as st

import pandas as pd

from database import YouTubeDatabase
from page_modules.shared import (
    _get_colors, _esc, ui_page_header, ui_section_divider, ui_section_header,
    ui_metric_card,
)


def show_comments_analysis(df: pd.DataFrame, channel_id: str = None):
    """Muestra análisis de comentarios, sentimiento, top fans y FAQ (11.x)."""
    COLORS = _get_colors()
    ui_page_header("💬", "Comentarios y Audiencia",
                   "Análisis de sentimiento, fans más activos y preguntas frecuentes")

    if not channel_id:
        st.info("Selecciona un canal para ver el análisis de comentarios.")
        return

    # ── Cargar datos ───────────────────────────────────────────────
    try:
        with YouTubeDatabase() as db:
            comment_stats = db.get_channel_comment_stats(channel_id)
            top_commenters = db.get_top_commenters(channel_id, top_n=20)
            faq_topics = db.get_faq_topics(channel_id)
            sentiment_time = db.get_comment_sentiment_over_time(channel_id)
    except Exception as e:
        st.error(f"Error cargando datos de comentarios: {e}")
        return

    total = comment_stats.get('total_comments', 0)
    if total == 0:
        st.info("No se han analizado comentarios aún. Ejecuta el pipeline para extraer y analizar comentarios.")
        return

    # ── KPIs ───────────────────────────────────────────────────────
    positive = comment_stats.get('positive', 0) or 0
    negative = comment_stats.get('negative', 0) or 0
    neutral = comment_stats.get('neutral', 0) or 0
    classified = positive + negative + neutral

    cols = st.columns(4)
    with cols[0]:
        st.markdown(ui_metric_card("💬", "Total comentarios", f"{total:,}"),
                    unsafe_allow_html=True)
    with cols[1]:
        pct_pos = f"{positive / classified * 100:.1f}%" if classified > 0 else "—"
        st.markdown(ui_metric_card("😊", "Positivos", pct_pos,
                                   delta=f"{positive:,}", delta_type="positive"),
                    unsafe_allow_html=True)
    with cols[2]:
        pct_neg = f"{negative / classified * 100:.1f}%" if classified > 0 else "—"
        st.markdown(ui_metric_card("😡", "Negativos", pct_neg,
                                   delta=f"{negative:,}", delta_type="negative"),
                    unsafe_allow_html=True)
    with cols[3]:
        pct_neu = f"{neutral / classified * 100:.1f}%" if classified > 0 else "—"
        st.markdown(ui_metric_card("😐", "Neutros", pct_neu,
                                   delta=f"{neutral:,}"),
                    unsafe_allow_html=True)

    ui_section_divider()

    # ── Distribución de sentimiento (donut) ────────────────────────
    if classified > 0:
        col_chart, col_fans = st.columns([1, 1])

        with col_chart:
            ui_section_header("📊", "Distribución de Sentimiento")
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Positivo', 'Negativo', 'Neutro'],
                values=[positive, negative, neutral],
                hole=0.55,
                marker=dict(colors=[COLORS['success'], COLORS['danger'], COLORS['warning']]),
                textinfo='label+percent',
                textfont=dict(size=13),
            )])
            fig_donut.update_layout(
                template='youtube_ai',
                height=350,
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # ── Top Fans (11.1) ─────────────────────────────────────────
        with col_fans:
            ui_section_header("⭐", "Top 20 Fans",
                              "Comentaristas más activos del canal")
            if not top_commenters.empty:
                display_df = top_commenters[['author_display_name', 'comment_count',
                                             'total_likes']].copy()
                display_df.columns = ['Nombre', 'Comentarios', 'Likes recibidos']
                display_df.index = range(1, len(display_df) + 1)
                st.dataframe(display_df, use_container_width=True, height=350)
            else:
                st.info("No hay datos de comentaristas aún.")

    ui_section_divider()

    # ── Sentimiento temporal (11.2) ────────────────────────────────
    if not sentiment_time.empty:
        ui_section_header("📈", "Tendencia de Sentimiento",
                          "Evolución mensual del sentimiento en comentarios")

        fig_sent = go.Figure()
        fig_sent.add_trace(go.Scatter(
            x=sentiment_time['month'], y=sentiment_time['positive'],
            name='Positivo', mode='lines+markers',
            line=dict(color=COLORS['success'], width=2),
            fill='tozeroy', fillcolor=f"rgba(16,185,129,0.1)",
        ))
        fig_sent.add_trace(go.Scatter(
            x=sentiment_time['month'], y=sentiment_time['negative'],
            name='Negativo', mode='lines+markers',
            line=dict(color=COLORS['danger'], width=2),
        ))
        fig_sent.add_trace(go.Scatter(
            x=sentiment_time['month'], y=sentiment_time['neutral'],
            name='Neutro', mode='lines+markers',
            line=dict(color=COLORS['warning'], width=2),
            opacity=0.6,
        ))
        fig_sent.update_layout(
            template='youtube_ai',
            height=400,
            xaxis_title="Mes",
            yaxis_title="Cantidad de comentarios",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_sent, use_container_width=True)

        ui_section_divider()

    # ── FAQ detectadas (11.3) ──────────────────────────────────────
    if not faq_topics.empty:
        ui_section_header("❓", "Preguntas Frecuentes Detectadas",
                          "Temas y preguntas recurrentes de tu audiencia")

        for _, topic_row in faq_topics.iterrows():
            topic_name = topic_row.get('topic', '')
            freq = topic_row.get('frequency', 0)
            samples = topic_row.get('sample_questions', '')

            with st.expander(f"**{_esc(topic_name)}** — {freq} menciones"):
                if samples:
                    for q in str(samples).split(';'):
                        q = q.strip()
                        if q:
                            st.markdown(f"- _{_esc(q)}_")
    elif classified > 0:
        st.info("No se detectaron preguntas frecuentes. Ejecuta el pipeline con más videos.")
