"""
Página de Histórico de Métricas del dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymysql.err

from database import YouTubeDatabase
from page_modules.shared import (
    _get_colors, ui_page_header, ui_section_divider,
)


def show_metrics_history(channel_id: str):
    """Evolución histórica de métricas del canal a lo largo de las ejecuciones."""
    COLORS = _get_colors()
    ui_page_header("📅", "Histórico de Métricas", "Evolución del canal a lo largo del tiempo")

    if not channel_id:
        st.warning("Selecciona un canal en el panel lateral.")
        return

    try:
        with YouTubeDatabase() as db:
            history_df = db.get_channel_metrics_history(channel_id)
            growth_df = db.get_top_videos_growth(channel_id, top_n=10)
    except pymysql.err.Error as e:
        st.error(f"Error de BD cargando histórico: {e}")
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
