"""
Página de Análisis de Performance del dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from page_modules.shared import (
    _get_colors, ui_page_header, ui_section_divider,
    VIDEO_TYPE_SEQUENCE,
)


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
