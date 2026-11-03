"""
Página: Series y Formatos (Mejora 17.x)
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database import YouTubeDatabase
from page_modules.shared import (
    _get_colors, _esc, ui_page_header, ui_section_divider, ui_section_header,
    ui_metric_card,
)


def show_series_analysis(df: pd.DataFrame, channel_id: str = None):
    """Muestra análisis de series detectadas y recomendaciones de formato."""
    COLORS = _get_colors()
    ui_page_header("📚", "Series y Formatos",
                   "Series detectadas automáticamente con métricas por episodio y recomendaciones de formato")

    # Cargar series
    try:
        with YouTubeDatabase() as db:
            series_df = db.get_all_series(channel_id)
    except Exception as e:
        st.error(f"Error cargando series: {e}")
        return

    if series_df.empty:
        st.info("No se han detectado series aún. Ejecuta el pipeline para detectarlas automáticamente.")
        return

    # ── KPIs ──────────────────────────────────────────────────────
    total_series = len(series_df)
    total_episodes = int(series_df['episode_count'].sum()) if 'episode_count' in series_df.columns else 0
    growing = len(series_df[series_df['trend'] == 'growing']) if 'trend' in series_df.columns else 0
    declining = len(series_df[series_df['trend'] == 'declining']) if 'trend' in series_df.columns else 0

    cols = st.columns(4)
    with cols[0]:
        st.markdown(ui_metric_card("📚", "Series detectadas", str(total_series)),
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(ui_metric_card("🎬", "Total episodios", str(total_episodes)),
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(ui_metric_card("📈", "En crecimiento", str(growing),
                                   delta_type="positive"),
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(ui_metric_card("📉", "En declive", str(declining),
                                   delta_type="negative"),
                    unsafe_allow_html=True)

    ui_section_divider()

    # ── Lista de series ───────────────────────────────────────────
    ui_section_header("📋", "Series detectadas",
                      "Ordenadas por número de episodios")

    for _, series_row in series_df.iterrows():
        trend = series_row.get('trend', 'stable')
        trend_icon = {'growing': '📈', 'declining': '📉', 'stable': '➡️'}.get(trend, '➡️')

        series_name = _esc(series_row.get('series_name', 'Sin nombre'))
        ep_count = series_row.get('episode_count', 0)
        avg_views = series_row.get('avg_views', 0)
        avg_eng = series_row.get('avg_engagement', 0)
        pattern = series_row.get('detected_pattern', '')
        pattern_label = 'Numeración' if pattern == 'numbering' else 'Similitud NLP'

        with st.expander(
            f"{trend_icon} **{series_name}** — {ep_count} episodios | "
            f"Promedio: {avg_views:,.0f} vistas | {avg_eng:.2f}% eng | "
            f"Detección: {pattern_label}",
            expanded=False,
        ):
            # Métricas de la serie
            scols = st.columns(3)
            with scols[0]:
                st.metric("Episodios", ep_count)
            with scols[1]:
                st.metric("Vistas promedio", f"{avg_views:,.0f}")
            with scols[2]:
                st.metric("Tendencia", trend.capitalize())

            # Gráfico de episodios
            series_id = series_row.get('series_id')
            if series_id:
                try:
                    with YouTubeDatabase() as db:
                        episodes_df = db.get_series_episodes(int(series_id))
                except Exception:
                    episodes_df = pd.DataFrame()

                if not episodes_df.empty and 'view_count' in episodes_df.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=episodes_df['episode_number'],
                        y=episodes_df['view_count'],
                        mode='lines+markers',
                        name='Vistas',
                        line=dict(color=COLORS.get('primary', '#6366F1'), width=3),
                        marker=dict(size=10),
                    ))

                    if 'engagement_rate' in episodes_df.columns:
                        fig.add_trace(go.Bar(
                            x=episodes_df['episode_number'],
                            y=episodes_df['engagement_rate'],
                            name='Engagement %',
                            marker_color=COLORS.get('secondary', '#EC4899'),
                            opacity=0.4,
                            yaxis='y2',
                        ))

                    fig.update_layout(
                        title=f"Rendimiento por episodio — {series_name}",
                        xaxis_title="Episodio",
                        yaxis_title="Vistas",
                        yaxis2=dict(
                            title="Engagement %",
                            overlaying='y',
                            side='right',
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=COLORS.get('text_primary', '#F8FAFC')),
                        height=350,
                        margin=dict(l=40, r=40, t=50, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Tabla de episodios
                    display_cols = ['episode_number', 'title', 'view_count', 'engagement_rate']
                    display_cols = [c for c in display_cols if c in episodes_df.columns]
                    st.dataframe(
                        episodes_df[display_cols].rename(columns={
                            'episode_number': 'Episodio',
                            'title': 'Título',
                            'view_count': 'Vistas',
                            'engagement_rate': 'Engagement %',
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

            # Recomendación AI
            ai_rec = series_row.get('ai_recommendation', '')
            if ai_rec and str(ai_rec).strip():
                st.markdown(f"""
                <div style="background:{COLORS.get('bg_secondary', '#1E293B')};
                            border-left:4px solid {COLORS.get('primary', '#6366F1')};
                            padding:1rem; border-radius:0.5rem; margin-top:0.5rem;">
                    <strong>🤖 Recomendación de formato:</strong><br>
                    {_esc(str(ai_rec))}
                </div>
                """, unsafe_allow_html=True)

    # ── Resumen de tendencias ─────────────────────────────────────
    ui_section_divider("Resumen de tendencias")

    if 'trend' in series_df.columns:
        trend_counts = series_df['trend'].value_counts().reset_index()
        trend_counts.columns = ['Tendencia', 'Cantidad']
        trend_map = {'growing': 'En crecimiento', 'declining': 'En declive', 'stable': 'Estable'}
        trend_counts['Tendencia'] = trend_counts['Tendencia'].map(
            lambda t: trend_map.get(t, t)
        )
        color_map = {
            'En crecimiento': COLORS.get('success', '#10B981'),
            'En declive': COLORS.get('danger', '#EF4444'),
            'Estable': COLORS.get('warning', '#F59E0B'),
        }
        fig2 = px.pie(
            trend_counts, names='Tendencia', values='Cantidad',
            color='Tendencia', color_discrete_map=color_map,
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS.get('text_primary', '#F8FAFC')),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)
