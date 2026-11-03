"""
Página: Comparador Temporal (Mejora 15.3)
"""
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from page_modules.shared import (
    _get_colors, ui_page_header, ui_section_divider, ui_metric_card,
)


def show_temporal_comparison(df: pd.DataFrame):
    """Compara métricas entre dos rangos de fecha con deltas visuales."""
    ui_page_header(
        "📆", "Comparador Temporal",
        "Compara métricas entre dos períodos para identificar tendencias"
    )

    if df.empty:
        st.warning("No hay datos disponibles.")
        return

    COLORS = _get_colors()

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
