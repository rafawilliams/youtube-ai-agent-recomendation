"""
Páginas de Analytics Avanzados y Tendencias de Temas del dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database import YouTubeDatabase
from trends_analyzer import TrendsAnalyzer, GEO_OPTIONS, TIMEFRAME_OPTIONS
from page_modules.shared import (
    _get_colors, ui_page_header, ui_section_divider,
    VIDEO_TYPE_COLORS, CHART_COLORS,
)


def show_advanced_analytics(channel_id: str):
    """Página de Analytics Avanzados — datos de YouTube Analytics API (OAuth)."""
    COLORS = _get_colors()
    ui_page_header("📊", "Analytics Avanzados", "Datos de YouTube Analytics API (OAuth)")

    if not channel_id:
        st.warning("Selecciona un canal en el panel lateral.")
        return

    try:
        db = YouTubeDatabase()
        analytics_df = db.get_video_analytics(channel_id)
        traffic_df   = db.get_traffic_sources(channel_id)
        db.close()
    except Exception as e:
        st.error(f"Error cargando analytics: {e}")
        return

    # ── Sin datos: mostrar instrucciones de configuración ────────────────
    if analytics_df.empty and traffic_df.empty:
        st.info(
            "**No hay datos de Analytics Avanzados para este canal.**\n\n"
            "Para habilitar esta sección:\n"
            "1. En [Google Cloud Console](https://console.cloud.google.com) "
            "habilita la **YouTube Analytics API**.\n"
            "2. Crea credenciales **OAuth 2.0 → Aplicación de escritorio**.\n"
            "3. Descarga el JSON y guárdalo como **`credentials.json`** en la "
            "raíz del proyecto.\n"
            "4. Ejecuta `python main.py` — el paso 4 abrirá el navegador para "
            "autorizar el acceso y guardará los datos en la base de datos."
        )
        return

    # ── Sección A: KPIs ──────────────────────────────────────────────────
    st.subheader("📌 Métricas Clave")
    col1, col2, col3, col4 = st.columns(4)

    if not analytics_df.empty:
        avg_retention = analytics_df['avg_view_percentage'].mean()
        total_shares  = int(analytics_df['shares'].sum())
        total_subs    = int(analytics_df['subscribers_gained'].sum())

        has_ctr = analytics_df['impression_ctr'].sum() > 0
        avg_ctr = analytics_df['impression_ctr'].mean() if has_ctr else None

        col1.metric("Retención media", f"{avg_retention:.1f}%")
        col2.metric("CTR promedio", f"{avg_ctr:.2f}%" if avg_ctr else "N/D",
                    help="Porcentaje de impresiones que se convirtieron en clic")
        col3.metric("Veces compartido", f"{total_shares:,}")
        col4.metric("Suscriptores ganados", f"{total_subs:,}")
    else:
        st.info("Sin datos de métricas por video.")

    ui_section_divider()

    # ── Sección B: Retención por video (top 20) ──────────────────────────
    if not analytics_df.empty:
        st.subheader("🔁 Retención media por video (Top 20)")

        top20 = analytics_df.nlargest(20, 'avg_view_percentage').copy()
        top20['title_short'] = top20['title'].str[:55]

        fig_ret = px.bar(
            top20.sort_values('avg_view_percentage'),
            x='avg_view_percentage',
            y='title_short',
            color='video_type',
            orientation='h',
            labels={
                'avg_view_percentage': '% Promedio visto',
                'title_short': 'Video',
                'video_type': 'Tipo',
            },
            color_discrete_map=VIDEO_TYPE_COLORS,
        )
        fig_ret.update_layout(height=520, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_ret, use_container_width=True)

        ui_section_divider()

        # ── Sección C: Scatter Vistas vs Retención ───────────────────────
        st.subheader("🔭 Alcance vs Retención")
        st.caption("Cuadrante superior derecho = videos con muchas vistas Y buena retención.")

        scatter_df = analytics_df.merge(
            analytics_df[['video_id']].assign(
                view_count=analytics_df.get('views', analytics_df.get('views', 0))
            ),
            on='video_id', how='left'
        ) if 'views' not in analytics_df.columns else analytics_df.copy()

        if 'views' in analytics_df.columns:
            fig_sc = px.scatter(
                analytics_df,
                x='views',
                y='avg_view_percentage',
                color='video_type',
                hover_name='title',
                labels={
                    'views': 'Vistas reales',
                    'avg_view_percentage': '% Retencion media',
                    'video_type': 'Tipo',
                },
                color_discrete_map=VIDEO_TYPE_COLORS,
                log_x=True,
            )
            fig_sc.update_layout(height=420)
            st.plotly_chart(fig_sc, use_container_width=True)

        ui_section_divider()

        # ── Sección D: CTR por video (si hay datos) ──────────────────────
        if analytics_df['impression_ctr'].sum() > 0:
            st.subheader("🖼 CTR de miniatura (Top 20 con más impresiones)")

            top_ctr = (
                analytics_df[analytics_df['impressions'] > 0]
                .nlargest(20, 'impressions')
                .copy()
            )
            top_ctr['title_short'] = top_ctr['title'].str[:55]

            fig_ctr = px.bar(
                top_ctr.sort_values('impression_ctr'),
                x='impression_ctr',
                y='title_short',
                color='video_type',
                orientation='h',
                labels={
                    'impression_ctr': 'CTR (%)',
                    'title_short': 'Video',
                    'video_type': 'Tipo',
                },
                color_discrete_map=VIDEO_TYPE_COLORS,
            )
            fig_ctr.update_layout(height=520, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_ctr, use_container_width=True)
            ui_section_divider()

    # ── Sección E: Fuentes de tráfico ────────────────────────────────────
    if not traffic_df.empty:
        st.subheader("🚦 Fuentes de tráfico")

        col_pie, col_table = st.columns([1, 1])

        with col_pie:
            fig_pie = px.pie(
                traffic_df,
                names='source_label',
                values='views',
                hole=0.4,
                color_discrete_sequence=CHART_COLORS,
            )
            fig_pie.update_layout(height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_table:
            total_views = traffic_df['views'].sum()
            display_df = traffic_df[['source_label', 'views', 'estimated_minutes']].copy()
            display_df['% del total'] = (display_df['views'] / total_views * 100).round(1)
            display_df.columns = ['Fuente', 'Vistas', 'Minutos vistos', '% Total']
            st.dataframe(display_df, use_container_width=True, hide_index=True)


def show_trends_analysis():
    """Análisis de tendencias de temas vía Google Trends (pytrends)."""
    COLORS = _get_colors()
    ui_page_header("📊", "Tendencias de Temas", "Compara popularidad de tus ideas en Google Trends")
    st.caption("Compara la popularidad de tus ideas de contenido en Google antes de publicar.")

    if not TrendsAnalyzer.is_available():
        st.error("❌ **pytrends no está instalado.**")
        st.code("pip install pytrends>=4.9.0")
        return

    # ── Configuración ────────────────────────────────────────────────
    st.subheader("🔍 Temas a comparar")
    st.info("Ingresa entre 1 y 5 temas para comparar su popularidad en Google Trends.")

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        geo_label = st.selectbox("🌍 País / Región", list(GEO_OPTIONS.keys()), index=0)
    with col_cfg2:
        tf_label = st.selectbox("📅 Período", list(TIMEFRAME_OPTIONS.keys()), index=1)

    geo = GEO_OPTIONS[geo_label]
    timeframe = TIMEFRAME_OPTIONS[tf_label]

    # Hasta 5 keyword inputs dinámicos
    if 'trends_n_kw' not in st.session_state:
        st.session_state.trends_n_kw = 2

    kw_cols = st.columns(5)
    keywords = []
    for i in range(st.session_state.trends_n_kw):
        val = kw_cols[i % 5].text_input(
            f"Tema {i + 1}", key=f"trends_kw_{i}",
            placeholder="ej: cocina panameña",
        )
        keywords.append(val)

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
    with btn_col1:
        if st.button("➕ Agregar tema") and st.session_state.trends_n_kw < 5:
            st.session_state.trends_n_kw += 1
            st.rerun()
    with btn_col2:
        if st.button("➖ Quitar") and st.session_state.trends_n_kw > 1:
            st.session_state.trends_n_kw -= 1
            keywords = keywords[:-1]
            st.rerun()

    keywords = [k.strip() for k in keywords if k.strip()]
    if not keywords:
        st.warning("Ingresa al menos un tema para analizar.")
        return

    if st.button("🔍 Analizar tendencias", type="primary"):
        st.session_state.trends_result = None   # limpiar caché previa
        with st.spinner("Consultando Google Trends… (puede tardar unos segundos)"):
            try:
                analyzer = TrendsAnalyzer()
                iot = analyzer.get_interest_over_time(keywords, geo=geo, timeframe=timeframe)
                scores = analyzer.get_trend_scores(keywords, geo=geo, timeframe=timeframe)
                related = analyzer.get_related_queries(keywords[0], geo=geo) if len(keywords) >= 1 else {}
                peak_days = TrendsAnalyzer.peak_day(iot) if not iot.empty else {}
                st.session_state.trends_result = {
                    'iot': iot,
                    'scores': scores,
                    'related': related,
                    'peak_days': peak_days,
                    'keywords': keywords,
                    'geo_label': geo_label,
                    'tf_label': tf_label,
                }
            except Exception as e:
                err = str(e)
                if '429' in err or 'Too Many Requests' in err:
                    st.error("⚠️ **Google Trends bloqueó la solicitud (demasiadas consultas).** Espera unos minutos e intenta de nuevo.")
                else:
                    st.error(f"Error al consultar Google Trends: {err}")
                return

    result = st.session_state.get('trends_result')
    if result is None:
        return

    iot        = result['iot']
    scores     = result['scores']
    related    = result['related']
    peak_days  = result['peak_days']
    kws        = result['keywords']

    ui_section_divider()

    # ── Sección A: Score promedio actual ─────────────────────────────
    st.subheader("📊 Popularidad promedio en el período")
    st.caption(f"País: **{result['geo_label']}** · Período: **{result['tf_label']}**")

    score_cols = st.columns(len(kws))
    for i, kw in enumerate(kws):
        score = scores.get(kw, 0)
        color = "normal" if score >= 50 else "off"
        score_cols[i].metric(
            label=kw,
            value=f"{score}/100",
            delta="🔥 En tendencia" if score >= 60 else ("📉 Baja demanda" if score < 25 else None),
            delta_color=color,
        )

    # ── Sección B: Interés en el tiempo ──────────────────────────────
    if not iot.empty:
        ui_section_divider()
        st.subheader("📈 Interés a lo largo del tiempo")

        fig_line = px.line(
            iot.reset_index(),
            x='date',
            y=kws,
            labels={'date': 'Fecha', 'value': 'Interés (0–100)', 'variable': 'Tema'},
            color_discrete_sequence=CHART_COLORS,
        )
        fig_line.update_layout(
            height=380,
            margin={'l': 0, 'r': 0, 't': 20, 'b': 0},
            legend={'title': 'Tema'},
            hovermode='x unified',
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Día de la semana con más interés por tema
        if peak_days:
            st.caption("**Día de la semana con mayor interés histórico:**")
            day_cols = st.columns(len(kws))
            for i, kw in enumerate(kws):
                day_cols[i].info(f"**{kw}**\n\n📅 {peak_days.get(kw, '—')}")

    # ── Sección C: Comparación en barra ─────────────────────────────
    ui_section_divider()
    st.subheader("📊 Comparación directa de popularidad")

    scores_df = pd.DataFrame({
        'Tema': list(scores.keys()),
        'Score': list(scores.values()),
    }).sort_values('Score', ascending=True)

    fig_bar = go.Figure(go.Bar(
        x=scores_df['Score'],
        y=scores_df['Tema'],
        orientation='h',
        marker_color=[
            COLORS["success"] if s >= 60 else COLORS["warning"] if s >= 30 else COLORS["danger"]
            for s in scores_df['Score']
        ],
        text=scores_df['Score'].apply(lambda s: f"{s}/100"),
        textposition='outside',
    ))
    fig_bar.update_layout(
        xaxis={'title': 'Popularidad promedio (0=nula, 100=máx)', 'range': [0, 110]},
        height=max(200, 60 * len(kws)),
        margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Sección D: Consultas relacionadas ────────────────────────────
    ui_section_divider()
    st.subheader(f"🔗 Consultas relacionadas con «{kws[0]}»")
    st.caption("Fuente: Google Trends · Período: últimos 3 meses")

    tab_top, tab_rising = st.tabs(["🏆 Más buscadas", "🚀 En ascenso"])

    with tab_top:
        top_df = related.get('top', pd.DataFrame())
        if not top_df.empty and 'query' in top_df.columns:
            top_df = top_df[['query', 'value']].head(15).copy()
            top_df.columns = ['Consulta', 'Popularidad']
            st.dataframe(top_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hay datos de consultas top disponibles.")

    with tab_rising:
        rising_df = related.get('rising', pd.DataFrame())
        if not rising_df.empty and 'query' in rising_df.columns:
            rising_df = rising_df[['query', 'value']].head(15).copy()
            rising_df.columns = ['Consulta', 'Crecimiento']
            rising_df['Crecimiento'] = rising_df['Crecimiento'].apply(
                lambda v: f"+{v}%" if isinstance(v, (int, float)) else str(v)
            )
            st.dataframe(rising_df, use_container_width=True, hide_index=True)
        else:
            st.info("No hay datos de consultas en ascenso disponibles.")

    # ── Sección E: Recomendación de contenido ─────────────────────────
    if scores:
        best_kw = max(scores, key=lambda k: scores[k])
        best_score = scores[best_kw]
        ui_section_divider()
        st.subheader("💡 Recomendación")
        if best_score >= 60:
            st.success(
                f"🚀 **«{best_kw}»** es el tema con mayor demanda actual ({best_score}/100). "
                f"Considera publicar sobre este tema pronto para aprovechar la tendencia."
            )
        elif best_score >= 30:
            st.warning(
                f"📊 **«{best_kw}»** tiene demanda moderada ({best_score}/100). "
                f"El tema funciona, pero no está en su punto máximo."
            )
        else:
            st.error(
                f"⚠️ Ninguno de los temas tiene alta demanda en {result['geo_label']} ahora mismo. "
                f"El más popular es **«{best_kw}»** ({best_score}/100). "
                f"Considera ampliar o cambiar los temas."
            )
