"""
Páginas de Análisis de Contenido y Comparar Canales del dashboard.
"""
import os
import re
from collections import Counter

import streamlit as st
import pandas as pd
import plotly.express as px

from database import YouTubeDatabase
from content_classifier import ContentClassifier
from page_modules.shared import (
    _get_colors, ui_page_header, ui_section_divider,
    VIDEO_TYPE_SEQUENCE, CHART_COLORS,
)


def _word_freq(titles: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """Frecuencia de palabras en títulos, filtrando stopwords en español."""
    stopwords = {
        'de', 'la', 'el', 'en', 'los', 'las', 'un', 'una', 'y', 'que',
        'con', 'del', 'por', 'para', 'se', 'es', 'mi', 'tu', 'su', 'al',
        'le', 'no', 'si', 'me', 'ya', 'o', 'a', 'e', 'i', 'lo', 'te',
        'nos', 'hay', 'muy', 'mas', 'pero', 'como', 'este', 'esta',
    }
    words = []
    for title in titles.dropna():
        for w in re.findall(r'[a-záéíóúüñ]+', str(title).lower()):
            if w not in stopwords and len(w) > 2:
                words.append(w)
    top = Counter(words).most_common(top_n)
    return pd.DataFrame(top, columns=['Palabra', 'Frecuencia'])


def _tag_freq(tags: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """Frecuencia de tags (separados por coma)."""
    all_tags = []
    for t in tags.dropna():
        for tag in str(t).split(','):
            tag = tag.strip().lower()
            if tag:
                all_tags.append(tag)
    top = Counter(all_tags).most_common(top_n)
    return pd.DataFrame(top, columns=['Tag', 'Frecuencia'])


def show_content_analysis(df: pd.DataFrame):
    """Análisis de palabras en títulos, tags más usados y longitud vs vistas."""
    COLORS = _get_colors()
    ui_page_header("🔤", "Análisis de Contenido", "Palabras, tags y patrones de rendimiento en títulos")

    if df.empty:
        st.warning("No hay datos disponibles.")
        return

    tab_words, tab_tags, tab_length, tab_categories = st.tabs([
        "📝 Palabras en Títulos",
        "🏷️ Tags más usados",
        "📏 Longitud vs Vistas",
        "📂 Categorías",
    ])

    # ── Palabras en títulos ───────────────────────────────────────────
    with tab_words:
        st.caption("Palabras más frecuentes en títulos, separadas por rendimiento.")
        median_views = df['view_count'].median()
        df_high = df[df['view_count'] >= median_views]
        df_low  = df[df['view_count'] <  median_views]

        col_h, col_l = st.columns(2)

        with col_h:
            st.subheader("🔝 Alta performance")
            freq_h = _word_freq(df_high['title'])
            if not freq_h.empty:
                fig = px.bar(
                    freq_h, x='Frecuencia', y='Palabra', orientation='h',
                    color_discrete_sequence=[COLORS["success"]],
                )
                fig.update_layout(
                    height=420, margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
                    yaxis={'categoryorder': 'total ascending'},
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_l:
            st.subheader("📉 Baja performance")
            freq_l = _word_freq(df_low['title'])
            if not freq_l.empty:
                fig = px.bar(
                    freq_l, x='Frecuencia', y='Palabra', orientation='h',
                    color_discrete_sequence=[COLORS["danger"]],
                )
                fig.update_layout(
                    height=420, margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
                    yaxis={'categoryorder': 'total ascending'},
                )
                st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Umbral de performance: **{int(median_views):,} vistas** (mediana del canal).")

    # ── Tags más usados ───────────────────────────────────────────────
    with tab_tags:
        freq_t = _tag_freq(df['tags'])
        if freq_t.empty:
            st.info("No hay tags disponibles en los videos del canal.")
        else:
            fig = px.bar(
                freq_t, x='Frecuencia', y='Tag', orientation='h',
                color_discrete_sequence=[COLORS["primary"]],
            )
            fig.update_layout(
                height=520, margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
                yaxis={'categoryorder': 'total ascending'},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Total de tags únicos: **{len(_tag_freq(df['tags'], top_n=9999))}**")

    # ── Longitud del título vs Vistas ─────────────────────────────────
    with tab_length:
        df_len = df.copy()
        df_len['len_titulo'] = df_len['title'].apply(
            lambda t: len(str(t)) if pd.notna(t) else 0
        )
        df_len['rango_longitud'] = pd.cut(
            df_len['len_titulo'],
            bins=[0, 30, 50, 70, 100, 200],
            labels=['< 30', '30–50', '50–70', '70–100', '> 100'],
        )

        # Promedios por rango
        bucket = (
            df_len.groupby('rango_longitud', observed=True)['view_count']
            .mean().reset_index()
        )
        bucket.columns = ['Longitud', 'Vistas promedio']

        col_b, col_s = st.columns(2)
        with col_b:
            st.subheader("Vistas promedio por rango de longitud")
            fig_b = px.bar(
                bucket, x='Longitud', y='Vistas promedio',
                color_discrete_sequence=[COLORS["secondary"]],
                text='Vistas promedio',
            )
            fig_b.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_b.update_layout(height=320, margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
            st.plotly_chart(fig_b, use_container_width=True)

        with col_s:
            st.subheader("Longitud del título vs Vistas (scatter)")
            fig_s = px.scatter(
                df_len, x='len_titulo', y='view_count', color='video_type',
                hover_data=['title'],
                labels={'len_titulo': 'Longitud del título (chars)', 'view_count': 'Vistas'},
                color_discrete_sequence=VIDEO_TYPE_SEQUENCE,
            )
            fig_s.update_layout(height=320, margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
            st.plotly_chart(fig_s, use_container_width=True)

    # ── Categorías de contenido (Mejora 12.2) ─────────────────────────
    with tab_categories:
        st.caption("Clasificación automática de videos por tipo de contenido y análisis de rendimiento por categoría.")

        # Verificar si hay channel_id disponible
        channel_ids = df['channel_id'].unique()
        if len(channel_ids) == 0:
            st.warning("No hay canal seleccionado.")
        else:
            ch_id = channel_ids[0]
            try:
                db_cat = YouTubeDatabase()

                # Contar clasificados vs no clasificados
                categorized = df[df['content_category'].notna() & (df['content_category'] != '')].copy() if 'content_category' in df.columns else pd.DataFrame()
                uncategorized = db_cat.get_videos_without_category(ch_id)

                col_info1, col_info2 = st.columns(2)
                col_info1.metric("✅ Clasificados", len(categorized))
                col_info2.metric("❓ Sin clasificar", len(uncategorized))

                # Botón para clasificar
                anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')
                if not uncategorized.empty and anthropic_key:
                    if st.button("🤖 Clasificar videos con IA", type="secondary", key="btn_classify"):
                        with st.spinner(f"Clasificando {len(uncategorized)} videos con Claude AI..."):
                            classifier = ContentClassifier(anthropic_key)
                            videos_list = uncategorized.to_dict('records')

                            all_categories: dict[str, str] = {}
                            # Clasificar en lotes de 25
                            for i in range(0, len(videos_list), 25):
                                batch = videos_list[i:i+25]
                                batch_result = classifier.classify_batch(batch)
                                all_categories.update(batch_result)

                            if all_categories:
                                db_cat.save_content_categories(all_categories)
                                st.success(f"✅ {len(all_categories)} videos clasificados")
                                st.rerun()

                # Mostrar análisis por categoría
                perf_df = db_cat.get_performance_by_category(ch_id)

                if not perf_df.empty:
                    ui_section_divider()
                    st.subheader("📊 Rendimiento por Categoría")

                    # Traducir categorías
                    perf_df['categoria'] = perf_df['content_category'].apply(
                        lambda c: f"{ContentClassifier.get_category_icon(c)} {ContentClassifier.get_category_label(c)}"
                    )

                    # Bar chart: vistas promedio
                    fig_views = px.bar(
                        perf_df.sort_values('avg_views', ascending=False),
                        x='categoria', y='avg_views',
                        labels={'categoria': 'Categoría', 'avg_views': 'Vistas Promedio'},
                        color_discrete_sequence=[COLORS["primary"]],
                    )
                    fig_views.update_layout(height=350)
                    st.plotly_chart(fig_views, use_container_width=True)

                    # Bar chart: engagement promedio
                    fig_eng = px.bar(
                        perf_df.sort_values('avg_engagement', ascending=False),
                        x='categoria', y='avg_engagement',
                        labels={'categoria': 'Categoría', 'avg_engagement': 'Engagement Promedio (%)'},
                        color_discrete_sequence=[COLORS["success"]],
                    )
                    fig_eng.update_layout(height=350)
                    st.plotly_chart(fig_eng, use_container_width=True)

                    # Pie chart: distribución
                    fig_pie = px.pie(
                        perf_df, values='video_count', names='categoria',
                        color_discrete_sequence=CHART_COLORS,
                    )
                    fig_pie.update_layout(height=350)
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Insight textual
                    if len(perf_df) >= 2:
                        best = perf_df.sort_values('avg_views', ascending=False).iloc[0]
                        worst = perf_df.sort_values('avg_views').iloc[0]
                        if best['avg_views'] > 0 and worst['avg_views'] > 0:
                            ratio = best['avg_views'] / worst['avg_views']
                            best_label = ContentClassifier.get_category_label(best['content_category'])
                            worst_label = ContentClassifier.get_category_label(worst['content_category'])
                            st.info(
                                f"💡 Tus **{best_label}** obtienen **{ratio:.1f}x** más vistas "
                                f"que tus **{worst_label}** en promedio."
                            )

                elif not categorized.empty:
                    st.info("Recarga los datos del dashboard para ver las categorías recién clasificadas.")

                db_cat.close()
            except Exception as e:
                st.error(f"Error al analizar categorías: {e}")


def show_channel_comparison(df_all: pd.DataFrame):
    """Vista side-by-side de todos los canales con benchmarking de métricas."""
    ui_page_header("🆚", "Comparar Canales", "Benchmarking side-by-side de todos tus canales")

    if df_all.empty:
        st.warning("No hay datos disponibles.")
        return

    channels = df_all['channel_id'].unique()
    if len(channels) < 2:
        st.info(
            "Se necesitan **al menos 2 canales** para comparar.\n\n"
            "Agrega más canales en tu `.env` (`CHANNEL_IDS`) y ejecuta `python main.py`."
        )
        return

    # ── KPIs por canal ────────────────────────────────────────────────
    stats = (
        df_all
        .groupby('channel_title', dropna=False)
        .agg(
            videos         = ('video_id',       'count'),
            total_vistas   = ('view_count',      'sum'),
            avg_vistas     = ('view_count',      'mean'),
            avg_engagement = ('engagement_rate', 'mean'),
            shorts         = ('is_short',        'sum'),
        )
        .reset_index()
    )
    stats['channel_title'] = stats['channel_title'].fillna('Canal sin nombre')

    st.subheader("📊 Resumen por canal")
    kpi_cols = st.columns(len(stats))
    for i, row in stats.iterrows():
        with kpi_cols[i]:
            st.markdown(f"### 📺 {row['channel_title']}")
            st.metric("Videos",          f"{int(row['videos']):,}")
            st.metric("Vistas totales",  f"{int(row['total_vistas']):,}")
            st.metric("Vistas promedio", f"{int(row['avg_vistas']):,}")
            st.metric("Engagement",      f"{row['avg_engagement']:.2f}%")
            st.metric("Shorts",          f"{int(row['shorts'])}")

    # ── Gráficos comparativos ────────────────────────────────────────
    ui_section_divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Vistas promedio por canal")
        fig1 = px.bar(
            stats, x='channel_title', y='avg_vistas',
            color='channel_title',
            labels={'channel_title': 'Canal', 'avg_vistas': 'Vistas promedio'},
            text='avg_vistas',
        )
        fig1.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig1.update_layout(showlegend=False, height=320,
                           margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Engagement promedio por canal")
        fig2 = px.bar(
            stats, x='channel_title', y='avg_engagement',
            color='channel_title',
            labels={'channel_title': 'Canal', 'avg_engagement': 'Engagement %'},
            text='avg_engagement',
        )
        fig2.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig2.update_layout(showlegend=False, height=320,
                           margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
        st.plotly_chart(fig2, use_container_width=True)

    # ── Distribución de vistas por canal ─────────────────────────────
    ui_section_divider()
    st.subheader("Distribución de vistas por canal")
    fig_box = px.box(
        df_all, x='channel_title', y='view_count', color='channel_title',
        labels={'channel_title': 'Canal', 'view_count': 'Vistas'},
    )
    fig_box.update_layout(showlegend=False, height=360,
                          margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Tabla resumen ─────────────────────────────────────────────────
    ui_section_divider()
    st.subheader("Tabla resumen")
    disp = stats.rename(columns={
        'channel_title':  'Canal',
        'videos':         'Videos',
        'total_vistas':   'Vistas totales',
        'avg_vistas':     'Vistas promedio',
        'avg_engagement': 'Engagement %',
        'shorts':         'Shorts',
    })
    disp['Vistas totales']  = disp['Vistas totales'].apply(lambda v: f"{int(v):,}")
    disp['Vistas promedio'] = disp['Vistas promedio'].apply(lambda v: f"{int(v):,}")
    disp['Engagement %']    = disp['Engagement %'].apply(lambda v: f"{v:.2f}%")
    st.dataframe(disp, use_container_width=True, hide_index=True)
