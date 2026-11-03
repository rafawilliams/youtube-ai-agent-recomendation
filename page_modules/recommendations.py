"""
Páginas de Recomendaciones de IA del dashboard.
"""
import json
import os
import re
import streamlit as st
import pandas as pd
import plotly.express as px
import pymysql.err

from database import YouTubeDatabase
from ai_analyzer import AIAnalyzer
from youtube_extractor import YouTubeDataExtractor
from trends_analyzer import TrendsAnalyzer
from page_modules.shared import (
    _get_colors, _esc, ui_page_header, ui_section_divider,
    PERF_LABEL_COLORS, _detect_channel_language,
)


def show_recommendations(channel_id=None):
    """Muestra las recomendaciones generadas por IA"""
    COLORS = _get_colors()
    ui_page_header("🎯", "Recomendaciones de IA", "Sugerencias generadas por Claude AI")

    try:
        db = YouTubeDatabase()

        if not channel_id:
            st.warning("Selecciona un canal en el panel lateral.")
            db.close()
            return

        recommendations_df = db.get_recommendations(channel_id, days=30)

        if recommendations_df.empty:
            st.info("No hay recomendaciones disponibles. Ejecuta el script main.py para generar una.")
        else:
            latest = recommendations_df.iloc[0]

            topic    = str(latest['recommended_topic'] or '').strip()
            reasoning = str(latest['reasoning'] or '').strip()

            # ── Cabecera con métricas clave ──────────────────────────────
            st.markdown(f"### 💡 Recomendación para: **{latest['recommendation_date']}**")
            col1, col2 = st.columns(2)
            _fmt = latest['recommended_type'] or '—'
            _aspect = '9:16' if _fmt == 'Short' else '16:9'
            col1.metric("🎬 Formato recomendado", f"{_fmt} ({_aspect})" if _fmt != '—' else '—')
            col2.metric("📊 Performance esperado", latest['predicted_performance'] or '—')

            if topic:
                st.info(f"**📝 Tema sugerido:** {topic}")

            # ── Sugerencias de título ─────────────────────────────────────
            title_suggestions_raw = latest.get('title_suggestions') if hasattr(latest, 'get') else None
            if title_suggestions_raw is None and 'title_suggestions' in latest.index:
                title_suggestions_raw = latest['title_suggestions']

            title_suggestions = []
            if title_suggestions_raw and str(title_suggestions_raw).strip():
                try:
                    title_suggestions = json.loads(title_suggestions_raw)
                except (json.JSONDecodeError, TypeError):
                    pass

            if title_suggestions:
                ui_section_divider()
                st.markdown("**✍ Sugerencias de Título:**")
                for i, sug in enumerate(title_suggestions, 1):
                    title_text = sug.get('title', '')
                    analysis_text = sug.get('analysis', '')
                    st.markdown(f"**{i}.** {title_text}")
                    if analysis_text:
                        st.caption(f"→ {analysis_text}")

            # ── Generador de guion ─────────────────────────────────────
            ui_section_divider()
            rec_id = int(latest['id']) if 'id' in latest.index and pd.notna(latest['id']) else None

            if rec_id:
                existing_outline = db.get_script_outline(rec_id)
            else:
                existing_outline = None

            if existing_outline:
                st.markdown("**📝 Guion / Outline:**")
                st.markdown(existing_outline['outline_text'])
            else:
                first_title = title_suggestions[0]['title'] if title_suggestions else (topic or '—')
                anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')

                if anthropic_key and rec_id:
                    # Detectar idioma del canal y obtener nombre real
                    ch_language_detected = _detect_channel_language(db, channel_id)
                    try:
                        cursor_ch = db.conn.cursor()
                        cursor_ch.execute(
                            "SELECT channel_name FROM channels WHERE channel_id = %s",
                            (channel_id,),
                        )
                        ch_row = cursor_ch.fetchone()
                        ch_display_name = ch_row['channel_name'] if ch_row else channel_id
                    except Exception:
                        ch_display_name = channel_id

                    _lang_opts = {'🇪🇸 Español': 'es', '🇺🇸 English': 'en', '🇧🇷 Português': 'pt'}
                    _lang_labels = list(_lang_opts.keys())
                    _default_label = next(
                        (k for k, v in _lang_opts.items() if v == ch_language_detected),
                        '🇪🇸 Español',
                    )
                    _sel_label = st.selectbox(
                        "Idioma del guion",
                        _lang_labels,
                        index=_lang_labels.index(_default_label),
                        key="script_lang_select",
                    )
                    ch_language = _lang_opts[_sel_label]

                    if st.button("📝 Generar Guion", type="secondary", key="btn_script"):
                        with st.spinner("Generando guion con IA..."):
                            analyzer = AIAnalyzer(anthropic_key)
                            outline = analyzer.generate_script_outline(
                                video_type=latest['recommended_type'] or 'Short',
                                topic=topic or '—',
                                title=first_title,
                                channel_name=ch_display_name,
                                language=ch_language,
                            )
                            if 'error' not in outline:
                                db.save_script_outline(channel_id, rec_id, outline)
                                st.markdown("**📝 Guion / Outline:**")
                                st.markdown(outline['outline_text'])
                                st.success("Guion guardado")
                            else:
                                st.error(f"Error: {outline['error']}")

            # ── Generador de SEO Content (Mejora 9.2 + 9.3) ───────────
            ui_section_divider()

            if rec_id:
                existing_seo = db.get_seo_content(rec_id)
            else:
                existing_seo = None

            if existing_seo:
                st.markdown("**📋 Descripción SEO Optimizada:**")
                st.code(existing_seo['seo_description'], language=None)

                if existing_seo['tags']:
                    st.markdown("**🏷 Tags Optimizados:**")
                    tags_html = ' '.join(
                        f'<span style="display:inline-block; background:{COLORS["bg_tertiary"]}; '
                        f'border:1px solid {COLORS["border"]}; border-radius:12px; '
                        f'padding:2px 10px; margin:2px; font-size:0.85rem; '
                        f'color:{COLORS["text_secondary"]};">{_esc(tag)}</span>'
                        for tag in existing_seo['tags']
                    )
                    st.markdown(tags_html, unsafe_allow_html=True)

                if existing_seo['hashtags']:
                    st.markdown(
                        "**#️⃣ Hashtags:** "
                        + '  '.join(existing_seo['hashtags'])
                    )

                if existing_seo['related_videos']:
                    st.markdown("**🔗 Videos Relacionados Sugeridos:**")
                    for rv in existing_seo['related_videos']:
                        st.markdown(f"- [{_esc(rv.get('title', ''))}]({rv.get('url', '')})")

                # Áreas de texto para copiar fácilmente
                with st.expander("📎 Copiar descripción y tags"):
                    st.text_area(
                        "Descripción SEO (selecciona y copia)",
                        value=existing_seo['seo_description'],
                        height=200,
                        key="seo_desc_copy",
                    )
                    if existing_seo['tags']:
                        st.text_area(
                            "Tags (separados por coma)",
                            value=', '.join(existing_seo['tags']),
                            height=80,
                            key="seo_tags_copy",
                        )
            else:
                seo_first_title = title_suggestions[0]['title'] if title_suggestions else (topic or '—')
                seo_api_key = os.getenv('ANTHROPIC_API_KEY', '')

                if seo_api_key and rec_id:
                    # Detectar idioma y nombre del canal (fuera del spinner para poder mostrar selector)
                    seo_language_detected = _detect_channel_language(db, channel_id)
                    try:
                        cursor_seo = db.conn.cursor()
                        cursor_seo.execute(
                            "SELECT channel_name FROM channels WHERE channel_id = %s",
                            (channel_id,),
                        )
                        seo_ch_row = cursor_seo.fetchone()
                        seo_ch_name = seo_ch_row['channel_name'] if seo_ch_row else channel_id
                    except Exception:
                        seo_ch_name = channel_id

                    _seo_lang_opts = {'🇪🇸 Español': 'es', '🇺🇸 English': 'en', '🇧🇷 Português': 'pt'}
                    _seo_lang_labels = list(_seo_lang_opts.keys())
                    _seo_default = next(
                        (k for k, v in _seo_lang_opts.items() if v == seo_language_detected),
                        '🇪🇸 Español',
                    )
                    _seo_sel = st.selectbox(
                        "Idioma del SEO",
                        _seo_lang_labels,
                        index=_seo_lang_labels.index(_seo_default),
                        key="seo_lang_select",
                    )
                    seo_language = _seo_lang_opts[_seo_sel]

                    if st.button("📋 Generar SEO (Descripción + Tags)", type="secondary", key="btn_seo"):
                        with st.spinner("Generando descripción SEO y tags optimizados..."):
                            seo_analyzer = AIAnalyzer(seo_api_key)

                            # 1. Tags más frecuentes del canal
                            channel_tags = db.get_top_tags_from_channel(channel_id)

                            # 2. Extraer keywords del título/tema
                            raw_keywords = re.findall(
                                r'\b[a-záéíóúñü]{4,}\b',
                                (seo_first_title + ' ' + (topic or '')).lower(),
                            )
                            _stopwords = {
                                'para', 'como', 'este', 'esta', 'estos', 'estas',
                                'sobre', 'cuando', 'donde', 'porque', 'pero',
                                'with', 'from', 'that', 'this', 'your', 'have',
                            }
                            seo_keywords = [w for w in dict.fromkeys(raw_keywords)
                                            if w not in _stopwords][:5]

                            # 3. Videos relacionados del canal
                            related_videos = db.get_related_videos_by_keywords(
                                channel_id, seo_keywords, limit=5,
                            )

                            # 4. Google Trends (opcional, no bloquear si falla)
                            trend_scores = None
                            rising_queries = None
                            if seo_keywords:
                                try:
                                    if TrendsAnalyzer.is_available():
                                        ta = TrendsAnalyzer()
                                        trend_scores = ta.get_trend_scores(
                                            seo_keywords[:5], geo='PA',
                                        )
                                        related_q = ta.get_related_queries(
                                            seo_keywords[0], geo='PA',
                                        )
                                        rising_df = related_q.get('rising', pd.DataFrame())
                                        if not rising_df.empty and 'query' in rising_df.columns:
                                            rising_queries = rising_df['query'].tolist()[:10]
                                except Exception:
                                    pass  # Trends no disponible, continuar sin datos

                            # 5. Generar con Claude
                            seo_result = seo_analyzer.generate_seo_content(
                                video_type=latest['recommended_type'] or 'Short',
                                topic=topic or '—',
                                title=seo_first_title,
                                channel_name=seo_ch_name,
                                channel_tags=channel_tags,
                                related_videos=related_videos,
                                trend_scores=trend_scores,
                                rising_queries=rising_queries,
                                language=seo_language,
                            )

                            if 'error' not in seo_result:
                                db.save_seo_content(channel_id, rec_id, seo_result)
                                st.success("✅ SEO content generado y guardado")
                                st.rerun()
                            else:
                                st.error(f"Error: {seo_result['error']}")

            # ── Análisis completo sin truncar ────────────────────────────
            ui_section_divider()
            st.markdown("**🧠 Análisis Detallado:**")
            st.markdown(reasoning if reasoning else '—')

            with st.expander("Ver historial de recomendaciones"):
                history_df = recommendations_df[['recommendation_date', 'recommended_type', 'predicted_performance', 'created_at']].copy()
                history_df['recommended_type'] = history_df['recommended_type'].apply(
                    lambda t: f"{t} (9:16)" if t == 'Short' else f"{t} (16:9)" if t else t
                )
                history_df['created_at'] = pd.to_datetime(history_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')

                st.dataframe(
                    history_df.rename(columns={
                        'recommendation_date': 'Fecha Objetivo',
                        'recommended_type': 'Tipo',
                        'predicted_performance': 'Performance Esperado',
                        'created_at': 'Generada el'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

        # ── Sección de retroalimentación del ciclo de aprendizaje ────────────
        ui_section_divider()
        st.subheader("🔄 Ciclo de Retroalimentación")
        st.caption(
            "Seguimiento de si las recomendaciones anteriores fueron aplicadas "
            "y cómo rindieron los videos publicados después."
        )

        results_df = db.get_recommendation_results(channel_id, limit=10)

        if results_df.empty:
            st.info(
                "Aún no hay resultados de retroalimentación. "
                "Se generan automáticamente cuando ejecutas **main.py** "
                "y se detectan videos publicados después de una recomendación."
            )
        else:
            # ── KPIs de resumen ───────────────────────────────────────────
            resolved = results_df[results_df['performance_label'].notna()]
            if not resolved.empty:
                above = (resolved['performance_label'] == 'above_average').sum()
                avg   = (resolved['performance_label'] == 'average').sum()
                below = (resolved['performance_label'] == 'below_average').sum()
                followed = resolved['followed_recommendation'].sum() if 'followed_recommendation' in resolved else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("✅ Sobre promedio",  int(above))
                c2.metric("🟡 En el promedio",  int(avg))
                c3.metric("❌ Bajo promedio",   int(below))
                c4.metric("🎯 Siguieron la rec.", int(followed))

            # ── Gráfico de performance_ratio ─────────────────────────────
            chart_df = resolved[resolved['performance_ratio'].notna()].copy()
            if not chart_df.empty:
                chart_df['fecha'] = pd.to_datetime(chart_df['recommendation_date']).dt.strftime('%Y-%m-%d')
                chart_df['color'] = chart_df['performance_label'].map(PERF_LABEL_COLORS).fillna(COLORS["text_muted"])

                fig_ratio = px.bar(
                    chart_df,
                    x='fecha',
                    y='performance_ratio',
                    color='performance_label',
                    color_discrete_map=PERF_LABEL_COLORS,
                    labels={
                        'fecha': 'Recomendación',
                        'performance_ratio': 'Ratio vs Promedio del Canal',
                        'performance_label': 'Resultado',
                    },
                    title='Performance del video publicado vs promedio del canal (por recomendación)',
                )
                fig_ratio.add_hline(y=1.0, line_dash='dash', line_color='white',
                                    annotation_text='Promedio del canal')
                fig_ratio.update_layout(showlegend=True)
                st.plotly_chart(fig_ratio, use_container_width=True)

            # ── Tabla detallada ───────────────────────────────────────────
            with st.expander("Ver tabla de resultados"):
                display_cols = {
                    'recommendation_date': 'Fecha Rec.',
                    'recommended_type':    'Tipo Rec.',
                    'video_type':          'Tipo Publicado',
                    'followed_recommendation': 'Siguió Rec.',
                    'performance_ratio':   'Ratio',
                    'performance_label':   'Resultado',
                    'title':               'Video Publicado',
                }
                show_df = results_df[[c for c in display_cols if c in results_df.columns]].rename(columns=display_cols).copy()
                # Formato de aspecto
                for _col in ('Tipo Rec.', 'Tipo Publicado'):
                    if _col in show_df.columns:
                        show_df[_col] = show_df[_col].apply(
                            lambda t: f"{t} (9:16)" if t == 'Short' else f"{t} (16:9)" if t else t
                        )
                # Formatear booleano
                if 'Siguió Rec.' in show_df.columns:
                    show_df['Siguió Rec.'] = show_df['Siguió Rec.'].map(
                        {1: '✅', 0: '❌'}
                    ).fillna('⏳')
                if 'Ratio' in show_df.columns:
                    show_df['Ratio'] = show_df['Ratio'].apply(
                        lambda v: f"{v:.2f}x" if pd.notna(v) else '—'
                    )
                label_map = {
                    'above_average':  '✅ Sobre promedio',
                    'average':        '🟡 Promedio',
                    'below_average':  '❌ Bajo promedio',
                }
                if 'Resultado' in show_df.columns:
                    show_df['Resultado'] = show_df['Resultado'].map(label_map).fillna('⏳ Pendiente')

                st.dataframe(show_df, use_container_width=True, hide_index=True)

        db.close()

    except pymysql.err.Error as e:
        st.error(f"Error de BD cargando recomendaciones: {e}")


def show_generate_new_recommendation():
    """Interfaz para generar nueva recomendación"""
    ui_page_header("🤖", "Generar Nueva Recomendación", "Crea una recomendación personalizada con IA")

    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

    if not anthropic_api_key:
        st.error("ANTHROPIC_API_KEY no configurada. Por favor configura tu API key en el archivo .env")
        return

    try:
        db = YouTubeDatabase()
        videos_df = db.get_all_videos()

        if videos_df.empty:
            st.warning("No hay datos disponibles. Ejecuta primero main.py para extraer datos de YouTube.")
            db.close()
            return

        channel_ids = videos_df['channel_id'].unique()

        # Selector de canal
        channel_options = {}
        for channel_id in channel_ids:
            channel_name = videos_df[videos_df['channel_id'] == channel_id].iloc[0]['channel_title']
            channel_options[channel_name] = channel_id

        selected_channel_name = st.selectbox(
            "Selecciona el canal",
            options=list(channel_options.keys())
        )

        selected_channel_id = channel_options[selected_channel_name]

        if st.button("🚀 Generar Recomendación", type="primary"):
            with st.spinner("Analizando datos y generando recomendación con IA..."):
                # Filtrar videos del canal seleccionado
                channel_videos = videos_df[videos_df['channel_id'] == selected_channel_id]

                # Actualizar subscriber_count y video_count desde YouTube API
                channel_info = {
                    'channel_id': selected_channel_id,
                    'channel_name': selected_channel_name,
                    'subscriber_count': 0
                }
                youtube_api_key = os.getenv('YOUTUBE_API_KEY')
                if youtube_api_key:
                    try:
                        extractor = YouTubeDataExtractor(youtube_api_key)
                        fresh_info = extractor.get_channel_info(selected_channel_id)
                        if fresh_info:
                            db.save_channel_data(fresh_info)
                            channel_info['subscriber_count'] = fresh_info['subscriber_count']
                    except Exception:
                        pass  # Si falla la actualización, continúa con los datos locales

                # Vincular videos recientes + recuperar historial de retroalimentación
                channel_avg_views = float(channel_videos['view_count'].mean()) if not channel_videos.empty else 0.0
                db.link_video_to_recommendation(selected_channel_id, channel_avg_views)
                past_results_df = db.get_recommendation_results(selected_channel_id, limit=5)
                past_results = past_results_df.to_dict('records') if not past_results_df.empty else []

                # Generar recomendación
                analyzer = AIAnalyzer(anthropic_api_key)
                recommendation = analyzer.generate_daily_recommendation(
                    channel_videos, channel_info, past_results=past_results
                )

                if 'error' not in recommendation:
                    # Guardar en base de datos
                    db.save_recommendation(selected_channel_id, recommendation)
                    db.save_recommendation_result(
                        selected_channel_id,
                        recommendation['recommendation_date'],
                        recommendation['recommended_type'],
                    )

                    st.success("✅ Recomendación generada exitosamente!")

                    st.markdown(f"""
                    <div class="success-box">
                        <h3>🎯 Nueva Recomendación Generada</h3>
                        <p><strong>📅 Para el día:</strong> {_esc(recommendation['recommendation_date'])}</p>
                        <p><strong>🎬 Formato:</strong> {_esc(recommendation['recommended_type'])} ({'9:16' if recommendation['recommended_type'] == 'Short' else '16:9'})</p>
                        <p><strong>📊 Performance Esperado:</strong> {_esc(recommendation['predicted_performance'])}</p>
                        <hr>
                        <p><strong>💡 Análisis Completo:</strong></p>
                        <p style="white-space: pre-wrap;">{_esc(recommendation['reasoning'])}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Mostrar sugerencias de título
                    title_suggestions = recommendation.get('title_suggestions', [])
                    if title_suggestions:
                        st.markdown("### ✍ Sugerencias de Título")
                        for i, sug in enumerate(title_suggestions, 1):
                            st.markdown(f"**{i}.** {sug.get('title', '')}")
                            analysis = sug.get('analysis', '')
                            if analysis:
                                st.caption(f"→ {analysis}")
                else:
                    st.error(f"Error: {recommendation['error']}")

        db.close()

    except Exception as e:
        st.error(f"Error: {str(e)}")
