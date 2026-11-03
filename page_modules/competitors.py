"""
Página: Análisis de Competencia (Mejora 7.1)
"""
import os
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st

from database import YouTubeDatabase
from ai_analyzer import AIAnalyzer
from page_modules.shared import (
    _get_colors, _esc, ui_page_header, ui_section_divider, ui_metric_card,
    CHART_COLORS,
)


def show_competitor_analysis(df: pd.DataFrame, channel_id: str | None):
    """Página de análisis comparativo contra canales competidores."""
    ui_page_header(
        "🕵", "Análisis de Competencia",
        "Compara tu canal contra la competencia directa — frecuencia, engagement y content gaps"
    )

    if not channel_id or df.empty:
        st.warning("Selecciona un canal con datos en el panel lateral.")
        return

    COLORS = _get_colors()

    # ── Cargar datos de competidores ────────────────────────────────────
    try:
        db = YouTubeDatabase()
        competitor_channels = db.get_competitor_channels()
    except Exception as e:
        st.error(f"Error al conectar con la base de datos: {e}")
        return

    if competitor_channels.empty:
        st.info(
            "**No hay competidores configurados.**\n\n"
            "Para activar esta funcionalidad:\n"
            "1. Agrega `COMPETITOR_CHANNEL_IDS=UCxxxx,UCyyyy` en tu archivo `.env`\n"
            "2. Ejecuta `python main.py` para extraer los datos de competidores\n"
            "3. Vuelve a esta página para ver el análisis"
        )
        db.close()
        return

    # Cargar videos propios y de competidores
    own_channel_ids = [channel_id]
    comp_channel_ids = competitor_channels['channel_id'].tolist()
    all_ids = own_channel_ids + comp_channel_ids

    all_videos_df = db.get_all_videos_with_competitor_flag(all_ids)
    db.close()

    if all_videos_df.empty:
        st.warning("No hay videos disponibles para analizar.")
        return

    own_videos = all_videos_df[all_videos_df['is_competitor'] == 0]
    comp_videos = all_videos_df[all_videos_df['is_competitor'] == 1]

    own_name = (
        own_videos.iloc[0]['channel_title']
        if not own_videos.empty and pd.notna(own_videos.iloc[0].get('channel_title'))
        else "Tu canal"
    )

    # ── Sección 1: Tabla comparativa ──────────────────────────────────
    ui_section_divider("Benchmarking")
    st.subheader("📊 Tu Canal vs Competidores")

    # Construir stats por canal
    rows = []
    for cid in all_ids:
        ch_df = all_videos_df[all_videos_df['channel_id'] == cid]
        if ch_df.empty:
            continue
        ch_name = ch_df.iloc[0]['channel_title'] if pd.notna(ch_df.iloc[0].get('channel_title')) else cid
        is_comp = int(ch_df.iloc[0].get('is_competitor', 0))
        subs = int(ch_df.iloc[0].get('subscriber_count', 0)) if pd.notna(ch_df.iloc[0].get('subscriber_count')) else 0

        # Últimos 30 días
        recent = ch_df[ch_df['published_at'] >= (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=30))]

        rows.append({
            'Canal': f"📺 {ch_name}" if not is_comp else f"🕵 {ch_name}",
            'Tipo': 'Propio' if not is_comp else 'Competidor',
            'Suscriptores': subs,
            'Videos totales': len(ch_df),
            'Videos (30d)': len(recent),
            'Vistas prom.': int(ch_df['view_count'].mean()) if not ch_df['view_count'].isna().all() else 0,
            'Engagement %': round(float(ch_df['engagement_rate'].mean()), 2) if not ch_df['engagement_rate'].isna().all() else 0,
            'Shorts %': round(float(ch_df['is_short'].sum()) / max(len(ch_df), 1) * 100, 1),
            '_channel_id': cid,
            '_avg_views': float(ch_df['view_count'].mean()) if not ch_df['view_count'].isna().all() else 0,
        })

    if not rows:
        st.warning("No hay datos suficientes.")
        return

    bench_df = pd.DataFrame(rows)

    # KPI cards comparativos
    own_row = bench_df[bench_df['Tipo'] == 'Propio']
    comp_rows = bench_df[bench_df['Tipo'] == 'Competidor']

    if not own_row.empty and not comp_rows.empty:
        own_avg_views = own_row.iloc[0]['Vistas prom.']
        comp_avg_views = comp_rows['Vistas prom.'].mean()
        own_eng = own_row.iloc[0]['Engagement %']
        comp_avg_eng = comp_rows['Engagement %'].mean()
        own_freq = own_row.iloc[0]['Videos (30d)']
        comp_avg_freq = comp_rows['Videos (30d)'].mean()

        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1:
            delta_views = ((own_avg_views / comp_avg_views - 1) * 100) if comp_avg_views > 0 else 0
            delta_str = f"{delta_views:+.1f}% vs competidores"
            st.markdown(ui_metric_card(
                "👁", "Vistas Promedio", f"{own_avg_views:,}",
                delta=delta_str,
                delta_type="positive" if delta_views >= 0 else "negative",
            ), unsafe_allow_html=True)
        with kc2:
            delta_eng = own_eng - comp_avg_eng
            st.markdown(ui_metric_card(
                "💬", "Engagement", f"{own_eng:.2f}%",
                delta=f"{delta_eng:+.2f}pp vs comp.",
                delta_type="positive" if delta_eng >= 0 else "negative",
            ), unsafe_allow_html=True)
        with kc3:
            delta_freq = own_freq - comp_avg_freq
            st.markdown(ui_metric_card(
                "📅", "Publicaciones (30d)", f"{own_freq}",
                delta=f"{delta_freq:+.0f} vs comp.",
                delta_type="positive" if delta_freq >= 0 else "negative",
            ), unsafe_allow_html=True)
        with kc4:
            st.markdown(ui_metric_card(
                "🕵", "Competidores", f"{len(comp_rows)}",
                delta="rastreados",
                delta_type="neutral",
            ), unsafe_allow_html=True)

    # Tabla
    display_df = bench_df.drop(columns=['_channel_id', '_avg_views'])
    display_df['Suscriptores'] = display_df['Suscriptores'].apply(lambda v: f"{v:,}")
    display_df['Vistas prom.'] = display_df['Vistas prom.'].apply(lambda v: f"{v:,}")
    display_df['Engagement %'] = display_df['Engagement %'].apply(lambda v: f"{v:.2f}%")
    display_df['Shorts %'] = display_df['Shorts %'].apply(lambda v: f"{v:.1f}%")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Sección 2: Gráficos comparativos ──────────────────────────────
    ui_section_divider("Gráficos Comparativos")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Vistas promedio por canal")
        fig_views = px.bar(
            bench_df, x='Canal', y='Vistas prom.',
            color='Tipo',
            color_discrete_map={'Propio': COLORS['primary'], 'Competidor': COLORS['secondary']},
            text='Vistas prom.',
        )
        fig_views.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_views.update_layout(showlegend=True, height=360,
                                margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
        st.plotly_chart(fig_views, use_container_width=True)

    with col_g2:
        st.subheader("Engagement promedio por canal")
        fig_eng = px.bar(
            bench_df, x='Canal', y='Engagement %',
            color='Tipo',
            color_discrete_map={'Propio': COLORS['primary'], 'Competidor': COLORS['secondary']},
            text='Engagement %',
        )
        fig_eng.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_eng.update_layout(showlegend=True, height=360,
                              margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
        st.plotly_chart(fig_eng, use_container_width=True)

    # ── Sección 3: Crecimiento relativo de vistas (últimos 30 videos) ─
    ui_section_divider("Crecimiento de Vistas Relativo")
    st.subheader("📈 Crecimiento de vistas — normalizado")
    st.caption(
        "Cada canal se normaliza a 100 en su primer video del periodo, "
        "para comparar crecimiento relativo sin importar tamaño absoluto."
    )

    growth_traces = []
    for cid in all_ids:
        ch_df = all_videos_df[all_videos_df['channel_id'] == cid].sort_values('published_at')
        if ch_df.empty or ch_df['view_count'].isna().all():
            continue
        ch_name = ch_df.iloc[0]['channel_title'] if pd.notna(ch_df.iloc[0].get('channel_title')) else cid
        # Tomar últimos 30 videos
        recent_vids = ch_df.tail(30).copy()
        if recent_vids.empty:
            continue
        # Normalizar: primer video = 100
        first_views = recent_vids.iloc[0]['view_count']
        if first_views and first_views > 0:
            recent_vids['normalized'] = (recent_vids['view_count'] / first_views) * 100
        else:
            recent_vids['normalized'] = 0
        recent_vids['video_num'] = range(1, len(recent_vids) + 1)
        recent_vids['canal'] = ch_name
        growth_traces.append(recent_vids[['video_num', 'normalized', 'canal']])

    if growth_traces:
        growth_df = pd.concat(growth_traces, ignore_index=True)
        fig_growth = px.line(
            growth_df, x='video_num', y='normalized', color='canal',
            labels={'video_num': 'Video #', 'normalized': 'Vistas (normalizado a 100)', 'canal': 'Canal'},
        )
        fig_growth.update_layout(height=400, margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
        st.plotly_chart(fig_growth, use_container_width=True)
    else:
        st.caption("No hay datos suficientes para el gráfico de crecimiento.")

    # ── Sección 4: Nube de palabras (títulos) ─────────────────────────
    ui_section_divider("Análisis de Títulos")
    st.subheader("☁ Palabras más usadas en títulos")

    col_wc1, col_wc2 = st.columns(2)

    # Stopwords comunes en español para filtrar
    _stopwords = {
        'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'del', 'un', 'una',
        'con', 'por', 'para', 'que', 'es', 'se', 'no', 'su', 'al', 'lo',
        'como', 'más', 'o', 'mi', 'si', 'ya', 'te', 'me', 'esto', 'esta',
        'the', 'and', 'to', 'of', 'in', 'is', 'for', 'on', 'it', 'with',
        'this', 'that', 'are', 'was', 'you', 'your', 'my', 'i', 'we',
        'shorts', 'short', 'video', 'ft', 'vs',
    }

    def _title_word_counts(titles_series, top_n=20):
        """Extrae las top N palabras de una serie de títulos."""
        all_words = []
        for title in titles_series.dropna():
            words = re.findall(r'[a-záéíóúñü]+', str(title).lower())
            all_words.extend(w for w in words if len(w) > 2 and w not in _stopwords)
        return Counter(all_words).most_common(top_n)

    with col_wc1:
        st.markdown(f"**📺 {_esc(own_name)}**")
        own_words = _title_word_counts(own_videos['title'])
        if own_words:
            own_wc_df = pd.DataFrame(own_words, columns=['Palabra', 'Frecuencia'])
            fig_own_wc = px.bar(
                own_wc_df, x='Frecuencia', y='Palabra', orientation='h',
                color_discrete_sequence=[COLORS['primary']],
            )
            fig_own_wc.update_layout(height=420, yaxis={'categoryorder': 'total ascending'},
                                     margin={'l': 0, 'r': 0, 't': 10, 'b': 0})
            st.plotly_chart(fig_own_wc, use_container_width=True)
        else:
            st.caption("Sin datos suficientes")

    with col_wc2:
        st.markdown("**🕵 Competidores (combinados)**")
        comp_words = _title_word_counts(comp_videos['title'])
        if comp_words:
            comp_wc_df = pd.DataFrame(comp_words, columns=['Palabra', 'Frecuencia'])
            fig_comp_wc = px.bar(
                comp_wc_df, x='Frecuencia', y='Palabra', orientation='h',
                color_discrete_sequence=[COLORS['secondary']],
            )
            fig_comp_wc.update_layout(height=420, yaxis={'categoryorder': 'total ascending'},
                                      margin={'l': 0, 'r': 0, 't': 10, 'b': 0})
            st.plotly_chart(fig_comp_wc, use_container_width=True)
        else:
            st.caption("Sin datos suficientes")

    # Detectar palabras que usan los competidores pero NO el canal propio
    own_word_set = {w for w, _ in own_words} if own_words else set()
    comp_word_set = {w for w, _ in comp_words} if comp_words else set()
    gap_words = comp_word_set - own_word_set

    if gap_words:
        st.markdown(
            f"**🔍 Palabras frecuentes en competidores que NO aparecen en tu canal:** "
            f"`{'`, `'.join(sorted(gap_words)[:15])}`"
        )

    # ── Sección 5: Temas de esta semana ───────────────────────────────
    ui_section_divider("Temas Recientes de Competidores")
    st.subheader("📅 Qué están publicando los competidores esta semana")

    week_ago = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=7)
    comp_recent = comp_videos[comp_videos['published_at'] >= week_ago].sort_values('view_count', ascending=False)

    if not comp_recent.empty:
        for _, row in comp_recent.head(15).iterrows():
            ch_name_r = row['channel_title'] if pd.notna(row.get('channel_title')) else ''
            views_r = int(row['view_count']) if pd.notna(row['view_count']) else 0
            eng_r = float(row['engagement_rate']) if pd.notna(row['engagement_rate']) else 0
            vtype_r = row.get('video_type', '')
            vtype_color = COLORS['secondary'] if vtype_r == 'Short' else COLORS['primary']
            st.markdown(
                f"<div style='background:{COLORS['bg_secondary']};border-radius:8px;"
                f"padding:0.8rem 1rem;margin-bottom:0.5rem;"
                f"border-left:3px solid {vtype_color}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div>"
                f"<div style='font-weight:600;color:{COLORS['text_primary']};font-size:0.95rem'>"
                f"{_esc(row['title'])}</div>"
                f"<div style='font-size:0.8rem;color:{COLORS['text_secondary']};margin-top:2px'>"
                f"🕵 {_esc(ch_name_r)} · {_esc(vtype_r)}</div>"
                f"</div>"
                f"<div style='text-align:right;min-width:120px'>"
                f"<div style='font-weight:600;color:{COLORS['text_primary']}'>{views_r:,} vistas</div>"
                f"<div style='font-size:0.8rem;color:{COLORS['text_secondary']}'>"
                f"Eng: {eng_r:.2f}%</div>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Los competidores no han publicado videos en los últimos 7 días.")

    # ── Sección 6: Content Gaps con IA ─────────────────────────────────
    ui_section_divider("Content Gaps con IA")
    st.subheader("🤖 Análisis de Brechas de Contenido")
    st.caption("Claude AI analiza qué temas explotan tus competidores y cuáles son tus oportunidades")

    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key:
        st.error("ANTHROPIC_API_KEY no configurada en .env")
        return

    if st.button("🔍 Generar Análisis de Content Gaps", key="btn_content_gaps"):
        with st.spinner("Claude está analizando a tus competidores..."):
            try:
                analyzer = AIAnalyzer(anthropic_api_key)

                # Preparar datos propios
                own_titles_list = own_videos['title'].dropna().tolist()[:30]
                own_stats = {
                    'total_videos': len(own_videos),
                    'avg_views': float(own_videos['view_count'].mean()) if not own_videos.empty else 0,
                    'avg_engagement': float(own_videos['engagement_rate'].mean()) if not own_videos.empty else 0,
                }

                # Preparar datos de cada competidor
                comp_data_list = []
                for comp_cid in comp_channel_ids:
                    c_vids = comp_videos[comp_videos['channel_id'] == comp_cid]
                    if c_vids.empty:
                        continue
                    c_name = c_vids.iloc[0]['channel_title'] if pd.notna(c_vids.iloc[0].get('channel_title')) else comp_cid
                    c_subs = int(c_vids.iloc[0].get('subscriber_count', 0)) if pd.notna(c_vids.iloc[0].get('subscriber_count')) else 0
                    comp_data_list.append({
                        'name': c_name,
                        'titles': c_vids['title'].dropna().tolist()[:20],
                        'avg_views': float(c_vids['view_count'].mean()),
                        'avg_engagement': float(c_vids['engagement_rate'].mean()),
                        'total_videos': len(c_vids),
                        'subscriber_count': c_subs,
                    })

                result = analyzer.analyze_competitor_gaps(
                    own_channel_name=own_name,
                    own_titles=own_titles_list,
                    own_stats=own_stats,
                    competitor_data=comp_data_list,
                )

                if result and not result.startswith('Error'):
                    st.markdown(result)
                else:
                    st.error(f"No se pudo generar el análisis: {result}")
            except Exception as e:
                st.error(f"Error al generar análisis: {e}")

    # ── Sección 7: Alertas de Videos Virales (7.2) ────────────────────
    ui_section_divider("Alertas de Videos Virales")
    st.subheader("🚨 Videos de Competidores con Rendimiento Excepcional")
    st.caption("Videos que superaron 2x el promedio de vistas de su canal")

    try:
        db_alerts = YouTubeDatabase()
        alerts_df = db_alerts.get_competitor_alerts(limit=20)
        db_alerts.close()
    except Exception as e:
        st.error(f"Error al cargar alertas: {e}")
        alerts_df = pd.DataFrame()

    if alerts_df.empty:
        st.info(
            "**Sin alertas todavía.** Las alertas se generan automáticamente cuando "
            "`python main.py` detecta un video de competidor con >2x su promedio de vistas."
        )
    else:
        for _, alert_row in alerts_df.iterrows():
            ratio_val = float(alert_row['ratio']) if pd.notna(alert_row.get('ratio')) else 0
            views_val = int(alert_row['view_count']) if pd.notna(alert_row.get('view_count')) else 0
            avg_val = float(alert_row['competitor_avg_views']) if pd.notna(alert_row.get('competitor_avg_views')) else 0
            a_title = _esc(str(alert_row.get('video_title', ''))[:80])
            a_channel = _esc(str(alert_row.get('channel_name', '')))
            a_date = str(alert_row.get('created_at', ''))[:10]
            notified_icon = "📨" if alert_row.get('notified') else "🔕"

            # Color del badge por ratio
            if ratio_val >= 5:
                badge_color = "#ef4444"
            elif ratio_val >= 3:
                badge_color = "#f97316"
            else:
                badge_color = "#eab308"

            st.markdown(
                f"<div style='background:{COLORS['bg_secondary']};border-radius:8px;"
                f"padding:1rem;margin-bottom:0.75rem;"
                f"border-left:4px solid {badge_color}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:start'>"
                f"<div>"
                f"<div style='font-weight:600;color:{COLORS['text_primary']};font-size:0.95rem'>"
                f"{a_title}</div>"
                f"<div style='font-size:0.8rem;color:{COLORS['text_secondary']};margin-top:2px'>"
                f"🕵 {a_channel} · {a_date} {notified_icon}</div>"
                f"</div>"
                f"<div style='text-align:right;min-width:140px'>"
                f"<div style='font-weight:700;color:{badge_color};font-size:1.1rem'>"
                f"{ratio_val:.1f}x</div>"
                f"<div style='font-size:0.8rem;color:{COLORS['text_secondary']}'>"
                f"{views_val:,} vistas (prom: {avg_val:,.0f})</div>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )

            # Expandir análisis de Claude
            ai_text = alert_row.get('ai_analysis', '')
            if ai_text and str(ai_text).strip():
                with st.expander(f"🤖 Ver análisis — {a_title[:40]}..."):
                    st.markdown(str(ai_text))
