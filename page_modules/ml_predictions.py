"""
Páginas de predicciones ML: Retención (12.1), Late Bloomer (12.3), Canibalización (12.4)
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from database import YouTubeDatabase
from retention_predictor import RetentionPredictor
from late_bloomer_detector import LateBloomerDetector
from cannibalization_detector import CannibalizationDetector
from page_modules.shared import (
    _get_colors, _esc, ui_page_header, ui_section_divider,
    _get_retention_predictor, CHART_COLORS,
)
from virality_predictor import WEEKDAY_LABELS


# ══════════════════════════════════════════════════════════════════════
# Predicción de Retención (Mejora 12.1)
# ══════════════════════════════════════════════════════════════════════

def show_retention_prediction(df: pd.DataFrame, channel_id: str):
    """Predicción de retención de audiencia con ML."""
    ui_page_header("📊", "Predicción de Retención",
                   "Modelo ML para estimar el % de video visto por la audiencia")

    if df.empty or not channel_id:
        st.warning("No hay datos disponibles para este canal.")
        return

    COLORS = _get_colors()

    # Cargar datos de analytics
    try:
        db = YouTubeDatabase()
        analytics_df = db.get_video_analytics(channel_id)
        db.close()
    except Exception:
        analytics_df = pd.DataFrame()

    if analytics_df.empty or 'avg_view_percentage' not in analytics_df.columns:
        st.warning(
            "⚠ Se requieren datos de YouTube Analytics API para entrenar este modelo. "
            "Ejecuta `python main.py` con `credentials.json` configurado."
        )
        st.info(
            "Este modelo predice la retención de audiencia (% del video visto). "
            "Necesita datos de la Analytics API que incluyan `avg_view_percentage`."
        )
        return

    predictor, from_cache = _get_retention_predictor(df, analytics_df, channel_id)

    if not predictor.is_trained():
        reason = predictor.get_train_metrics().get('reason', 'No se pudo entrenar.')
        st.warning(f"⚠ {reason}")
        return

    metrics = predictor.get_train_metrics()
    n_splits = metrics.get('cv_splits', '?')

    if from_cache:
        st.info(f"Modelo cargado desde caché — {metrics.get('samples', '?')} videos.")
    else:
        st.success(f"Modelo reentrenado con **{metrics.get('samples', '?')} videos** · {n_splits}-fold CV temporal.")

    # ── Métricas de precisión ─────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Videos usados", f"{metrics['samples']}",
                   help=f"CV temporal con {n_splits} folds.")
    with col2:
        st.metric("MAE — CV", f"{metrics.get('cv_mae', 0):.1f}%",
                   help="Error absoluto medio: cuántos puntos porcentuales se desvía en promedio.")
    with col3:
        cv_r2 = metrics.get('cv_r2', 0)
        st.metric("R² — CV", f"{cv_r2:.3f}",
                   help="Coeficiente de determinación. 1.0 = predicción perfecta, 0 = igual al promedio.")

    # ── Scores de videos ──────────────────────────────────────────────
    ui_section_divider()
    st.subheader("🎯 Retención Predicha por Video")

    merged = df.merge(
        analytics_df[['video_id', 'avg_view_percentage']],
        on='video_id', how='left',
    )
    scored_df = predictor.predict(merged)

    # Guardar en DB
    try:
        db2 = YouTubeDatabase()
        db2.save_retention_predictions(
            scored_df[['video_id', 'channel_id', 'predicted_retention']].dropna(subset=['predicted_retention'])
        )
        db2.close()
    except Exception:
        pass

    display_cols = ['title', 'video_type', 'avg_view_percentage', 'predicted_retention']
    available_cols = [c for c in display_cols if c in scored_df.columns]
    top_df = scored_df[available_cols].sort_values('predicted_retention', ascending=False).head(20).copy()
    if 'avg_view_percentage' in top_df.columns:
        top_df['avg_view_percentage'] = top_df['avg_view_percentage'].apply(
            lambda v: f"{v:.1f}%" if pd.notna(v) and v > 0 else '—'
        )
    top_df['predicted_retention'] = top_df['predicted_retention'].apply(lambda v: f"{v:.1f}%")

    st.dataframe(
        top_df.rename(columns={
            'title': 'Título', 'video_type': 'Tipo',
            'avg_view_percentage': 'Retención Real',
            'predicted_retention': 'Retención Predicha',
        }),
        use_container_width=True, hide_index=True,
    )

    # ── Feature importance ────────────────────────────────────────────
    ui_section_divider()
    importance = predictor.get_feature_importance()
    if importance:
        st.subheader("📊 Importancia de Features")
        imp_df = pd.DataFrame(
            sorted(importance.items(), key=lambda x: x[1], reverse=True),
            columns=['Feature', 'Importancia'],
        )
        fig = px.bar(imp_df, x='Importancia', y='Feature', orientation='h',
                     color_discrete_sequence=[COLORS["primary"]])
        fig.update_layout(yaxis={'autorange': 'reversed'}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ── What-If Simulator ─────────────────────────────────────────────
    ui_section_divider()
    st.subheader("🔬 Simulador What-If: Retención")

    wi_col1, wi_col2, wi_col3 = st.columns(3)
    with wi_col1:
        wi_hour = st.slider("Hora de publicación", 0, 23, 12, key="ret_hour")
        wi_weekday = st.selectbox("Día", list(range(7)),
                                  format_func=lambda d: WEEKDAY_LABELS[d], key="ret_day")
        wi_is_short = st.toggle("Es Short", key="ret_short")
        wi_duration = st.number_input("Duración (seg)", 15, 7200, 600, key="ret_dur")
        wi_tags = st.slider("Número de tags", 0, 30, 10, key="ret_tags")
    with wi_col2:
        wi_title_len = st.slider("Longitud título", 10, 100, 60, key="ret_tlen")
        wi_has_num = st.toggle("Título con número", key="ret_num")
        wi_has_q = st.toggle("Título con pregunta", key="ret_q")
        wi_desc_len = st.slider("Longitud descripción", 0, 5000, 500, key="ret_desc")
        wi_days = st.slider("Días desde última subida", 1, 60, 7, key="ret_days")
    with wi_col3:
        wi_hook = st.toggle("Título con hook (cómo, secreto, truco)", key="ret_hook")
        wi_tutorial = st.toggle("Es tutorial", key="ret_tut")
        wi_entertain = st.toggle("Es entretenimiento", key="ret_ent")

    pred_ret = predictor.predict_single(
        hour=wi_hour, weekday=wi_weekday, is_short=wi_is_short,
        duration_seconds=wi_duration, tags_count=wi_tags,
        title_length=wi_title_len, title_has_number=int(wi_has_num),
        title_has_question=int(wi_has_q), description_length=wi_desc_len,
        days_since_last_upload=wi_days, has_hook_title=int(wi_hook),
        is_tutorial=int(wi_tutorial), is_entertainment=int(wi_entertain),
    )

    color = RetentionPredictor.score_color(pred_ret)
    st.markdown(
        f"<div style='text-align:center; padding:1.5rem; background:{COLORS['bg_tertiary']}; "
        f"border-radius:12px; border:2px solid {color};'>"
        f"<div style='font-size:3rem; font-weight:bold; color:{color};'>{pred_ret:.1f}%</div>"
        f"<div style='color:{COLORS['text_secondary']};'>Retención estimada</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# Late Bloomer Detection (Mejora 12.3)
# ══════════════════════════════════════════════════════════════════════

def show_late_bloomer_analysis(df: pd.DataFrame, channel_id: str):
    """Análisis de videos con patrón de despegue tardío."""
    ui_page_header("🌱", "Despegue Tardío",
                   "Detecta videos que crecen lentamente al inicio pero despegan después")

    if df.empty or not channel_id:
        st.warning("No hay datos disponibles.")
        return

    COLORS = _get_colors()

    try:
        db = YouTubeDatabase()
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return

    try:
        detector = LateBloomerDetector()
        results = detector.analyze_channel(channel_id, db)

        if not results:
            st.info(
                "No hay suficientes snapshots de métricas para analizar patrones. "
                "Ejecuta `python main.py` al menos 3 veces en días diferentes para acumular datos."
            )
            return

        # ── Alertas recientes ─────────────────────────────────────────────
        alerts = detector.get_recent_alerts(results, days_lookback=30)

        if alerts:
            st.subheader("🚨 Alertas — Videos Recientes con Despegue Tardío")
            for alert in alerts:
                st.success(
                    f"🌱 **{_esc(alert['title'])}** — "
                    f"Aceleración: **{alert['acceleration_factor']:.1f}x** · "
                    f"Vistas primeras 48h: {alert['early_views']:,} → "
                    f"Vistas actuales: {alert['current_views']:,}\n\n"
                    f"_Este video muestra un patrón SEO-driven. No lo descartes._"
                )
            ui_section_divider()

        # ── Tabla completa ────────────────────────────────────────────────
        st.subheader("📋 Análisis de Patrones de Crecimiento")

        table_data = []
        for r in results:
            label = LateBloomerDetector.PATTERN_LABELS_ES.get(r['pattern'], r['pattern'])
            table_data.append({
                'Título': r['title'],
                'Tipo': r.get('video_type', ''),
                'Publicado': r['published_at'][:10] if r['published_at'] else '',
                'Patrón': label,
                'Aceleración': f"{r['acceleration_factor']:.1f}x",
                'Vistas 48h': f"{r['early_views']:,}",
                'Vistas Actual': f"{r['current_views']:,}",
                'Snapshots': r['total_snapshots'],
            })

        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

        # ── Curva de crecimiento de un video ──────────────────────────────
        ui_section_divider()
        st.subheader("📈 Curva de Crecimiento")

        video_options = {r['title'][:60]: r['video_id'] for r in results if r['total_snapshots'] >= 3}
        if video_options:
            selected_title = st.selectbox("Selecciona un video", list(video_options.keys()),
                                          key="lb_video_select")
            selected_vid = video_options[selected_title]

            hist_df = db.get_video_metrics_history(selected_vid)
            if not hist_df.empty:
                hist_df = hist_df.sort_values('recorded_at')
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist_df['recorded_at'], y=hist_df['view_count'],
                    mode='lines+markers', name='Vistas',
                    line=dict(color=COLORS["primary"], width=2),
                ))
                fig.update_layout(
                    title="Evolución de vistas en el tiempo",
                    xaxis_title="Fecha", yaxis_title="Vistas",
                    template="youtube_ai",
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── Estadísticas de patrones ──────────────────────────────────────
        ui_section_divider()
        st.subheader("📊 Distribución de Patrones")

        pattern_counts = {}
        for r in results:
            label = LateBloomerDetector.PATTERN_LABELS_ES.get(r['pattern'], r['pattern'])
            pattern_counts[label] = pattern_counts.get(label, 0) + 1

        if pattern_counts:
            pc_df = pd.DataFrame(list(pattern_counts.items()), columns=['Patrón', 'Videos'])
            fig = px.pie(pc_df, values='Videos', names='Patrón',
                         color_discrete_sequence=CHART_COLORS)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            late_pct = sum(1 for r in results if r['is_late_bloomer']) / len(results) * 100
            if late_pct > 20:
                st.info(
                    f"📊 **{late_pct:.0f}%** de tus videos analizados muestran despegue tardío. "
                    "Esto sugiere que tu contenido tiene buena discoverability via búsqueda (SEO)."
                )
    finally:
        db.close()


# ══════════════════════════════════════════════════════════════════════
# Cannibalization Detection (Mejora 12.4)
# ══════════════════════════════════════════════════════════════════════

def show_cannibalization_analysis(df: pd.DataFrame, channel_id: str):
    """Análisis de canibalización de contenido entre videos."""
    ui_page_header("🔀", "Canibalización de Contenido",
                   "Detecta videos con títulos/tags similares que compiten por la misma audiencia")

    if df.empty or not channel_id:
        st.warning("No hay datos disponibles.")
        return

    COLORS = _get_colors()

    # ── Controles ─────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        threshold = st.slider(
            "Umbral de similitud", 0.50, 1.00, 0.80, 0.05,
            help="Solo se muestran pares con similitud ≥ este valor.",
            key="canib_threshold",
        )
    with col_ctrl2:
        max_days = st.slider(
            "Máx. días entre publicaciones", 1, 180, 30,
            help="Solo pares publicados dentro de este rango de días.",
            key="canib_days",
        )

    detector = CannibalizationDetector(threshold=threshold, max_days_apart=max_days)
    results = detector.detect(df)

    if not results:
        st.success(
            "✅ No se detectó canibalización con los parámetros actuales. "
            "Tu diversidad de contenido es buena."
        )
        return

    # ── Alertas recientes ─────────────────────────────────────────────
    alerts = detector.get_recent_alerts(results, days_lookback=60)
    if alerts:
        st.subheader("🚨 Alertas de Canibalización Reciente")
        for a in alerts[:5]:
            severity_color = COLORS["danger"] if a['similarity'] >= 0.90 else COLORS["warning"]
            terms = ', '.join(a['shared_terms'][:3]) if a['shared_terms'] else '—'
            st.markdown(
                f"<div style='padding:0.8rem; margin:0.5rem 0; border-left:4px solid {severity_color}; "
                f"background:{COLORS['bg_tertiary']}; border-radius:0 8px 8px 0;'>"
                f"<strong>{_esc(a['title_a'])}</strong><br>"
                f"<strong>{_esc(a['title_b'])}</strong><br>"
                f"<span style='color:{severity_color}; font-weight:bold;'>"
                f"Similitud: {a['similarity']:.0%}</span> · "
                f"{a['days_apart']} días de diferencia · "
                f"Términos: {_esc(terms)}</div>",
                unsafe_allow_html=True,
            )
        ui_section_divider()

    # ── Tabla completa ────────────────────────────────────────────────
    st.subheader(f"📋 {len(results)} pares detectados")

    table_data = []
    for r in results:
        table_data.append({
            'Video A': r['title_a'][:50],
            'Video B': r['title_b'][:50],
            'Similitud': f"{r['similarity']:.0%}",
            'Días': r['days_apart'],
            'Términos Comunes': ', '.join(r['shared_terms'][:3]),
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    # ── Heatmap de similitud ──────────────────────────────────────────
    ui_section_divider()
    st.subheader("🗺 Mapa de Similitud (últimos 30 videos)")

    recent_df = df.sort_values('published_at', ascending=False).head(30).reset_index(drop=True)
    if len(recent_df) >= 2:
        corpus = CannibalizationDetector._build_text_corpus(recent_df)
        vectorizer = TfidfVectorizer(max_features=5000)
        try:
            tfidf_m = vectorizer.fit_transform(corpus)
            sim_m = cosine_similarity(tfidf_m)

            labels = [t[:30] for t in recent_df['title'].tolist()]
            fig = px.imshow(
                sim_m, x=labels, y=labels,
                color_continuous_scale='YlOrRd',
                zmin=0, zmax=1,
                labels=dict(color='Similitud'),
            )
            fig.update_layout(height=600, xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        except ValueError:
            st.info("No se pudo generar el mapa (datos insuficientes).")

    # ── Recomendaciones ───────────────────────────────────────────────
    ui_section_divider()
    if results:
        st.info(
            "💡 **Recomendación:** Considera espaciar temas similares al menos 30 días. "
            "Videos sobre el mismo tema compiten por las mismas búsquedas y pueden "
            "dividir la audiencia potencial."
        )
