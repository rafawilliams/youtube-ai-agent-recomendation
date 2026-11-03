"""
Páginas de predicción de viralidad y vistas del dashboard.
"""
import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pymysql.err

from database import YouTubeDatabase
from virality_predictor import WEEKDAY_LABELS
from page_modules.shared import (
    _get_colors, ui_page_header, ui_section_divider,
    VIDEO_TYPE_SEQUENCE,
    _get_virality_predictor, _get_view_predictor,
)

log = logging.getLogger(__name__)


def show_virality_prediction(df: pd.DataFrame, channel_id: str):
    """Predicción de viralidad con ML"""
    COLORS = _get_colors()
    ui_page_header("🔮", "Predicción de Viralidad", "Modelo ML para estimar potencial viral")

    if df.empty or not channel_id:
        st.warning("No hay datos disponibles para este canal.")
        return

    predictor, from_cache = _get_virality_predictor(df, channel_id)

    # ── Sección A: Puntuar videos existentes ──────────────────────────
    st.subheader("🧠 Scores de Viralidad — Videos del Canal")

    if not predictor.is_trained():
        train_result = predictor.get_train_metrics()
        st.warning(f"⚠ {train_result.get('reason', 'No se pudo entrenar el modelo.')}")
        st.info("Ejecuta `python main.py` para obtener más datos de videos.")
        return

    train_result = predictor.get_train_metrics()
    n_splits = train_result.get('cv_splits', '?')
    if from_cache:
        st.info(f"Modelo cargado desde caché — {train_result.get('samples', '?')} videos.")
    else:
        st.success(f"Modelo reentrenado con **{train_result.get('samples', '?')} videos** · {n_splits}-fold CV temporal.")

    cv_mae_p = train_result.get('cv_mae_percentile')
    if cv_mae_p is not None:
        st.metric(
            "Error CV (MAE percentil)",
            f"±{cv_mae_p:.1f} pts",
            help="Error absoluto medio de validación cruzada temporal sobre el percentil (0–100). Menor = más preciso.",
        )

    scored_df = predictor.predict(df)

    # Guardar en DB
    try:
        with YouTubeDatabase() as db:
            db.save_virality_predictions(
                scored_df[['video_id', 'channel_id', 'virality_score']].copy()
            )
    except pymysql.err.Error as e:
        log.debug("No se pudieron guardar predicciones de viralidad: %s", e)

    # Tabla top 20
    display_cols = ['title', 'video_type', 'view_count', 'engagement_rate', 'virality_score']
    top_df = scored_df[display_cols].sort_values('virality_score', ascending=False).head(20).copy()
    top_df['virality_score'] = top_df['virality_score'].apply(lambda s: f"{s:.1f} / 10")

    st.dataframe(
        top_df.rename(columns={
            'title': 'Título',
            'video_type': 'Tipo',
            'view_count': 'Vistas',
            'engagement_rate': 'Engagement %',
            'virality_score': '🔥 Score Viral',
        }),
        use_container_width=True,
        hide_index=True,
    )

    ui_section_divider()
    col1, col2 = st.columns(2)

    # Importancia de features
    with col1:
        st.subheader("📊 Factores que más influyen")
        fi = predictor.get_feature_importance()
        labels_map = {
            'hour': 'Hora de publicación (Panamá)',
            'weekday_num': 'Día de la semana',
            'is_short': 'Formato (Short/Largo)',
            'duration_seconds': 'Duración (seg)',
            'tags_count': 'Cantidad de tags',
            'title_length': 'Longitud del título',
            'title_has_number': 'Número en el título',
            'title_has_question': 'Pregunta en el título (?)',
            'description_length': 'Longitud de descripción',
            'days_since_last_upload': 'Días desde última subida',
            'channel_age_days': 'Antigüedad del canal (días)',
        }
        fi_df = pd.DataFrame({
            'Factor': [labels_map[k] for k in fi],
            'Importancia': list(fi.values()),
        }).sort_values('Importancia', ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi_df['Importancia'],
            y=fi_df['Factor'],
            orientation='h',
            marker_color=COLORS["primary"],
        ))
        fig_fi.update_layout(
            xaxis_title='Importancia relativa',
            height=420,
            margin={'l': 0, 'r': 0, 't': 10, 'b': 0},
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Scatter score vs vistas reales
    with col2:
        st.subheader("📈 Score vs Vistas reales")
        fig_sc = px.scatter(
            scored_df,
            x='virality_score',
            y='view_count',
            color='video_type',
            hover_data=['title'],
            labels={'virality_score': 'Score Viral', 'view_count': 'Vistas'},
            color_discrete_sequence=VIDEO_TYPE_SEQUENCE,
        )
        fig_sc.update_layout(height=300, margin={'l': 0, 'r': 0, 't': 10, 'b': 0})
        st.plotly_chart(fig_sc, use_container_width=True)

    # ── Sección B: Predictor interactivo ──────────────────────────────
    ui_section_divider()
    st.subheader("🎯 ¿Qué tan viral sería tu próximo video?")

    col1, col2, col3 = st.columns(3)
    with col1:
        dia = st.selectbox("📅 Día de publicación", WEEKDAY_LABELS)
        hora = st.slider("🕐 Hora de publicación (Panamá UTC-5)", 0, 23, 18)
        tiene_numero = st.checkbox("🔢 El título contiene números", value=False)
        tiene_pregunta = st.checkbox("❓ El título es una pregunta (?)", value=False)
    with col2:
        formato = st.radio("🎬 Formato", ["Short", "Video Largo"])
        duracion = st.number_input(
            "⏱ Duración (segundos)",
            min_value=1, max_value=3600,
            value=60 if formato == "Short" else 600,
        )
        len_descripcion = st.number_input(
            "📝 Longitud de descripción (caracteres)",
            min_value=0, max_value=5000, value=500,
        )
    with col3:
        num_tags = st.number_input("🏷 Cantidad de tags", min_value=0, max_value=50, value=10)
        len_titulo = st.number_input("✍ Longitud del título (caracteres)", min_value=5, max_value=200, value=60)
        dias_desde_ultimo = st.number_input(
            "📆 Días desde última subida",
            min_value=0, max_value=365, value=7,
        )

    weekday_num = WEEKDAY_LABELS.index(dia)
    is_short = formato == "Short"

    # Edad del canal: calculada desde los datos reales del canal
    dt_utc = pd.to_datetime(df['published_at'], utc=True)
    canal_age_days = int((dt_utc.max() - dt_utc.min()).days) if len(df) > 1 else 365

    score = predictor.predict_single(
        hour=hora,
        weekday=weekday_num,
        is_short=is_short,
        duration_seconds=int(duracion),
        tags_count=int(num_tags),
        title_length=int(len_titulo),
        title_has_number=int(tiene_numero),
        title_has_question=int(tiene_pregunta),
        description_length=int(len_descripcion),
        days_since_last_upload=int(dias_desde_ultimo),
        channel_age_days=canal_age_days,
    )

    color = predictor.score_color(score)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Potencial Viral", 'font': {'size': 20}},
        number={'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [1, 10], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [1, 4], 'color': COLORS["bg_secondary"]},
                {'range': [4, 7], 'color': 'rgba(245,158,11,0.1)'},
                {'range': [7, 10], 'color': 'rgba(16,185,129,0.1)'},
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 3},
                'thickness': 0.75,
                'value': score,
            },
        }
    ))
    fig_gauge.update_layout(height=320, margin={'l': 20, 'r': 20, 't': 40, 'b': 0})
    st.plotly_chart(fig_gauge, use_container_width=True)

    if score >= 8:
        st.success(f"🚀 **Score {score}/10** — Alto potencial viral. ¡Excelente combinación!")
    elif score >= 5:
        st.warning(f"📊 **Score {score}/10** — Potencial moderado. Ajusta hora o formato.")
    else:
        st.error(f"⚠ **Score {score}/10** — Potencial bajo. Considera publicar en otro momento.")

    # ── What-If: Predicción cruzada de vistas ──────────────────────────
    view_pred, _ = _get_view_predictor(df, channel_id)
    if view_pred.is_trained():
        st.markdown("---")
        st.markdown("##### 👁 Estimación de vistas para este escenario")
        vp_result = view_pred.predict_single(
            hour=hora,
            weekday=weekday_num,
            is_short=is_short,
            duration_seconds=int(duracion),
            tags_count=int(num_tags),
            title_length=int(len_titulo),
            title_has_number=int(tiene_numero),
            title_has_question=int(tiene_pregunta),
            description_length=int(len_descripcion),
            days_since_last_upload=int(dias_desde_ultimo),
            channel_age_days=canal_age_days,
        )
        vp_avg = int(view_pred.get_train_metrics().get('channel_avg_views', 0))
        vp_pred = vp_result['predicted']
        vp_delta = vp_pred - vp_avg

        col_wif1, col_wif2 = st.columns(2)
        with col_wif1:
            st.metric(
                label="Vistas estimadas",
                value=f"~{vp_pred:,}",
                delta=f"{vp_delta:+,} vs promedio del canal",
            )
            st.caption(f"Rango de confianza: **{vp_result['low']:,} — {vp_result['high']:,}** vistas")
        with col_wif2:
            vp_ratio = vp_pred / vp_avg if vp_avg > 0 else 1.0
            if vp_ratio >= 1.5:
                st.success(f"🚀 **{vp_ratio:.1f}x** el promedio del canal — Alto potencial")
            elif vp_ratio >= 0.8:
                st.warning(f"📊 **{vp_ratio:.1f}x** el promedio del canal — Rendimiento moderado")
            else:
                st.error(f"⚠ **{vp_ratio:.1f}x** el promedio del canal — Bajo el promedio")


def show_view_prediction(df: pd.DataFrame, channel_id: str):
    """Predicción de vistas con ML — número esperado + rango de confianza + heatmap."""
    COLORS = _get_colors()
    ui_page_header("👁", "Predicción de Vistas", "Estimación de vistas con rango de confianza")

    if df.empty or not channel_id:
        st.warning("No hay datos disponibles para este canal.")
        return

    predictor, from_cache = _get_view_predictor(df, channel_id)

    if not predictor.is_trained():
        reason = predictor.get_train_metrics().get('reason', 'No se pudo entrenar el modelo.')
        st.warning(f"⚠ {reason}")
        st.info("Ejecuta `python main.py` para obtener más datos de videos.")
        return

    # ── Sección A: Métricas de precisión + scatter ─────────────────────
    st.subheader("🎯 Precisión del modelo en datos históricos")

    metrics = predictor.get_train_metrics()
    n_splits = metrics.get('cv_splits', '?')
    if from_cache:
        st.info(f"Modelo cargado desde caché — {metrics.get('samples', '?')} videos.")
    else:
        st.success(f"Modelo reentrenado con **{metrics.get('samples', '?')} videos** · {n_splits}-fold CV temporal.")
    col1, col2, col3 = st.columns(3)
    channel_avg = int(metrics.get('channel_avg_views', 0))
    with col1:
        st.metric(
            "Videos usados para entrenar",
            f"{metrics['samples']}",
            help=f"Validación cruzada temporal con {n_splits} folds (TimeSeriesSplit).",
        )
    with col2:
        cv_mae = metrics.get('cv_mae', metrics.get('mae', 0))
        st.metric(
            "MAE — Validación cruzada",
            f"{int(cv_mae):,} vistas",
            help="Error absoluto medio promedio de los folds de CV temporal. Mide cuántas vistas se desvía la predicción en promedio.",
        )
    with col3:
        cv_mape = metrics.get('cv_mape', metrics.get('mape', 0))
        st.metric(
            "MAPE — Validación cruzada",
            f"{cv_mape:.1f}%",
            help="Error porcentual absoluto medio promedio de los folds. Indica el % de error relativo al valor real.",
        )

    # Predecir sobre todos los videos y guardar
    scored_df = predictor.predict(df)

    try:
        with YouTubeDatabase() as db:
            db.save_view_predictions(
                scored_df[['video_id', 'channel_id', 'predicted_views', 'predicted_low', 'predicted_high']].copy()
            )
    except pymysql.err.Error as e:
        log.debug("No se pudieron guardar predicciones de vistas: %s", e)

    # Scatter: Vistas reales vs predichas
    fig_scatter = px.scatter(
        scored_df,
        x='view_count',
        y='predicted_views',
        color='video_type',
        hover_data=['title'],
        labels={'view_count': 'Vistas Reales', 'predicted_views': 'Vistas Predichas'},
        color_discrete_sequence=VIDEO_TYPE_SEQUENCE,
    )
    # Línea de referencia 45° (predicción perfecta)
    max_val = max(scored_df['view_count'].max(), scored_df['predicted_views'].max())
    fig_scatter.add_shape(
        type='line', x0=0, y0=0, x1=max_val, y1=max_val,
        line={'color': '#888', 'dash': 'dash', 'width': 1}
    )
    fig_scatter.update_layout(height=380, margin={'l': 0, 'r': 0, 't': 20, 'b': 0})
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Sección B: Predictor interactivo ───────────────────────────────
    ui_section_divider()
    st.subheader("🔢 ¿Cuántas vistas tendría tu próximo video?")

    col1, col2, col3 = st.columns(3)
    with col1:
        dia = st.selectbox("📅 Día de publicación", WEEKDAY_LABELS, key="vp_dia")
        hora = st.slider("🕐 Hora de publicación (Panamá UTC-5)", 0, 23, 18, key="vp_hora")
        tiene_numero = st.checkbox("🔢 El título contiene números", value=False, key="vp_num")
        tiene_pregunta = st.checkbox("❓ El título es una pregunta (?)", value=False, key="vp_q")
    with col2:
        formato = st.radio("🎬 Formato", ["Short", "Video Largo"], key="vp_formato")
        duracion = st.number_input(
            "⏱ Duración (segundos)",
            min_value=1, max_value=3600,
            value=60 if formato == "Short" else 600,
            key="vp_duracion",
        )
        len_descripcion = st.number_input(
            "📝 Longitud de descripción (caracteres)",
            min_value=0, max_value=5000, value=500, key="vp_desc",
        )
    with col3:
        num_tags = st.number_input("🏷 Cantidad de tags", min_value=0, max_value=50, value=10, key="vp_tags")
        len_titulo = st.number_input("✍ Longitud del título (caracteres)", min_value=5, max_value=200, value=60, key="vp_titulo")
        dias_desde_ultimo = st.number_input(
            "📆 Días desde última subida",
            min_value=0, max_value=365, value=7, key="vp_dias",
        )

    weekday_num = WEEKDAY_LABELS.index(dia)
    is_short = formato == "Short"

    # Edad del canal: calculada desde los datos reales del canal
    dt_utc_vp = pd.to_datetime(df['published_at'], utc=True)
    canal_age_days_vp = int((dt_utc_vp.max() - dt_utc_vp.min()).days) if len(df) > 1 else 365

    result = predictor.predict_single(
        hour=hora,
        weekday=weekday_num,
        is_short=is_short,
        duration_seconds=int(duracion),
        tags_count=int(num_tags),
        title_length=int(len_titulo),
        title_has_number=int(tiene_numero),
        title_has_question=int(tiene_pregunta),
        description_length=int(len_descripcion),
        days_since_last_upload=int(dias_desde_ultimo),
        channel_age_days=canal_age_days_vp,
    )

    pred = result['predicted']
    low = result['low']
    high = result['high']
    delta_vs_avg = pred - channel_avg

    col_res1, col_res2 = st.columns([1, 1])
    with col_res1:
        st.metric(
            label="Vistas estimadas",
            value=f"~{pred:,}",
            delta=f"{delta_vs_avg:+,} vs promedio del canal",
        )
        st.caption(f"Rango de confianza: **{low:,} — {high:,}** vistas")
    with col_res2:
        ratio = pred / channel_avg if channel_avg > 0 else 1.0
        if ratio >= 1.5:
            st.success(f"🚀 **{ratio:.1f}x** el promedio del canal — Alto potencial")
        elif ratio >= 0.8:
            st.warning(f"📊 **{ratio:.1f}x** el promedio del canal — Rendimiento moderado")
        else:
            st.error(f"⚠ **{ratio:.1f}x** el promedio del canal — Bajo el promedio")

    # ── What-If: Predicción cruzada de viralidad ───────────────────────
    vir_pred, _ = _get_virality_predictor(df, channel_id)
    if vir_pred.is_trained():
        st.markdown("---")
        st.markdown("##### 🔮 Score de viralidad para este escenario")
        vir_score = vir_pred.predict_single(
            hour=hora,
            weekday=weekday_num,
            is_short=is_short,
            duration_seconds=int(duracion),
            tags_count=int(num_tags),
            title_length=int(len_titulo),
            title_has_number=int(tiene_numero),
            title_has_question=int(tiene_pregunta),
            description_length=int(len_descripcion),
            days_since_last_upload=int(dias_desde_ultimo),
            channel_age_days=canal_age_days_vp,
        )
        vir_color = vir_pred.score_color(vir_score)

        fig_gauge_vp = go.Figure(go.Indicator(
            mode="gauge+number",
            value=vir_score,
            title={'text': "Potencial Viral", 'font': {'size': 18}},
            number={'font': {'size': 40, 'color': vir_color}},
            gauge={
                'axis': {'range': [1, 10], 'tickwidth': 1},
                'bar': {'color': vir_color},
                'steps': [
                    {'range': [1, 4], 'color': COLORS["bg_secondary"]},
                    {'range': [4, 7], 'color': 'rgba(245,158,11,0.1)'},
                    {'range': [7, 10], 'color': 'rgba(16,185,129,0.1)'},
                ],
                'threshold': {
                    'line': {'color': 'white', 'width': 3},
                    'thickness': 0.75,
                    'value': vir_score,
                },
            }
        ))
        fig_gauge_vp.update_layout(height=260, margin={'l': 20, 'r': 20, 't': 30, 'b': 0})
        st.plotly_chart(fig_gauge_vp, use_container_width=True)

        if vir_score >= 8:
            st.success(f"🚀 **Score {vir_score}/10** — Alto potencial viral")
        elif vir_score >= 5:
            st.warning(f"📊 **Score {vir_score}/10** — Potencial moderado")
        else:
            st.error(f"⚠ **Score {vir_score}/10** — Potencial bajo")

    # ── Sección C: Heatmap ─────────────────────────────────────────────
    ui_section_divider()
    st.subheader("🗓 Mejor momento para publicar")
    st.caption("Vistas estimadas según día y hora (timezone Panamá UTC-5)")

    tab_short, tab_largo = st.tabs(["📱 Shorts", "🎬 Videos Largos"])

    for tab, fmt_is_short, fmt_dur in [
        (tab_short, True, 45),
        (tab_largo, False, 600),
    ]:
        with tab:
            heatmap_df = predictor.get_publishing_heatmap(
                is_short=fmt_is_short,
                duration_seconds=fmt_dur,
                tags_count=10,
                title_length=60,
            )
            fig_hm = px.imshow(
                heatmap_df,
                labels={'x': 'Hora (Panamá)', 'y': 'Día', 'color': 'Vistas'},
                x=[str(h) for h in range(24)],
                y=WEEKDAY_LABELS,
                color_continuous_scale='YlOrRd',
                aspect='auto',
            )
            fig_hm.update_layout(height=300, margin={'l': 0, 'r': 0, 't': 10, 'b': 0})
            st.plotly_chart(fig_hm, use_container_width=True)
