"""
Página: Miniaturas — Galería y Comparador A/B (Mejoras 8.1, 8.2)
"""
import base64
import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from database import YouTubeDatabase
from page_modules.shared import (
    _get_colors, _esc, ui_page_header, ui_section_divider, ui_section_header,
    ui_metric_card,
)


def show_thumbnail_gallery(df: pd.DataFrame, channel_id: str):
    """Galería de miniaturas con scores de Claude Vision (8.1)."""

    COLORS = _get_colors()
    ui_page_header("🖼", "Galería de Miniaturas",
                   "Análisis de miniaturas con Claude Vision — scores de clickability y sugerencias")

    if df.empty or not channel_id:
        st.warning("No hay datos disponibles. Ejecuta el pipeline primero.")
        return

    with YouTubeDatabase() as db:
        scores_df = db.get_thumbnail_scores(channel_id)

    if scores_df.empty:
        st.info("Aún no se han analizado miniaturas para este canal. "
                "Ejecuta el pipeline para analizar automáticamente.")

        if st.button("🔍 Analizar miniaturas ahora", key="btn_analyze_thumbnails"):
            api_key = os.getenv('ANTHROPIC_API_KEY', '')
            if not api_key:
                st.error("ANTHROPIC_API_KEY no configurada.")
                return
            from thumbnail_analyzer import ThumbnailScorer
            scorer = ThumbnailScorer(api_key)
            with st.spinner("Analizando miniaturas con Claude Vision..."):
                with YouTubeDatabase() as db2:
                    results = scorer.analyze_channel(channel_id, db2, max_per_run=20)
            if results:
                st.success(f"✓ {len(results)} miniaturas analizadas")
                st.rerun()
            else:
                st.warning("No se encontraron miniaturas pendientes o hubo un error.")
        return

    # ── Validar columna clave ─────────────────────────────────────────
    if 'clickability' not in scores_df.columns:
        st.warning("Los datos de thumbnails no contienen la columna 'clickability'.")
        return

    # ── KPIs ──────────────────────────────────────────────────────────
    click_series = scores_df['clickability'].dropna()
    avg_click = float(click_series.mean()) if not click_series.empty else 0
    best_click = float(click_series.max()) if not click_series.empty else 0
    worst_click = float(click_series.min()) if not click_series.empty else 0
    total_analyzed = len(scores_df)

    cols = st.columns(4)
    with cols[0]:
        st.markdown(ui_metric_card("🖼", "Analizadas", str(total_analyzed)),
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(ui_metric_card("📊", "Clickability prom.",
                                   f"{avg_click:.1f}/10"),
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(ui_metric_card("🏆", "Mejor score",
                                   f"{best_click:.1f}/10"),
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(ui_metric_card("⚠", "Peor score",
                                   f"{worst_click:.1f}/10"),
                    unsafe_allow_html=True)

    ui_section_divider()

    # ── Distribución de scores ────────────────────────────────────────
    ui_section_header("📊", "Distribución de Clickability")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores_df['clickability'],
        nbinsx=10,
        marker_color=COLORS.get('primary', '#6366F1'),
        opacity=0.8,
    ))
    fig.update_layout(
        xaxis_title="Clickability Score",
        yaxis_title="Cantidad de videos",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS.get('text_primary', '#F8FAFC')),
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    ui_section_divider()

    # ── Radar por dimensión ───────────────────────────────────────────
    ui_section_header("🎯", "Scores promedio por dimensión")

    dimensions = ['text_clarity', 'color_contrast', 'composition', 'clickability']
    dim_labels = ['Claridad de Texto', 'Contraste de Color', 'Composición', 'Clickability']

    if 'facial_expression' in scores_df.columns:
        face_scores = scores_df['facial_expression'].dropna()
        if not face_scores.empty:
            dimensions.insert(1, 'facial_expression')
            dim_labels.insert(1, 'Expresión Facial')

    avgs = []
    for dim in dimensions:
        val = scores_df[dim].dropna().mean() if dim in scores_df.columns else 0
        avgs.append(round(float(val), 1))

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=avgs + [avgs[0]],
        theta=dim_labels + [dim_labels[0]],
        fill='toself',
        fillcolor='rgba(99,102,241,0.2)',
        line=dict(color=COLORS.get('primary', '#6366F1'), width=2),
        name='Promedio',
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 10], dtick=2,
                            gridcolor=COLORS.get('border', '#334155')),
            angularaxis=dict(gridcolor=COLORS.get('border', '#334155')),
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS.get('text_primary', '#F8FAFC')),
        height=350,
        margin=dict(l=60, r=60, t=30, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    ui_section_divider()

    # ── Galería de miniaturas ─────────────────────────────────────────
    ui_section_header("🖼", "Galería ordenada por Clickability")

    sort_order = st.radio(
        "Ordenar por", ["Mejor → Peor", "Peor → Mejor"],
        horizontal=True, key="thumb_sort",
    )
    ascending = sort_order == "Peor → Mejor"
    sorted_df = scores_df.sort_values('clickability', ascending=ascending)

    rows_to_show = min(len(sorted_df), 30)
    for row_start in range(0, rows_to_show, 3):
        grid_cols = st.columns(3)
        for col_idx in range(3):
            idx = row_start + col_idx
            if idx >= rows_to_show:
                break
            row = sorted_df.iloc[idx]
            video_id = row['video_id']
            thumb_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
            raw_title = row.get('title')
            title = _esc(str(raw_title if pd.notna(raw_title) else video_id)[:50])
            raw_click = row.get('clickability')
            click_score = float(raw_click) if pd.notna(raw_click) else 0.0
            raw_views = row.get('view_count')
            views = int(raw_views) if pd.notna(raw_views) else 0

            if click_score >= 7:
                badge_color = COLORS.get('success', '#10B981')
            elif click_score >= 4:
                badge_color = COLORS.get('warning', '#F59E0B')
            else:
                badge_color = COLORS.get('danger', '#EF4444')

            with grid_cols[col_idx]:
                st.markdown(f"""
                <div style="background: {COLORS.get('bg_secondary', '#1E293B')};
                            border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem;
                            border: 1px solid {COLORS.get('border', '#334155')};">
                    <img src="{thumb_url}" style="width: 100%; border-radius: 6px;
                         aspect-ratio: 16/9; object-fit: cover;" />
                    <div style="padding: 0.4rem 0.2rem 0;">
                        <div style="font-size: 0.8rem; font-weight: 600;
                                    color: {COLORS.get('text_primary', '#F8FAFC')};
                                    white-space: nowrap; overflow: hidden;
                                    text-overflow: ellipsis;">{title}</div>
                        <div style="display: flex; justify-content: space-between;
                                    align-items: center; margin-top: 4px;">
                            <span style="background: {badge_color}; color: #fff;
                                         padding: 2px 8px; border-radius: 12px;
                                         font-size: 0.75rem; font-weight: 700;">
                                {click_score:.1f}/10
                            </span>
                            <span style="font-size: 0.7rem;
                                         color: {COLORS.get('text_muted', '#94A3B8')};">
                                {views:,} vistas
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    ui_section_divider()

    # ── Detalle expandible por video ──────────────────────────────────
    ui_section_header("🔍", "Detalle y Sugerencias de Mejora")

    worst_df = scores_df.nsmallest(10, 'clickability')

    for _, row in worst_df.iterrows():
        raw_t = row.get('title')
        title = str(raw_t if pd.notna(raw_t) else row['video_id'])[:60]
        raw_c = row.get('clickability')
        click_score = float(raw_c) if pd.notna(raw_c) else 0.0

        with st.expander(f"⚠ {title} — {click_score:.1f}/10"):
            detail_cols = st.columns([1, 2])
            with detail_cols[0]:
                vid = row['video_id']
                st.image(f"https://img.youtube.com/vi/{vid}/hqdefault.jpg",
                         use_container_width=True)

            with detail_cols[1]:
                def _fmt(val):
                    return f"{val:.1f}" if pd.notna(val) else '-'

                scores_text = (
                    f"- **Claridad de texto:** {_fmt(row.get('text_clarity'))}/10\n"
                    f"- **Contraste de color:** {_fmt(row.get('color_contrast'))}/10\n"
                    f"- **Composición:** {_fmt(row.get('composition'))}/10\n"
                    f"- **Clickability:** {_fmt(row.get('clickability'))}/10\n"
                )
                if pd.notna(row.get('facial_expression')):
                    scores_text += f"- **Expresión facial:** {row['facial_expression']}/10\n"

                st.markdown(scores_text)

                suggestions = []
                raw_sug = row.get('improvement_suggestions', '[]')
                if isinstance(raw_sug, str):
                    try:
                        suggestions = json.loads(raw_sug)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(raw_sug, list):
                    suggestions = raw_sug

                if suggestions:
                    st.markdown("**💡 Sugerencias de mejora:**")
                    for sug in suggestions:
                        st.markdown(f"- {sug}")


def show_thumbnail_comparator(df: pd.DataFrame, channel_id: str):
    """Comparador A/B de miniaturas con Claude Vision (8.2)."""

    COLORS = _get_colors()
    ui_page_header("🔀", "Comparador A/B de Miniaturas",
                   "Sube 2-3 opciones de miniatura y Claude Vision predice cuál tendrá mayor CTR")

    if not channel_id:
        st.warning("Selecciona un canal en la barra lateral.")
        return

    # ── Upload de imágenes ────────────────────────────────────────────
    ui_section_header("📤", "Sube tus miniaturas", "Sube 2 o 3 imágenes para comparar")

    comparison_name = st.text_input(
        "Nombre de la comparación (opcional)",
        placeholder="Ej: Thumbnail para video de recetas",
        key="thumb_ab_name",
    )

    upload_cols = st.columns(3)
    uploaded_files = []
    labels = ["Miniatura A", "Miniatura B", "Miniatura C (opcional)"]

    for i, col in enumerate(upload_cols):
        with col:
            f = st.file_uploader(
                labels[i],
                type=["png", "jpg", "jpeg", "webp"],
                key=f"thumb_upload_{i}",
            )
            if f:
                uploaded_files.append((chr(65 + i), f))
                st.image(f, caption=f"Miniatura {chr(65 + i)}", use_container_width=True)

    if len(uploaded_files) < 2:
        st.info("Sube al menos 2 imágenes para comparar.")
    else:
        if st.button("🔍 Comparar miniaturas", key="btn_compare_thumbs",
                      type="primary"):
            api_key = os.getenv('ANTHROPIC_API_KEY', '')
            if not api_key:
                st.error("ANTHROPIC_API_KEY no configurada.")
                return

            from thumbnail_analyzer import ThumbnailComparator
            comparator = ThumbnailComparator(api_key)

            images = []
            for label, f in uploaded_files:
                f.seek(0)
                data = base64.b64encode(f.read()).decode('utf-8')
                ext = f.name.rsplit('.', 1)[-1].lower() if '.' in f.name else 'png'
                media_types = {'png': 'image/png', 'jpg': 'image/jpeg',
                               'jpeg': 'image/jpeg', 'webp': 'image/webp'}
                mt = media_types.get(ext, 'image/png')
                images.append({
                    "source": {
                        "type": "base64",
                        "media_type": mt,
                        "data": data,
                    }
                })

            with st.spinner("Analizando con Claude Vision..."):
                result = comparator.compare(images)

            if not result:
                st.error("No se pudo completar el análisis. Intenta de nuevo.")
                return

            with YouTubeDatabase() as db:
                db.save_thumbnail_comparison({
                    'channel_id': channel_id,
                    'comparison_name': comparison_name or 'Sin nombre',
                    'thumbnails': [l for l, _ in uploaded_files],
                    'winner': result.get('winner', '?'),
                    'analysis': json.dumps(result, ensure_ascii=False),
                })

            # ── Mostrar resultado ─────────────────────────────────────
            st.success(f"🏆 **Ganadora: Miniatura {result.get('winner', '?')}**")

            ui_section_divider()

            scores = result.get('scores', {})
            score_cols = st.columns(len(uploaded_files))
            for idx, (label, f) in enumerate(uploaded_files):
                with score_cols[idx]:
                    is_winner = (label == result.get('winner'))
                    border = COLORS.get('success', '#10B981') if is_winner else COLORS.get('border', '#334155')
                    winner_badge = ' 🏆' if is_winner else ''

                    st.markdown(f"""
                    <div style="background: {COLORS.get('bg_secondary', '#1E293B')};
                                border: 2px solid {border}; border-radius: 8px;
                                padding: 0.8rem; text-align: center;">
                        <div style="font-weight: 700; font-size: 1.1rem;
                                    color: {COLORS.get('text_primary', '#F8FAFC')};">
                            Miniatura {label}{winner_badge}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    s = scores.get(label, {})
                    for metric, metric_label in [
                        ('clickability', 'Clickability'),
                        ('mobile_legibility', 'Legibilidad móvil'),
                        ('visual_hierarchy', 'Jerarquía visual'),
                        ('engagement_potential', 'Potencial de engagement'),
                    ]:
                        val = s.get(metric, 0)
                        bar_pct = val * 10
                        bar_color = COLORS.get('success', '#10B981') if val >= 7 else (
                            COLORS.get('warning', '#F59E0B') if val >= 4 else COLORS.get('danger', '#EF4444')
                        )
                        st.markdown(f"""
                        <div style="margin: 4px 0;">
                            <div style="display: flex; justify-content: space-between;
                                        font-size: 0.75rem;
                                        color: {COLORS.get('text_muted', '#94A3B8')};">
                                <span>{metric_label}</span><span>{val}/10</span>
                            </div>
                            <div style="background: {COLORS.get('bg_tertiary', '#0F172A')};
                                        border-radius: 4px; height: 6px; margin-top: 2px;">
                                <div style="background: {bar_color}; width: {bar_pct}%;
                                            height: 100%; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            ui_section_divider()

            reasoning = result.get('reasoning', '')
            if reasoning:
                ui_section_header("💡", "Razonamiento")
                st.markdown(reasoning)

            suggestions = result.get('suggestions', [])
            if suggestions:
                ui_section_header("📝", "Sugerencias")
                for sug in suggestions:
                    st.markdown(f"- {sug}")

    # ── Historial de comparaciones ────────────────────────────────────
    ui_section_divider()
    ui_section_header("📜", "Historial de Comparaciones")

    with YouTubeDatabase() as db:
        history_df = db.get_thumbnail_comparisons(channel_id)

    if history_df.empty:
        st.info("Sin comparaciones previas.")
    else:
        for _, row in history_df.iterrows():
            name = row.get('comparison_name', 'Sin nombre')
            winner = row.get('winner', '?')
            created = str(row.get('created_at', ''))[:16]

            with st.expander(f"🏆 {name} — Ganadora: {winner} ({created})"):
                analysis = {}
                raw = row.get('analysis', '{}')
                if isinstance(raw, str):
                    try:
                        analysis = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if analysis.get('reasoning'):
                    st.markdown(analysis['reasoning'])
