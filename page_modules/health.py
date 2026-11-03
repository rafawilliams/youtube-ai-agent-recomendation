"""
Página: Salud del Canal (🏥)
"""
import json
import os

import pandas as pd
import streamlit as st

from database import YouTubeDatabase
from ai_analyzer import AIAnalyzer
from page_modules.shared import (
    _get_colors, _esc, ui_page_header, ui_section_divider, _detect_channel_language,
)


def _compute_health_metrics(df: pd.DataFrame, db, channel_id: str) -> list[dict]:
    """Calcula las métricas de salud del canal y retorna lista de dicts."""
    metrics: list[dict] = []
    # Usar timezone-aware si los datos lo son, sino naive
    _pub = pd.to_datetime(df['published_at'], errors='coerce') if 'published_at' in df.columns else pd.Series(dtype='datetime64[ns]')
    if _pub.dt.tz is not None:
        now = pd.Timestamp.now(tz='UTC')
    else:
        now = pd.Timestamp.now()

    # ── 1. Frecuencia de publicación (videos/semana últimos 30 días) ───
    if not df.empty and 'published_at' in df.columns:
        recent = df[_pub >= (now - pd.Timedelta(days=30))]
        weeks = max((now - _pub.min()).days / 7, 1)
        vids_per_week = round(len(recent) / min(weeks, 4.3), 1)
        if vids_per_week >= 3:
            status = 'green'
        elif vids_per_week >= 1:
            status = 'yellow'
        else:
            status = 'red'
        metrics.append({
            'name': 'Frecuencia de publicación',
            'icon': '📅',
            'value': f'{vids_per_week} vid/sem',
            'status': status,
            'detail': f'{len(recent)} videos en últimos 30 días',
            'raw': vids_per_week,
        })

    # ── 2. Frescura del canal (días desde último video) ────────────────
    if not df.empty and 'published_at' in df.columns:
        last_pub = _pub.max()
        days_ago = (now - last_pub).days if pd.notna(last_pub) else 999
        if days_ago <= 3:
            status = 'green'
        elif days_ago <= 7:
            status = 'yellow'
        else:
            status = 'red'
        metrics.append({
            'name': 'Frescura del canal',
            'icon': '🕐',
            'value': f'Hace {days_ago} días',
            'status': status,
            'detail': f'Último video: {last_pub.strftime("%Y-%m-%d")}',
            'raw': days_ago,
        })

    # ── 3. Tendencia de engagement (últimos 10 vs anteriores 10) ───────
    if not df.empty and 'engagement_rate' in df.columns:
        sorted_df = df.sort_values('published_at', ascending=False)
        eng_col = pd.to_numeric(sorted_df['engagement_rate'], errors='coerce')
        if len(eng_col.dropna()) >= 10:
            recent_eng = eng_col.iloc[:10].mean()
            older_eng = eng_col.iloc[10:20].mean() if len(eng_col) >= 20 else eng_col.iloc[10:].mean()
            if older_eng and older_eng > 0:
                change_pct = round(((recent_eng - older_eng) / older_eng) * 100, 1)
            else:
                change_pct = 0.0
            if change_pct >= 0:
                status = 'green'
            elif change_pct >= -10:
                status = 'yellow'
            else:
                status = 'red'
            sign = '+' if change_pct >= 0 else ''
            metrics.append({
                'name': 'Tendencia de engagement',
                'icon': '📈',
                'value': f'{sign}{change_pct}%',
                'status': status,
                'detail': f'Recientes: {recent_eng:.2f}% vs anteriores: {older_eng:.2f}%',
                'raw': change_pct,
            })

    # ── 4. Vistas promedio recientes vs general ────────────────────────
    if not df.empty and 'view_count' in df.columns:
        views = pd.to_numeric(df['view_count'], errors='coerce').dropna()
        sorted_df = df.sort_values('published_at', ascending=False)
        recent_views = pd.to_numeric(sorted_df['view_count'].iloc[:10], errors='coerce').dropna()
        avg_all = views.mean()
        avg_recent = recent_views.mean()
        if avg_all > 0:
            ratio = avg_recent / avg_all
            if ratio >= 1.0:
                status = 'green'
            elif ratio >= 0.7:
                status = 'yellow'
            else:
                status = 'red'
            metrics.append({
                'name': 'Vistas promedio recientes',
                'icon': '👁',
                'value': f'{int(avg_recent):,}',
                'status': status,
                'detail': f'{ratio:.0%} del promedio general ({int(avg_all):,})',
                'raw': ratio,
            })

    # ── 5. Ratio de viralidad (% videos sobre la mediana) ─────────────
    if not df.empty and 'view_count' in df.columns:
        views = pd.to_numeric(df['view_count'], errors='coerce').dropna()
        if len(views) >= 5:
            median_views = views.median()
            above_median = (views > median_views).sum()
            ratio_pct = round((above_median / len(views)) * 100, 1)
            # Por definición la mediana da ~50%, así que ajustamos thresholds
            # Miramos los últimos 10 videos vs mediana global
            recent_views = pd.to_numeric(
                df.sort_values('published_at', ascending=False)['view_count'].iloc[:10],
                errors='coerce').dropna()
            above_recent = (recent_views > median_views).sum()
            recent_pct = round((above_recent / max(len(recent_views), 1)) * 100, 1)
            if recent_pct >= 50:
                status = 'green'
            elif recent_pct >= 30:
                status = 'yellow'
            else:
                status = 'red'
            metrics.append({
                'name': 'Ratio de viralidad',
                'icon': '🔥',
                'value': f'{recent_pct}%',
                'status': status,
                'detail': f'{above_recent}/{len(recent_views)} videos recientes sobre mediana ({int(median_views):,} vistas)',
                'raw': recent_pct,
            })

    # ── 6. Consistencia de engagement ─────────────────────────────────
    if not df.empty and 'engagement_rate' in df.columns:
        eng = pd.to_numeric(df['engagement_rate'], errors='coerce').dropna()
        if len(eng) >= 5 and eng.mean() > 0:
            cv = eng.std() / eng.mean()  # Coeficiente de variación
            if cv < 0.5:
                status = 'green'
            elif cv < 1.0:
                status = 'yellow'
            else:
                status = 'red'
            metrics.append({
                'name': 'Consistencia de engagement',
                'icon': '🎯',
                'value': f'CV {cv:.2f}',
                'status': status,
                'detail': 'Baja variación = más predecible' if cv < 0.5 else 'Alta variación entre videos',
                'raw': cv,
            })

    # ── 7-8. Bonus: Retención y CTR si hay datos de Analytics API ─────
    try:
        analytics_df = db.get_video_analytics(channel_id)
        if not analytics_df.empty:
            # Retención promedio
            avg_ret = pd.to_numeric(analytics_df['avg_view_percentage'], errors='coerce').mean()
            if pd.notna(avg_ret):
                if avg_ret >= 50:
                    status = 'green'
                elif avg_ret >= 35:
                    status = 'yellow'
                else:
                    status = 'red'
                metrics.append({
                    'name': 'Retención promedio',
                    'icon': '⏱',
                    'value': f'{avg_ret:.1f}%',
                    'status': status,
                    'detail': 'Benchmark: 50%+ es excelente',
                    'raw': avg_ret,
                })
            # CTR de miniaturas
            ctr_vals = pd.to_numeric(analytics_df['impression_ctr'], errors='coerce').dropna()
            if len(ctr_vals) > 0:
                avg_ctr = ctr_vals.mean()
                if avg_ctr >= 5:
                    status = 'green'
                elif avg_ctr >= 3:
                    status = 'yellow'
                else:
                    status = 'red'
                metrics.append({
                    'name': 'CTR de miniaturas',
                    'icon': '🖼',
                    'value': f'{avg_ctr:.1f}%',
                    'status': status,
                    'detail': 'Benchmark: 5%+ es bueno',
                    'raw': avg_ctr,
                })
    except Exception:
        pass  # Analytics API no disponible — no pasa nada

    return metrics


def _health_score(metrics: list[dict]) -> int:
    """Calcula score 0-100 promediando los estados de las métricas."""
    if not metrics:
        return 0
    score_map = {'green': 100, 'yellow': 50, 'red': 0}
    total = sum(score_map.get(m['status'], 50) for m in metrics)
    return round(total / len(metrics))


def _sparkline_svg(values: list[float], color: str = '#6366F1',
                   width: int = 80, height: int = 24) -> str:
    """Genera un mini SVG sparkline inline."""
    if not values or len(values) < 2:
        return ''
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    points = []
    step = width / (len(values) - 1)
    for i, v in enumerate(values):
        x = round(i * step, 1)
        y = round(height - ((v - mn) / rng) * (height - 2) - 1, 1)
        points.append(f'{x},{y}')
    polyline = ' '.join(points)
    return (
        f'<svg width="{width}" height="{height}" style="vertical-align:middle;">'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        f'</svg>'
    )


def show_channel_health(df: pd.DataFrame, channel_id: str):
    """Página Salud del Canal — semáforo de métricas + diagnóstico IA."""
    ui_page_header("🏥", "Salud del Canal", "Diagnóstico rápido del estado de tu canal")

    if df.empty or not channel_id:
        st.warning("No hay datos suficientes para evaluar la salud del canal.")
        return

    try:
        db = YouTubeDatabase()
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return

    COLORS = _get_colors()

    # ── Calcular métricas ──────────────────────────────────────────────
    metrics = _compute_health_metrics(df, db, channel_id)
    if not metrics:
        st.warning("No se pudieron calcular métricas de salud.")
        db.close()
        return

    score = _health_score(metrics)

    # ── Score General ──────────────────────────────────────────────────
    if score >= 70:
        score_color = COLORS['success']
        score_label = 'Saludable'
    elif score >= 40:
        score_color = COLORS['warning']
        score_label = 'Necesita atención'
    else:
        score_color = COLORS['danger']
        score_label = 'Crítico'

    st.markdown(f"""
    <div style="background: {COLORS['bg_secondary']}; border-radius: 12px;
                padding: 1.5rem; margin-bottom: 1.5rem;
                border-left: 4px solid {score_color};">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div style="font-size: 0.8rem; color: {COLORS['text_muted']};
                            text-transform: uppercase; letter-spacing: 0.1em;">
                    Score General</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {score_color};
                            line-height: 1.1;">{score}<span style="font-size: 1rem; opacity: 0.6;">/100</span></div>
                <div style="font-size: 0.9rem; color: {COLORS['text_secondary']};">{score_label}</div>
            </div>
            <div style="flex: 1; min-width: 200px; max-width: 400px;">
                <div style="background: {COLORS['bg_tertiary']}; border-radius: 8px;
                            height: 16px; overflow: hidden;">
                    <div style="width: {score}%; height: 100%; border-radius: 8px;
                                background: linear-gradient(90deg, {score_color}, {score_color}dd);
                                transition: width 0.5s;"></div>
                </div>
                <div style="display: flex; justify-content: space-between;
                            font-size: 0.7rem; color: {COLORS['text_muted']}; margin-top: 4px;">
                    <span>🔴 Crítico</span><span>🟡 Atención</span><span>🟢 Saludable</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tarjetas de métricas ──────────────────────────────────────────
    status_colors = {
        'green': COLORS['success'],
        'yellow': COLORS['warning'],
        'red': COLORS['danger'],
    }
    status_emojis = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}

    # Obtener sparklines de engagement histórico
    try:
        hist_df = db.get_channel_metrics_history(channel_id)
        eng_history = hist_df['avg_engagement'].tolist() if not hist_df.empty and 'avg_engagement' in hist_df.columns else []
        views_history = hist_df['total_views'].tolist() if not hist_df.empty and 'total_views' in hist_df.columns else []
    except Exception:
        eng_history = []
        views_history = []

    # Sparkline para cada métrica según su tipo
    sparkline_map = {
        'Tendencia de engagement': eng_history[-14:] if eng_history else [],
        'Vistas promedio recientes': views_history[-14:] if views_history else [],
    }

    # Renderizar en grilla 3 columnas
    cols_per_row = 3
    for row_start in range(0, len(metrics), cols_per_row):
        cols = st.columns(cols_per_row)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx >= len(metrics):
                break
            m = metrics[idx]
            sc = status_colors.get(m['status'], COLORS['info'])
            emoji = status_emojis.get(m['status'], '⚪')
            spark_data = sparkline_map.get(m['name'], [])
            spark_html = _sparkline_svg(spark_data, sc) if spark_data else ''

            card_html = (
                f'<div style="background:{COLORS["bg_secondary"]};border-radius:10px;'
                f'padding:1rem;border-left:3px solid {sc};margin-bottom:0.5rem;">'
                f'<div style="display:flex;justify-content:space-between;align-items:start;">'
                f'<div style="font-size:0.75rem;color:{COLORS["text_muted"]};'
                f'text-transform:uppercase;letter-spacing:0.05em;">'
                f'{_esc(m["icon"])} {_esc(m["name"])}</div>'
                f'<span style="font-size:1.1rem;">{emoji}</span></div>'
                f'<div style="font-size:1.5rem;font-weight:700;color:{sc};margin:0.4rem 0;">'
                f'{_esc(m["value"])}</div>'
                f'<div style="font-size:0.7rem;color:{COLORS["text_muted"]};">'
                f'{_esc(m["detail"])}</div></div>'
            )
            col.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)

    # ── Acciones sugeridas (automáticas) ──────────────────────────────
    red_metrics = [m for m in metrics if m['status'] == 'red']
    yellow_metrics = [m for m in metrics if m['status'] == 'yellow']
    actions = []

    for m in red_metrics:
        if 'Frecuencia' in m['name']:
            actions.append(f"🔴 **Publica más seguido** — solo {m['value']}. Objetivo: 3+ videos/semana.")
        elif 'Frescura' in m['name']:
            actions.append(f"🔴 **Publica un video ahora** — llevas {m['value']} sin contenido nuevo.")
        elif 'engagement' in m['name'].lower():
            actions.append(f"🔴 **Engagement cayendo** ({m['value']}) — revisa CTAs y hooks de tus últimos videos.")
        elif 'Vistas' in m['name']:
            actions.append(f"🔴 **Vistas por debajo del promedio** — experimenta con nuevos temas o formatos.")
        elif 'Retención' in m['name']:
            actions.append(f"🔴 **Retención baja** ({m['value']}) — mejora los primeros 3 segundos del video.")
        elif 'CTR' in m['name']:
            actions.append(f"🔴 **CTR bajo** ({m['value']}) — rediseña tus miniaturas y títulos.")
        else:
            actions.append(f"🔴 **{m['name']}** necesita atención: {m['value']}.")

    for m in yellow_metrics:
        if 'engagement' in m['name'].lower():
            actions.append(f"🟡 **Engagement estable pero sin crecer** ({m['value']}) — prueba preguntas en video para generar comentarios.")
        elif 'Consistencia' in m['name']:
            actions.append(f"🟡 **Resultados irregulares** — tu engagement varía mucho entre videos. Identifica qué formato funciona mejor.")
        elif 'Frescura' in m['name']:
            actions.append(f"🟡 **Han pasado {m['value']}** desde tu último video — no pierdas el ritmo.")
        else:
            actions.append(f"🟡 **{m['name']}** puede mejorar: {m['value']}.")

    if actions:
        ui_section_divider()
        st.markdown("### ⚡ Acciones sugeridas")
        for action in actions[:5]:
            st.markdown(action)
    elif all(m['status'] == 'green' for m in metrics):
        ui_section_divider()
        st.markdown("### ✅ ¡Excelente!")
        st.success("Todas las métricas están en verde. ¡Sigue así!")

    # ── Diagnóstico IA ────────────────────────────────────────────────
    ui_section_divider()
    st.markdown("### 🧠 Diagnóstico con IA")

    anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')

    # Buscar el reporte más reciente (cualquier antigüedad)
    existing_report = db.get_latest_health_report(channel_id)
    existing_diagnosis = None
    report_id = None
    if existing_report and existing_report.get('ai_diagnosis'):
        existing_diagnosis = existing_report['ai_diagnosis']
        report_id = existing_report['id']

    # ── Función interna para generar/regenerar diagnóstico ───────────
    def _run_diagnosis():
        ch_lang = _detect_channel_language(db, channel_id)
        try:
            cur = db.conn.cursor()
            cur.execute("SELECT channel_name FROM channels WHERE channel_id = %s", (channel_id,))
            row = cur.fetchone()
            ch_name = row['channel_name'] if row else channel_id
        except Exception:
            ch_name = channel_id

        metrics_json = json.dumps(
            [{k: v for k, v in m.items() if k != 'raw'} for m in metrics],
            ensure_ascii=False,
        )
        rid = db.save_health_report(channel_id, score, metrics_json)
        analyzer = AIAnalyzer(anthropic_key)
        diagnosis = analyzer.generate_health_diagnosis(
            channel_name=ch_name,
            metrics=metrics,
            health_score=score,
            language=ch_lang,
        )
        if not diagnosis.startswith("Error"):
            db.update_health_diagnosis(rid, diagnosis)
            return diagnosis, None
        return None, diagnosis

    if existing_diagnosis:
        # Mostrar diagnóstico guardado
        st.markdown(existing_diagnosis)
        created_at = str(existing_report.get('created_at', '—'))[:16]
        col_ts, col_btn = st.columns([3, 1])
        col_ts.caption(f"Generado: {created_at}")

        if anthropic_key:
            if col_btn.button("🔄 Regenerar", key="btn_health_regen", help="Genera un nuevo diagnóstico con las métricas actuales"):
                with st.spinner("Regenerando diagnóstico con IA..."):
                    new_diag, err = _run_diagnosis()
                if new_diag:
                    st.success("Diagnóstico actualizado")
                    st.rerun()
                else:
                    st.error(f"Error: {err}")
    else:
        if anthropic_key:
            ch_language = _detect_channel_language(db, channel_id)
            lang_label = {'en': '🇺🇸 EN', 'es': '🇪🇸 ES', 'pt': '🇧🇷 PT'}.get(ch_language, ch_language)

            if st.button(f"🧠 Analizar con IA ({lang_label})", type="secondary", key="btn_health_ai"):
                with st.spinner("Analizando canal con IA..."):
                    new_diag, err = _run_diagnosis()
                if new_diag:
                    st.markdown(new_diag)
                    st.success("Diagnóstico guardado")
                else:
                    st.error(f"Error: {err}")
        else:
            st.info("Configura ANTHROPIC_API_KEY en .env para habilitar el diagnóstico con IA.")

    db.close()
