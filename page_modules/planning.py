"""
Página: Plan Semanal (Mejora 4.2)
"""
import json
import os

import pandas as pd
import streamlit as st

from database import YouTubeDatabase
from ai_analyzer import AIAnalyzer
from google_integrations import GoogleCalendarExporter, GoogleSheetsExporter
from page_modules.shared import (
    _esc, ui_page_header, ui_section_divider, _detect_channel_language,
)


def show_weekly_plan(df: pd.DataFrame, channel_id: str | None):
    """Página de planificación semanal generada por Claude."""
    ui_page_header("🗓", "Plan Semanal", "Planificación de contenido generada por Claude AI")

    if not channel_id or df.empty:
        st.warning("Selecciona un canal con datos en el panel lateral.")
        return

    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key:
        st.error("ANTHROPIC_API_KEY no configurada en el archivo .env.")
        return

    try:
        db = YouTubeDatabase()

        # ── Mostrar plan guardado más reciente ────────────────────────────
        plans_df = db.get_weekly_plans(channel_id, limit=4)

        if not plans_df.empty:
            latest = plans_df.iloc[0]
            try:
                days_data = json.loads(latest['plan_json'])
            except (json.JSONDecodeError, TypeError):
                days_data = []

            st.markdown(f"**Semana del {latest['week_start_date']}**  •  "
                        f"Generado el {str(latest['generated_at'])[:16]}")

            if latest.get('strategy'):
                st.info(f"**Estrategia:** {latest['strategy']}")

            # ── Calendario visual 7 columnas ─────────────────────────────
            _render_weekly_calendar(days_data)

            # ── Historial de planes anteriores ───────────────────────────
            if len(plans_df) > 1:
                with st.expander("📂 Ver planes anteriores"):
                    for _, row in plans_df.iloc[1:].iterrows():
                        st.markdown(f"**Semana del {row['week_start_date']}**")
                        try:
                            old_days = json.loads(row['plan_json'])
                            _render_weekly_calendar(old_days, compact=True)
                        except (json.JSONDecodeError, TypeError):
                            st.caption("(datos no disponibles)")
                        ui_section_divider()

            # ── Exportar plan actual ──────────────────────────────────────
            ui_section_divider()
            st.subheader("Exportar plan")
            exp_col1, exp_col2 = st.columns(2)

            with exp_col1:
                if st.button("📅 Exportar a Google Calendar", key="btn_gcal"):
                    try:
                        with st.spinner("Exportando a Google Calendar..."):
                            gcal = GoogleCalendarExporter()
                            gcal_result = gcal.export_weekly_plan(
                                days_data,
                                strategy=latest.get('strategy', ''),
                            )
                        if gcal_result['created'] > 0:
                            st.success(f"✅ {gcal_result['created']} evento(s) creado(s) en Google Calendar")
                            for link in gcal_result['event_links']:
                                if link:
                                    st.markdown(f"[Abrir evento]({link})")
                        if gcal_result['skipped'] > 0:
                            st.info(f"ℹ️ {gcal_result['skipped']} evento(s) ya existían (omitidos)")
                        if gcal_result['errors']:
                            for err in gcal_result['errors']:
                                st.warning(f"⚠️ {err}")
                        if gcal_result['created'] == 0 and gcal_result['skipped'] == 0 and not gcal_result['errors']:
                            st.info("No hay días de publicación en este plan.")
                    except FileNotFoundError as e:
                        st.error(f"❌ {e}")
                    except Exception as e:
                        st.error(f"❌ Error al exportar a Calendar: {e}")

            with exp_col2:
                if st.button("📊 Exportar a Google Sheets", key="btn_gsheets_plan"):
                    try:
                        with st.spinner("Creando Google Sheet..."):
                            gsheets = GoogleSheetsExporter()
                            ch_name = (
                                df.iloc[0]['channel_title']
                                if not df.empty and 'channel_title' in df.columns
                                else channel_id
                            )
                            gsheets_result = gsheets.export_weekly_plan(
                                days_data,
                                strategy=latest.get('strategy', ''),
                                week_start_date=str(latest['week_start_date']),
                                channel_name=ch_name,
                            )
                        st.success("✅ Google Sheet creado exitosamente")
                        st.markdown(f"[📊 Abrir Google Sheet]({gsheets_result['spreadsheet_url']})")
                    except FileNotFoundError as e:
                        st.error(f"❌ {e}")
                    except Exception as e:
                        st.error(f"❌ Error al exportar a Sheets: {e}")

        else:
            st.info("No hay planes semanales guardados. Genera el primero con el botón de abajo.")

        # ── Botón de generación ───────────────────────────────────────────
        ui_section_divider()
        st.subheader("Generar nuevo plan")

        plan_lang_detected = _detect_channel_language(db, channel_id)
        _plan_lang_opts = {'🇪🇸 Español': 'es', '🇺🇸 English': 'en', '🇧🇷 Português': 'pt'}
        _plan_lang_labels = list(_plan_lang_opts.keys())
        _plan_default = next(
            (k for k, v in _plan_lang_opts.items() if v == plan_lang_detected),
            '🇪🇸 Español',
        )
        _plan_sel = st.selectbox(
            "Idioma del plan",
            _plan_lang_labels,
            index=_plan_lang_labels.index(_plan_default),
            key="plan_lang_select",
        )
        plan_language = _plan_lang_opts[_plan_sel]

        if st.button("🗓 Generar plan para los próximos 7 días", type="primary"):
            with st.spinner("Claude está analizando el canal y diseñando el plan semanal..."):
                past_results_df = db.get_recommendation_results(channel_id, limit=5)
                past_results    = past_results_df.to_dict('records') if not past_results_df.empty else []

                channel_info = {
                    'channel_id':       channel_id,
                    'channel_name':     df.iloc[0]['channel_title'] if 'channel_title' in df.columns else channel_id,
                    'subscriber_count': 0,
                }

                analyzer = AIAnalyzer(anthropic_api_key)
                plan     = analyzer.generate_weekly_plan(
                    df, channel_info, past_results=past_results, language=plan_language,
                )

                if 'error' in plan:
                    st.error(plan['error'])
                else:
                    db.save_weekly_plan(
                        channel_id,
                        plan['week_start_date'],
                        json.dumps(plan['days'], ensure_ascii=False),
                        plan.get('strategy', ''),
                        plan['generated_at'],
                    )
                    st.success("✅ Plan semanal guardado")
                    st.rerun()

        db.close()

    except Exception as e:
        st.error(f"Error en plan semanal: {e}")


def _render_weekly_calendar(days_data: list, compact: bool = False):
    """Renderiza el calendario semanal en filas de 3 tarjetas con texto legible."""
    if not days_data:
        st.caption("(sin datos de días)")
        return

    COLS_PER_ROW = 3
    days_list = days_data[:7]

    for row_start in range(0, len(days_list), COLS_PER_ROW):
        row_days = days_list[row_start:row_start + COLS_PER_ROW]
        cols = st.columns(COLS_PER_ROW)
        for i, day in enumerate(row_days):
            col = cols[i]
            publish = day.get('publish', False)
            vtype   = day.get('type') or ''
            topic   = day.get('topic', '')
            hour    = day.get('hour')
            reason  = day.get('reason', '')

            if publish:
                css_class = 'publish-long' if vtype == 'Video Largo' else 'publish-short'
                badge = '🎬' if vtype == 'Video Largo' else '📱'
                label = vtype
            else:
                css_class = 'rest'
                badge, label = '💤', 'Descanso'

            hour_str = f"{hour}:00" if hour is not None else ''

            if compact:
                col.markdown(
                    f"<div class='calendar-card {css_class}' style='min-height:70px;font-size:13px'>"
                    f"<b>{_esc(day.get('day',''))}</b><br>{_esc(badge)} {_esc(label)}"
                    f"{'<br>' + _esc(hour_str) if hour_str else ''}"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                topic_html = (
                    f"<p style='font-size:15px;margin:8px 0 6px;color:var(--text-secondary);line-height:1.45'>"
                    f"{_esc(topic[:100])}{'…' if len(topic)>100 else ''}</p>"
                    if topic else ""
                )
                reason_html = (
                    f"<p style='font-size:13.5px;color:var(--text-muted);margin:6px 0 0;line-height:1.4'>"
                    f"{_esc(reason[:140])}{'…' if len(reason)>140 else ''}</p>"
                    if reason else ""
                )
                col.markdown(
                    f"<div class='calendar-card {css_class}'>"
                    f"<b style='font-size:18px'>{_esc(day.get('day',''))}</b>"
                    f"<p style='font-size:14px;color:var(--text-muted);margin:4px 0'>{_esc(day.get('date',''))}</p>"
                    f"<p style='font-size:30px;margin:10px 0'>{_esc(badge)}</p>"
                    f"<p style='font-size:17px;font-weight:bold;margin:2px 0'>{_esc(label)}</p>"
                    f"{'<p style=\"font-size:15px;color:var(--text-secondary);margin:5px 0\">⏰ ' + _esc(hour_str) + '</p>' if hour_str else ''}"
                    f"{topic_html}"
                    f"{reason_html}"
                    "</div>",
                    unsafe_allow_html=True,
                )

    if not compact:
        st.markdown("")
        rows = []
        for day in days_data[:7]:
            rows.append({
                'Día':     day.get('day', ''),
                'Fecha':   day.get('date', ''),
                'Publica': '✅' if day.get('publish') else '—',
                'Formato': (lambda t: f"{t} (9:16)" if t == 'Short' else f"{t} (16:9)" if t else '—')(day.get('type')),
                'Tema':    day.get('topic', '') or '—',
                'Hora':    f"{day['hour']}:00" if day.get('hour') is not None else '—',
                'Razón':   day.get('reason', '') or '—',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
