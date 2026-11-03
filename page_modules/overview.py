"""
Página de Resumen General (Overview) del dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io

from database import YouTubeDatabase
from google_integrations import GoogleSheetsExporter
from page_modules.shared import (
    _get_colors, _esc, ui_page_header, ui_section_divider,
    VIDEO_TYPE_SEQUENCE, _load_prefs, _save_prefs,
)
from page_modules.shared.data import _DEFAULT_PREFS


# ─── Overview Widgets (Mejora 15.1) ────────────────────────────────────────

_WIDGET_REGISTRY = {}


def _register_widget(key, label):
    """Decorator para registrar widgets del overview."""
    def decorator(fn):
        _WIDGET_REGISTRY[key] = {'fn': fn, 'label': label}
        return fn
    return decorator


@_register_widget('kpis', '📊 KPIs principales')
def _widget_kpis(df, channel_name, subscriber_count):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        subs_display = f"{subscriber_count:,}" if subscriber_count else "No disponible"
        st.metric("👥 Suscriptores", subs_display)
    with col2:
        st.metric("Total Videos", f"{len(df)}")
    with col3:
        st.metric("Vistas Totales", f"{df['view_count'].sum():,.0f}")
    with col4:
        st.metric("Promedio por Video", f"{df['view_count'].mean():,.0f}")
    with col5:
        st.metric("Engagement Rate", f"{df['engagement_rate'].mean():.2f}%")


@_register_widget('distribution', '🎬 Distribución de Contenido')
def _widget_distribution(df, channel_name, subscriber_count):
    COLORS = _get_colors()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎬 Distribución de Contenido")
        type_counts = df['video_type'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values, names=type_counts.index,
            title="Shorts vs Videos Largos",
            color_discrete_sequence=VIDEO_TYPE_SEQUENCE,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.subheader("📈 Performance por Tipo")
        perf = df.groupby('video_type').agg(
            {'view_count': 'mean', 'engagement_rate': 'mean'}
        ).reset_index()
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='Vistas Promedio',
            x=perf['video_type'], y=perf['view_count'],
            marker_color=COLORS['primary'],
        ))
        fig_bar.update_layout(title="Vistas Promedio por Tipo", yaxis_title="Vistas", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)


@_register_widget('performance', '🏆 Top Videos')
def _widget_performance(df, channel_name, subscriber_count):
    st.subheader("🏆 Top 10 Videos con Mejor Performance")
    top10 = df.nlargest(10, 'view_count')[
        ['title', 'video_type', 'view_count', 'engagement_rate', 'published_at']
    ].copy()
    top10['published_at'] = top10['published_at'].dt.strftime('%Y-%m-%d')
    st.dataframe(
        top10.rename(columns={
            'title': 'Título', 'video_type': 'Tipo', 'view_count': 'Vistas',
            'engagement_rate': 'Engagement %', 'published_at': 'Publicado',
        }),
        use_container_width=True, hide_index=True,
    )


@_register_widget('export', '📥 Exportar datos')
def _widget_export(df, channel_name, subscriber_count):
    st.subheader("📥 Exportar datos")
    export_cols = ['title', 'video_type', 'published_at', 'view_count',
                   'like_count', 'comment_count', 'engagement_rate', 'duration_seconds',
                   'tags', 'description']
    export_df = df[[c for c in export_cols if c in df.columns]].copy()
    if 'published_at' in export_df.columns:
        export_df['published_at'] = export_df['published_at'].dt.strftime('%Y-%m-%d %H:%M')

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name='Videos', index=False)
        summary = pd.DataFrame({
            'Métrica': ['Canal', 'Total videos', 'Vistas totales', 'Vistas promedio', 'Engagement %'],
            'Valor': [channel_name, len(df), int(df['view_count'].sum()),
                      int(df['view_count'].mean()), round(float(df['engagement_rate'].mean()), 2)]
        })
        summary.to_excel(writer, sheet_name='Resumen', index=False)

    st.download_button(
        label="📥 Descargar Excel (.xlsx)", data=buffer.getvalue(),
        file_name=f"{channel_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if st.button("📊 Exportar a Google Sheets", key="btn_gsheets_metrics"):
        try:
            with st.spinner("Creando Google Sheet..."):
                gsheets = GoogleSheetsExporter()
                result = gsheets.export_video_metrics(df, channel_name=channel_name)
            st.success("✅ Google Sheet creado")
            st.markdown(f"[📊 Abrir Google Sheet]({result['spreadsheet_url']})")
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Error: {e}")


def show_overview(df):
    """Muestra el resumen general del canal con widgets personalizables (15.1)."""
    ui_page_header("📊", "Resumen General", "Vista rápida de las métricas principales del canal")

    if df.empty:
        st.warning("No hay datos disponibles. Ejecuta primero el script main.py para extraer datos.")
        return

    channel_name = df['channel_title'].iloc[0] if pd.notna(df['channel_title'].iloc[0]) else "Canal"
    raw_subs = df['subscriber_count'].iloc[0] if 'subscriber_count' in df.columns else None
    subscriber_count = int(raw_subs) if raw_subs is not None and pd.notna(raw_subs) else None

    st.subheader(f"📺 {channel_name}")

    # Widget order from prefs
    prefs = st.session_state.get('prefs', _DEFAULT_PREFS)
    widget_order = prefs.get('widget_order', _DEFAULT_PREFS['widget_order'])
    all_keys = list(_WIDGET_REGISTRY.keys())
    widget_order = [k for k in widget_order if k in _WIDGET_REGISTRY] + \
                   [k for k in all_keys if k not in widget_order]

    # Layout customizer (collapsed expander)
    with st.expander("⚙ Personalizar layout", expanded=False):
        st.caption("Selecciona el orden de las secciones del resumen")
        new_order = []
        for i, key in enumerate(widget_order):
            options = [_WIDGET_REGISTRY[k]['label'] for k in widget_order]
            selected = st.selectbox(
                f"Posición {i + 1}", options, index=i,
                key=f"widget_pos_{i}",
            )
            sel_key = next(k for k, v in _WIDGET_REGISTRY.items() if v['label'] == selected)
            new_order.append(sel_key)

        if st.button("💾 Guardar layout", key="btn_save_layout"):
            seen = set()
            deduped = []
            for k in new_order:
                if k not in seen:
                    deduped.append(k)
                    seen.add(k)
            st.session_state.prefs['widget_order'] = deduped
            _save_prefs(st.session_state.prefs)
            st.success("Layout guardado")
            st.rerun()

    ui_section_divider()

    # Render widgets in order
    for key in widget_order:
        if key in _WIDGET_REGISTRY:
            _WIDGET_REGISTRY[key]['fn'](df, channel_name, subscriber_count)
            ui_section_divider()
