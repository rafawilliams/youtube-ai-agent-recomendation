"""
Página: Revenue y ROI (Mejoras 16.1, 16.2, 16.3)
"""
import logging
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database import YouTubeDatabase
from page_modules.shared import (
    _get_colors, _esc, ui_page_header, ui_section_divider, ui_section_header,
    ui_metric_card,
)
from revenue_analyzer import RevenueEstimator, ContentROICalculator, EvergreenDetector

log = logging.getLogger(__name__)


def show_revenue_analysis(df: pd.DataFrame, channel_id: str):
    """Estimación de ingresos, ROI por tipo de contenido y detección evergreen."""

    COLORS = _get_colors()
    ui_page_header("💰", "Revenue y ROI",
                   "Estimación de ingresos por CPM, ROI por tipo de contenido y videos evergreen")

    if df.empty or not channel_id:
        st.warning("No hay datos disponibles. Ejecuta el pipeline primero.")
        return

    # ── Sidebar: Configuración CPM y horas de producción ─────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💰 Configuración Revenue**")
    default_cpm = float(os.getenv('ESTIMATED_CPM', '4.50'))
    cpm = st.sidebar.number_input(
        "CPM estimado (USD)", min_value=0.1, max_value=100.0,
        value=default_cpm, step=0.5, key="revenue_cpm",
        help="Costo por mil impresiones. Ajusta según tu nicho y región."
    )
    st.sidebar.markdown("**⏱ Horas de producción**")
    hours_short = st.sidebar.number_input(
        "Short (horas)", min_value=0.1, max_value=50.0,
        value=1.0, step=0.5, key="revenue_hours_short"
    )
    hours_long = st.sidebar.number_input(
        "Video Largo (horas)", min_value=0.1, max_value=100.0,
        value=8.0, step=1.0, key="revenue_hours_long"
    )

    tab1, tab2, tab3 = st.tabs([
        "💵 Estimación de Ingresos",
        "📊 ROI por Tipo",
        "🌲 Videos Evergreen",
    ])

    # ── Tab 1: Estimación de Ingresos (16.1) ─────────────────────────
    with tab1:
        ui_section_header("💵", "Estimación de Ingresos",
                          f"Basado en CPM de ${cpm:.2f} USD")

        estimator = RevenueEstimator(cpm=cpm)
        df_rev = df.copy()
        df_rev['view_count'] = pd.to_numeric(df_rev['view_count'], errors='coerce').fillna(0)
        df_rev['estimated_revenue'] = df_rev['view_count'].apply(estimator.estimate_video_revenue)

        total_rev = float(df_rev['estimated_revenue'].sum())
        avg_rev = float(df_rev['estimated_revenue'].mean())
        best_idx = df_rev['estimated_revenue'].idxmax() if not df_rev.empty else None
        best_title = _esc(str(df_rev.loc[best_idx, 'title'])[:40]) if best_idx is not None else '-'
        best_rev = float(df_rev.loc[best_idx, 'estimated_revenue']) if best_idx is not None else 0

        # KPIs
        cols = st.columns(4)
        with cols[0]:
            st.markdown(ui_metric_card("💰", "Ingresos totales est.",
                                       f"${total_rev:,.2f}"),
                        unsafe_allow_html=True)
        with cols[1]:
            st.markdown(ui_metric_card("📊", "Promedio por video",
                                       f"${avg_rev:,.2f}"),
                        unsafe_allow_html=True)
        with cols[2]:
            st.markdown(ui_metric_card("🏆", "Mejor video",
                                       f"${best_rev:,.2f}"),
                        unsafe_allow_html=True)
        with cols[3]:
            st.markdown(ui_metric_card("📈", "CPM configurado",
                                       f"${cpm:.2f}"),
                        unsafe_allow_html=True)

        ui_section_divider()

        # Gráfico mensual
        monthly_df = estimator.estimate_channel_monthly(df_rev)
        if not monthly_df.empty:
            ui_section_header("📅", "Ingresos mensuales estimados")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=monthly_df['month'],
                y=monthly_df['estimated_revenue'],
                name='Ingresos ($)',
                marker_color=COLORS.get('success', '#10B981'),
                opacity=0.8,
            ))
            fig.add_trace(go.Scatter(
                x=monthly_df['month'],
                y=monthly_df['video_count'],
                name='Videos publicados',
                yaxis='y2',
                mode='lines+markers',
                line=dict(color=COLORS.get('primary', '#6366F1'), width=2),
                marker=dict(size=7),
            ))
            fig.update_layout(
                yaxis_title="Ingresos estimados (USD)",
                yaxis2=dict(title="Videos", overlaying='y', side='right'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS.get('text_primary', '#F8FAFC')),
                height=380,
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Proyecciones
            projections = estimator.project_revenue(monthly_df)
            if any(v > 0 for v in projections.values()):
                ui_section_header("🔮", "Proyecciones de ingresos",
                                  "Basadas en tendencia lineal de los últimos 6 meses")

                pcols = st.columns(3)
                labels = {3: "3 meses", 6: "6 meses", 12: "12 meses"}
                icons = {3: "📅", 6: "📆", 12: "🗓"}
                for i, (months, label) in enumerate(labels.items()):
                    with pcols[i]:
                        val = projections.get(months, 0)
                        st.markdown(
                            ui_metric_card(icons[months], label,
                                           f"${val:,.2f}"),
                            unsafe_allow_html=True,
                        )

        ui_section_divider()

        # Top 10 videos por ingresos
        ui_section_header("🏆", "Top 10 videos por ingresos estimados")
        top10 = df_rev.nlargest(10, 'estimated_revenue')
        display_df = top10[['title', 'video_type', 'view_count', 'estimated_revenue']].copy()
        display_df.columns = ['Título', 'Tipo', 'Vistas', 'Ingresos Est. ($)']
        display_df['Ingresos Est. ($)'] = display_df['Ingresos Est. ($)'].apply(
            lambda x: f"${x:,.2f}"
        )
        display_df['Vistas'] = display_df['Vistas'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── Tab 2: ROI por Tipo (16.2) ───────────────────────────────────
    with tab2:
        ui_section_header("📊", "ROI por Tipo de Contenido",
                          "Ingreso estimado por hora de producción")

        calculator = ContentROICalculator(production_hours={
            'Short': hours_short,
            'Video Largo': hours_long,
        })
        roi_df = calculator.calculate_roi(df, cpm)

        if roi_df.empty:
            st.info("No hay datos suficientes para calcular ROI por tipo.")
        else:
            # KPIs
            best_type = roi_df.iloc[0]['video_type'] if not roi_df.empty else '-'
            rcols = st.columns(3)
            with rcols[0]:
                st.markdown(
                    ui_metric_card("🏆", "Mejor ROI",
                                   f"{best_type}",
                                   delta_type="positive"),
                    unsafe_allow_html=True,
                )

            for i, vtype in enumerate(['Short', 'Video Largo']):
                row = roi_df[roi_df['video_type'] == vtype]
                if not row.empty and i + 1 < len(rcols):
                    rph = float(row['revenue_per_hour'].iloc[0])
                    with rcols[i + 1]:
                        st.markdown(
                            ui_metric_card(
                                "⚡" if vtype == 'Short' else "🎬",
                                f"{vtype} $/hora",
                                f"${rph:,.2f}",
                            ),
                            unsafe_allow_html=True,
                        )

            ui_section_divider()

            # Gráfico comparativo
            type_colors = {
                'Short': COLORS.get('secondary', '#EC4899'),
                'Video Largo': COLORS.get('primary', '#6366F1'),
            }
            fig_roi = go.Figure()
            for _, row in roi_df.iterrows():
                vtype = row['video_type']
                fig_roi.add_trace(go.Bar(
                    x=[row['revenue_per_hour']],
                    y=[vtype],
                    orientation='h',
                    name=vtype,
                    marker_color=type_colors.get(vtype,
                                                 COLORS.get('accent', '#F59E0B')),
                    text=[f"${row['revenue_per_hour']:.2f}/h"],
                    textposition='auto',
                ))
            fig_roi.update_layout(
                title="Ingreso estimado por hora de producción",
                xaxis_title="USD / hora",
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS.get('text_primary', '#F8FAFC')),
                height=250,
                margin=dict(l=120, r=40, t=50, b=40),
            )
            st.plotly_chart(fig_roi, use_container_width=True)

            # Tabla detallada
            detail = roi_df[['video_type', 'total_videos', 'avg_views',
                             'avg_revenue', 'production_hours',
                             'revenue_per_hour']].copy()
            detail.columns = ['Tipo', 'Videos', 'Vistas prom.',
                              'Ingreso prom. ($)', 'Horas prod.',
                              'USD/hora']
            detail['Vistas prom.'] = detail['Vistas prom.'].apply(
                lambda x: f"{x:,.0f}")
            detail['Ingreso prom. ($)'] = detail['Ingreso prom. ($)'].apply(
                lambda x: f"${x:,.2f}")
            detail['USD/hora'] = detail['USD/hora'].apply(
                lambda x: f"${x:,.2f}")
            st.dataframe(detail, use_container_width=True, hide_index=True)

            # Insight
            short_row = roi_df[roi_df['video_type'] == 'Short']
            long_row = roi_df[roi_df['video_type'] == 'Video Largo']
            if not short_row.empty and not long_row.empty:
                short_rph = float(short_row['revenue_per_hour'].iloc[0])
                long_rph = float(long_row['revenue_per_hour'].iloc[0])
                if short_rph > long_rph:
                    st.info(
                        f"💡 **Insight:** Los Shorts generan **${short_rph:.2f}/h** vs "
                        f"**${long_rph:.2f}/h** de videos largos. "
                        f"Considera aumentar la producción de Shorts para maximizar eficiencia."
                    )
                else:
                    st.info(
                        f"💡 **Insight:** Los videos largos generan **${long_rph:.2f}/h** vs "
                        f"**${short_rph:.2f}/h** de Shorts. "
                        f"Los videos largos son más rentables por hora de producción."
                    )

    # ── Tab 3: Videos Evergreen (16.3) ────────────────────────────────
    with tab3:
        ui_section_header("🌲", "Videos Evergreen",
                          "Videos que siguen generando vistas meses después de publicados")

        # Cargar scores de la BD
        eg_df = pd.DataFrame()
        try:
            with YouTubeDatabase() as db:
                eg_df = db.get_evergreen_scores(channel_id)
        except Exception as e:
            log.warning("Error cargando evergreen scores: %s", e)

        if eg_df.empty:
            st.info(
                "No hay datos evergreen aún. Ejecuta el pipeline para calcular "
                "los scores automáticamente (requiere historial de métricas)."
            )
        else:
            # KPIs
            total_analyzed = len(eg_df)
            eg_count = len(eg_df[eg_df['classification'] == 'evergreen'])
            spike_count = len(eg_df[eg_df['classification'] == 'spike_decline'])
            avg_score = float(eg_df['evergreen_score'].mean())

            ecols = st.columns(4)
            with ecols[0]:
                st.markdown(
                    ui_metric_card("🎬", "Videos analizados",
                                   str(total_analyzed)),
                    unsafe_allow_html=True,
                )
            with ecols[1]:
                st.markdown(
                    ui_metric_card("🌲", "Evergreen",
                                   str(eg_count),
                                   delta_type="positive"),
                    unsafe_allow_html=True,
                )
            with ecols[2]:
                st.markdown(
                    ui_metric_card("📉", "Pico y caída",
                                   str(spike_count),
                                   delta_type="negative"),
                    unsafe_allow_html=True,
                )
            with ecols[3]:
                st.markdown(
                    ui_metric_card("📊", "Score promedio",
                                   f"{avg_score:.2f}"),
                    unsafe_allow_html=True,
                )

            ui_section_divider()

            # Pie chart de clasificaciones
            class_labels = EvergreenDetector.CLASSIFICATION_LABELS_ES
            class_counts = eg_df['classification'].value_counts().reset_index()
            class_counts.columns = ['Clasificación', 'Cantidad']
            class_counts['Clasificación'] = class_counts['Clasificación'].map(
                lambda c: class_labels.get(c, c)
            )
            color_map = {
                '🌲 Evergreen': COLORS.get('success', '#10B981'),
                '📉 Pico y caída': COLORS.get('danger', '#EF4444'),
                '📊 Estable': COLORS.get('warning', '#F59E0B'),
                '❓ Datos insuficientes': COLORS.get('text_secondary', '#94A3B8'),
            }
            fig_pie = px.pie(
                class_counts, names='Clasificación', values='Cantidad',
                color='Clasificación', color_discrete_map=color_map,
            )
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS.get('text_primary', '#F8FAFC')),
                height=300,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            ui_section_divider()

            # Ranking de evergreen
            ui_section_header("🏆", "Ranking de videos evergreen",
                              "Ordenados por score (mayor = más evergreen)")

            display_cols = ['title', 'video_type', 'classification',
                           'evergreen_score', 'days_tracked',
                           'recent_daily_views']
            avail = [c for c in display_cols if c in eg_df.columns]
            if 'view_count' in eg_df.columns:
                avail.append('view_count')

            eg_display = eg_df[avail].copy()
            rename_map = {
                'title': 'Título',
                'video_type': 'Tipo',
                'classification': 'Clasificación',
                'evergreen_score': 'Score',
                'days_tracked': 'Días trackeados',
                'recent_daily_views': 'Vistas diarias recientes',
                'view_count': 'Vistas totales',
            }
            eg_display = eg_display.rename(columns=rename_map)
            if 'Clasificación' in eg_display.columns:
                eg_display['Clasificación'] = eg_display['Clasificación'].map(
                    lambda c: class_labels.get(c, c)
                )
            st.dataframe(eg_display, use_container_width=True, hide_index=True)

            # Impacto en revenue
            if 'view_count' in eg_df.columns:
                ui_section_divider()
                ui_section_header("💰", "Impacto en revenue",
                                  "Contribución de videos evergreen al ingreso total estimado")

                eg_df['view_count'] = pd.to_numeric(
                    eg_df['view_count'], errors='coerce'
                ).fillna(0)
                eg_df['est_revenue'] = eg_df['view_count'] * cpm / 1000

                total_eg_rev = float(
                    eg_df[eg_df['classification'] == 'evergreen']['est_revenue'].sum()
                )
                total_all_rev = float(eg_df['est_revenue'].sum())
                pct = (total_eg_rev / total_all_rev * 100) if total_all_rev > 0 else 0

                icols = st.columns(3)
                with icols[0]:
                    st.markdown(
                        ui_metric_card("🌲", "Revenue evergreen",
                                       f"${total_eg_rev:,.2f}"),
                        unsafe_allow_html=True,
                    )
                with icols[1]:
                    st.markdown(
                        ui_metric_card("💰", "Revenue total analizado",
                                       f"${total_all_rev:,.2f}"),
                        unsafe_allow_html=True,
                    )
                with icols[2]:
                    st.markdown(
                        ui_metric_card("📊", "% Evergreen",
                                       f"{pct:.1f}%",
                                       delta_type="positive" if pct > 40 else "neutral"),
                        unsafe_allow_html=True,
                    )

                if pct > 40:
                    st.success(
                        f"🌲 **Excelente:** El {pct:.0f}% de tus ingresos proviene de "
                        f"videos evergreen. Tu contenido tiene gran valor a largo plazo."
                    )
                elif pct > 20:
                    st.info(
                        f"📊 **Bien:** El {pct:.0f}% de tus ingresos viene de evergreen. "
                        f"Busca crear más contenido SEO-driven para aumentar esta cifra."
                    )
                else:
                    st.warning(
                        f"⚠ Solo el {pct:.0f}% de tus ingresos es evergreen. "
                        f"Considera crear más tutoriales, guías o contenido atemporal."
                    )
