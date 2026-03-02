"""
Motor de análisis con IA para generar recomendaciones
"""
import os
import re
import json
import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import pytz
from anthropic import Anthropic
from typing import Dict, List, Optional
from retry_config import retry_anthropic

log = logging.getLogger(__name__)

PANAMA_TZ = pytz.timezone('America/Panama')


class AIAnalyzer:
    """Analizador inteligente que usa Claude para generar insights y recomendaciones"""
    
    def __init__(self, anthropic_api_key: str):
        """
        Inicializa el analizador con la API de Anthropic
        
        Args:
            anthropic_api_key: API key de Anthropic
        """
        self.client = Anthropic(api_key=anthropic_api_key)
        
    def analyze_channel_performance(self, videos_df: pd.DataFrame, channel_info: Dict) -> Dict:
        """
        Analiza el rendimiento general del canal
        
        Args:
            videos_df: DataFrame con videos y métricas
            channel_info: Información del canal
            
        Returns:
            Diccionario con análisis completo
        """
        if videos_df.empty:
            return {'error': 'No hay datos para analizar'}
        
        # Preparar estadísticas para el análisis
        stats = self._calculate_statistics(videos_df)
        
        # Generar análisis con IA
        prompt = self._create_analysis_prompt(stats, channel_info)
        
        analysis = self._call_claude(prompt)
        
        return {
            'statistics': stats,
            'ai_insights': analysis,
            'generated_at': datetime.now().isoformat()
        }
    
    def generate_daily_recommendation(
        self,
        videos_df: pd.DataFrame,
        channel_info: Dict,
        past_results: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Genera recomendación específica para el día siguiente

        Args:
            videos_df: DataFrame con videos históricos
            channel_info: Información del canal
            past_results: Lista de dicts con historial de retroalimentación previo
                          (campos: recommendation_date, recommended_type, video_type,
                           followed_recommendation, performance_label, performance_ratio)

        Returns:
            Diccionario con recomendación detallada
        """
        if videos_df.empty or len(videos_df) < 5:
            return {
                'error': 'Necesitas al menos 5 videos publicados para generar recomendaciones',
                'recommendation': 'Publica más contenido para obtener insights basados en datos'
            }

        # Calcular estadísticas y tendencias
        stats = self._calculate_statistics(videos_df)
        trends = self._identify_trends(videos_df)
        best_performers = self._get_best_performers(videos_df)
        cadence_insights = self.analyze_cadence(videos_df)
        hourly_insights = self.analyze_hourly_saturation(videos_df)

        # Generar prompt específico para recomendación
        prompt = self._create_recommendation_prompt(
            stats, trends, best_performers, channel_info, past_results or [],
            cadence_insights=cadence_insights,
            hourly_insights=hourly_insights,
        )

        # Obtener recomendación de Claude (más tokens para incluir sugerencias de título)
        recommendation_text = self._call_claude(prompt, max_tokens=2500)

        # Parsear recomendación
        recommendation = self._parse_recommendation(recommendation_text, stats)

        return recommendation
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calcula estadísticas del canal"""
        
        # Filtrar últimos 30 días
        df_recent = df[df['published_at'] >= (datetime.now(timezone.utc) - timedelta(days=30))]
        
        # Separar por tipo
        shorts = df[df['is_short'] == True]
        long_videos = df[df['is_short'] == False]
        
        stats = {
            'total_videos': len(df),
            'total_shorts': len(shorts),
            'total_long_videos': len(long_videos),
            'videos_last_30_days': len(df_recent),
            
            # Métricas generales
            'avg_views': df['view_count'].mean(),
            'median_views': df['view_count'].median(),
            'total_views': df['view_count'].sum(),
            'avg_engagement_rate': df['engagement_rate'].mean(),
            
            # Comparación Shorts vs Videos Largos
            'shorts_avg_views': shorts['view_count'].mean() if len(shorts) > 0 else 0,
            'long_videos_avg_views': long_videos['view_count'].mean() if len(long_videos) > 0 else 0,
            'shorts_avg_engagement': shorts['engagement_rate'].mean() if len(shorts) > 0 else 0,
            'long_videos_avg_engagement': long_videos['engagement_rate'].mean() if len(long_videos) > 0 else 0,
            
            # Tendencias recientes
            'recent_avg_views': df_recent['view_count'].mean() if len(df_recent) > 0 else 0,
            'recent_shorts_count': len(df_recent[df_recent['is_short'] == True]),
            'recent_long_videos_count': len(df_recent[df_recent['is_short'] == False]),
            
            # Mejores performers
            'best_video_views': df['view_count'].max(),
            'best_video_title': df.loc[df['view_count'].idxmax(), 'title'] if len(df) > 0 else '',
            'best_video_type': df.loc[df['view_count'].idxmax(), 'video_type'] if len(df) > 0 else '',
        }
        
        return stats
    
    def _identify_trends(self, df: pd.DataFrame) -> Dict:
        """Identifica tendencias en los datos"""
        
        # Ordenar por fecha
        df_sorted = df.sort_values('published_at')
        
        # Últimos 10 videos
        recent_10 = df_sorted.tail(10)
        
        # Calcular tendencias
        trends = {
            'views_trend': 'increasing' if recent_10.tail(5)['view_count'].mean() > recent_10.head(5)['view_count'].mean() else 'decreasing',
            'engagement_trend': 'increasing' if recent_10.tail(5)['engagement_rate'].mean() > recent_10.head(5)['engagement_rate'].mean() else 'decreasing',
            'shorts_ratio_last_10': (recent_10['is_short'].sum() / len(recent_10)) * 100,
            'avg_time_between_uploads': (df_sorted['published_at'].diff().mean().days if len(df_sorted) > 1 else 0),
        }
        
        # Día de semana y hora con mejor performance (en timezone de Panamá UTC-5)
        # Usamos df_tz independiente para evitar SettingWithCopyWarning sobre el df original
        published_panama = pd.to_datetime(df['published_at'], utc=True).dt.tz_convert(PANAMA_TZ)
        df_tz = df[['view_count']].copy()
        df_tz['weekday'] = published_panama.dt.day_name()
        best_day = df_tz.groupby('weekday')['view_count'].mean().idxmax()
        trends['best_weekday'] = best_day

        df_tz['hour'] = published_panama.dt.hour
        trends['best_hour'] = df_tz.groupby('hour')['view_count'].mean().idxmax()
        
        return trends

    # ------------------------------------------------------------------
    # Cadencia y Saturación Horaria (Mejoras 13.2 / 13.3)
    # ------------------------------------------------------------------

    def analyze_cadence(self, videos_df: pd.DataFrame) -> Dict:
        """
        Analiza la correlación entre days_since_last_upload y performance
        por tipo de video (Short vs Video Largo).

        Returns:
            {cadence_by_type, optimal_cadence, summary_text}
        """
        if len(videos_df) < 10:
            return {'cadence_by_type': {}, 'optimal_cadence': {}, 'summary_text': ''}

        df = videos_df.copy()
        df = df.sort_values('published_at')

        # Calcular días desde el último upload
        dt_utc = pd.to_datetime(df['published_at'], utc=True)
        df['days_since_last'] = dt_utc.diff().dt.total_seconds().div(86400).fillna(30)

        # Asignar buckets de cadencia
        bins = [0, 1.5, 2.5, 3.5, 5.5, 7.5, 14.5, float('inf')]
        labels = ['1 día', '2 días', '3 días', '4-5 días', '6-7 días', '8-14 días', '15+ días']
        df['cadence_bucket'] = pd.cut(df['days_since_last'], bins=bins, labels=labels)

        cadence_by_type = {}
        optimal_cadence = {}
        summary_parts = []

        for vtype in ['Short', 'Video Largo']:
            type_df = df[df['video_type'] == vtype]
            if len(type_df) < 5:
                continue

            grouped = type_df.groupby('cadence_bucket', observed=True).agg(
                count=('view_count', 'size'),
                avg_views=('view_count', 'mean'),
                avg_engagement=('engagement_rate', 'mean'),
            ).reset_index()

            grouped = grouped.rename(columns={'cadence_bucket': 'bucket_label'})
            cadence_by_type[vtype] = grouped.to_dict('records')

            # Óptimo: bucket con mayor avg_views y al menos 3 videos
            valid = grouped[grouped['count'] >= 3]
            if not valid.empty:
                best = valid.loc[valid['avg_views'].idxmax()]
                optimal_cadence[vtype] = {
                    'bucket': str(best['bucket_label']),
                    'avg_views': round(float(best['avg_views'])),
                    'description': f"{best['avg_views']:,.0f} vistas promedio ({int(best['count'])} videos)",
                }
                summary_parts.append(
                    f"{vtype}s: cadencia óptima de {best['bucket_label']} "
                    f"({best['avg_views']:,.0f} vistas prom.)"
                )

        summary_text = ". ".join(summary_parts) if summary_parts else ''

        return {
            'cadence_by_type': cadence_by_type,
            'optimal_cadence': optimal_cadence,
            'summary_text': summary_text,
        }

    def analyze_hourly_saturation(
        self,
        videos_df: pd.DataFrame,
        trends_keywords: Optional[List[str]] = None,
        trends_geo: str = 'PA',
    ) -> Dict:
        """
        Analiza saturación horaria propia: dónde publicas más vs dónde rinde mejor.
        Opcionalmente cruza con Google Trends.

        Returns:
            {publishing_frequency, actual_performance, opportunity_score,
             recommendations, saturated_slots, summary_text}
        """
        weekday_labels = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
        hours = list(range(24))

        empty_matrix = pd.DataFrame(0, index=weekday_labels, columns=hours)
        empty_result = {
            'publishing_frequency': empty_matrix.copy(),
            'actual_performance': empty_matrix.copy(),
            'opportunity_score': empty_matrix.copy(),
            'recommendations': [],
            'saturated_slots': [],
            'summary_text': '',
        }

        if len(videos_df) < 10:
            return empty_result

        df = videos_df.copy()
        published_panama = pd.to_datetime(df['published_at'], utc=True).dt.tz_convert(PANAMA_TZ)
        df['weekday'] = published_panama.dt.weekday   # 0=Lunes
        df['hour'] = published_panama.dt.hour

        # Matriz de frecuencia (cuántos videos publicados por día/hora)
        freq = df.groupby(['weekday', 'hour']).size().unstack(fill_value=0)
        freq = freq.reindex(index=range(7), columns=hours, fill_value=0)
        freq.index = weekday_labels

        # Matriz de performance (vistas promedio por día/hora)
        perf = df.groupby(['weekday', 'hour'])['view_count'].mean().unstack()
        perf = perf.reindex(index=range(7), columns=hours)
        perf.index = weekday_labels
        # NaN para celdas con menos de 2 videos
        count_matrix = df.groupby(['weekday', 'hour']).size().unstack(fill_value=0)
        count_matrix = count_matrix.reindex(index=range(7), columns=hours, fill_value=0)
        perf = perf.where(count_matrix.values >= 2)

        # Score de oportunidad: alto rendimiento + baja frecuencia
        freq_pct = freq.rank(pct=True) * 100
        perf_filled = perf.fillna(0)
        perf_pct = perf_filled.rank(pct=True) * 100
        # Oportunidad = rendimiento alto - frecuencia alta (normalizado 0-100)
        raw_opp = perf_pct - freq_pct
        opportunity = ((raw_opp - raw_opp.min().min()) /
                       max(raw_opp.max().max() - raw_opp.min().min(), 1) * 100).round(0)

        # Ponderar con Google Trends si disponible
        if trends_keywords:
            try:
                from trends_analyzer import TrendsAnalyzer
                ta = TrendsAnalyzer()
                interest_df = ta.get_interest_over_time(trends_keywords, geo=trends_geo)
                if interest_df is not None and not interest_df.empty:
                    interest_df.index = pd.to_datetime(interest_df.index)
                    interest_by_day = interest_df.mean(axis=1).groupby(
                        interest_df.index.dayofweek
                    ).mean()
                    interest_by_day = interest_by_day.reindex(range(7), fill_value=50)
                    # Normalizar 0-100
                    interest_norm = (interest_by_day / max(interest_by_day.max(), 1) * 100)
                    # Ponderar: opportunity * (0.5 + 0.5 * interest_weight)
                    for i, day_label in enumerate(weekday_labels):
                        weight = 0.5 + 0.5 * (interest_norm.iloc[i] / 100)
                        opportunity.loc[day_label] = (opportunity.loc[day_label] * weight).round(0)
            except Exception:
                pass

        # Extraer top 5 oportunidades
        recommendations = []
        opp_flat = opportunity.stack().sort_values(ascending=False)
        for (day_label, hour_val), score in opp_flat.head(10).items():
            freq_val = int(freq.loc[day_label, hour_val])
            perf_val = perf.loc[day_label, hour_val]
            if pd.isna(perf_val) and freq_val == 0:
                continue
            recommendations.append({
                'day': day_label,
                'hour': int(hour_val),
                'opportunity_score': int(score),
                'avg_views': int(perf_val) if pd.notna(perf_val) else 0,
                'times_published': freq_val,
            })
            if len(recommendations) >= 5:
                break

        # Extraer top 5 slots saturados (alta frecuencia + bajo rendimiento)
        saturated_slots = []
        saturation_score = freq_pct - perf_pct
        sat_flat = saturation_score.stack().sort_values(ascending=False)
        for (day_label, hour_val), _ in sat_flat.head(10).items():
            freq_val = int(freq.loc[day_label, hour_val])
            if freq_val < 2:
                continue
            perf_val = perf.loc[day_label, hour_val]
            saturated_slots.append({
                'day': day_label,
                'hour': int(hour_val),
                'avg_views': int(perf_val) if pd.notna(perf_val) else 0,
                'times_published': freq_val,
            })
            if len(saturated_slots) >= 5:
                break

        # Resumen texto
        summary_parts = []
        if recommendations:
            top = recommendations[0]
            summary_parts.append(
                f"Mejor oportunidad: {top['day']} a las {top['hour']}:00 "
                f"(score {top['opportunity_score']}/100)"
            )
        if saturated_slots:
            sat = saturated_slots[0]
            summary_parts.append(
                f"Franja más saturada: {sat['day']} a las {sat['hour']}:00 "
                f"({sat['times_published']} videos, {sat['avg_views']:,} vistas prom.)"
            )

        return {
            'publishing_frequency': freq,
            'actual_performance': perf.fillna(0),
            'opportunity_score': opportunity,
            'recommendations': recommendations,
            'saturated_slots': saturated_slots,
            'summary_text': ". ".join(summary_parts),
        }

    def _format_opportunity_slots(self, hourly_insights: Optional[Dict]) -> str:
        """Formatea las top ventanas de oportunidad como texto para prompts."""
        if not hourly_insights or not hourly_insights.get('recommendations'):
            return "(sin datos suficientes)"
        lines = []
        for slot in hourly_insights['recommendations'][:5]:
            lines.append(
                f"- {slot['day']} a las {slot['hour']}:00 "
                f"(oportunidad: {slot['opportunity_score']}/100, "
                f"prom: {slot['avg_views']:,} vistas, "
                f"publicado {slot['times_published']}x)"
            )
        return "\n".join(lines)

    def _build_cadence_prompt_section(
        self,
        cadence_insights: Optional[Dict],
        hourly_insights: Optional[Dict],
    ) -> str:
        """Construye la sección de cadencia/horarios para inyectar en prompts."""
        parts = []
        if cadence_insights and cadence_insights.get('summary_text'):
            parts.append(f"\nCADENCIA ÓPTIMA:\n{cadence_insights['summary_text']}")
        if hourly_insights and hourly_insights.get('recommendations'):
            parts.append(
                f"\nHORARIOS DE ALTA OPORTUNIDAD:\n"
                f"{self._format_opportunity_slots(hourly_insights)}"
            )
        return "\n".join(parts) + "\n" if parts else ""

    def _get_best_performers(self, df: pd.DataFrame, top_n: int = 5) -> List[Dict]:
        """Obtiene los videos con mejor performance"""
        
        top_videos = df.nlargest(top_n, 'view_count')
        
        performers = []
        for _, video in top_videos.iterrows():
            performers.append({
                'title': video['title'],
                'type': video['video_type'],
                'views': video['view_count'],
                'engagement_rate': video['engagement_rate'],
                'published_days_ago': (datetime.now(timezone.utc) - video['published_at']).days
            })
        
        return performers
    
    def _create_analysis_prompt(self, stats: Dict, channel_info: Dict) -> str:
        """Crea el prompt para análisis general"""
        
        prompt = f"""Eres un experto en estrategia de contenido para YouTube. Analiza los siguientes datos de un canal y proporciona insights detallados.

INFORMACIÓN DEL CANAL:
- Nombre: {channel_info.get('channel_name', 'N/A')}
- Suscriptores: {channel_info.get('subscriber_count', 0):,}
- Videos totales: {stats['total_videos']}

ESTADÍSTICAS:
- Total de Shorts: {stats['total_shorts']}
- Total de Videos Largos: {stats['total_long_videos']}
- Videos publicados últimos 30 días: {stats['videos_last_30_days']}

RENDIMIENTO:
- Vistas promedio: {stats['avg_views']:.0f}
- Vistas totales: {stats['total_views']:,.0f}
- Engagement rate promedio: {stats['avg_engagement_rate']:.2f}%

COMPARACIÓN SHORTS VS VIDEOS LARGOS:
- Shorts - Promedio de vistas: {stats['shorts_avg_views']:.0f}
- Videos Largos - Promedio de vistas: {stats['long_videos_avg_views']:.0f}
- Shorts - Engagement: {stats['shorts_avg_engagement']:.2f}%
- Videos Largos - Engagement: {stats['long_videos_avg_engagement']:.2f}%

MEJOR VIDEO:
- Título: {stats['best_video_title']}
- Tipo: {stats['best_video_type']}
- Vistas: {stats['best_video_views']:,.0f}

Por favor proporciona:
1. Análisis de qué tipo de contenido funciona mejor (Shorts vs Videos Largos)
2. Principales fortalezas del canal
3. Áreas de oportunidad
4. Insights sobre el engagement
5. Recomendaciones estratégicas generales

Sé específico y usa los datos proporcionados."""

        return prompt
    
    def _create_recommendation_prompt(
        self,
        stats: Dict,
        trends: Dict,
        best_performers: List[Dict],
        channel_info: Dict,
        past_results: List[Dict],
        cadence_insights: Optional[Dict] = None,
        hourly_insights: Optional[Dict] = None,
    ) -> str:
        """Crea el prompt para recomendación diaria (incluye historial de retroalimentación)."""

        best_videos_text = "\n".join([
            f"- {p['title']} ({p['type']}) - {p['views']:,} vistas, {p['engagement_rate']:.2f}% engagement"
            for p in best_performers[:3]
        ])

        # ── Sección de retroalimentación histórica ──────────────────────────
        feedback_section = ""
        if past_results:
            lines = []
            for r in past_results[:5]:   # máximo 5 entradas para no sobrecargar el prompt
                date  = r.get('recommendation_date', '')[:10]
                rtype = r.get('recommended_type', '?')
                vtype = r.get('video_type') or '(sin datos aún)'
                ratio = r.get('performance_ratio')
                label = r.get('performance_label', '')
                followed = r.get('followed_recommendation')

                # Emoji de resultado
                if label == 'above_average':
                    emoji = '✅'
                elif label == 'average':
                    emoji = '🟡'
                elif label == 'below_average':
                    emoji = '❌'
                else:
                    emoji = '⏳'

                ratio_str = f"{ratio:.2f}x promedio" if ratio is not None else 'sin resultado aún'

                if followed is None:
                    follow_str = '(pendiente de publicación)'
                elif followed:
                    follow_str = f"publicaron {vtype} — {ratio_str} {emoji}"
                else:
                    follow_str = f"publicaron {vtype} (diferente al recomendado) — {ratio_str} {emoji}"

                lines.append(f"• {date}: Recomendé {rtype} → {follow_str}")

            feedback_section = (
                "\nHISTORIAL DE RETROALIMENTACIÓN (últimas recomendaciones):\n"
                + "\n".join(lines)
                + "\n\nUsa este historial para ajustar tu recomendación: si el canal ha tendido a "
                "ignorar el formato recomendado o si cierto tipo de video rindió peor de lo esperado, "
                "considera corregir el rumbo o reforzar la estrategia que ha dado mejores resultados.\n"
            )
        # ────────────────────────────────────────────────────────────────────

        # ── Sección de cadencia y horarios ────────────────────────────────
        cadence_section = ""
        if cadence_insights and cadence_insights.get('summary_text'):
            cadence_section += f"\nCADENCIA ÓPTIMA DE PUBLICACIÓN:\n{cadence_insights['summary_text']}\n"
        if hourly_insights and hourly_insights.get('summary_text'):
            cadence_section += (
                f"\nHORARIOS SUBUTILIZADOS (ventanas de oportunidad):\n"
                f"{self._format_opportunity_slots(hourly_insights)}\n"
            )
        if cadence_section:
            cadence_section += (
                "\nConsidera la cadencia óptima y los horarios de oportunidad al "
                "recomendar el día y hora de publicación.\n"
            )
        # ────────────────────────────────────────────────────────────────────

        prompt = f"""Eres un estratega de contenido para YouTube especializado en maximizar audiencia y suscriptores.

DATOS DEL CANAL:
- Suscriptores actuales: {channel_info.get('subscriber_count', 0):,}
- Videos publicados: {stats['total_videos']}

PERFORMANCE ACTUAL:
- Shorts publicados: {stats['total_shorts']} (Promedio: {stats['shorts_avg_views']:.0f} vistas)
- Videos Largos: {stats['total_long_videos']} (Promedio: {stats['long_videos_avg_views']:.0f} vistas)
- Engagement Shorts: {stats['shorts_avg_engagement']:.2f}%
- Engagement Videos Largos: {stats['long_videos_avg_engagement']:.2f}%

TENDENCIAS:
- Tendencia de vistas: {trends['views_trend']}
- Tendencia de engagement: {trends['engagement_trend']}
- Ratio de Shorts últimos 10 videos: {trends['shorts_ratio_last_10']:.1f}%
- Mejor día para publicar: {trends['best_weekday']}
- Mejor hora para publicar: {trends['best_hour']}:00 (hora de Panamá, UTC-5)

TOP 3 VIDEOS CON MEJOR PERFORMANCE:
{best_videos_text}
{feedback_section}{cadence_section}
OBJETIVO: Maximizar audiencia y ganar nuevos suscriptores

Por favor genera una recomendación ESPECÍFICA para el contenido que debería publicar MAÑANA. La recomendación debe incluir:

1. FORMATO RECOMENDADO: ¿Short o Video Largo? (Basado en datos)
2. TEMA/TIPO DE CONTENIDO: Qué tipo de contenido específico recomiendas (basado en lo que ha funcionado)
3. RAZÓN: Por qué esta es la mejor opción basándose en los datos
4. PREDICCIÓN: Cómo esperas que performe en comparación con el promedio
5. HORARIO: Mejor momento para publicar
6. SUGERENCIAS DE TÍTULO: Propón exactamente 3 opciones de título para el video recomendado. Para CADA título incluye un análisis breve de por qué podría funcionar bien, considerando:
   - Palabras clave relevantes
   - Claridad del mensaje
   - Factor de curiosidad o enganche

Formato para los títulos (respeta este formato exacto):
TÍTULO 1: [título aquí]
ANÁLISIS: [por qué funciona]
TÍTULO 2: [título aquí]
ANÁLISIS: [por qué funciona]
TÍTULO 3: [título aquí]
ANÁLISIS: [por qué funciona]

Sé ESPECÍFICO y basado en DATOS. No des recomendaciones genéricas."""

        return prompt
    
    def _call_claude(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Llama a la API de Claude con retry automático en errores transitorios.

        Args:
            prompt: Prompt para Claude
            max_tokens: Máximo de tokens en la respuesta

        Returns:
            Respuesta de Claude o string de error
        """
        try:
            return self._call_claude_with_retry(prompt, max_tokens)
        except Exception as e:
            return f"Error al generar análisis: {str(e)}"

    @retry_anthropic
    def _call_claude_with_retry(self, prompt: str, max_tokens: int) -> str:
        """Llamada interna a Claude protegida con retry."""
        message = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    
    def _parse_recommendation(self, recommendation_text: str, stats: Dict) -> Dict:
        """
        Parsea la recomendación de texto a estructura

        Args:
            recommendation_text: Texto de la recomendación
            stats: Estadísticas del canal

        Returns:
            Diccionario estructurado con la recomendación
        """
        # Intentar determinar el tipo recomendado
        recommended_type = "Video Largo"
        if "short" in recommendation_text.lower() and "no short" not in recommendation_text.lower():
            recommended_type = "Short"

        # Extraer sugerencias de título
        title_suggestions = self._extract_title_suggestions(recommendation_text)

        return {
            'recommendation_date': (datetime.now() + timedelta(days=1)).date().isoformat(),
            'recommended_type': recommended_type,
            'recommended_topic': self._extract_topic(recommendation_text),
            'reasoning': recommendation_text,
            'predicted_performance': self._predict_performance(recommended_type, stats),
            'title_suggestions': title_suggestions,
            'generated_at': datetime.now().isoformat()
        }
    
    def _extract_topic(self, text: str) -> str:
        """Extrae el tema recomendado del texto"""
        # Buscar patrones comunes
        lines = text.split('\n')
        for line in lines:
            if 'tema' in line.lower() or 'contenido' in line.lower() or 'topic' in line.lower():
                return line.strip()

        # Si no encuentra, retorna las primeras 100 caracteres
        return text[:100] + "..."

    def _extract_title_suggestions(self, text: str) -> List[Dict]:
        """
        Extrae las 3 sugerencias de título del texto de la recomendación.

        Returns:
            Lista de dicts con 'title' y 'analysis' por cada sugerencia.
        """
        suggestions: List[Dict] = []

        # Patrón: TÍTULO N: ... seguido opcionalmente de ANÁLISIS: ...
        pattern = re.compile(
            r'T[ÍI]TULO\s*(\d)\s*:\s*(.+?)(?:\n\s*AN[ÁA]LISIS\s*:\s*(.+?))?(?=\nT[ÍI]TULO\s*\d|$)',
            re.IGNORECASE | re.DOTALL,
        )

        for match in pattern.finditer(text):
            title = match.group(2).strip().strip('"\'""«»')
            analysis = (match.group(3) or '').strip()
            if title:
                suggestions.append({'title': title, 'analysis': analysis})

        return suggestions[:3]
    
    def _predict_performance(self, video_type: str, stats: Dict) -> str:
        """Predice el performance esperado"""

        if video_type == "Short":
            avg_views = stats['shorts_avg_views']
            avg_engagement = stats['shorts_avg_engagement']
        else:
            avg_views = stats['long_videos_avg_views']
            avg_engagement = stats['long_videos_avg_engagement']

        return f"Vistas esperadas: ~{avg_views:.0f} | Engagement esperado: ~{avg_engagement:.1f}%"

    # ------------------------------------------------------------------
    # Planificación semanal
    # ------------------------------------------------------------------

    def generate_weekly_plan(
        self,
        videos_df: pd.DataFrame,
        channel_info: Dict,
        past_results: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Genera un plan de contenido para los próximos 7 días.

        Returns:
            {
              'week_start_date': 'YYYY-MM-DD',
              'strategy': str,
              'days': [
                  {
                      'date': 'YYYY-MM-DD',
                      'day': 'Lunes',
                      'publish': bool,
                      'type': 'Short' | 'Video Largo' | None,
                      'topic': str,
                      'hour': int,
                      'reason': str,
                  },
                  ... (7 elementos)
              ],
              'generated_at': str,
              'error': str  (solo si hay error)
            }
        """
        if videos_df.empty or len(videos_df) < 5:
            return {
                'error': 'Necesitas al menos 5 videos publicados para generar un plan semanal.',
            }

        stats          = self._calculate_statistics(videos_df)
        trends         = self._identify_trends(videos_df)
        best_performers = self._get_best_performers(videos_df)
        cadence_insights = self.analyze_cadence(videos_df)
        hourly_insights  = self.analyze_hourly_saturation(videos_df)

        # Calcular los próximos 7 días a partir de mañana
        today     = datetime.now(PANAMA_TZ).date()
        next_days = [today + timedelta(days=i) for i in range(1, 8)]
        day_names = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

        prompt = self._create_weekly_plan_prompt(
            stats, trends, best_performers, channel_info,
            next_days, day_names, past_results or [],
            cadence_insights=cadence_insights,
            hourly_insights=hourly_insights,
        )

        raw = self._call_claude(prompt, max_tokens=3000)
        plan_data = self._parse_weekly_plan(raw, next_days, day_names)
        plan_data['week_start_date'] = next_days[0].isoformat()
        plan_data['generated_at']    = datetime.now().isoformat()
        return plan_data

    def _create_weekly_plan_prompt(
        self,
        stats: Dict,
        trends: Dict,
        best_performers: List[Dict],
        channel_info: Dict,
        next_days: list,
        day_names: List[str],
        past_results: List[Dict],
        cadence_insights: Optional[Dict] = None,
        hourly_insights: Optional[Dict] = None,
    ) -> str:
        """Crea el prompt para el plan semanal. Pide respuesta en JSON estricto."""
        best_videos_text = "\n".join([
            f"- {p['title']} ({p['type']}) — {p['views']:,} vistas"
            for p in best_performers[:3]
        ])

        # Construir lista de días con nombre para el prompt
        days_list = "\n".join(
            f"  {d.isoformat()} ({day_names[d.weekday()]})"
            for d in next_days
        )

        # Historial de retroalimentación (igual que en la recomendación diaria)
        feedback_lines = []
        for r in past_results[:5]:
            date  = r.get('recommendation_date', '')[:10]
            rtype = r.get('recommended_type', '?')
            ratio = r.get('performance_ratio')
            label = r.get('performance_label', '')
            emoji = {'above_average': '✅', 'average': '🟡', 'below_average': '❌'}.get(label, '⏳')
            ratio_str = f"{ratio:.2f}x" if ratio is not None else 'pendiente'
            feedback_lines.append(f"  • {date}: Recomendé {rtype} → {ratio_str} {emoji}")
        feedback_section = (
            "\nHISTORIAL DE RESULTADOS:\n" + "\n".join(feedback_lines) + "\n"
        ) if feedback_lines else ""

        return f"""Eres un estratega de contenido para YouTube. Genera un plan de publicación para los próximos 7 días basado en datos reales del canal.

DATOS DEL CANAL:
- Suscriptores: {channel_info.get('subscriber_count', 0):,}
- Videos totales: {stats['total_videos']}
- Shorts promedio: {stats['shorts_avg_views']:.0f} vistas | {stats['shorts_avg_engagement']:.2f}% engagement
- Videos largos promedio: {stats['long_videos_avg_views']:.0f} vistas | {stats['long_videos_avg_engagement']:.2f}% engagement
- Mejor día histórico: {trends['best_weekday']}
- Mejor hora histórica: {trends['best_hour']}:00 (hora de Panamá UTC-5)
- Tendencia de vistas: {trends['views_trend']}
- Ratio Shorts (últimos 10 videos): {trends['shorts_ratio_last_10']:.1f}%

TOP 3 VIDEOS:
{best_videos_text}
{feedback_section}
DÍAS A PLANIFICAR:
{days_list}
{self._build_cadence_prompt_section(cadence_insights, hourly_insights)}
REGLAS:
- No es obligatorio publicar todos los días; elige 3-5 días estratégicos.
- Alterna entre Shorts y Videos Largos según el rendimiento histórico.
- Respeta la cadencia óptima entre videos del mismo tipo.
- Prioriza los horarios de alta oportunidad sobre los saturados.
- Cada tema debe ser específico, no genérico.

RESPONDE ÚNICAMENTE con un JSON válido con esta estructura (sin markdown, sin explicación extra):
{{
  "strategy": "Texto breve (2-3 oraciones) con la estrategia general de la semana",
  "days": [
    {{
      "date": "YYYY-MM-DD",
      "day": "NombreDía",
      "publish": true,
      "type": "Short",
      "topic": "Tema específico del video",
      "hour": 18,
      "reason": "Razón breve basada en datos"
    }},
    {{
      "date": "YYYY-MM-DD",
      "day": "NombreDía",
      "publish": false,
      "type": null,
      "topic": "",
      "hour": null,
      "reason": "Día de descanso / sin publicación recomendada"
    }}
  ]
}}

Incluye exactamente 7 objetos en "days", uno por cada día listado arriba."""

    def _parse_weekly_plan(self, raw: str, next_days: list, day_names: List[str]) -> Dict:
        """
        Parsea la respuesta de Claude intentando extraer JSON.
        Retorna un dict con 'strategy' y 'days'. En caso de fallo, crea
        una estructura mínima para no romper el dashboard.
        """
        # Intentar extraer bloque JSON de la respuesta
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            try:
                data = json.loads(json_match.group())
                strategy = str(data.get('strategy', ''))
                days_raw = data.get('days', [])
                days = []
                for entry in days_raw[:7]:
                    days.append({
                        'date':    entry.get('date', ''),
                        'day':     entry.get('day', ''),
                        'publish': bool(entry.get('publish', False)),
                        'type':    entry.get('type'),
                        'topic':   entry.get('topic', ''),
                        'hour':    entry.get('hour'),
                        'reason':  entry.get('reason', ''),
                    })
                return {'strategy': strategy, 'days': days}
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: estructura vacía con los días calculados y el texto raw como estrategia
        days = [
            {
                'date':    d.isoformat(),
                'day':     day_names[d.weekday()],
                'publish': False,
                'type':    None,
                'topic':   '',
                'hour':    None,
                'reason':  'No se pudo parsear la respuesta.',
            }
            for d in next_days
        ]
        return {'strategy': raw[:500], 'days': days}

    # ------------------------------------------------------------------
    # Script / Outline Generator
    # ------------------------------------------------------------------

    def generate_script_outline(
        self,
        video_type: str,
        topic: str,
        title: str,
        channel_name: str,
        language: str = 'es',
    ) -> Dict:
        """
        Genera un guion/outline para un video basandose en el tipo, tema y titulo.

        Args:
            video_type: "Short" o "Video Largo"
            topic: Tema del video
            title: Titulo sugerido
            channel_name: Nombre del canal
            language: Codigo de idioma del canal ('es', 'en', etc.)

        Returns:
            dict con {video_type, topic, title, outline_text, created_at}
            o {error: str} si falla
        """
        prompt = self._create_script_outline_prompt(video_type, topic, title, channel_name, language)
        response = self._call_claude(prompt, max_tokens=2000)

        if response.startswith("Error"):
            return {'error': response}

        return {
            'video_type': video_type,
            'topic': topic,
            'title': title,
            'outline_text': response.strip(),
            'created_at': datetime.now().isoformat(),
        }

    # Mapa de idiomas soportados
    _LANGUAGE_NAMES = {
        'es': 'espanol',
        'en': 'English',
        'pt': 'portugues',
        'fr': 'francais',
    }

    def _create_script_outline_prompt(
        self,
        video_type: str,
        topic: str,
        title: str,
        channel_name: str,
        language: str = 'es',
    ) -> str:
        """Construye el prompt para generar el outline del video."""

        lang_name = self._LANGUAGE_NAMES.get(language, language)

        if video_type == 'Short':
            if language == 'en':
                format_instructions = """
REQUIRED STRUCTURE FOR SHORT (max 60 seconds):

1. HOOK (0-3 seconds)
   - Opening line that captures attention immediately
   - Must generate curiosity or surprise

2. DEVELOPMENT (3-45 seconds)
   - Main point of the video
   - Maximum 2-3 concrete ideas
   - Direct and fast-paced language

3. CTA / CLOSE (45-60 seconds)
   - Clear call to action (follow, like, comment)
   - Memorable closing line or cliffhanger

ON-SCREEN TEXT:
   - 3-5 key texts that should appear overlaid on the video"""
            else:
                format_instructions = """
ESTRUCTURA REQUERIDA PARA SHORT (maximo 60 segundos):

1. HOOK (0-3 segundos)
   - Frase de apertura que capture la atencion inmediatamente
   - Debe generar curiosidad o sorpresa

2. DESARROLLO (3-45 segundos)
   - Punto principal del video
   - Maximo 2-3 ideas concretas
   - Lenguaje directo y rapido

3. CTA / CIERRE (45-60 segundos)
   - Call to action claro (seguir, like, comentar)
   - Frase de cierre memorable o cliffhanger

TEXTO EN PANTALLA:
   - 3-5 textos clave que deben aparecer superpuestos en el video"""
        else:
            if language == 'en':
                format_instructions = """
REQUIRED STRUCTURE FOR LONG VIDEO:

1. HOOK + PREVIEW (0:00 - 0:30)
   - Opening hook to engage the viewer
   - Preview of what they'll learn/see (value promise)
   - "In this video you'll discover..."

2. INTRO (0:30 - 1:00)
   - Brief topic introduction
   - Why it's relevant/important
   - Necessary context

3. MAIN SECTIONS (1:00 - end)
   - Divide the content into 3-5 clear sections
   - For each section include:
     * Section title
     * Estimated timestamp (e.g.: "3:00 - 5:00")
     * 3-4 key points to cover
     * Transition to the next section

4. CONCLUSION + CTA (last 60 seconds)
   - Quick summary of main points
   - Call to action (subscribe, comment, watch another video)
   - Suggestion for next video on the channel"""
            else:
                format_instructions = """
ESTRUCTURA REQUERIDA PARA VIDEO LARGO:

1. HOOK + PREVIEW (0:00 - 0:30)
   - Gancho inicial que enganche al espectador
   - Preview de lo que va a aprender/ver (promesa de valor)
   - "En este video vas a descubrir..."

2. INTRO (0:30 - 1:00)
   - Presentacion breve del tema
   - Por que es relevante/importante
   - Contexto necesario

3. SECCIONES PRINCIPALES (1:00 - final)
   - Divide el contenido en 3-5 secciones claras
   - Para cada seccion incluye:
     * Titulo de la seccion
     * Timestamp estimado (ej: "3:00 - 5:00")
     * 3-4 puntos clave a cubrir
     * Transicion a la siguiente seccion

4. CONCLUSION + CTA (ultimos 60 segundos)
   - Resumen rapido de los puntos principales
   - Call to action (suscribirse, comentar, ver otro video)
   - Sugerencia de video siguiente del canal"""

        prompt = f"""You are a professional YouTube scriptwriter for the channel "{channel_name}".

TASK: Generate a detailed script/outline for the following video.
CRITICAL: The ENTIRE outline must be written in **{lang_name}** — all narration, on-screen text, CTA, and notes.

VIDEO DATA:
- Type: {video_type}
- Topic: {topic}
- Title: "{title}"

{format_instructions}

RULES:
- Write EVERYTHING in {lang_name} (narration, on-screen text, CTA, notes)
- Be specific: don't give generic instructions, give CONCRETE EXAMPLES of what to say
- Adapt the tone to the channel's content type
- Include suggestions for visual elements or B-roll where relevant
- The outline must be actionable: the creator should be able to record it immediately"""

        return prompt

    # ------------------------------------------------------------------
    # SEO Content Generator (Mejora 9.2 + 9.3)
    # ------------------------------------------------------------------

    def generate_seo_content(
        self,
        video_type: str,
        topic: str,
        title: str,
        channel_name: str,
        channel_tags: List[str],
        related_videos: List[Dict],
        trend_scores: Optional[Dict[str, float]] = None,
        rising_queries: Optional[List[str]] = None,
        language: str = 'es',
    ) -> Dict:
        """
        Genera descripción SEO optimizada y lista de tags para un video de YouTube.

        Args:
            video_type: "Short" o "Video Largo"
            topic: Tema del video
            title: Título del video
            channel_name: Nombre del canal
            channel_tags: Tags más frecuentes de los videos exitosos del canal
            related_videos: Lista de dicts {video_id, title, url} de videos relacionados
            trend_scores: Dict {keyword: score 0-100} de Google Trends (opcional)
            rising_queries: Lista de consultas en ascenso de Google Trends (opcional)
            language: Código de idioma del canal ('es', 'en', etc.)

        Returns:
            dict con {title, seo_description, tags, hashtags, related_videos, created_at}
            o {error: str} si falla
        """
        prompt = self._create_seo_content_prompt(
            video_type, topic, title, channel_name,
            channel_tags, related_videos,
            trend_scores, rising_queries, language,
        )
        response = self._call_claude(prompt, max_tokens=2500)

        if response.startswith("Error"):
            return {'error': response}

        return self._parse_seo_content(response, title, related_videos)

    def _create_seo_content_prompt(
        self,
        video_type: str,
        topic: str,
        title: str,
        channel_name: str,
        channel_tags: List[str],
        related_videos: List[Dict],
        trend_scores: Optional[Dict[str, float]] = None,
        rising_queries: Optional[List[str]] = None,
        language: str = 'es',
    ) -> str:
        """Construye el prompt para generar descripción SEO + tags."""

        lang_name = self._LANGUAGE_NAMES.get(language, language)

        # Formatear tags existentes del canal
        channel_tags_text = ', '.join(channel_tags[:20]) if channel_tags else '(sin datos)'

        # Formatear videos relacionados
        if related_videos:
            related_lines = [
                f'  - "{rv["title"]}" -> {rv["url"]}'
                for rv in related_videos[:5]
            ]
            related_text = '\n'.join(related_lines)
        else:
            related_text = '(no hay videos relacionados disponibles)'

        # Formatear datos de tendencias
        trends_text = ''
        if trend_scores:
            trends_lines = [f'  - {kw}: {score}/100' for kw, score in trend_scores.items()]
            trends_text = 'TENDENCIAS ACTUALES (Google Trends, score 0-100):\n' + '\n'.join(trends_lines)
        if rising_queries:
            trends_text += '\nCONSULTAS EN ASCENSO:\n  - ' + '\n  - '.join(rising_queries[:10])

        prompt = f"""You are a YouTube SEO expert for the channel "{channel_name}".

TASK: Generate TWO things for the following video:
1. An SEO-optimized YouTube description
2. A list of 15-20 optimized tags ordered by relevance

CRITICAL: Write EVERYTHING in **{lang_name}**.

VIDEO DATA:
- Type: {video_type}
- Title: "{title}"
- Topic: {topic}

CHANNEL'S MOST SUCCESSFUL TAGS (from top-performing videos):
{channel_tags_text}

RELATED VIDEOS FROM THIS CHANNEL (for link suggestions):
{related_text}

{trends_text}

===================================================================
PART 1: SEO DESCRIPTION
===================================================================

Generate a complete YouTube description following these rules:
- FIRST 2 LINES: Must contain the most important keywords naturally. These are visible without expanding and are critical for SEO and CTR.
- TIMESTAMPS: Include suggested timestamps for the video sections (even if estimated).
- HASHTAGS: Include exactly 3 relevant hashtags at the end (YouTube recommends max 3). Format: #hashtag1 #hashtag2 #hashtag3
- RELATED LINKS: If related videos from the channel are available, include 2-3 links to them with a short label.
- Include a brief call-to-action (subscribe, like, comment).
- Total length: 800-1500 characters (YouTube's sweet spot for SEO).

===================================================================
PART 2: OPTIMIZED TAGS
===================================================================

Generate 15-20 tags following these rules:
- Order by relevance (most important first)
- Mix of: exact-match keywords, broad keywords, channel-specific tags, and trending terms
- Include the video title (or close variation) as the first tag
- Include 2-3 long-tail keywords (3+ words)
- If trend data is available, prioritize keywords with higher trend scores
- Reuse tags from the channel's successful videos when relevant to this topic
- Each tag should be lowercase, no # prefix

RESPONSE FORMAT (use this exact JSON structure):
```json
{{
  "seo_description": "The full description text here, with line breaks as \\n",
  "tags": ["tag1", "tag2", "tag3", "...up to 20 tags"],
  "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3"]
}}
```

IMPORTANT: Respond ONLY with the JSON block, no extra text before or after."""

        return prompt

    def _parse_seo_content(
        self, response: str, title: str, related_videos: List[Dict]
    ) -> Dict:
        """Parsea la respuesta de Claude con descripción SEO y tags."""

        # Intentar extraer JSON del response
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            log.warning("No se pudo parsear JSON de SEO content, usando respuesta cruda")
            return {
                'title': title,
                'seo_description': response.strip(),
                'tags': [],
                'hashtags': [],
                'related_videos': [
                    {'video_id': rv['video_id'], 'title': rv['title'], 'url': rv['url']}
                    for rv in related_videos[:3]
                ],
                'created_at': datetime.now().isoformat(),
            }

        return {
            'title': title,
            'seo_description': data.get('seo_description', '').replace('\\n', '\n'),
            'tags': data.get('tags', [])[:20],
            'hashtags': data.get('hashtags', [])[:3],
            'related_videos': [
                {'video_id': rv['video_id'], 'title': rv['title'], 'url': rv['url']}
                for rv in related_videos[:3]
            ],
            'created_at': datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Channel Health Diagnosis
    # ------------------------------------------------------------------

    def generate_health_diagnosis(
        self,
        channel_name: str,
        metrics: List[Dict],
        health_score: int,
        language: str = 'es',
    ) -> str:
        """
        Genera un diagnóstico de salud del canal usando Claude.

        Args:
            channel_name: Nombre del canal
            metrics: Lista de dicts con {name, value, status, trend}
            health_score: Puntuación general 0-100
            language: Código de idioma ('es', 'en', etc.)

        Returns:
            Texto del diagnóstico en markdown, o string con 'Error' si falla
        """
        lang_name = self._LANGUAGE_NAMES.get(language, language)

        metrics_block = ""
        for m in metrics:
            emoji = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}.get(m['status'], '⚪')
            metrics_block += f"- {emoji} **{m['name']}**: {m['value']} — {m.get('detail', '')}\n"

        prompt = f"""You are a YouTube channel strategist analyzing the channel "{channel_name}".

CHANNEL HEALTH SCORE: {health_score}/100

CURRENT METRICS:
{metrics_block}

TASK: Write a concise but insightful health diagnosis for this channel.
Write EVERYTHING in **{lang_name}**.

STRUCTURE (use markdown headers):
## Diagnóstico General
One paragraph (3-4 sentences) summarizing the overall state of the channel.

## Fortalezas
2-3 bullet points about what's working well (based on green metrics).

## Áreas Críticas
2-3 bullet points about what needs immediate attention (based on red/yellow metrics).
For each, explain WHY it matters and WHAT the creator should do.

## Plan de Acción (próximos 7 días)
3-4 specific, actionable steps ordered by priority.
Each step should be concrete (not generic advice).

RULES:
- Be data-driven: reference the actual numbers from the metrics
- Be specific to THIS channel, not generic YouTube advice
- Keep it concise — max 400 words total
- Use the channel name naturally in the text"""

        return self._call_claude(prompt, max_tokens=1500)

    # ------------------------------------------------------------------
    # Análisis de Competencia (Mejora 7.1)
    # ------------------------------------------------------------------

    def analyze_competitor_gaps(
        self,
        own_channel_name: str,
        own_titles: List[str],
        own_stats: Dict,
        competitor_data: List[Dict],
        language: str = 'es',
    ) -> str:
        """
        Identifica content gaps entre el canal propio y los competidores.

        Args:
            own_channel_name: Nombre del canal propio
            own_titles: Lista de títulos recientes del canal propio
            own_stats: {avg_views, avg_engagement, total_videos, top_topics}
            competitor_data: Lista de dicts, cada uno:
                {name, titles, avg_views, avg_engagement, total_videos, subscriber_count}
            language: Código de idioma

        Returns:
            Texto markdown con el análisis de brechas de contenido.
        """
        lang_name = self._LANGUAGE_NAMES.get(language, language)

        own_titles_block = "\n".join(f"- {t}" for t in own_titles[:30])

        comp_blocks = []
        for comp in competitor_data:
            titles_list = "\n".join(f"  - {t}" for t in comp['titles'][:20])
            comp_blocks.append(
                f"### {comp['name']} ({comp['subscriber_count']:,} subs)\n"
                f"  Videos: {comp['total_videos']} | "
                f"Vistas prom: {comp['avg_views']:,.0f} | "
                f"Engagement: {comp['avg_engagement']:.2f}%\n"
                f"  Títulos recientes:\n{titles_list}"
            )
        competitors_block = "\n\n".join(comp_blocks)

        prompt = f"""You are a YouTube competitive analyst for the channel "{own_channel_name}".

YOUR CHANNEL STATS:
- Videos: {own_stats['total_videos']}
- Avg views: {own_stats['avg_views']:,.0f}
- Avg engagement: {own_stats['avg_engagement']:.2f}%
- Recent titles:
{own_titles_block}

COMPETITORS:
{competitors_block}

TASK: Perform a competitive content gap analysis. Write EVERYTHING in **{lang_name}**.

STRUCTURE (use markdown headers):

## Resumen Competitivo
One paragraph comparing the channel's positioning vs competitors.

## Temas que Funcionan en Competidores
3-5 bullet points of topics/themes that competitors cover successfully
that YOUR channel does NOT cover. For each:
- Name the specific topic
- Which competitor covers it
- Why it works (estimated from title patterns)

## Oportunidades de Contenido (Content Gaps)
3-5 specific video ideas that fill gaps — topics your competitors haven't
fully exploited OR that you could cover better. For each:
- Suggested title
- Format (Short / Video Largo)
- Why this gap exists

## Ventajas Competitivas
2-3 things YOUR channel does that competitors don't — strengths to double down on.

## Alertas
Any concerning patterns: competitors growing faster, covering your niche, etc.

RULES:
- Be data-driven: reference actual titles and numbers
- Suggest SPECIFIC video titles, not generic ideas
- Keep analysis actionable and concise (max 600 words)
- Use the channel names naturally"""

        return self._call_claude(prompt, max_tokens=2500)

    # ------------------------------------------------------------------
    # Alertas de Competidores (Mejora 7.2)
    # ------------------------------------------------------------------

    def analyze_viral_competitor_video(
        self,
        video_title: str,
        channel_name: str,
        view_count: int,
        competitor_avg_views: float,
        ratio: float,
        video_type: str = '',
        language: str = 'es',
    ) -> str:
        """
        Analiza por qué un video de competidor despegó y cómo replicar el éxito.

        Args:
            video_title: Título del video viral
            channel_name: Nombre del canal competidor
            view_count: Vistas actuales del video
            competitor_avg_views: Promedio de vistas del competidor
            ratio: Multiplicador sobre el promedio (view_count / avg)
            video_type: 'Short' o 'Video'
            language: Código de idioma

        Returns:
            Texto markdown con análisis y recomendaciones.
        """
        lang_name = self._LANGUAGE_NAMES.get(language, language)

        prompt = f"""You are a YouTube strategist analyzing a competitor's viral video.

COMPETITOR VIDEO:
- Title: "{video_title}"
- Channel: {channel_name}
- Views: {view_count:,}
- Channel average views: {competitor_avg_views:,.0f}
- Performance: {ratio:.1f}x above average
- Type: {video_type or 'Unknown'}

TASK: Analyze WHY this video went viral and HOW to replicate its success.
Write EVERYTHING in **{lang_name}**.

STRUCTURE (use markdown):

## Por qué despegó
2-3 bullet points analyzing title, topic, and format factors that explain the
performance spike. Be specific about the title structure, emotional hooks,
trending topics, or format choices.

## Cómo replicar el éxito
3 concrete video ideas inspired by this viral video. For each:
- Suggested title
- Format (Short / Video Largo)
- What angle to take that differentiates from the competitor

## Acción inmediata
ONE specific action to take THIS WEEK to capitalize on the trend.

RULES:
- Be data-driven and specific (reference the actual title)
- Suggest SPECIFIC titles, not generic ideas
- Max 300 words total
- Write in {lang_name} only"""

        return self._call_claude(prompt, max_tokens=1200)

    def recommend_series_format(
        self,
        series_name: str,
        channel_name: str,
        episodes: list[dict],
        language: str = 'es',
    ) -> str:
        """
        Genera recomendación de formato para una serie (Mejora 17.2).

        Args:
            series_name: Nombre de la serie
            channel_name: Nombre del canal
            episodes: Lista de dicts con title, episode_number, view_count,
                      engagement_rate, published_at
        """
        ep_lines = []
        for ep in episodes:
            views = ep.get('view_count', 0) or 0
            eng = ep.get('engagement_rate', 0) or 0
            ep_lines.append(
                f"  Ep {ep.get('episode_number', '?')}: "
                f"\"{ep.get('title', '')}\" — "
                f"{views:,.0f} vistas, {eng:.2f}% engagement"
            )
        ep_text = '\n'.join(ep_lines)

        prompt = f"""Analiza esta serie de videos del canal "{channel_name}" y da recomendaciones de formato.

Serie: "{series_name}"
Episodios ({len(episodes)} total):
{ep_text}

Responde en {language}, máximo 250 palabras, con estos puntos:
1. **Tendencia de audiencia**: ¿las vistas crecen, decrecen o se mantienen entre episodios?
2. **Punto de fatiga**: ¿En qué episodio se pierde audiencia? ¿Cuántos episodios es lo óptimo?
3. **Recomendación concreta**: Una acción específica (series más cortas, cambiar formato, nuevo ángulo, etc.)
"""

        return self._call_claude(prompt, max_tokens=800)


if __name__ == "__main__":
    # Test del analizador
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    
    if not api_key:
        print("Error: ANTHROPIC_API_KEY no configurada")
        exit(1)
    
    analyzer = AIAnalyzer(api_key)
    
    # Crear datos de prueba
    test_data = {
        'video_id': ['1', '2', '3'],
        'title': ['Test 1', 'Test 2', 'Test 3'],
        'is_short': [True, False, True],
        'video_type': ['Short', 'Video Largo', 'Short'],
        'view_count': [1000, 5000, 800],
        'engagement_rate': [5.0, 3.5, 6.0],
        'published_at': pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-10'])
    }
    
    df = pd.DataFrame(test_data)
    channel_info = {'channel_name': 'Test Channel', 'subscriber_count': 1000}
    
    print("Generando recomendación de prueba...")
    recommendation = analyzer.generate_daily_recommendation(df, channel_info)
    
    print("\n=== RECOMENDACIÓN GENERADA ===")
    print(json.dumps(recommendation, indent=2, ensure_ascii=False))
