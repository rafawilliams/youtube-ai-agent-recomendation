"""
Script principal para ejecutar el agente de análisis de YouTube
"""
import os
import sys
import logging
import pandas as pd
from dotenv import load_dotenv

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from logger import setup_logging
from youtube_extractor import YouTubeDataExtractor
from database import YouTubeDatabase
from ai_analyzer import AIAnalyzer
from analytics_extractor import YouTubeAnalyticsExtractor
from telegram_notifier import TelegramNotifier
from series_detector import SeriesDetector
from revenue_analyzer import EvergreenDetector

log = logging.getLogger('pipeline')


def _validate_config():
    """Valida las variables de entorno requeridas. Retorna (yt_key, anthropic_key, channel_ids) o None."""
    youtube_api_key = os.getenv('YOUTUBE_API_KEY')
    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    channel_ids = os.getenv('YOUTUBE_CHANNEL_IDS', '').split(',')

    if not youtube_api_key:
        log.error("YOUTUBE_API_KEY no configurada")
        return None
    if not anthropic_api_key:
        log.error("ANTHROPIC_API_KEY no configurada")
        return None
    if not channel_ids or channel_ids[0] == '':
        log.error("YOUTUBE_CHANNEL_IDS no configurado")
        return None

    channel_ids = [ch.strip() for ch in channel_ids if ch.strip()]

    max_videos = int(os.getenv('MAX_VIDEOS_PER_CHANNEL', '200'))

    log.info("Configuración cargada — Canales: %d | Máx. videos por canal: %d", len(channel_ids), max_videos)
    return youtube_api_key, anthropic_api_key, channel_ids, max_videos


def _step1_extract(extractor, channel_ids, max_videos_per_channel):
    """Extrae videos de todos los canales y retorna el DataFrame."""
    log.info("PASO 1: Extrayendo datos de YouTube API...")

    videos_df = extractor.extract_all_data(channel_ids, max_videos_per_channel=max_videos_per_channel)

    if videos_df.empty:
        log.error("No se pudieron extraer datos de los canales")
        return None

    log.info("Datos extraídos — %d videos (%d Shorts / %d Largos)",
             len(videos_df),
             len(videos_df[videos_df['is_short']]),
             len(videos_df[~videos_df['is_short']]))
    return videos_df


def _build_channel_info(channel_id, channel_videos, api_info):
    """Construye el dict de canal priorizando datos reales de la API."""
    if api_info:
        return {
            'channel_id': channel_id,
            'channel_name': api_info['channel_name'],
            'description': api_info['description'],
            'subscriber_count': api_info['subscriber_count'],
            'video_count': api_info['video_count'],
            'view_count': api_info['view_count'],
            'created_at': api_info['created_at'],
        }
    return {
        'channel_id': channel_id,
        'channel_name': channel_videos.iloc[0]['channel_title'],
        'description': '',
        'subscriber_count': 0,
        'video_count': len(channel_videos),
        'view_count': int(channel_videos['view_count'].sum()),
        'created_at': channel_videos['published_at'].min().isoformat(),
    }


def _step2_save(extractor, videos_df):
    """Guarda canales y videos en la base de datos."""
    log.info("PASO 2: Guardando datos en base de datos...")

    with YouTubeDatabase() as db:
        for channel_id in videos_df['channel_id'].unique():
            channel_videos = videos_df[videos_df['channel_id'] == channel_id]
            api_info = extractor.get_channel_info(channel_id)
            channel_info = _build_channel_info(channel_id, channel_videos, api_info)
            db.save_channel_data(channel_info)
            if api_info:
                log.info("  %s: %s suscriptores", channel_info['channel_name'], f"{channel_info['subscriber_count']:,}")

        db.save_videos_data(videos_df)

    log.info("Datos guardados en base de datos MariaDB")


def _log_analysis_stats(stats):
    """Registra las estadísticas clave del análisis."""
    log.info("  📈 Estadísticas Clave:")
    log.info("     Total vistas: %s", f"{stats['total_views']:,.0f}")
    log.info("     Promedio por video: %s", f"{stats['avg_views']:.0f}")
    log.info("     Engagement rate: %.2f%%", stats['avg_engagement_rate'])
    log.info("  🎬 Comparación:")
    log.info("     Shorts: %s vistas promedio", f"{stats['shorts_avg_views']:.0f}")
    log.info("     Videos Largos: %s vistas promedio", f"{stats['long_videos_avg_views']:.0f}")
    log.info("  🏆 Mejor video:")
    log.info("     %s...", stats['best_video_title'][:50])
    log.info("     %s vistas", f"{stats['best_video_views']:,.0f}")


def _log_recommendation(recommendation):
    """Registra la recomendación generada."""
    log.info("  " + "=" * 56)
    log.info("  🎯 RECOMENDACIÓN PARA MAÑANA")
    log.info("  " + "=" * 56)
    log.info("  📅 Fecha: %s", recommendation['recommendation_date'])
    log.info("  🎬 Formato: %s", recommendation['recommended_type'])
    log.info("  📊 Performance esperado: %s", recommendation['predicted_performance'])
    log.info("  💡 Análisis completo:")
    log.info("  " + "-" * 56)
    for line in recommendation['reasoning'].split('\n')[:15]:
        if line.strip():
            log.info("  %s", line)

    # Mostrar sugerencias de título
    title_suggestions = recommendation.get('title_suggestions', [])
    if title_suggestions:
        log.info("  " + "=" * 56)
        log.info("  ✍ SUGERENCIAS DE TÍTULO")
        log.info("  " + "=" * 56)
        for i, sug in enumerate(title_suggestions, 1):
            log.info("  %d. %s", i, sug['title'])
            if sug.get('analysis'):
                log.info("     → %s", sug['analysis'])


def _analyze_channel(analyzer, channel_id, channel_videos, notifier=None):
    """Ejecuta análisis + recomendación para un canal."""
    channel_info = {
        'channel_id': channel_id,
        'channel_name': channel_videos.iloc[0]['channel_title'],
        'subscriber_count': 0,
    }
    log.info("📊 Analizando: %s", channel_info['channel_name'])

    analysis = analyzer.analyze_channel_performance(channel_videos, channel_info)
    analysis_stats = None
    if 'error' not in analysis:
        log.info("  ✓ Análisis completado")
        analysis_stats = analysis['statistics']
        _log_analysis_stats(analysis_stats)

    log.info("  Generando recomendación para mañana...")

    with YouTubeDatabase() as db:
        # Vincular el video publicado más reciente a recomendaciones anteriores pendientes
        channel_avg_views = float(channel_videos['view_count'].mean()) if not channel_videos.empty else 0.0
        linked = db.link_video_to_recommendation(channel_id, channel_avg_views)
        if linked:
            log.info("  🔗 %d recomendación(es) vinculadas con videos publicados", linked)

        # Recuperar historial de retroalimentación para enriquecer el prompt
        past_results_df = db.get_recommendation_results(channel_id, limit=5)
        past_results = past_results_df.to_dict('records') if not past_results_df.empty else []

        recommendation = analyzer.generate_daily_recommendation(
            channel_videos, channel_info, past_results=past_results
        )

        if 'error' not in recommendation:
            log.info("  ✓ Recomendación generada")
            db.save_recommendation(channel_id, recommendation)
            # Registrar la recomendación en el ciclo de retroalimentación
            db.save_recommendation_result(
                channel_id,
                recommendation['recommendation_date'],
                recommendation['recommended_type'],
            )
            _log_recommendation(recommendation)

            # Notificar por Telegram
            if notifier and notifier.is_enabled() and analysis_stats:
                sent = notifier.notify_recommendation(
                    channel_info['channel_name'], recommendation, analysis_stats
                )
                if sent:
                    log.info("  📨 Notificación enviada por Telegram")
        else:
            log.warning("  %s", recommendation['error'])


def _step3_analyze(analyzer, videos_df, channel_ids, notifier=None):
    """Genera análisis y recomendaciones para cada canal."""
    log.info("PASO 3: Generando análisis con IA...")

    for channel_id in channel_ids:
        channel_videos = videos_df[videos_df['channel_id'] == channel_id]
        if channel_videos.empty:
            log.warning("No hay datos para el canal %s", channel_id)
            continue
        _analyze_channel(analyzer, channel_id, channel_videos, notifier=notifier)


def _step_competitors(extractor, max_videos_per_channel: int):
    """Extrae datos de canales competidores (solo Data API pública)."""
    raw = os.getenv('COMPETITOR_CHANNEL_IDS', '').strip()
    if not raw:
        log.warning("Sin competidores configurados — agrega COMPETITOR_CHANNEL_IDS en .env")
        return

    competitor_ids = [cid.strip() for cid in raw.split(',') if cid.strip()]
    if not competitor_ids:
        return

    log.info("PASO COMPETIDORES: Extrayendo datos de %d competidor(es)...", len(competitor_ids))

    comp_videos_df = extractor.extract_all_data(
        competitor_ids, max_videos_per_channel=max_videos_per_channel
    )

    if comp_videos_df.empty:
        log.warning("No se pudieron extraer datos de competidores")
        return

    # Guardar en BD marcando is_competitor=True
    with YouTubeDatabase() as db:
        for channel_id in comp_videos_df['channel_id'].unique():
            channel_videos = comp_videos_df[comp_videos_df['channel_id'] == channel_id]
            api_info = extractor.get_channel_info(channel_id)
            channel_info = _build_channel_info(channel_id, channel_videos, api_info)
            channel_info['is_competitor'] = 1
            db.save_channel_data(channel_info)
            if api_info:
                log.info("  🕵 Competidor: %s (%s suscriptores)",
                         channel_info['channel_name'],
                         f"{channel_info['subscriber_count']:,}")

        db.save_videos_data(comp_videos_df)

    log.info("Datos de competidores guardados — %d videos", len(comp_videos_df))


def _step_competitor_alerts(analyzer, notifier):
    """Detecta videos de competidores con rendimiento excepcional y envía alertas."""
    log.info("PASO ALERTAS: Verificando videos virales de competidores...")

    with YouTubeDatabase() as db:
        recent_df = db.get_recent_competitor_videos(days=7)

    if recent_df.empty:
        log.info("  Sin videos recientes de competidores")
        return

    alert_count = 0

    for _, row in recent_df.iterrows():
        video_id = row['video_id']
        view_count = int(row['view_count']) if pd.notna(row['view_count']) else 0
        avg_views = float(row['competitor_avg_views']) if pd.notna(row['competitor_avg_views']) else 0

        if avg_views <= 0 or view_count <= 0:
            continue

        ratio = view_count / avg_views

        if ratio < 2.0:
            continue

        # Verificar que no se haya enviado ya esta alerta
        with YouTubeDatabase() as db:
            if db.is_alert_already_sent(video_id):
                continue

        channel_name = row.get('channel_name', '')
        video_title = row.get('title', '')
        video_type = row.get('video_type', '')

        log.info("  🚨 Video viral detectado: \"%s\" de %s (%.1fx promedio)",
                 video_title[:60], channel_name, ratio)

        # Análisis con Claude
        ai_analysis = analyzer.analyze_viral_competitor_video(
            video_title=video_title,
            channel_name=channel_name,
            view_count=view_count,
            competitor_avg_views=avg_views,
            ratio=ratio,
            video_type=video_type,
        )

        # Guardar alerta en BD
        alert_data = {
            'video_id': video_id,
            'channel_id': row['channel_id'],
            'channel_name': channel_name,
            'video_title': video_title,
            'view_count': view_count,
            'competitor_avg_views': avg_views,
            'ratio': ratio,
            'ai_analysis': ai_analysis,
            'notified': 0,
        }

        with YouTubeDatabase() as db:
            db.save_competitor_alert(alert_data)

        # Notificar por Telegram
        if notifier and notifier.is_enabled():
            sent = notifier.notify_competitor_alert(
                channel_name=channel_name,
                video_title=video_title,
                view_count=view_count,
                competitor_avg_views=avg_views,
                ratio=ratio,
                ai_analysis=ai_analysis,
            )
            if sent:
                log.info("  📨 Alerta enviada por Telegram")
                with YouTubeDatabase() as db:
                    db.conn.cursor().execute(
                        "UPDATE competitor_alerts SET notified = 1 WHERE video_id = %s",
                        (video_id,),
                    )
                    db.conn.commit()

        alert_count += 1

    if alert_count:
        log.info("  🚨 %d alerta(s) de competidores generadas", alert_count)
    else:
        log.info("  Sin videos virales de competidores detectados")


def _step_series_detection(analyzer):
    """Detecta series automáticamente y genera recomendaciones de formato."""
    log.info("PASO SERIES: Detectando series de videos...")

    with YouTubeDatabase() as db:
        all_videos = db.get_all_videos()

    if all_videos.empty:
        log.info("  Sin videos para analizar")
        return

    detector = SeriesDetector()
    series_list = detector.detect(all_videos)

    if not series_list:
        log.info("  No se detectaron series")
        return

    log.info("  📚 %d series detectadas", len(series_list))

    for s in series_list:
        with YouTubeDatabase() as db:
            # Calcular métricas agregadas de la serie
            ep_ids = [e['video_id'] for e in s['episodes']]
            ep_videos = all_videos[all_videos['video_id'].isin(ep_ids)]
            avg_views = float(ep_videos['view_count'].mean()) if not ep_videos.empty and 'view_count' in ep_videos.columns else 0
            avg_eng = float(ep_videos['engagement_rate'].mean()) if not ep_videos.empty and 'engagement_rate' in ep_videos.columns else 0

            # Determinar tendencia (comparar primera mitad vs segunda mitad)
            trend = 'stable'
            if len(ep_videos) >= 4 and 'view_count' in ep_videos.columns:
                sorted_eps = ep_videos.sort_values('published_at')
                mid = len(sorted_eps) // 2
                first_half = sorted_eps.iloc[:mid]['view_count'].mean()
                second_half = sorted_eps.iloc[mid:]['view_count'].mean()
                if first_half > 0:
                    ratio = second_half / first_half
                    if ratio > 1.15:
                        trend = 'growing'
                    elif ratio < 0.85:
                        trend = 'declining'

            # Generar recomendación AI si hay >=3 episodios
            ai_rec = ''
            if len(s['episodes']) >= 3 and analyzer:
                ep_data = []
                for ep in s['episodes']:
                    vid = ep_videos[ep_videos['video_id'] == ep['video_id']]
                    ep_data.append({
                        'title': ep['title'],
                        'episode_number': ep['episode_number'],
                        'view_count': float(vid['view_count'].iloc[0]) if not vid.empty and 'view_count' in vid.columns else 0,
                        'engagement_rate': float(vid['engagement_rate'].iloc[0]) if not vid.empty and 'engagement_rate' in vid.columns else 0,
                        'published_at': ep['published_at'],
                    })
                channel_name = ep_videos['channel_title'].iloc[0] if 'channel_title' in ep_videos.columns and not ep_videos.empty else ''
                ai_rec = analyzer.recommend_series_format(
                    s['series_name'], channel_name, ep_data,
                )

            series_data = {
                'channel_id': s['channel_id'],
                'series_name': s['series_name'],
                'detected_pattern': s['detected_pattern'],
                'episode_count': len(s['episodes']),
                'avg_views': avg_views,
                'avg_engagement': avg_eng,
                'trend': trend,
                'ai_recommendation': ai_rec,
            }

            series_id = db.save_series(series_data)

            # Guardar episodios
            for ep in s['episodes']:
                db.save_series_episode(
                    series_id, ep['video_id'], ep['episode_number'],
                )

        log.info("  📚 Serie: %s (%d episodios, tendencia: %s)",
                 s['series_name'][:50], len(s['episodes']), trend)


def _step_revenue_analysis():
    """Calcula scores evergreen para videos con suficiente historial (Mejora 16.3)."""
    log.info("PASO REVENUE: Calculando scores evergreen...")

    detector = EvergreenDetector()

    with YouTubeDatabase() as db:
        own_channels = db.get_own_channels()

    if own_channels.empty:
        log.info("  Sin canales propios para analizar")
        return

    total_scored = 0
    for _, ch in own_channels.iterrows():
        channel_id = ch['channel_id']
        with YouTubeDatabase() as db:
            results = detector.analyze_channel(channel_id, db)

        if not results:
            continue

        evergreen_count = sum(1 for r in results
                              if r['classification'] == 'evergreen')
        log.info("  🌲 %s: %d videos analizados, %d evergreen",
                 ch.get('channel_name', channel_id), len(results), evergreen_count)

        with YouTubeDatabase() as db:
            for r in results:
                db.save_evergreen_score({
                    'video_id': r['video_id'],
                    'channel_id': channel_id,
                    'evergreen_score': r['evergreen_score'],
                    'classification': r['classification'],
                    'days_tracked': r['days_tracked'],
                    'recent_daily_views': r['recent_daily_views'],
                    'early_daily_views': r['early_daily_views'],
                    'decay_rate': r['decay_rate'],
                })
        total_scored += len(results)

    log.info("  Scores evergreen calculados para %d videos", total_scored)


def _step4_analytics(videos_df, channel_ids):
    """Extrae analytics avanzados via OAuth (opcional — requiere credentials.json)."""
    log.info("PASO 4: Extrayendo analytics avanzados (YouTube Analytics API)...")

    extractor = YouTubeAnalyticsExtractor()

    if not extractor.is_configured():
        log.warning("Analytics avanzados omitidos — coloca credentials.json en la raíz del proyecto. "
                     "Ver: https://console.cloud.google.com/apis/credentials")
        return

    if not extractor.authenticate():
        log.warning("No se pudo autenticar para Analytics avanzados. Omitiendo paso 4.")
        return

    with YouTubeDatabase() as db:
        for channel_id in channel_ids:
            channel_videos = videos_df[videos_df['channel_id'] == channel_id]
            if channel_videos.empty:
                continue

            log.info("  Canal %s:", channel_id)

            analytics_df = extractor.get_video_analytics(channel_id)
            if not analytics_df.empty:
                # Filtrar solo videos ya almacenados en la BD
                analytics_df = analytics_df[
                    analytics_df['video_id'].isin(channel_videos['video_id'])
                ]
            if not analytics_df.empty:
                db.save_video_analytics(analytics_df)
                log.info("    ✓ %d videos con analytics guardados", len(analytics_df))
            else:
                log.warning("    Sin datos de analytics por video")

            traffic_df = extractor.get_traffic_sources(channel_id)
            if not traffic_df.empty:
                db.save_traffic_sources(traffic_df)
                log.info("    ✓ %d fuentes de tráfico guardadas", len(traffic_df))
            else:
                log.warning("    Sin datos de fuentes de tráfico")

    log.info("Analytics avanzados guardados")


def main():
    """Función principal que ejecuta el pipeline completo"""
    load_dotenv(override=True)
    setup_logging()

    log.info("=" * 60)
    log.info("AGENTE DE IA PARA ANÁLISIS DE YOUTUBE")
    log.info("=" * 60)

    config = _validate_config()
    if not config:
        return

    youtube_api_key, anthropic_api_key, channel_ids, max_videos = config

    try:
        extractor = YouTubeDataExtractor(youtube_api_key)

        videos_df = _step1_extract(extractor, channel_ids, max_videos)
        if videos_df is None:
            return

        _step2_save(extractor, videos_df)

        # Extraer datos de competidores (si están configurados)
        _step_competitors(extractor, max_videos)

        notifier = TelegramNotifier()
        if notifier.is_enabled():
            log.info("Telegram: notificaciones activadas")
        else:
            log.info("Telegram: no configurado (opcional — ver .env)")

        analyzer = AIAnalyzer(anthropic_api_key)

        # Alertas de videos virales de competidores (Mejora 7.2)
        _step_competitor_alerts(analyzer, notifier)

        # Detección de series y recomendaciones de formato (Mejora 17.x)
        _step_series_detection(analyzer)

        # Análisis de revenue y detección de evergreen (Mejora 16.x)
        _step_revenue_analysis()

        _step3_analyze(analyzer, videos_df, channel_ids, notifier=notifier)

        _step4_analytics(videos_df, channel_ids)

        # Notificación final por Telegram
        if notifier.is_enabled():
            notifier.notify_pipeline_complete(
                channels_count=len(channel_ids),
                total_videos=len(videos_df),
            )

        log.info("=" * 60)
        log.info("✓ PROCESO COMPLETADO EXITOSAMENTE")
        log.info("=" * 60)
        log.info("Próximos pasos:")
        log.info("   1. Revisa las recomendaciones generadas")
        log.info("   2. Ejecuta 'streamlit run dashboard.py' para ver el dashboard")
        log.info("   3. Ejecuta este script diariamente para obtener nuevas recomendaciones")

    except Exception as e:
        log.error("Error durante la ejecución: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
