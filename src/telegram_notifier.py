"""
Notificaciones por Telegram para el pipeline de YouTube AI Agent.

Envía recomendaciones diarias y métricas del canal via Telegram Bot API.
Si TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID no están configurados, se omite silenciosamente.

Configuración en .env:
    TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
    TELEGRAM_CHAT_ID=987654321
"""
import os
import logging
import requests

from retry_config import retry_http

log = logging.getLogger(__name__)


class TelegramNotifier:
    """Envía notificaciones al chat de Telegram configurado."""

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self):
        self._token = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
        self._chat_id = os.getenv('TELEGRAM_CHAT_ID', '').strip()
        self.enabled = bool(self._token and self._chat_id)

    def is_enabled(self) -> bool:
        return self.enabled

    def send_message(self, text: str) -> bool:
        """
        Envía un mensaje Markdown al chat configurado.
        Retorna True si se envió correctamente, False en caso contrario.
        Nunca lanza excepciones — el pipeline no debe fallar por Telegram.
        """
        if not self.enabled:
            return False

        try:
            self._post_message(text)
            return True
        except Exception as e:
            log.warning("Telegram: error definitivo tras reintentos — %s", e)
            return False

    @retry_http
    def _post_message(self, text: str) -> None:
        """Envía el mensaje HTTP con retry automático en errores transitorios."""
        resp = requests.post(
            self.API_URL.format(token=self._token),
            json={
                'chat_id': self._chat_id,
                'text': text,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True,
            },
            timeout=10,
        )
        resp.raise_for_status()

    def notify_recommendation(self, channel_name: str, recommendation: dict, stats: dict) -> bool:
        """
        Envía la recomendación diaria + métricas clave del canal.

        Args:
            channel_name: Nombre del canal
            recommendation: dict con recommendation_date, recommended_type,
                            predicted_performance, title_suggestions, reasoning
            stats: dict con total_views, avg_views, avg_engagement_rate,
                   best_video_title, best_video_views
        """
        # Título sugerido (primer sugerencia si existe)
        title_suggestions = recommendation.get('title_suggestions', [])
        suggested_title = title_suggestions[0]['title'] if title_suggestions else '—'

        # Escapar caracteres problemáticos de Markdown
        channel_name_safe = _md_escape(channel_name)
        suggested_title_safe = _md_escape(suggested_title)
        best_title = _md_escape(stats.get('best_video_title', '—')[:50])

        lines = [
            f"🎯 *Recomendación — {channel_name_safe}*",
            "",
            f"📅 {recommendation.get('recommendation_date', '—')}",
            f"🎬 Formato: {recommendation.get('recommended_type', '—')}",
            f"📊 Performance esperado: {_md_escape(recommendation.get('predicted_performance', '—'))}",
            "",
            "✍ *Título sugerido:*",
            f"_{suggested_title_safe}_",
            "",
            "📈 *Métricas del canal:*",
            f"• Vistas totales: {stats.get('total_views', 0):,.0f}",
            f"• Promedio/video: {stats.get('avg_views', 0):,.0f}",
            f"• Engagement: {stats.get('avg_engagement_rate', 0):.2f}%",
            f"• Mejor video: \"{best_title}\" ({stats.get('best_video_views', 0):,.0f} vistas)",
        ]

        return self.send_message('\n'.join(lines))

    def notify_pipeline_complete(self, channels_count: int, total_videos: int) -> bool:
        """Mensaje breve de confirmación al final del pipeline."""
        text = (
            "✅ *Pipeline completado*\n"
            f"Canales: {channels_count} | Videos procesados: {total_videos:,}"
        )
        return self.send_message(text)

    def notify_competitor_alert(
        self,
        channel_name: str,
        video_title: str,
        view_count: int,
        competitor_avg_views: float,
        ratio: float,
        ai_analysis: str,
    ) -> bool:
        """
        Envía alerta cuando un video de competidor supera 2x su promedio.

        Args:
            channel_name: Nombre del canal competidor
            video_title: Título del video viral
            view_count: Vistas actuales
            competitor_avg_views: Promedio de vistas del competidor
            ratio: Multiplicador sobre el promedio
            ai_analysis: Análisis de Claude (markdown)
        """
        channel_safe = _md_escape(channel_name)
        title_safe = _md_escape(video_title[:80])

        # Truncar análisis para caber en Telegram (max ~4000 chars)
        analysis_short = ai_analysis[:2500] if ai_analysis else '(sin análisis)'

        lines = [
            f"🚨 *Alerta Competidor — {channel_safe}*",
            "",
            f"📺 _{title_safe}_",
            f"👀 Vistas: {view_count:,} ({ratio:.1f}x su promedio de {competitor_avg_views:,.0f})",
            "",
            "🤖 *Análisis:*",
            analysis_short,
        ]

        return self.send_message('\n'.join(lines))


def _md_escape(text: str) -> str:
    """Escapa caracteres especiales de Markdown v1 para Telegram."""
    for ch in ('_', '*', '`', '['):
        text = text.replace(ch, '\\' + ch)
    return text
