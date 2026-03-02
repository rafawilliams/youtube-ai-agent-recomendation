"""
Detector de videos con "despegue tardío" (Mejora 12.3).

Analiza el histórico de métricas (video_metrics) para identificar videos
que crecen lentamente en las primeras 48h pero aceleran después — patrón
típico de contenido SEO-driven.
"""
import logging
from datetime import datetime

import pandas as pd

log = logging.getLogger(__name__)


class LateBloomerDetector:
    """Detecta videos con patrón de despegue tardío analizando
    las snapshots de video_metrics a lo largo del tiempo."""

    MIN_SNAPSHOTS = 3
    EARLY_WINDOW_HOURS = 48
    ACCELERATION_THRESHOLD = 2.0

    PATTERN_LATE_BLOOMER = 'late_bloomer'
    PATTERN_FRONT_LOADED = 'front_loaded'
    PATTERN_STEADY = 'steady'
    PATTERN_INSUFFICIENT = 'insufficient_data'

    PATTERN_LABELS_ES = {
        'late_bloomer': '🌱 Despegue tardío',
        'front_loaded': '🚀 Carga frontal',
        'steady': '📈 Crecimiento constante',
        'insufficient_data': '❓ Datos insuficientes',
    }

    # ------------------------------------------------------------------
    # Análisis de un solo video
    # ------------------------------------------------------------------

    def detect(self, metrics_df: pd.DataFrame, published_at) -> dict:
        """
        Analiza el patrón de crecimiento de un video a partir de sus snapshots.

        Args:
            metrics_df: DataFrame con columnas [view_count, recorded_at]
                        de video_metrics para UN solo video.
            published_at: Fecha de publicación del video (str o datetime).

        Returns:
            dict con is_late_bloomer, pattern, growth rates, etc.
        """
        default = {
            'is_late_bloomer': False,
            'early_growth_rate': 0.0,
            'late_growth_rate': 0.0,
            'acceleration_factor': 0.0,
            'total_snapshots': 0,
            'early_views': 0,
            'current_views': 0,
            'pattern': self.PATTERN_INSUFFICIENT,
        }

        if metrics_df.empty or len(metrics_df) < self.MIN_SNAPSHOTS:
            return default

        df = metrics_df.copy()

        # Asegurar tipos
        df['recorded_at'] = pd.to_datetime(df['recorded_at'], utc=True, errors='coerce')
        pub_dt = pd.to_datetime(published_at, utc=True)
        df['view_count'] = pd.to_numeric(df['view_count'], errors='coerce').fillna(0)

        df = df.sort_values('recorded_at').reset_index(drop=True)

        # Calcular horas desde publicación
        df['hours_since'] = (df['recorded_at'] - pub_dt).dt.total_seconds() / 3600

        # Filtrar snapshots anteriores a la publicación (datos erróneos)
        df = df[df['hours_since'] >= 0]
        if len(df) < self.MIN_SNAPSHOTS:
            return default

        # Split early / late
        early = df[df['hours_since'] <= self.EARLY_WINDOW_HOURS]
        late = df[df['hours_since'] > self.EARLY_WINDOW_HOURS]

        if early.empty or late.empty:
            return {
                **default,
                'total_snapshots': len(df),
                'current_views': int(df['view_count'].iloc[-1]),
                'pattern': self.PATTERN_INSUFFICIENT,
            }

        # Growth rates (vistas por hora)
        early_min = float(early['view_count'].iloc[0])
        early_max = float(early['view_count'].iloc[-1])
        early_hours = max(float(early['hours_since'].iloc[-1] - early['hours_since'].iloc[0]), 1.0)
        early_growth = (early_max - early_min) / early_hours

        late_max = float(late['view_count'].iloc[-1])
        late_hours = max(float(late['hours_since'].iloc[-1] - early['hours_since'].iloc[-1]), 1.0)
        late_growth = (late_max - early_max) / late_hours

        # Evitar división por cero
        if early_growth <= 0:
            acceleration = late_growth / 0.1 if late_growth > 0 else 0.0
        else:
            acceleration = late_growth / early_growth

        # Determinar patrón
        if acceleration >= self.ACCELERATION_THRESHOLD:
            pattern = self.PATTERN_LATE_BLOOMER
        elif acceleration < 0.5:
            pattern = self.PATTERN_FRONT_LOADED
        else:
            pattern = self.PATTERN_STEADY

        return {
            'is_late_bloomer': pattern == self.PATTERN_LATE_BLOOMER,
            'early_growth_rate': round(early_growth, 2),
            'late_growth_rate': round(late_growth, 2),
            'acceleration_factor': round(acceleration, 2),
            'total_snapshots': len(df),
            'early_views': int(early_max),
            'current_views': int(df['view_count'].iloc[-1]),
            'pattern': pattern,
        }

    # ------------------------------------------------------------------
    # Análisis de canal completo
    # ------------------------------------------------------------------

    def analyze_channel(self, channel_id: str, db) -> list[dict]:
        """
        Analiza todos los videos del canal que tengan suficientes snapshots.

        Args:
            channel_id: ID del canal.
            db: Instancia de YouTubeDatabase.

        Returns:
            Lista de dicts con metadata del video + resultado de detect().
        """
        # Obtener videos con suficientes snapshots
        videos_df = db.get_videos_with_snapshot_counts(channel_id, self.MIN_SNAPSHOTS)
        if videos_df.empty:
            return []

        results: list[dict] = []
        for _, row in videos_df.iterrows():
            video_id = row['video_id']
            try:
                metrics_df = db.get_video_metrics_history(video_id)
                detection = self.detect(metrics_df, row['published_at'])
                results.append({
                    'video_id': video_id,
                    'title': row.get('title', ''),
                    'video_type': row.get('video_type', ''),
                    'published_at': str(row.get('published_at', '')),
                    'snapshot_count': int(row.get('snapshot_count', 0)),
                    **detection,
                })
            except Exception as e:
                log.debug("LateBloomer: error analizando %s — %s", video_id, e)
                continue

        results.sort(key=lambda r: r.get('acceleration_factor', 0), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Alertas
    # ------------------------------------------------------------------

    @staticmethod
    def get_recent_alerts(results: list[dict], days_lookback: int = 30) -> list[dict]:
        """Filtra videos recientes con patrón de despegue tardío."""
        cutoff = datetime.now().date()
        alerts: list[dict] = []
        for r in results:
            if not r.get('is_late_bloomer'):
                continue
            try:
                pub = pd.to_datetime(r['published_at']).date()
            except (ValueError, TypeError):
                continue
            if (cutoff - pub).days <= days_lookback:
                alerts.append(r)
        return alerts
