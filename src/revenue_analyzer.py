"""
Estimador de ingresos, ROI por tipo de contenido y detector de videos
evergreen (Mejora 16.x).

- RevenueEstimator (16.1): estima ingresos por video basado en CPM configurable.
- ContentROICalculator (16.2): calcula ingreso por hora de producción por tipo.
- EvergreenDetector (16.3): identifica videos que siguen generando vistas
  meses después de publicados, analizando el histórico de video_metrics.

CPM configurable via .env: ESTIMATED_CPM=4.50
"""
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_CPM = 4.50


# ══════════════════════════════════════════════════════════════════════
# 16.1 — Estimador de ingresos
# ══════════════════════════════════════════════════════════════════════

class RevenueEstimator:
    """Estima ingresos por video y canal basado en CPM configurable."""

    def __init__(self, cpm: float | None = None):
        if cpm is not None:
            self.cpm = cpm
        else:
            self.cpm = float(os.getenv('ESTIMATED_CPM', str(DEFAULT_CPM)))
        log.debug("RevenueEstimator inicializado con CPM=%.2f", self.cpm)

    def estimate_video_revenue(self, view_count: int) -> float:
        """Estima ingresos de un video: views * CPM / 1000."""
        return (view_count or 0) * self.cpm / 1000

    def estimate_channel_monthly(self, videos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrupa videos por mes de publicación y calcula ingresos estimados.

        Returns:
            DataFrame con columnas [month, video_count, total_views, estimated_revenue]
        """
        if videos_df.empty or 'published_at' not in videos_df.columns:
            return pd.DataFrame(columns=['month', 'video_count', 'total_views',
                                         'estimated_revenue'])

        df = videos_df.copy()
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        df = df.dropna(subset=['published_at'])

        if df.empty:
            return pd.DataFrame(columns=['month', 'video_count', 'total_views',
                                         'estimated_revenue'])

        df['month'] = df['published_at'].dt.to_period('M')
        df['view_count'] = pd.to_numeric(df['view_count'], errors='coerce').fillna(0)

        monthly = df.groupby('month').agg(
            video_count=('video_id', 'count'),
            total_views=('view_count', 'sum'),
        ).reset_index()

        monthly['estimated_revenue'] = monthly['total_views'] * self.cpm / 1000
        monthly['month'] = monthly['month'].dt.to_timestamp()
        monthly = monthly.sort_values('month').reset_index(drop=True)

        return monthly

    def project_revenue(
        self,
        monthly_df: pd.DataFrame,
        months_ahead: list[int] | None = None,
    ) -> dict:
        """
        Proyecta ingresos futuros usando regresión lineal sobre los últimos 6 meses.

        Args:
            monthly_df: DataFrame de estimate_channel_monthly()
            months_ahead: Lista de horizontes en meses (default [3, 6, 12])

        Returns:
            dict con claves por horizonte: {3: float, 6: float, 12: float}
        """
        if months_ahead is None:
            months_ahead = [3, 6, 12]

        result = {m: 0.0 for m in months_ahead}

        if monthly_df.empty or len(monthly_df) < 2:
            return result

        # Usar últimos 6 meses para la tendencia
        recent = monthly_df.tail(6).copy()
        if len(recent) < 2:
            return result

        # X = índice secuencial, Y = ingresos mensuales
        x = np.arange(len(recent), dtype=float)
        y = recent['estimated_revenue'].values.astype(float)

        try:
            slope, intercept = np.polyfit(x, y, 1)
        except (np.linalg.LinAlgError, ValueError):
            log.debug("No se pudo ajustar regresión lineal para proyecciones")
            return result

        last_x = float(len(recent) - 1)
        last_monthly = float(y[-1])

        for m in months_ahead:
            # Proyectar ingresos mensuales futuros y sumar
            total = 0.0
            for i in range(1, m + 1):
                projected_monthly = max(intercept + slope * (last_x + i), 0)
                total += projected_monthly
            result[m] = round(total, 2)

        return result


# ══════════════════════════════════════════════════════════════════════
# 16.2 — ROI por tipo de contenido
# ══════════════════════════════════════════════════════════════════════

class ContentROICalculator:
    """Calcula ingreso estimado por hora de producción por tipo de video."""

    DEFAULT_HOURS = {
        'Short': 1.0,
        'Video Largo': 8.0,
    }

    def __init__(self, production_hours: dict | None = None):
        self.production_hours = production_hours or self.DEFAULT_HOURS.copy()

    def calculate_roi(self, videos_df: pd.DataFrame, cpm: float) -> pd.DataFrame:
        """
        Calcula ROI por tipo de contenido.

        Returns:
            DataFrame con columnas [video_type, total_videos, avg_views,
                                    avg_revenue, revenue_per_hour, production_hours]
        """
        if videos_df.empty or 'video_type' not in videos_df.columns:
            return pd.DataFrame()

        df = videos_df.copy()
        df['view_count'] = pd.to_numeric(df['view_count'], errors='coerce').fillna(0)
        df['estimated_revenue'] = df['view_count'] * cpm / 1000

        grouped = df.groupby('video_type').agg(
            total_videos=('video_id', 'count'),
            avg_views=('view_count', 'mean'),
            avg_revenue=('estimated_revenue', 'mean'),
        ).reset_index()

        # Asignar horas de producción y calcular ROI
        grouped['production_hours'] = grouped['video_type'].map(
            self.production_hours
        ).fillna(4.0)  # Default 4h si el tipo no está mapeado
        grouped['revenue_per_hour'] = grouped['avg_revenue'] / grouped['production_hours']

        return grouped.sort_values('revenue_per_hour', ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# 16.3 — Detector de videos evergreen
# ══════════════════════════════════════════════════════════════════════

class EvergreenDetector:
    """Detecta videos que siguen generando vistas meses después de publicados,
    analizando el histórico de video_metrics a lo largo del tiempo.

    Patrón similar a LateBloomerDetector pero con ventanas de 30 días
    en lugar de 48h."""

    MIN_SNAPSHOTS = 5
    MIN_DAYS_TRACKED = 30
    EARLY_WINDOW_DAYS = 30
    RECENT_WINDOW_DAYS = 30
    EVERGREEN_RATIO_THRESHOLD = 0.5

    CLASSIFICATION_EVERGREEN = 'evergreen'
    CLASSIFICATION_SPIKE_DECLINE = 'spike_decline'
    CLASSIFICATION_STEADY = 'steady'
    CLASSIFICATION_INSUFFICIENT = 'insufficient_data'

    CLASSIFICATION_LABELS_ES = {
        'evergreen': '🌲 Evergreen',
        'spike_decline': '📉 Pico y caída',
        'steady': '📊 Estable',
        'insufficient_data': '❓ Datos insuficientes',
    }

    # ------------------------------------------------------------------
    # Análisis de un solo video
    # ------------------------------------------------------------------

    def detect(self, metrics_df: pd.DataFrame, published_at) -> dict:
        """
        Analiza el patrón de longevidad de un video a partir de sus snapshots.

        Args:
            metrics_df: DataFrame con columnas [view_count, recorded_at]
                        de video_metrics para UN solo video.
            published_at: Fecha de publicación del video (str o datetime).

        Returns:
            dict con evergreen_score, classification, days_tracked, etc.
        """
        default = {
            'evergreen_score': 0.0,
            'classification': self.CLASSIFICATION_INSUFFICIENT,
            'days_tracked': 0,
            'recent_daily_views': 0.0,
            'early_daily_views': 0.0,
            'decay_rate': 0.0,
        }

        if metrics_df.empty or len(metrics_df) < self.MIN_SNAPSHOTS:
            return default

        df = metrics_df.copy()

        # Asegurar tipos
        df['recorded_at'] = pd.to_datetime(df['recorded_at'], utc=True, errors='coerce')
        pub_dt = pd.to_datetime(published_at, utc=True)
        df['view_count'] = pd.to_numeric(df['view_count'], errors='coerce').fillna(0)

        df = df.sort_values('recorded_at').reset_index(drop=True)

        # Días desde publicación
        df['days_since'] = (df['recorded_at'] - pub_dt).dt.total_seconds() / 86400

        # Filtrar snapshots anteriores a la publicación
        df = df[df['days_since'] >= 0]
        if len(df) < self.MIN_SNAPSHOTS:
            return default

        days_tracked = int(df['days_since'].iloc[-1])
        if days_tracked < self.MIN_DAYS_TRACKED:
            return {**default, 'days_tracked': days_tracked}

        # Calcular ganancia diaria de vistas entre snapshots consecutivos
        df['view_diff'] = df['view_count'].diff().fillna(0).clip(lower=0)
        df['day_diff'] = df['days_since'].diff().fillna(1).clip(lower=0.01)
        df['daily_views'] = df['view_diff'] / df['day_diff']

        # Ventana temprana (primeros 30 días)
        early = df[df['days_since'] <= self.EARLY_WINDOW_DAYS]
        # Ventana reciente (últimos 30 días)
        last_day = df['days_since'].iloc[-1]
        recent = df[df['days_since'] >= (last_day - self.RECENT_WINDOW_DAYS)]

        early_daily = float(early['daily_views'].mean()) if not early.empty else 0.0
        recent_daily = float(recent['daily_views'].mean()) if not recent.empty else 0.0

        # Ratio reciente/temprano
        if early_daily > 0:
            ratio = recent_daily / early_daily
        else:
            ratio = 1.0 if recent_daily > 0 else 0.0

        # Decay rate (pendiente de regresión lineal sobre vistas diarias)
        decay_rate = 0.0
        if len(df) >= 3:
            x = df['days_since'].values.astype(float)
            y = df['daily_views'].values.astype(float)
            try:
                slope, _ = np.polyfit(x, y, 1)
                decay_rate = round(float(slope), 4)
            except (np.linalg.LinAlgError, ValueError):
                pass

        # Score y clasificación
        evergreen_score = round(min(max(ratio, 0.0), 1.0), 3)

        if ratio >= self.EVERGREEN_RATIO_THRESHOLD:
            classification = self.CLASSIFICATION_EVERGREEN
        elif ratio < 0.2:
            classification = self.CLASSIFICATION_SPIKE_DECLINE
        else:
            classification = self.CLASSIFICATION_STEADY

        return {
            'evergreen_score': evergreen_score,
            'classification': classification,
            'days_tracked': days_tracked,
            'recent_daily_views': round(recent_daily, 2),
            'early_daily_views': round(early_daily, 2),
            'decay_rate': decay_rate,
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
                log.debug("Evergreen: error analizando %s — %s", video_id, e)
                continue

        results.sort(key=lambda r: r.get('evergreen_score', 0), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @staticmethod
    def get_top_evergreen(results: list[dict], top_n: int = 10) -> list[dict]:
        """Filtra solo los videos clasificados como evergreen, top N por score."""
        evergreen = [r for r in results if r.get('classification') == 'evergreen']
        return evergreen[:top_n]
