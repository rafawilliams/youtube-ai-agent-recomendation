"""
Tests para src/revenue_analyzer.py — RevenueEstimator, ContentROICalculator, EvergreenDetector.

Cubre: estimación de ingresos, agrupación mensual, proyecciones, ROI por tipo,
detección evergreen con diferentes patrones.
"""
import pytest
import pandas as pd
import numpy as np
from revenue_analyzer import RevenueEstimator, ContentROICalculator, EvergreenDetector


# ===========================================================================
# RevenueEstimator
# ===========================================================================

class TestRevenueEstimator:
    """Tests para RevenueEstimator."""

    def test_estimate_video_revenue_known_cpm(self):
        """Calcula ingresos correctamente: views * CPM / 1000."""
        estimator = RevenueEstimator(cpm=5.0)
        revenue = estimator.estimate_video_revenue(10000)
        assert revenue == pytest.approx(50.0)

    def test_estimate_video_revenue_zero_views(self):
        """Con 0 views, el ingreso es 0."""
        estimator = RevenueEstimator(cpm=5.0)
        assert estimator.estimate_video_revenue(0) == 0.0

    def test_estimate_channel_monthly_groups_by_month(self):
        """Agrupa videos por mes y calcula ingresos estimados."""
        df = pd.DataFrame({
            "video_id": ["v1", "v2", "v3", "v4"],
            "published_at": pd.to_datetime([
                "2025-01-15", "2025-01-20",
                "2025-02-10", "2025-02-25",
            ]),
            "view_count": [10000, 20000, 15000, 5000],
        })
        estimator = RevenueEstimator(cpm=4.0)
        result = estimator.estimate_channel_monthly(df)

        assert len(result) == 2  # 2 meses
        assert "estimated_revenue" in result.columns
        assert "total_views" in result.columns
        assert "video_count" in result.columns

        # Enero: 30000 views * 4 / 1000 = 120
        jan = result.iloc[0]
        assert jan["total_views"] == 30000
        assert jan["estimated_revenue"] == pytest.approx(120.0)

    def test_estimate_channel_monthly_empty_df(self):
        """DataFrame vacío retorna DataFrame vacío con columnas correctas."""
        estimator = RevenueEstimator(cpm=4.0)
        result = estimator.estimate_channel_monthly(pd.DataFrame())
        assert result.empty
        assert "estimated_revenue" in result.columns

    def test_project_revenue_insufficient_data(self):
        """Con <2 meses de datos, las proyecciones son 0."""
        estimator = RevenueEstimator(cpm=4.0)
        monthly_df = pd.DataFrame({
            "month": [pd.Timestamp("2025-01-01")],
            "estimated_revenue": [100.0],
        })
        result = estimator.project_revenue(monthly_df)
        assert result[3] == 0.0
        assert result[6] == 0.0
        assert result[12] == 0.0

    def test_project_revenue_with_data(self):
        """Con suficientes meses, project_revenue devuelve proyecciones > 0."""
        estimator = RevenueEstimator(cpm=4.0)
        monthly_df = pd.DataFrame({
            "month": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
            "estimated_revenue": [100.0, 150.0, 200.0],
        })
        result = estimator.project_revenue(monthly_df, months_ahead=[3, 6])
        # Con tendencia positiva, las proyecciones deben ser > 0
        assert result[3] > 0
        assert result[6] > 0


# ===========================================================================
# ContentROICalculator
# ===========================================================================

class TestContentROICalculator:
    """Tests para ContentROICalculator."""

    def test_calculate_roi_separates_types(self):
        """Calcula ROI separando Short vs Video Largo."""
        df = pd.DataFrame({
            "video_id": ["v1", "v2", "v3", "v4"],
            "video_type": ["Short", "Short", "Video Largo", "Video Largo"],
            "view_count": [5000, 8000, 20000, 15000],
        })
        calc = ContentROICalculator()
        result = calc.calculate_roi(df, cpm=5.0)

        assert len(result) == 2
        assert "revenue_per_hour" in result.columns
        assert "production_hours" in result.columns

        # Verificar que Short tiene 1h de producción y Video Largo 8h
        short_row = result[result["video_type"] == "Short"].iloc[0]
        long_row = result[result["video_type"] == "Video Largo"].iloc[0]
        assert short_row["production_hours"] == 1.0
        assert long_row["production_hours"] == 8.0

    def test_calculate_roi_empty_df(self):
        """DataFrame vacío retorna DataFrame vacío."""
        calc = ContentROICalculator()
        result = calc.calculate_roi(pd.DataFrame(), cpm=5.0)
        assert result.empty


# ===========================================================================
# EvergreenDetector
# ===========================================================================

class TestEvergreenDetector:
    """Tests para EvergreenDetector."""

    def test_insufficient_data(self):
        """Con pocos snapshots, classification es 'insufficient_data'."""
        detector = EvergreenDetector()
        metrics_df = pd.DataFrame({
            "view_count": [100, 200],
            "recorded_at": pd.to_datetime(["2025-01-01", "2025-01-05"], utc=True),
        })
        result = detector.detect(metrics_df, "2025-01-01")
        assert result["classification"] == "insufficient_data"

    def test_steady_views_pattern(self):
        """Video con vistas estables se clasifica como 'steady' o 'evergreen'."""
        detector = EvergreenDetector()
        pub_date = "2025-01-01"
        base = pd.Timestamp("2025-01-01", tz="UTC")
        # Crear snapshots con vistas incrementales constantes (steady growth)
        days = list(range(0, 91, 3))  # 0, 3, 6, ..., 90 (31 snapshots)
        views = [1000 + d * 50 for d in days]  # Crecimiento lineal
        metrics_df = pd.DataFrame({
            "view_count": views,
            "recorded_at": [base + pd.Timedelta(days=d) for d in days],
        })

        result = detector.detect(metrics_df, pub_date)
        assert result["classification"] in ("steady", "evergreen")
        assert result["days_tracked"] >= 30

    def test_spike_decline_pattern(self):
        """Video con pico inicial y caída fuerte se clasifica como 'spike_decline'."""
        detector = EvergreenDetector()
        pub_date = "2025-01-01"
        base = pd.Timestamp("2025-01-01", tz="UTC")

        # Pico inicial: muchas vistas los primeros días, luego casi nada
        days = list(range(0, 91, 3))  # 31 snapshots
        views = []
        cumulative = 0
        for d in days:
            if d < 10:
                cumulative += 5000  # Alto crecimiento temprano
            else:
                cumulative += 10   # Casi nada después
            views.append(cumulative)

        metrics_df = pd.DataFrame({
            "view_count": views,
            "recorded_at": [base + pd.Timedelta(days=d) for d in days],
        })
        result = detector.detect(metrics_df, pub_date)
        assert result["classification"] == "spike_decline"
        assert result["evergreen_score"] < 0.2

    def test_empty_metrics_returns_insufficient(self):
        """DataFrame de métricas vacío retorna insufficient_data."""
        detector = EvergreenDetector()
        empty_df = pd.DataFrame(columns=["view_count", "recorded_at"])
        result = detector.detect(empty_df, "2025-01-01")
        assert result["classification"] == "insufficient_data"
