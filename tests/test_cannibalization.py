"""
Tests para src/cannibalization_detector.py — CannibalizationDetector.

Cubre: detección de pares similares (>=0.80), exclusión de pares diferentes (<0.80),
exclusión temporal (>30 días), y edge cases.
"""
import pytest
import pandas as pd
import numpy as np
from cannibalization_detector import CannibalizationDetector


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_df(rows):
    """Construye un DataFrame de videos desde dicts parciales."""
    defaults = {
        "tags": "",
        "published_at": pd.Timestamp("2025-10-15", tz="UTC"),
    }
    full_rows = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full_rows)


# ===========================================================================
# Detección de pares canibalizados
# ===========================================================================

class TestCannibalizationDetection:
    """Tests para CannibalizationDetector.detect()."""

    def test_nearly_identical_titles_detected(self):
        """Títulos casi idénticos (>0.80 similitud) se detectan como canibalización."""
        df = _make_df([
            {
                "video_id": "v1",
                "title": "Tutorial completo de Python para principiantes 2025",
                "tags": "python,tutorial,programación,principiantes",
                "published_at": pd.Timestamp("2025-10-10", tz="UTC"),
            },
            {
                "video_id": "v2",
                "title": "Tutorial completo de Python para principiantes 2025 actualizado",
                "tags": "python,tutorial,programación,principiantes",
                "published_at": pd.Timestamp("2025-10-12", tz="UTC"),
            },
        ])
        detector = CannibalizationDetector(threshold=0.80, max_days_apart=30)
        results = detector.detect(df)
        assert len(results) >= 1
        assert results[0]["similarity"] >= 0.80

    def test_different_titles_not_detected(self):
        """Títulos completamente diferentes no se detectan."""
        df = _make_df([
            {
                "video_id": "v1",
                "title": "Receta de pasta italiana con tomate fresco",
                "tags": "cocina,pasta,receta",
                "published_at": pd.Timestamp("2025-10-10", tz="UTC"),
            },
            {
                "video_id": "v2",
                "title": "Cómo programar en Rust desde cero para backend",
                "tags": "rust,programación,backend",
                "published_at": pd.Timestamp("2025-10-12", tz="UTC"),
            },
        ])
        detector = CannibalizationDetector(threshold=0.80, max_days_apart=30)
        results = detector.detect(df)
        assert len(results) == 0

    def test_videos_over_30_days_apart_excluded(self):
        """Videos con >30 días de diferencia se excluyen aunque sean similares."""
        df = _make_df([
            {
                "video_id": "v1",
                "title": "Tutorial completo de Python para principiantes",
                "tags": "python,tutorial,programación",
                "published_at": pd.Timestamp("2025-06-01", tz="UTC"),
            },
            {
                "video_id": "v2",
                "title": "Tutorial completo de Python para principiantes actualizado",
                "tags": "python,tutorial,programación",
                "published_at": pd.Timestamp("2025-09-15", tz="UTC"),
            },
        ])
        detector = CannibalizationDetector(threshold=0.80, max_days_apart=30)
        results = detector.detect(df)
        assert len(results) == 0

    def test_less_than_2_videos_returns_empty(self):
        """Con menos de 2 videos, retorna lista vacía."""
        df = _make_df([
            {"video_id": "v1", "title": "Un solo video"},
        ])
        detector = CannibalizationDetector()
        results = detector.detect(df)
        assert results == []

    def test_empty_corpus_returns_empty(self):
        """Videos sin título ni tags retornan lista vacía."""
        df = _make_df([
            {"video_id": "v1", "title": "", "tags": ""},
            {"video_id": "v2", "title": "", "tags": ""},
        ])
        detector = CannibalizationDetector()
        results = detector.detect(df)
        assert results == []

    def test_results_sorted_by_similarity_desc(self):
        """Los resultados están ordenados por similitud descendente."""
        df = _make_df([
            {
                "video_id": "v1",
                "title": "Python tutorial completo para principiantes básico",
                "tags": "python,tutorial",
                "published_at": pd.Timestamp("2025-10-01", tz="UTC"),
            },
            {
                "video_id": "v2",
                "title": "Python tutorial completo para principiantes avanzado",
                "tags": "python,tutorial",
                "published_at": pd.Timestamp("2025-10-05", tz="UTC"),
            },
            {
                "video_id": "v3",
                "title": "Python tutorial completo para principiantes intermedio",
                "tags": "python,tutorial",
                "published_at": pd.Timestamp("2025-10-10", tz="UTC"),
            },
        ])
        detector = CannibalizationDetector(threshold=0.50, max_days_apart=30)
        results = detector.detect(df)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["similarity"] >= results[i + 1]["similarity"]
