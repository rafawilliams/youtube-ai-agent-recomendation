"""
Tests para src/series_detector.py — SeriesDetector.

Cubre: 8 patrones de numeración, clustering TF-IDF, min_series_size,
separación por canal, y DataFrame vacío.
"""
import pytest
import pandas as pd
import numpy as np
from series_detector import SeriesDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(rows):
    """Construye un DataFrame de videos a partir de una lista de dicts parciales."""
    defaults = {
        "channel_id": "UC_test_001",
        "tags": "tag1,tag2",
        "published_at": pd.Timestamp("2025-10-01", tz="UTC"),
    }
    full_rows = []
    for r in rows:
        row = {**defaults, **r}
        full_rows.append(row)
    return pd.DataFrame(full_rows)


# ===========================================================================
# Patrones de numeración
# ===========================================================================

class TestNumberingPatterns:
    """Verifica que cada uno de los 8 patrones regex sea detectado."""

    def test_hashtag_pattern(self):
        """#1, #2 se detectan."""
        df = _make_df([
            {"video_id": "v1", "title": "Serie genial #1"},
            {"video_id": "v2", "title": "Serie genial #2"},
        ])
        detector = SeriesDetector()
        series = detector.detect(df)
        assert len(series) >= 1
        assert series[0]["detected_pattern"] == "numbering"
        assert len(series[0]["episodes"]) == 2

    def test_parte_pattern(self):
        """Parte 1, Parte 2 se detectan."""
        df = _make_df([
            {"video_id": "v1", "title": "Aventura épica Parte 1"},
            {"video_id": "v2", "title": "Aventura épica Parte 2"},
        ])
        series = SeriesDetector().detect(df)
        assert len(series) >= 1
        ep_nums = [e["episode_number"] for e in series[0]["episodes"]]
        assert 1 in ep_nums and 2 in ep_nums

    def test_part_pattern(self):
        """Part 1, Part 2 (inglés) se detectan."""
        df = _make_df([
            {"video_id": "v1", "title": "Amazing tutorial Part 1"},
            {"video_id": "v2", "title": "Amazing tutorial Part 2"},
        ])
        series = SeriesDetector().detect(df)
        assert len(series) >= 1

    def test_episodio_pattern(self):
        """Episodio 1, Episodio 2 se detectan."""
        df = _make_df([
            {"video_id": "v1", "title": "Mi serie Episodio 1"},
            {"video_id": "v2", "title": "Mi serie Episodio 2"},
        ])
        series = SeriesDetector().detect(df)
        assert len(series) >= 1

    def test_ep_pattern(self):
        """Ep 1, Ep. 2 se detectan."""
        df = _make_df([
            {"video_id": "v1", "title": "Documental Ep 1"},
            {"video_id": "v2", "title": "Documental Ep. 2"},
        ])
        series = SeriesDetector().detect(df)
        assert len(series) >= 1

    def test_capitulo_pattern(self):
        """Capítulo 1, Cap 2 se detectan."""
        df = _make_df([
            {"video_id": "v1", "title": "Historia loca Capítulo 1"},
            {"video_id": "v2", "title": "Historia loca Cap 2"},
        ])
        series = SeriesDetector().detect(df)
        assert len(series) >= 1

    def test_volumen_pattern(self):
        """Vol 1, Volumen 2 se detectan."""
        df = _make_df([
            {"video_id": "v1", "title": "Compilación de fails Vol 1"},
            {"video_id": "v2", "title": "Compilación de fails Vol 2"},
        ])
        series = SeriesDetector().detect(df)
        assert len(series) >= 1

    def test_dash_number_pattern(self):
        """'- 1', '- 2' al final se detectan."""
        df = _make_df([
            {"video_id": "v1", "title": "Recetas saludables - 1"},
            {"video_id": "v2", "title": "Recetas saludables - 2"},
        ])
        series = SeriesDetector().detect(df)
        assert len(series) >= 1

    def test_parentheses_fraction_pattern(self):
        """(1/5), (2/5) se detectan."""
        df = _make_df([
            {"video_id": "v1", "title": "Curso de Git (1/5)"},
            {"video_id": "v2", "title": "Curso de Git (2/5)"},
        ])
        series = SeriesDetector().detect(df)
        assert len(series) >= 1

    def test_bare_fraction_pattern(self):
        """1/5, 2/5 sin paréntesis se detectan."""
        df = _make_df([
            {"video_id": "v1", "title": "Viaje a Japón 1/5"},
            {"video_id": "v2", "title": "Viaje a Japón 2/5"},
        ])
        series = SeriesDetector().detect(df)
        assert len(series) >= 1


# ===========================================================================
# TF-IDF clustering
# ===========================================================================

class TestTFIDFClustering:
    """Tests para agrupación por similitud TF-IDF."""

    def test_groups_similar_titles(self):
        """Títulos muy similares (sin numeración) se agrupan por TF-IDF."""
        df = _make_df([
            {"video_id": "v1", "title": "Receta de pasta italiana deliciosa"},
            {"video_id": "v2", "title": "Receta de pasta italiana casera"},
            {"video_id": "v3", "title": "Programación en Rust desde cero"},
        ])
        detector = SeriesDetector(similarity_threshold=0.50)
        series = detector.detect(df)
        # Al menos las dos recetas deben agruparse
        recipe_series = [s for s in series if any("pasta" in e["title"].lower() for e in s["episodes"])]
        assert len(recipe_series) >= 1
        assert len(recipe_series[0]["episodes"]) >= 2


# ===========================================================================
# Reglas de negocio
# ===========================================================================

class TestBusinessRules:
    """Tests para min_series_size, separación por canal y edge cases."""

    def test_min_series_size_excludes_singles(self):
        """Un solo video con patrón #1 no forma serie (min_series_size=2)."""
        df = _make_df([
            {"video_id": "v1", "title": "Video único #1"},
            {"video_id": "v2", "title": "Algo completamente diferente sin número"},
        ])
        series = SeriesDetector().detect(df)
        # No debería haber series de 1 episodio
        for s in series:
            assert len(s["episodes"]) >= 2

    def test_videos_from_different_channels_dont_mix(self):
        """Videos del mismo nombre pero de canales distintos no se mezclan."""
        df = _make_df([
            {"video_id": "v1", "title": "Tutorial React #1", "channel_id": "UC_A"},
            {"video_id": "v2", "title": "Tutorial React #2", "channel_id": "UC_A"},
            {"video_id": "v3", "title": "Tutorial React #1", "channel_id": "UC_B"},
            {"video_id": "v4", "title": "Tutorial React #2", "channel_id": "UC_B"},
        ])
        series = SeriesDetector().detect(df)
        # Debe haber 2 series separadas, una por canal
        assert len(series) == 2
        channel_ids = {s["channel_id"] for s in series}
        assert channel_ids == {"UC_A", "UC_B"}

    def test_empty_dataframe_returns_empty(self):
        """DataFrame vacío retorna lista vacía."""
        df = pd.DataFrame(columns=["video_id", "title", "tags", "published_at", "channel_id"])
        series = SeriesDetector().detect(df)
        assert series == []
