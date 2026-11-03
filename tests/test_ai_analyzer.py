"""
Tests para src/ai_analyzer.py — AIAnalyzer.

Cubre: _parse_recommendation, _extract_title_suggestions, _extract_topic, _predict_performance.
Todos los tests evitan llamadas reales a la API de Anthropic.
"""
import pytest
from unittest.mock import patch, MagicMock
from ai_analyzer import AIAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer():
    """Crea un AIAnalyzer con Anthropic mockeado."""
    with patch("ai_analyzer.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        analyzer = AIAnalyzer(anthropic_api_key="fake-key-for-testing")
    return analyzer


# ===========================================================================
# _parse_recommendation
# ===========================================================================

class TestParseRecommendation:
    """Tests para _parse_recommendation."""

    def test_detects_short_type(self, sample_stats):
        """Si el texto menciona 'short' sin 'no short', el tipo es Short."""
        analyzer = _make_analyzer()
        text = "Recomiendo publicar un Short de 60 segundos sobre Python."
        result = analyzer._parse_recommendation(text, sample_stats)
        assert result["recommended_type"] == "Short"

    def test_detects_video_largo_when_no_short_mentioned(self, sample_stats):
        """Si el texto no menciona 'short', el tipo es Video Largo."""
        analyzer = _make_analyzer()
        text = "Recomiendo publicar un video largo tutorial sobre React."
        result = analyzer._parse_recommendation(text, sample_stats)
        assert result["recommended_type"] == "Video Largo"

    def test_no_short_phrase_means_video_largo(self, sample_stats):
        """Si el texto dice 'no short', se interpreta como Video Largo."""
        analyzer = _make_analyzer()
        text = "Hoy no short es la mejor opción. Mejor un video largo de análisis."
        result = analyzer._parse_recommendation(text, sample_stats)
        assert result["recommended_type"] == "Video Largo"

    def test_result_has_required_keys(self, sample_stats):
        """El resultado contiene todas las claves esperadas."""
        analyzer = _make_analyzer()
        text = "Publica un Short sobre productividad."
        result = analyzer._parse_recommendation(text, sample_stats)
        expected_keys = {
            "recommendation_date",
            "recommended_type",
            "recommended_topic",
            "reasoning",
            "predicted_performance",
            "title_suggestions",
            "generated_at",
        }
        assert expected_keys.issubset(result.keys())

    def test_reasoning_preserves_original_text(self, sample_stats):
        """El campo reasoning guarda el texto original."""
        analyzer = _make_analyzer()
        text = "Texto completo de la recomendación con datos."
        result = analyzer._parse_recommendation(text, sample_stats)
        assert result["reasoning"] == text


# ===========================================================================
# _extract_title_suggestions
# ===========================================================================

class TestExtractTitleSuggestions:
    """Tests para _extract_title_suggestions."""

    def test_extracts_three_titles_with_accents(self, mock_claude_response):
        """Parsea 3 sugerencias con acentos (TÍTULO/ANÁLISIS)."""
        analyzer = _make_analyzer()
        suggestions = analyzer._extract_title_suggestions(mock_claude_response)
        assert len(suggestions) == 3
        for s in suggestions:
            assert "title" in s
            assert "analysis" in s
            assert len(s["title"]) > 0

    def test_title_without_accents(self):
        """Acepta TITULO sin acento (I en lugar de I con tilde)."""
        analyzer = _make_analyzer()
        text = (
            "TITULO 1: Mi primer video\n"
            "ANALISIS: Funciona por curiosidad\n"
            "TITULO 2: Segundo video genial\n"
            "ANALISIS: Bueno para SEO\n"
        )
        suggestions = analyzer._extract_title_suggestions(text)
        assert len(suggestions) == 2
        assert suggestions[0]["title"] == "Mi primer video"
        assert suggestions[1]["title"] == "Segundo video genial"

    def test_empty_text_returns_empty_list(self):
        """Texto vacío devuelve lista vacía."""
        analyzer = _make_analyzer()
        suggestions = analyzer._extract_title_suggestions("")
        assert suggestions == []

    def test_only_two_titles_returns_two(self):
        """Si solo hay 2 títulos, retorna 2."""
        analyzer = _make_analyzer()
        text = (
            "TÍTULO 1: Primer título interesante\n"
            "ANÁLISIS: Genera curiosidad.\n"
            "TÍTULO 2: Segundo título atractivo\n"
            "ANÁLISIS: Tiene buen SEO.\n"
        )
        suggestions = analyzer._extract_title_suggestions(text)
        assert len(suggestions) == 2

    def test_strips_quotes_from_titles(self):
        """Remueve comillas del título extraído."""
        analyzer = _make_analyzer()
        text = 'TÍTULO 1: "Mi título con comillas"\nANÁLISIS: Buen enganche.\n'
        suggestions = analyzer._extract_title_suggestions(text)
        assert len(suggestions) == 1
        assert '"' not in suggestions[0]["title"]


# ===========================================================================
# _extract_topic
# ===========================================================================

class TestExtractTopic:
    """Tests para _extract_topic."""

    def test_detects_line_with_tema_keyword(self):
        """Detecta la linea que contiene 'tema'."""
        analyzer = _make_analyzer()
        text = "Línea sin relevancia\nEl tema recomendado es productividad.\nOtra línea."
        topic = analyzer._extract_topic(text)
        assert "tema" in topic.lower()
        assert "productividad" in topic.lower()

    def test_detects_line_with_contenido_keyword(self):
        """Detecta la linea que contiene 'contenido'."""
        analyzer = _make_analyzer()
        text = "Intro del texto.\nTipo de contenido: Tutorial de Python.\nFin."
        topic = analyzer._extract_topic(text)
        assert "contenido" in topic.lower()

    def test_detects_line_with_topic_keyword(self):
        """Detecta la linea que contiene 'topic'."""
        analyzer = _make_analyzer()
        text = "Something here.\nSuggested topic for the video: AI coding.\nEnd."
        topic = analyzer._extract_topic(text)
        assert "topic" in topic.lower()

    def test_no_keywords_returns_first_100_chars(self):
        """Sin keywords, retorna los primeros 100 caracteres + '...'."""
        analyzer = _make_analyzer()
        text = "A" * 200
        topic = analyzer._extract_topic(text)
        assert len(topic) == 103  # 100 chars + "..."
        assert topic.endswith("...")


# ===========================================================================
# _predict_performance
# ===========================================================================

class TestPredictPerformance:
    """Tests para _predict_performance."""

    def test_uses_shorts_stats_for_short(self, sample_stats):
        """Para tipo Short, usa shorts_avg_views y shorts_avg_engagement."""
        analyzer = _make_analyzer()
        result = analyzer._predict_performance("Short", sample_stats)
        assert "20000" in result or "20,000" in result or "20000" in result.replace(",", "")
        assert "6.5" in result

    def test_uses_long_stats_for_video_largo(self, sample_stats):
        """Para tipo Video Largo, usa long_videos_avg_views y long_videos_avg_engagement."""
        analyzer = _make_analyzer()
        result = analyzer._predict_performance("Video Largo", sample_stats)
        assert "12000" in result or "12,000" in result or "12000" in result.replace(",", "")
        assert "4.8" in result

    def test_returns_string_with_expected_format(self, sample_stats):
        """El resultado contiene 'Vistas esperadas' y 'Engagement esperado'."""
        analyzer = _make_analyzer()
        result = analyzer._predict_performance("Short", sample_stats)
        assert "Vistas esperadas" in result
        assert "Engagement esperado" in result
