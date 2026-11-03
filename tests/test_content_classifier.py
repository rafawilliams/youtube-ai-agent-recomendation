"""
Tests para src/content_classifier.py — ContentClassifier.classify_single_by_keywords().

Cubre: 10 categorías con títulos/tags representativos y orden de prioridad.
"""
import pytest
from unittest.mock import patch, MagicMock
from content_classifier import ContentClassifier


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _classify(title="", tags=""):
    """Shortcut para classify_single_by_keywords (método estático)."""
    return ContentClassifier.classify_single_by_keywords(title, tags)


# ===========================================================================
# Test de cada categoría
# ===========================================================================

class TestCategoryDetection:
    """Verifica que cada categoría se detecta con títulos/tags representativos."""

    def test_reaction_detected(self):
        """Título con 'reacción' se clasifica como reaction."""
        assert _classify("Mi reacción al nuevo trailer") == "reaction"

    def test_reaction_english(self):
        """Título con 'react' se clasifica como reaction."""
        assert _classify("Reacting to viral TikToks") == "reaction"

    def test_challenge_detected(self):
        """Título con 'challenge' o 'reto' se clasifica como challenge."""
        assert _classify("Reto de 24 horas sin dormir") == "challenge"

    def test_challenge_vs(self):
        """Título con 'vs ' se clasifica como challenge."""
        assert _classify("Comida de $1 vs $1000") == "challenge"

    def test_listicle_detected(self):
        """Título con 'top ' o 'mejores' se clasifica como listicle."""
        assert _classify("Top 10 mejores películas del año") == "listicle"

    def test_listicle_peores(self):
        """Título con 'peores' se clasifica como listicle."""
        assert _classify("Los peores errores de programación") == "listicle"

    def test_review_detected(self):
        """Título con 'review' o 'reseña' se clasifica como review."""
        assert _classify("Review del nuevo MacBook Pro") == "review"

    def test_review_unboxing(self):
        """Título con 'unboxing' se clasifica como review."""
        assert _classify("Unboxing de la PS6 edición limitada") == "review"

    def test_tutorial_detected(self):
        """Título con 'tutorial' se clasifica como tutorial."""
        assert _classify("Tutorial de Docker para principiantes") == "tutorial"

    def test_tutorial_curso(self):
        """Título con 'curso' se clasifica como tutorial."""
        assert _classify("Curso completo de Python") == "tutorial"

    def test_howto_detected(self):
        """Título con 'cómo' o 'how to' se clasifica como how-to."""
        assert _classify("Cómo configurar tu servidor Linux") == "how-to"

    def test_howto_tips(self):
        """Título con 'tips' se clasifica como how-to."""
        assert _classify("Tips para mejorar tu productividad") == "how-to"

    def test_news_detected(self):
        """Título con 'noticia' o 'update' se clasifica como news."""
        assert _classify("Noticias de tecnología esta semana") == "news"

    def test_vlog_detected(self):
        """Título con 'vlog' o 'día en' se clasifica como vlog."""
        assert _classify("Vlog: un día en mi vida como dev") == "vlog"

    def test_vlog_rutina(self):
        """Título con 'rutina' se clasifica como vlog."""
        assert _classify("Mi rutina de mañana productiva") == "vlog"

    def test_entertainment_detected(self):
        """Título con 'divertido' o 'prank' se clasifica como entertainment."""
        assert _classify("Broma épica a mi compañero de trabajo") == "entertainment"

    def test_other_for_unknown(self):
        """Título sin keywords conocidos se clasifica como other."""
        assert _classify("Un video sobre temas varios") == "other"

    def test_empty_string_returns_other(self):
        """String vacío se clasifica como other."""
        assert _classify("") == "other"
        assert _classify("", "") == "other"


# ===========================================================================
# Orden de prioridad
# ===========================================================================

class TestPriorityOrder:
    """Verifica que la prioridad (reaction > challenge > ... > other) se respeta."""

    def test_reaction_over_challenge(self):
        """'reaction' tiene prioridad sobre 'challenge'."""
        result = _classify("Reacción al reto de comida challenge")
        assert result == "reaction"

    def test_challenge_over_listicle(self):
        """'challenge' tiene prioridad sobre 'listicle'."""
        result = _classify("Top 5 mejores challenges de YouTube", "challenge,reto")
        assert result == "challenge"

    def test_listicle_over_review(self):
        """'listicle' tiene prioridad sobre 'review'."""
        result = _classify("Top 10 mejores reviews de celulares")
        assert result == "listicle"

    def test_review_over_tutorial(self):
        """'review' tiene prioridad sobre 'tutorial'."""
        result = _classify("Review y tutorial del nuevo iPhone")
        assert result == "review"

    def test_tutorial_over_howto(self):
        """'tutorial' tiene prioridad sobre 'how-to'."""
        result = _classify("Tutorial: cómo hacer un sitio web")
        assert result == "tutorial"

    def test_howto_over_news(self):
        """'how-to' tiene prioridad sobre 'news'."""
        result = _classify("Cómo hacer que las noticias no te abrumen")
        assert result == "how-to"

    def test_tags_also_considered(self):
        """Los tags contribuyen a la clasificación."""
        result = _classify("Video interesante", "tutorial,python,curso")
        assert result == "tutorial"
