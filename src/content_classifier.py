"""
Clasificador automático de categoría de contenido (Mejora 12.2).

Usa Claude AI para clasificar videos en categorías internas:
tutorial, review, vlog, entertainment, news, how-to, listicle, reaction, challenge, other.

Permite análisis de performance por categoría y recomendaciones más específicas.
"""
import json
import logging
import re
from typing import Optional

from anthropic import Anthropic
from retry_config import retry_anthropic

log = logging.getLogger(__name__)


class ContentClassifier:
    """Clasifica videos en categorías de contenido usando Claude AI."""

    CATEGORIES = [
        'tutorial', 'review', 'vlog', 'entertainment', 'news',
        'how-to', 'listicle', 'reaction', 'challenge', 'other',
    ]

    CATEGORY_LABELS_ES = {
        'tutorial': 'Tutorial',
        'review': 'Review/Reseña',
        'vlog': 'Vlog',
        'entertainment': 'Entretenimiento',
        'news': 'Noticias',
        'how-to': 'How-To/Guía',
        'listicle': 'Listicle/Top',
        'reaction': 'Reacción',
        'challenge': 'Challenge/Reto',
        'other': 'Otro',
    }

    CATEGORY_ICONS = {
        'tutorial': '📚',
        'review': '⭐',
        'vlog': '🎥',
        'entertainment': '🎭',
        'news': '📰',
        'how-to': '🔧',
        'listicle': '📋',
        'reaction': '😱',
        'challenge': '🏆',
        'other': '📦',
    }

    def __init__(self, anthropic_api_key: str):
        self._client = Anthropic(api_key=anthropic_api_key)

    # ------------------------------------------------------------------
    # Clasificación con Claude
    # ------------------------------------------------------------------

    def classify_batch(self, videos: list[dict]) -> dict[str, str]:
        """
        Clasifica un lote de hasta 25 videos usando Claude AI.

        Args:
            videos: Lista de dicts con {video_id, title, tags, description (opcional)}

        Returns:
            dict {video_id: category_str}
        """
        if not videos:
            return {}

        batch = videos[:25]
        prompt = self._build_classify_prompt(batch)

        try:
            response = self._call_claude(prompt, max_tokens=1500)
        except Exception as e:
            log.warning("ContentClassifier: error Claude — %s. Usando fallback.", e)
            return {
                v['video_id']: self.classify_single_by_keywords(
                    v.get('title', ''), v.get('tags', '')
                )
                for v in batch
            }

        return self._parse_classify_response(response, batch)

    def _build_classify_prompt(self, videos: list[dict]) -> str:
        """Construye el prompt para clasificación en lote."""
        categories_str = ', '.join(self.CATEGORIES)

        lines = []
        for i, v in enumerate(videos, 1):
            title = v.get('title', '(sin título)')
            tags = v.get('tags', '')[:100]
            desc = v.get('description', '')[:150]
            lines.append(f"{i}. [{v['video_id']}] Title: \"{title}\" | Tags: \"{tags}\" | Desc: \"{desc}\"")

        videos_text = '\n'.join(lines)

        return f"""Classify each video into EXACTLY ONE of these categories: {categories_str}

Rules:
- "tutorial" = teaches step-by-step, educational content
- "review" = evaluates/reviews a product, service, or media
- "vlog" = personal diary, daily life, casual talking
- "entertainment" = comedy, skits, pranks, fun content
- "news" = current events, updates, announcements
- "how-to" = practical guide, tips, "cómo hacer", DIY
- "listicle" = "top N", "X mejores", ranked lists
- "reaction" = reacting to other content, "reacción a"
- "challenge" = challenges, "reto", competitive format
- "other" = doesn't clearly fit any category

Videos:
{videos_text}

Respond ONLY with a valid JSON object mapping video_id to category:
{{"video_id1": "category", "video_id2": "category", ...}}"""

    @retry_anthropic
    def _call_claude(self, prompt: str, max_tokens: int = 1500) -> str:
        """Llamada a Claude con retry automático."""
        message = self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _parse_classify_response(self, response: str, videos: list[dict]) -> dict[str, str]:
        """Parsea la respuesta JSON de Claude. Fallback a keywords si falla."""
        result: dict[str, str] = {}

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            log.warning("ContentClassifier: no se pudo parsear JSON, usando fallback keywords")
            return {
                v['video_id']: self.classify_single_by_keywords(
                    v.get('title', ''), v.get('tags', '')
                )
                for v in videos
            }

        # Validar que las categorías sean válidas
        valid = set(self.CATEGORIES)
        for v in videos:
            vid = v['video_id']
            category = data.get(vid, '').lower().strip()
            if category in valid:
                result[vid] = category
            else:
                result[vid] = self.classify_single_by_keywords(
                    v.get('title', ''), v.get('tags', '')
                )

        return result

    # ------------------------------------------------------------------
    # Clasificación por keywords (fallback sin API)
    # ------------------------------------------------------------------

    @staticmethod
    def classify_single_by_keywords(title: str, tags: str) -> str:
        """
        Clasificación rápida por keywords. No requiere API.
        Útil como fallback o para clasificar muchos videos sin costo.
        """
        text = f"{title} {tags}".lower()

        # Orden de prioridad (de más específico a más genérico)
        rules = [
            ('reaction', ['reaccion', 'reacción', 'react', 'reacting']),
            ('challenge', ['challenge', 'reto', 'desafío', 'desafio', 'vs ']),
            ('listicle', ['top ', 'mejores', 'peores', 'ranking', ' best ', ' worst ']),
            ('review', ['review', 'reseña', 'resena', 'unboxing', 'análisis de']),
            ('tutorial', ['tutorial', 'curso', 'aprende', 'clase ', 'lección']),
            ('how-to', ['cómo', 'como hacer', 'how to', 'paso a paso', 'guía', 'guia', 'tips', 'trucos']),
            ('news', ['noticia', 'actualización', 'update', 'breaking', 'último momento']),
            ('vlog', ['vlog', 'día en', 'dia en', 'daily', 'mi vida', 'rutina']),
            ('entertainment', ['divertido', 'funny', 'comedia', 'prank', 'broma', 'sketch']),
        ]

        for category, keywords in rules:
            if any(kw in text for kw in keywords):
                return category

        return 'other'

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @classmethod
    def get_category_label(cls, category: str) -> str:
        """Retorna el label en español de una categoría."""
        return cls.CATEGORY_LABELS_ES.get(category, category.title())

    @classmethod
    def get_category_icon(cls, category: str) -> str:
        """Retorna el icono de una categoría."""
        return cls.CATEGORY_ICONS.get(category, '📦')
