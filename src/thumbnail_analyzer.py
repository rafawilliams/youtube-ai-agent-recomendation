"""
Análisis de miniaturas de YouTube con Claude Vision (Mejora 8.x).

- ThumbnailScorer (8.1): analiza miniaturas existentes usando Claude Vision
  multimodal para evaluar claridad de texto, expresión facial, contraste,
  composición y clickability (1-10).
- ThumbnailComparator (8.2): compara 2-3 opciones de miniaturas para
  predecir cuál tiene mayor probabilidad de CTR.
"""
import json
import logging
import time
from typing import Dict, List, Optional

from anthropic import Anthropic
from retry_config import retry_anthropic

log = logging.getLogger(__name__)

THUMBNAIL_URL_TEMPLATE = "https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"


# ══════════════════════════════════════════════════════════════════════
# 8.1 — Thumbnail Scorer
# ══════════════════════════════════════════════════════════════════════

class ThumbnailScorer:
    """Evalúa miniaturas de YouTube usando Claude Vision."""

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)

    @staticmethod
    def get_thumbnail_url(video_id: str) -> str:
        """Construye la URL de la miniatura en máxima resolución."""
        return THUMBNAIL_URL_TEMPLATE.format(video_id=video_id)

    @retry_anthropic
    def _call_vision(self, thumbnail_url: str) -> str:
        """Envía la miniatura a Claude Vision y retorna el texto de respuesta."""
        message = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": thumbnail_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Analiza esta miniatura de YouTube. "
                            "Responde SOLO con un JSON válido (sin markdown, sin ```), "
                            "con esta estructura exacta:\n"
                            "{\n"
                            '  "text_clarity": <1-10>,\n'
                            '  "facial_expression": <1-10 si hay rostro, null si no>,\n'
                            '  "color_contrast": <1-10>,\n'
                            '  "composition": <1-10>,\n'
                            '  "clickability": <1-10>,\n'
                            '  "has_text": true/false,\n'
                            '  "has_face": true/false,\n'
                            '  "dominant_colors": ["color1", "color2", "color3"],\n'
                            '  "improvement_suggestions": ["sugerencia1", "sugerencia2"]\n'
                            "}\n\n"
                            "Criterios:\n"
                            "- text_clarity: legibilidad del texto en pantalla móvil\n"
                            "- facial_expression: si hay rostro, evalúa emoción/engagement\n"
                            "- color_contrast: contraste y saturación visual\n"
                            "- composition: regla de tercios, balance visual\n"
                            "- clickability: puntuación general de atractivo para clic\n"
                            "- dominant_colors: 3 colores dominantes en español\n"
                            "- improvement_suggestions: 2-3 sugerencias concretas en español"
                        ),
                    },
                ],
            }],
        )
        return message.content[0].text

    def analyze(self, video_id: str) -> Optional[Dict]:
        """
        Analiza la miniatura de un video.

        Returns:
            Dict con scores y sugerencias, o None si falla.
        """
        url = self.get_thumbnail_url(video_id)
        try:
            raw = self._call_vision(url)
            # Limpiar posibles bloques markdown
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            result = json.loads(cleaned)
            result['video_id'] = video_id
            return result
        except json.JSONDecodeError:
            log.warning("Respuesta no-JSON para %s: %s", video_id, raw[:200])
            return None
        except Exception as e:
            log.error("Error analizando thumbnail de %s: %s", video_id, e)
            return None

    def analyze_channel(self, channel_id: str, db, max_per_run: int = 50) -> List[Dict]:
        """
        Analiza thumbnails pendientes de un canal.

        Args:
            channel_id: ID del canal
            db: instancia de YouTubeDatabase
            max_per_run: máximo de thumbnails a analizar por ejecución

        Returns:
            Lista de resultados de análisis
        """
        pending = db.get_videos_without_thumbnail_score(channel_id)
        if pending.empty:
            log.info("  Sin thumbnails pendientes para %s", channel_id)
            return []

        pending = pending.head(max_per_run)
        results = []

        for i, row in pending.iterrows():
            video_id = row['video_id']
            result = self.analyze(video_id)
            if result:
                result['channel_id'] = channel_id
                db.save_thumbnail_score(result)
                results.append(result)
                log.debug("  🖼 %s: clickability=%.1f",
                          row.get('title', video_id)[:40],
                          result.get('clickability', 0))

            # Rate limiting: pausa cada 5 análisis
            if (len(results) % 5 == 0) and len(results) > 0:
                time.sleep(2)

        return results


# ══════════════════════════════════════════════════════════════════════
# 8.2 — Thumbnail Comparator (A/B)
# ══════════════════════════════════════════════════════════════════════

class ThumbnailComparator:
    """Compara 2-3 miniaturas para predecir cuál tendrá mayor CTR."""

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)

    @retry_anthropic
    def _call_vision_compare(self, image_contents: list) -> str:
        """Envía múltiples imágenes a Claude Vision para comparación."""
        content_blocks = []
        for idx, img in enumerate(image_contents):
            label = chr(65 + idx)  # A, B, C
            content_blocks.append({
                "type": "text",
                "text": f"--- Miniatura {label} ---",
            })
            content_blocks.append({
                "type": "image",
                "source": img["source"],
            })

        content_blocks.append({
            "type": "text",
            "text": (
                "\nCompara estas miniaturas de YouTube para CTR. "
                "Responde SOLO con JSON válido (sin markdown, sin ```):\n"
                "{\n"
                '  "winner": "A" o "B" o "C",\n'
                '  "scores": {\n'
                '    "A": {"clickability": <1-10>, "mobile_legibility": <1-10>, '
                '"visual_hierarchy": <1-10>, "engagement_potential": <1-10>},\n'
                '    "B": {"clickability": <1-10>, ...}\n'
                "  },\n"
                '  "reasoning": "explicación breve en español de por qué la ganadora es mejor",\n'
                '  "suggestions": ["sugerencia para mejorar la miniatura perdedora"]\n'
                "}"
            ),
        })

        message = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": content_blocks}],
        )
        return message.content[0].text

    def compare(self, images: List[Dict]) -> Optional[Dict]:
        """
        Compara 2-3 imágenes de miniaturas.

        Args:
            images: lista de dicts con 'source' key.
                    Cada source puede ser:
                    - {"type": "url", "url": "https://..."}
                    - {"type": "base64", "media_type": "image/png", "data": "..."}

        Returns:
            Dict con winner, scores, reasoning, suggestions. None si falla.
        """
        if len(images) < 2 or len(images) > 3:
            log.error("Se requieren 2 o 3 imágenes para comparar")
            return None

        try:
            raw = self._call_vision_compare(images)
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning("Respuesta no-JSON en comparación: %s", raw[:200])
            return None
        except Exception as e:
            log.error("Error comparando thumbnails: %s", e)
            return None
