"""
Analizador de comentarios y sentimiento de audiencia (Mejora 11.x).

- Clasificación de sentimiento con Claude (positivo/negativo/neutro)
- Ranking de top commenters
- Detección de FAQ/temas recurrentes con Claude
"""
import json
import logging
import os
import re
from datetime import datetime

from anthropic import Anthropic
from retry_config import retry_anthropic

log = logging.getLogger(__name__)

ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-6')


class CommentAnalyzer:
    """Analiza comentarios de YouTube: sentimiento, top fans, FAQ."""

    BATCH_SIZE = 25

    def __init__(self, anthropic_api_key: str):
        self._client = Anthropic(api_key=anthropic_api_key)

    # ------------------------------------------------------------------
    # 11.2 — Clasificación de sentimiento
    # ------------------------------------------------------------------

    def classify_sentiment_batch(self, comments: list[dict]) -> list[dict]:
        """
        Clasifica sentimiento de un lote de comentarios usando Claude.

        Args:
            comments: Lista de dicts con al menos {comment_id, text}

        Returns:
            Lista de dicts {comment_id, sentiment} donde sentiment es
            'positivo', 'negativo', o 'neutro'.
        """
        results = []
        for i in range(0, len(comments), self.BATCH_SIZE):
            batch = comments[i:i + self.BATCH_SIZE]
            batch_results = self._classify_batch(batch)
            results.extend(batch_results)
        return results

    def _classify_batch(self, batch: list[dict]) -> list[dict]:
        """Clasifica un batch de hasta 25 comentarios."""
        if not batch:
            return []

        lines = []
        for idx, c in enumerate(batch, 1):
            text = c.get('text', '')[:300]  # Truncar textos largos
            lines.append(f'{idx}. [{c["comment_id"]}] "{text}"')

        prompt = f"""Clasifica el sentimiento de cada comentario de YouTube como 'positivo', 'negativo' o 'neutro'.

Reglas:
- 'positivo': elogios, agradecimiento, entusiasmo, apoyo, emojis positivos
- 'negativo': quejas, críticas, insultos, decepción, frustración
- 'neutro': preguntas objetivas, observaciones sin carga emocional, timestamps

Comentarios:
{chr(10).join(lines)}

Responde SOLO con un JSON válido mapeando comment_id a sentimiento:
{{"comment_id1": "positivo", "comment_id2": "negativo", ...}}"""

        try:
            response = self._call_claude(prompt, max_tokens=1500)
            return self._parse_sentiment_response(response, batch)
        except Exception as e:
            log.warning("CommentAnalyzer: error Claude sentimiento — %s. Usando fallback.", e)
            return [{'comment_id': c['comment_id'], 'sentiment': 'neutro'} for c in batch]

    def _parse_sentiment_response(self, response: str, batch: list[dict]) -> list[dict]:
        """Parsea la respuesta JSON de Claude para sentimiento."""
        valid_sentiments = {'positivo', 'negativo', 'neutro'}
        results = []

        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            log.warning("CommentAnalyzer: no se pudo parsear JSON de sentimiento")
            return [{'comment_id': c['comment_id'], 'sentiment': 'neutro'} for c in batch]

        for c in batch:
            cid = c['comment_id']
            sentiment = data.get(cid, 'neutro').lower().strip()
            if sentiment not in valid_sentiments:
                sentiment = 'neutro'
            results.append({'comment_id': cid, 'sentiment': sentiment})

        return results

    # ------------------------------------------------------------------
    # 11.1 — Top commenters
    # ------------------------------------------------------------------

    @staticmethod
    def rank_top_commenters(comments: list[dict], top_n: int = 20) -> list[dict]:
        """
        Rankea los comentaristas más activos.

        Args:
            comments: Lista de dicts con author_display_name, like_count, sentiment

        Returns:
            Lista de dicts con author, comment_count, total_likes, avg_sentiment
        """
        from collections import defaultdict, Counter

        author_data: dict[str, dict] = defaultdict(lambda: {
            'count': 0, 'total_likes': 0, 'sentiments': Counter()
        })

        for c in comments:
            author = c.get('author_display_name', '').strip()
            if not author:
                continue
            author_data[author]['count'] += 1
            author_data[author]['total_likes'] += c.get('like_count', 0)
            sentiment = c.get('sentiment')
            if sentiment:
                author_data[author]['sentiments'][sentiment] += 1

        ranked = sorted(
            author_data.items(),
            key=lambda x: x[1]['count'],
            reverse=True,
        )[:top_n]

        result = []
        for author, data in ranked:
            sentiments = data['sentiments']
            total_s = sum(sentiments.values())
            dominant = sentiments.most_common(1)[0][0] if total_s > 0 else 'neutro'
            result.append({
                'author': author,
                'comment_count': data['count'],
                'total_likes': data['total_likes'],
                'dominant_sentiment': dominant,
            })
        return result

    # ------------------------------------------------------------------
    # 11.3 — Detección de FAQ / temas recurrentes
    # ------------------------------------------------------------------

    def extract_faq_topics(self, comments: list[dict], channel_id: str) -> list[dict]:
        """
        Usa Claude para identificar preguntas frecuentes en los comentarios.

        Args:
            comments: Lista de dicts con text
            channel_id: ID del canal

        Returns:
            Lista de dicts con topic, sample_questions, frequency, extracted_at
        """
        if len(comments) < 5:
            return []

        # Filtrar solo comentarios con signos de pregunta o palabras interrogativas
        question_comments = [
            c for c in comments
            if '?' in c.get('text', '') or
            any(w in c.get('text', '').lower() for w in
                ['cómo', 'como', 'qué', 'que es', 'por qué', 'cuándo', 'donde', 'cuál',
                 'how', 'what', 'why', 'when', 'where', 'which'])
        ]

        if len(question_comments) < 3:
            return []

        # Tomar máximo 100 preguntas para el prompt
        sample = question_comments[:100]
        lines = [f'{i+1}. "{c["text"][:200]}"' for i, c in enumerate(sample)]

        prompt = f"""Analiza estos comentarios/preguntas de un canal de YouTube e identifica los temas o preguntas más frecuentes.

Comentarios:
{chr(10).join(lines)}

Agrupa las preguntas similares en temas. Para cada tema, indica:
- El tema general
- 2-3 preguntas de ejemplo
- Cuántas preguntas están relacionadas con ese tema

Responde SOLO con un JSON array válido:
[
  {{"topic": "Tema 1", "sample_questions": "Pregunta 1; Pregunta 2", "frequency": 5}},
  {{"topic": "Tema 2", "sample_questions": "Pregunta 3; Pregunta 4", "frequency": 3}}
]

Máximo 10 temas, ordenados por frecuencia descendente."""

        try:
            response = self._call_claude(prompt, max_tokens=2000)
            return self._parse_faq_response(response, channel_id)
        except Exception as e:
            log.warning("CommentAnalyzer: error en FAQ detection — %s", e)
            return []

    def _parse_faq_response(self, response: str, channel_id: str) -> list[dict]:
        """Parsea la respuesta JSON de Claude para FAQ topics."""
        now = datetime.now().isoformat()

        try:
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
        except (json.JSONDecodeError, ValueError):
            log.warning("CommentAnalyzer: no se pudo parsear FAQ JSON")
            return []

        results = []
        for item in data[:10]:
            if not isinstance(item, dict) or 'topic' not in item:
                continue
            results.append({
                'topic': str(item['topic'])[:500],
                'sample_questions': str(item.get('sample_questions', ''))[:2000],
                'frequency': int(item.get('frequency', 1)),
                'extracted_at': now,
            })

        return results

    # ------------------------------------------------------------------
    # Claude API
    # ------------------------------------------------------------------

    @retry_anthropic
    def _call_claude(self, prompt: str, max_tokens: int = 1500) -> str:
        """Llamada a Claude con retry automático."""
        message = self._client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
