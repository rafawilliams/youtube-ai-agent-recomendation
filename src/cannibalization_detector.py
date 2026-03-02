"""
Detector de canibalización de contenido (Mejora 12.4).

Usa TF-IDF + cosine similarity de scikit-learn para detectar pares de videos
con títulos/tags muy similares publicados en un período cercano, lo cual puede
indicar que están compitiendo por la misma audiencia.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)


class CannibalizationDetector:
    """Detecta pares de videos con alta similitud semántica que pueden
    estar canibalizando audiencia entre sí."""

    SIMILARITY_THRESHOLD = 0.80
    MAX_DAYS_APART = 30

    def __init__(self, threshold: float = 0.80, max_days_apart: int = 30):
        self.threshold = threshold
        self.max_days_apart = max_days_apart

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------

    @staticmethod
    def _build_text_corpus(videos_df: pd.DataFrame) -> list[str]:
        """Concatena título + tags de cada video en un texto para TF-IDF."""
        corpus: list[str] = []
        for _, row in videos_df.iterrows():
            title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
            tags = str(row.get('tags', '')) if pd.notna(row.get('tags')) else ''
            tags_clean = tags.replace(',', ' ')
            corpus.append(f"{title} {tags_clean}".strip().lower())
        return corpus

    # ------------------------------------------------------------------
    # Detección
    # ------------------------------------------------------------------

    def detect(self, videos_df: pd.DataFrame) -> list[dict]:
        """
        Detecta pares de videos con similitud ≥ threshold publicados
        dentro de max_days_apart días entre sí.

        Args:
            videos_df: DataFrame con columnas video_id, title, tags, published_at

        Returns:
            Lista de dicts ordenados por similitud descendente.
        """
        if len(videos_df) < 2:
            return []

        df = videos_df.reset_index(drop=True)
        corpus = self._build_text_corpus(df)

        # Filtrar documentos vacíos
        if all(not doc.strip() for doc in corpus):
            return []

        # TF-IDF
        vectorizer = TfidfVectorizer(max_features=5000)
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            log.debug("TF-IDF: corpus vacío, no se puede vectorizar")
            return []

        # Cosine similarity (matriz completa)
        sim_matrix = cosine_similarity(tfidf_matrix)

        # Parsear fechas
        dates = pd.to_datetime(df['published_at'], utc=True, errors='coerce')

        results: list[dict] = []

        # Iterar triángulo superior
        n = len(df)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim < self.threshold:
                    continue

                # Verificar cercanía temporal
                if pd.isna(dates.iloc[i]) or pd.isna(dates.iloc[j]):
                    continue
                days_apart = abs((dates.iloc[i] - dates.iloc[j]).days)
                if days_apart > self.max_days_apart:
                    continue

                shared = self._get_shared_terms(vectorizer, tfidf_matrix, i, j)

                results.append({
                    'video_id_a': df.iloc[i]['video_id'],
                    'title_a': df.iloc[i]['title'],
                    'published_a': str(dates.iloc[i].date()),
                    'video_id_b': df.iloc[j]['video_id'],
                    'title_b': df.iloc[j]['title'],
                    'published_b': str(dates.iloc[j].date()),
                    'similarity': round(sim, 3),
                    'days_apart': days_apart,
                    'shared_terms': shared,
                })

        results.sort(key=lambda r: r['similarity'], reverse=True)
        return results

    @staticmethod
    def _get_shared_terms(vectorizer, tfidf_matrix, idx_a: int, idx_b: int,
                          top_n: int = 5) -> list[str]:
        """Retorna los top N términos TF-IDF compartidos entre dos documentos."""
        feature_names = vectorizer.get_feature_names_out()
        vec_a = np.asarray(tfidf_matrix[idx_a].todense()).flatten()
        vec_b = np.asarray(tfidf_matrix[idx_b].todense()).flatten()

        # Peso combinado solo donde ambos tienen valor > 0
        mask = (vec_a > 0) & (vec_b > 0)
        if not mask.any():
            return []

        combined = (vec_a + vec_b) * mask
        top_indices = combined.argsort()[::-1][:top_n]
        return [feature_names[i] for i in top_indices if combined[i] > 0]

    # ------------------------------------------------------------------
    # Alertas
    # ------------------------------------------------------------------

    @staticmethod
    def get_recent_alerts(results: list[dict], days_lookback: int = 60) -> list[dict]:
        """Filtra pares donde al menos un video fue publicado recientemente."""
        cutoff = datetime.now().date()
        alerts: list[dict] = []
        for r in results:
            try:
                date_a = datetime.strptime(r['published_a'], '%Y-%m-%d').date()
                date_b = datetime.strptime(r['published_b'], '%Y-%m-%d').date()
            except (ValueError, TypeError):
                continue
            diff_a = (cutoff - date_a).days
            diff_b = (cutoff - date_b).days
            if diff_a <= days_lookback or diff_b <= days_lookback:
                alerts.append(r)
        return alerts
