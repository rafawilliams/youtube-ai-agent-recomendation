"""
Detector automático de series de videos (Mejora 17.1).

Agrupa videos en series usando:
1. Patrones de numeración en títulos (regex: #N, Parte N, Ep N, etc.)
2. Similitud TF-IDF + coseno para títulos sin numeración explícita

Umbral de similitud: 0.65 (más bajo que canibalización porque series
comparten temática pero pueden tener títulos ligeramente distintos).
"""
import re
import logging
from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

# Patrones regex para detectar numeración de episodios
_EPISODE_PATTERNS = [
    r'#\s*(\d+)',                          # #1, # 12
    r'[Pp]arte?\s*(\d+)',                  # Parte 1, Part 2
    r'[Ee]p(?:isodio)?\s*\.?\s*(\d+)',     # Ep 1, Episodio 3, Ep. 5
    r'[Cc]ap(?:[íi]tulo)?\s*\.?\s*(\d+)',  # Cap 1, Capítulo 3
    r'[Vv]ol(?:umen)?\s*\.?\s*(\d+)',      # Vol 1, Volumen 2
    r'[-–—]\s*(\d+)\s*(?:de\s+\d+)?$',    # - 1, — 3 de 5 (at end)
    r'\((\d+)\s*(?:/|de)\s*\d+\)',         # (1/5), (2 de 10)
    r'\b(\d+)\s*(?:/|de)\s*\d+\b',        # 1/5, 2 de 10
]

SIMILARITY_THRESHOLD = 0.65
MIN_SERIES_SIZE = 2


class SeriesDetector:
    """Detecta y agrupa videos en series automáticamente."""

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        min_series_size: int = MIN_SERIES_SIZE,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_series_size = min_series_size

    def detect(self, videos_df: pd.DataFrame) -> list[dict]:
        """
        Detecta series en un DataFrame de videos.

        Args:
            videos_df: DataFrame con columnas title, video_id, tags,
                       published_at, channel_id

        Returns:
            Lista de series detectadas, cada una con:
            - series_name: str
            - detected_pattern: str (regex o 'tfidf')
            - channel_id: str
            - episodes: list[dict] con video_id, title, episode_number, published_at
        """
        if videos_df.empty or len(videos_df) < self.min_series_size:
            return []

        all_series: list[dict] = []

        for channel_id, channel_df in videos_df.groupby('channel_id'):
            if len(channel_df) < self.min_series_size:
                continue

            # Fase 1: patrones de numeración (alta confianza)
            numbered_series, matched_ids = self._detect_by_numbering(channel_df)
            all_series.extend(numbered_series)

            # Fase 2: similitud TF-IDF (los que no fueron agrupados en fase 1)
            remaining = channel_df[~channel_df['video_id'].isin(matched_ids)]
            if len(remaining) >= self.min_series_size:
                tfidf_series = self._detect_by_similarity(remaining)
                all_series.extend(tfidf_series)

        log.info("Series detectadas: %d", len(all_series))
        return all_series

    def _detect_by_numbering(
        self, df: pd.DataFrame
    ) -> tuple[list[dict], set[str]]:
        """
        Agrupa videos que comparten un título base + patrón numérico.
        Retorna (series_list, set de video_ids agrupados).
        """
        groups: dict[str, list[dict]] = defaultdict(list)
        matched_ids: set[str] = set()

        for _, row in df.iterrows():
            title = str(row.get('title', ''))
            for pattern in _EPISODE_PATTERNS:
                match = re.search(pattern, title)
                if match:
                    ep_num = int(match.group(1))
                    base = re.sub(pattern, '', title).strip()
                    base = re.sub(r'\s+', ' ', base).strip(' -–—:|')
                    base_key = base.lower()

                    if base_key and len(base_key) > 3:
                        groups[base_key].append({
                            'video_id': row['video_id'],
                            'title': title,
                            'episode_number': ep_num,
                            'published_at': str(row.get('published_at', '')),
                            'channel_id': row['channel_id'],
                            'pattern': pattern,
                        })
                        matched_ids.add(row['video_id'])
                    break

        series_list = []
        for base_key, episodes in groups.items():
            if len(episodes) < self.min_series_size:
                for ep in episodes:
                    matched_ids.discard(ep['video_id'])
                continue

            episodes.sort(key=lambda e: e['episode_number'])

            # Nombre legible: usar el título del primer episodio como base
            series_name = re.sub(
                _EPISODE_PATTERNS[0], '', episodes[0]['title']
            ).strip(' -–—:|#')
            for pattern in _EPISODE_PATTERNS[1:]:
                series_name = re.sub(pattern, '', series_name).strip(' -–—:|')
            series_name = re.sub(r'\s+', ' ', series_name).strip()

            if not series_name:
                series_name = base_key.title()

            series_list.append({
                'series_name': series_name,
                'detected_pattern': 'numbering',
                'channel_id': episodes[0]['channel_id'],
                'episodes': [
                    {
                        'video_id': e['video_id'],
                        'title': e['title'],
                        'episode_number': e['episode_number'],
                        'published_at': e['published_at'],
                    }
                    for e in episodes
                ],
            })

        return series_list, matched_ids

    def _detect_by_similarity(self, df: pd.DataFrame) -> list[dict]:
        """
        Agrupa videos por similitud de títulos usando TF-IDF + coseno.
        Solo para videos que NO fueron agrupados por numeración.
        """
        corpus = self._build_corpus(df)
        if len(corpus) < self.min_series_size:
            return []

        vectorizer = TfidfVectorizer(max_features=5000)
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            log.debug("TF-IDF: corpus vacío")
            return []

        sim_matrix = cosine_similarity(tfidf_matrix)
        n = len(df)

        # Union-Find para agrupar videos similares
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if float(sim_matrix[i, j]) >= self.similarity_threshold:
                    union(i, j)

        clusters: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)

        series_list = []
        for members in clusters.values():
            if len(members) < self.min_series_size:
                continue

            member_rows = [df.iloc[m] for m in members]
            member_rows.sort(
                key=lambda r: str(r.get('published_at', ''))
            )

            titles = [str(r.get('title', '')) for r in member_rows]
            series_name = self._extract_common_name(titles)

            episodes = []
            for ep_num, row in enumerate(member_rows, 1):
                episodes.append({
                    'video_id': row['video_id'],
                    'title': str(row.get('title', '')),
                    'episode_number': ep_num,
                    'published_at': str(row.get('published_at', '')),
                })

            series_list.append({
                'series_name': series_name,
                'detected_pattern': 'tfidf',
                'channel_id': str(member_rows[0]['channel_id']),
                'episodes': episodes,
            })

        return series_list

    @staticmethod
    def _build_corpus(df: pd.DataFrame) -> list[str]:
        """Concatena título + tags para TF-IDF."""
        corpus = []
        for _, row in df.iterrows():
            title = str(row.get('title', '')) if pd.notna(row.get('title')) else ''
            tags = str(row.get('tags', '')) if pd.notna(row.get('tags')) else ''
            tags_clean = tags.replace(',', ' ')
            corpus.append(f"{title} {tags_clean}".strip().lower())
        return corpus

    @staticmethod
    def _extract_common_name(titles: list[str]) -> str:
        """Extrae el nombre común más largo entre los títulos."""
        if not titles:
            return 'Serie sin nombre'

        word_counts: dict[str, int] = defaultdict(int)
        for t in titles:
            words = set(t.lower().split())
            for w in words:
                if len(w) > 2:
                    word_counts[w] += 1

        threshold = len(titles) * 0.5
        common = [w for w, c in word_counts.items() if c >= threshold]

        if not common:
            return min(titles, key=len)[:80]

        first_words = titles[0].lower().split()
        name_parts = [w for w in first_words if w.lower() in common]
        name = ' '.join(name_parts).strip().title()
        return name[:120] if name else titles[0][:80]
