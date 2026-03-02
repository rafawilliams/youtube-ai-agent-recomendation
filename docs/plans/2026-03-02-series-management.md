# Gestión de Series y Formatos (17.1 + 17.2) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Auto-detect video series via NLP title similarity, track episode metrics, and recommend format adjustments based on audience retention trends across episodes.

**Architecture:** New `src/series_detector.py` module uses TF-IDF + cosine similarity (same stack as cannibalization_detector) to cluster videos into series by title patterns. Two new DB tables (`video_series`, `series_episodes`) store detected series. `ai_analyzer.py` gets a new method for format recommendations. Dashboard gets a new "Series y Formatos" page. Pipeline runs detection after content classification.

**Tech Stack:** scikit-learn (TF-IDF, cosine_similarity), re (regex for numbering patterns), pandas, Claude AI for format recommendations, Streamlit + Plotly for dashboard.

---

### Task 1: Database — Create series tables

**Files:**
- Modify: `src/database.py` — `_create_tables()` method + new query methods

**Step 1: Add `video_series` and `series_episodes` tables in `_create_tables()`**

Add after the `competitor_alerts` table creation (before the indexes section):

```python
cursor.execute("""
    CREATE TABLE IF NOT EXISTS video_series (
        series_id INT AUTO_INCREMENT PRIMARY KEY,
        channel_id VARCHAR(255) NOT NULL,
        series_name VARCHAR(500) NOT NULL,
        detected_pattern VARCHAR(500),
        episode_count INT DEFAULT 0,
        avg_views FLOAT DEFAULT 0,
        avg_engagement FLOAT DEFAULT 0,
        trend VARCHAR(20) DEFAULT 'stable',
        ai_recommendation TEXT,
        created_at VARCHAR(50) NOT NULL,
        updated_at VARCHAR(50) NOT NULL,
        UNIQUE KEY uq_series_channel_name (channel_id, series_name),
        FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS series_episodes (
        id INT AUTO_INCREMENT PRIMARY KEY,
        series_id INT NOT NULL,
        video_id VARCHAR(255) NOT NULL,
        episode_number INT DEFAULT 0,
        detected_at VARCHAR(50) NOT NULL,
        UNIQUE KEY uq_episode_video (video_id),
        FOREIGN KEY (series_id) REFERENCES video_series(series_id),
        FOREIGN KEY (video_id) REFERENCES videos(video_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")
```

**Step 2: Add CRUD methods**

Add a new section `# ── Series (Mejora 17.x) ──` before `close()`:

```python
# ── Series (Mejora 17.x) ────────────────────────────────────────

def save_series(self, series: dict) -> int:
    """Guarda o actualiza una serie. Retorna series_id."""
    cursor = self.conn.cursor()
    now = datetime.now().isoformat()
    try:
        cursor.execute("""
            INSERT INTO video_series
            (channel_id, series_name, detected_pattern, episode_count,
             avg_views, avg_engagement, trend, ai_recommendation,
             created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                episode_count = VALUES(episode_count),
                avg_views = VALUES(avg_views),
                avg_engagement = VALUES(avg_engagement),
                trend = VALUES(trend),
                ai_recommendation = VALUES(ai_recommendation),
                updated_at = VALUES(updated_at)
        """, (
            series['channel_id'],
            series['series_name'],
            series.get('detected_pattern', ''),
            series.get('episode_count', 0),
            series.get('avg_views', 0),
            series.get('avg_engagement', 0),
            series.get('trend', 'stable'),
            series.get('ai_recommendation', ''),
            now, now,
        ))
        self.conn.commit()
        # Get the series_id (last insert or existing)
        if cursor.lastrowid:
            return cursor.lastrowid
        cursor.execute("""
            SELECT series_id FROM video_series
            WHERE channel_id = %s AND series_name = %s
        """, (series['channel_id'], series['series_name']))
        row = cursor.fetchone()
        return row['series_id'] if row else 0
    except Exception as e:
        self.conn.rollback()
        log.warning("Error guardando serie: %s", e)
        raise

def save_series_episode(self, series_id: int, video_id: str, episode_number: int):
    """Vincula un video a una serie como episodio. Ignora duplicados."""
    cursor = self.conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO series_episodes
            (series_id, video_id, episode_number, detected_at)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                series_id = VALUES(series_id),
                episode_number = VALUES(episode_number)
        """, (series_id, video_id, episode_number, datetime.now().isoformat()))
        self.conn.commit()
    except Exception as e:
        self.conn.rollback()
        if 'Duplicate' not in str(e):
            raise

def get_all_series(self, channel_id: str = None) -> pd.DataFrame:
    """Retorna todas las series, opcionalmente filtradas por canal."""
    cursor = self.conn.cursor()
    query = """
        SELECT s.*, c.channel_name
        FROM video_series s
        JOIN channels c ON s.channel_id = c.channel_id
    """
    if channel_id:
        query += " WHERE s.channel_id = %s ORDER BY s.episode_count DESC"
        cursor.execute(query, (channel_id,))
    else:
        query += " ORDER BY s.episode_count DESC"
        cursor.execute(query)
    rows = cursor.fetchall()
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def get_series_episodes(self, series_id: int) -> pd.DataFrame:
    """Retorna los episodios de una serie con métricas de video."""
    cursor = self.conn.cursor()
    cursor.execute("""
        SELECT
            se.episode_number,
            v.video_id, v.title, v.published_at, v.video_type,
            m.view_count, m.like_count, m.comment_count, m.engagement_rate
        FROM series_episodes se
        JOIN videos v ON se.video_id = v.video_id
        LEFT JOIN (
            SELECT video_id, view_count, like_count, comment_count,
                   engagement_rate,
                   ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY recorded_at DESC) AS rn
            FROM video_metrics
        ) m ON v.video_id = m.video_id AND m.rn = 1
        WHERE se.series_id = %s
        ORDER BY se.episode_number
    """, (series_id,))
    rows = cursor.fetchall()
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        df['published_at'] = pd.to_datetime(df['published_at'])
    return df
```

**Step 3: Run syntax check**

Run: `.venv/Scripts/python.exe -c "import py_compile; py_compile.compile('src/database.py', doraise=True)"`
Expected: No output (success)

**Step 4: Commit**

```bash
git add src/database.py
git commit -m "feat(17.x): add video_series and series_episodes tables + CRUD"
```

---

### Task 2: Series Detector — NLP clustering module

**Files:**
- Create: `src/series_detector.py`

**Step 1: Create the series detector module**

```python
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
    r'#\s*(\d+)',                    # #1, # 12
    r'[Pp]arte?\s*(\d+)',           # Parte 1, Part 2
    r'[Ee]p(?:isodio)?\s*\.?\s*(\d+)',  # Ep 1, Episodio 3, Ep. 5
    r'[Cc]ap(?:[íi]tulo)?\s*\.?\s*(\d+)',  # Cap 1, Capítulo 3
    r'[Vv]ol(?:umen)?\s*\.?\s*(\d+)',  # Vol 1, Volumen 2
    r'[-–—]\s*(\d+)\s*(?:de\s+\d+)?$',  # - 1, — 3 de 5 (at end)
    r'\((\d+)\s*(?:/|de)\s*\d+\)',  # (1/5), (2 de 10)
    r'\b(\d+)\s*(?:/|de)\s*\d+\b',  # 1/5, 2 de 10
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

        # Detectar por canal
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
        # Para cada video, extraer base del título (sin el número)
        groups: dict[str, list[dict]] = defaultdict(list)
        matched_ids: set[str] = set()

        for _, row in df.iterrows():
            title = str(row.get('title', ''))
            for pattern in _EPISODE_PATTERNS:
                match = re.search(pattern, title)
                if match:
                    ep_num = int(match.group(1))
                    # Base = título sin el patrón numérico, normalizado
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
                    break  # Solo usar el primer patrón que haga match

        series_list = []
        for base_key, episodes in groups.items():
            if len(episodes) < self.min_series_size:
                # Devolver los IDs al pool de no agrupados
                for ep in episodes:
                    matched_ids.discard(ep['video_id'])
                continue

            # Ordenar por número de episodio
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
        indices = list(df.index)

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

        # Agrupar por raíz
        clusters: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)

        series_list = []
        for members in clusters.values():
            if len(members) < self.min_series_size:
                continue

            # Ordenar por fecha de publicación
            member_rows = [df.iloc[m] for m in members]
            member_rows.sort(
                key=lambda r: str(r.get('published_at', ''))
            )

            # Nombre: palabras comunes entre títulos
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

        # Tokenizar y buscar palabras que aparecen en >50% de títulos
        word_counts: dict[str, int] = defaultdict(int)
        for t in titles:
            words = set(t.lower().split())
            for w in words:
                if len(w) > 2:  # Ignorar palabras muy cortas
                    word_counts[w] += 1

        threshold = len(titles) * 0.5
        common = [w for w, c in word_counts.items() if c >= threshold]

        if not common:
            # Fallback: usar el título más corto
            return min(titles, key=len)[:80]

        # Reconstruir nombre manteniendo orden del primer título
        first_words = titles[0].lower().split()
        name_parts = [w for w in first_words if w.lower() in common]
        name = ' '.join(name_parts).strip().title()
        return name[:120] if name else titles[0][:80]
```

**Step 2: Run syntax check**

Run: `.venv/Scripts/python.exe -c "import py_compile; py_compile.compile('src/series_detector.py', doraise=True)"`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/series_detector.py
git commit -m "feat(17.1): add SeriesDetector with numbering + TF-IDF clustering"
```

---

### Task 3: AI Analyzer — Format recommendation method

**Files:**
- Modify: `src/ai_analyzer.py` — add `recommend_series_format()` method

**Step 1: Add the method after `analyze_viral_competitor_video()`**

```python
def recommend_series_format(
    self,
    series_name: str,
    channel_name: str,
    episodes: list[dict],
    language: str = 'es',
) -> str:
    """
    Genera recomendación de formato para una serie (Mejora 17.2).

    Args:
        series_name: Nombre de la serie
        channel_name: Nombre del canal
        episodes: Lista de dicts con title, episode_number, view_count,
                  engagement_rate, published_at
    """
    # Preparar resumen de episodios
    ep_lines = []
    for ep in episodes:
        views = ep.get('view_count', 0) or 0
        eng = ep.get('engagement_rate', 0) or 0
        ep_lines.append(
            f"  Ep {ep.get('episode_number', '?')}: "
            f"\"{ep.get('title', '')}\" — "
            f"{views:,.0f} vistas, {eng:.2f}% engagement"
        )
    ep_text = '\n'.join(ep_lines)

    prompt = f"""Analiza esta serie de videos del canal "{channel_name}" y da recomendaciones de formato.

Serie: "{series_name}"
Episodios ({len(episodes)} total):
{ep_text}

Responde en {language}, máximo 250 palabras, con estos puntos:
1. **Tendencia de audiencia**: ¿las vistas crecen, decrecen o se mantienen entre episodios?
2. **Punto de fatiga**: ¿En qué episodio se pierde audiencia? ¿Cuántos episodios es lo óptimo?
3. **Recomendación concreta**: Una acción específica (series más cortas, cambiar formato, nuevo ángulo, etc.)
"""

    return self._call_claude(prompt, max_tokens=800)
```

**Step 2: Run syntax check**

Run: `.venv/Scripts/python.exe -c "import py_compile; py_compile.compile('src/ai_analyzer.py', doraise=True)"`
Expected: No output (success)

**Step 3: Commit**

```bash
git add src/ai_analyzer.py
git commit -m "feat(17.2): add recommend_series_format() to AIAnalyzer"
```

---

### Task 4: Pipeline — Wire series detection into main.py

**Files:**
- Modify: `main.py` — add `_step_series_detection()` and wire it

**Step 1: Add import at top (after existing src imports)**

```python
from series_detector import SeriesDetector
```

**Step 2: Add step function after `_step_competitor_alerts()`**

```python
def _step_series_detection(analyzer):
    """Detecta series automáticamente y genera recomendaciones de formato."""
    log.info("PASO SERIES: Detectando series de videos...")

    with YouTubeDatabase() as db:
        all_videos = db.get_all_videos()

    if all_videos.empty:
        log.info("  Sin videos para analizar")
        return

    detector = SeriesDetector()
    series_list = detector.detect(all_videos)

    if not series_list:
        log.info("  No se detectaron series")
        return

    log.info("  📚 %d series detectadas", len(series_list))

    for s in series_list:
        # Guardar serie en DB
        with YouTubeDatabase() as db:
            # Calcular métricas agregadas de la serie
            ep_videos = all_videos[
                all_videos['video_id'].isin(
                    [e['video_id'] for e in s['episodes']]
                )
            ]
            avg_views = float(ep_videos['view_count'].mean()) if not ep_videos.empty and 'view_count' in ep_videos.columns else 0
            avg_eng = float(ep_videos['engagement_rate'].mean()) if not ep_videos.empty and 'engagement_rate' in ep_videos.columns else 0

            # Determinar tendencia (comparar primera mitad vs segunda mitad)
            trend = 'stable'
            if len(ep_videos) >= 4 and 'view_count' in ep_videos.columns:
                sorted_eps = ep_videos.sort_values('published_at')
                mid = len(sorted_eps) // 2
                first_half = sorted_eps.iloc[:mid]['view_count'].mean()
                second_half = sorted_eps.iloc[mid:]['view_count'].mean()
                if first_half > 0:
                    ratio = second_half / first_half
                    if ratio > 1.15:
                        trend = 'growing'
                    elif ratio < 0.85:
                        trend = 'declining'

            # Generar recomendación AI si hay >=3 episodios
            ai_rec = ''
            if len(s['episodes']) >= 3 and analyzer:
                ep_data = []
                for ep in s['episodes']:
                    vid = ep_videos[ep_videos['video_id'] == ep['video_id']]
                    ep_data.append({
                        'title': ep['title'],
                        'episode_number': ep['episode_number'],
                        'view_count': float(vid['view_count'].iloc[0]) if not vid.empty and 'view_count' in vid.columns else 0,
                        'engagement_rate': float(vid['engagement_rate'].iloc[0]) if not vid.empty and 'engagement_rate' in vid.columns else 0,
                        'published_at': ep['published_at'],
                    })
                channel_name = ep_videos['channel_title'].iloc[0] if 'channel_title' in ep_videos.columns and not ep_videos.empty else ''
                ai_rec = analyzer.recommend_series_format(
                    s['series_name'], channel_name, ep_data,
                )

            series_data = {
                'channel_id': s['channel_id'],
                'series_name': s['series_name'],
                'detected_pattern': s['detected_pattern'],
                'episode_count': len(s['episodes']),
                'avg_views': avg_views,
                'avg_engagement': avg_eng,
                'trend': trend,
                'ai_recommendation': ai_rec,
            }

            series_id = db.save_series(series_data)

            # Guardar episodios
            for ep in s['episodes']:
                db.save_series_episode(
                    series_id, ep['video_id'], ep['episode_number'],
                )

        log.info("  📚 Serie: %s (%d episodios, tendencia: %s)",
                 s['series_name'][:50], len(s['episodes']), trend)
```

**Step 3: Wire into `main()` function**

Add call after `_step_competitor_alerts(analyzer, notifier)` and before `_step3_analyze()`:

```python
_step_series_detection(analyzer)
```

**Step 4: Run syntax check**

Run: `.venv/Scripts/python.exe -c "import py_compile; py_compile.compile('main.py', doraise=True)"`
Expected: No output (success)

**Step 5: Commit**

```bash
git add main.py
git commit -m "feat(17.x): wire series detection into pipeline"
```

---

### Task 5: Dashboard — Series y Formatos page

**Files:**
- Modify: `dashboard.py` — add page constant, show function, navigation entry, route

**Step 1: Add page constant after `PAGE_TEMPORAL`**

```python
PAGE_SERIES            = "📚 Series y Formatos"
```

**Step 2: Add `show_series_analysis()` function**

Add after `show_temporal_comparison()` (or before the main function):

```python
def show_series_analysis(df: pd.DataFrame, channel_id: str = None):
    """Muestra análisis de series detectadas y recomendaciones de formato."""
    COLORS = _get_colors()
    ui_page_header("📚", "Series y Formatos",
                   "Series detectadas automáticamente con métricas por episodio y recomendaciones de formato")

    # Cargar series
    try:
        with YouTubeDatabase() as db:
            series_df = db.get_all_series(channel_id)
    except Exception as e:
        st.error(f"Error cargando series: {e}")
        return

    if series_df.empty:
        st.info("No se han detectado series aún. Ejecuta el pipeline para detectarlas automáticamente.")
        return

    # ── KPIs ──────────────────────────────────────────────────────
    total_series = len(series_df)
    total_episodes = int(series_df['episode_count'].sum()) if 'episode_count' in series_df.columns else 0
    avg_ep_per_series = total_episodes / total_series if total_series > 0 else 0

    growing = len(series_df[series_df['trend'] == 'growing']) if 'trend' in series_df.columns else 0
    declining = len(series_df[series_df['trend'] == 'declining']) if 'trend' in series_df.columns else 0

    cols = st.columns(4)
    with cols[0]:
        st.markdown(ui_metric_card("📚", "Series detectadas", str(total_series)),
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(ui_metric_card("🎬", "Total episodios", str(total_episodes)),
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(ui_metric_card("📈", "En crecimiento", str(growing),
                                   delta_type="positive"),
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(ui_metric_card("📉", "En declive", str(declining),
                                   delta_type="negative"),
                    unsafe_allow_html=True)

    ui_section_divider()

    # ── Lista de series ───────────────────────────────────────────
    ui_section_header("📋", "Series detectadas",
                      "Ordenadas por número de episodios")

    for _, series_row in series_df.iterrows():
        trend = series_row.get('trend', 'stable')
        trend_icon = {'growing': '📈', 'declining': '📉', 'stable': '➡️'}.get(trend, '➡️')
        trend_color = {
            'growing': COLORS.get('success', '#10B981'),
            'declining': COLORS.get('danger', '#EF4444'),
            'stable': COLORS.get('warning', '#F59E0B'),
        }.get(trend, COLORS.get('text_secondary', '#94A3B8'))

        series_name = _esc(series_row.get('series_name', 'Sin nombre'))
        ep_count = series_row.get('episode_count', 0)
        avg_views = series_row.get('avg_views', 0)
        avg_eng = series_row.get('avg_engagement', 0)
        pattern = series_row.get('detected_pattern', '')
        pattern_label = 'Numeración' if pattern == 'numbering' else 'Similitud NLP'

        with st.expander(
            f"{trend_icon} **{series_name}** — {ep_count} episodios | "
            f"Promedio: {avg_views:,.0f} vistas | {avg_eng:.2f}% eng | "
            f"Detección: {pattern_label}",
            expanded=False,
        ):
            # Métricas de la serie
            scols = st.columns(3)
            with scols[0]:
                st.metric("Episodios", ep_count)
            with scols[1]:
                st.metric("Vistas promedio", f"{avg_views:,.0f}")
            with scols[2]:
                st.metric("Tendencia", trend.capitalize())

            # Gráfico de episodios
            series_id = series_row.get('series_id')
            if series_id:
                try:
                    with YouTubeDatabase() as db:
                        episodes_df = db.get_series_episodes(int(series_id))
                except Exception:
                    episodes_df = pd.DataFrame()

                if not episodes_df.empty and 'view_count' in episodes_df.columns:
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=episodes_df['episode_number'],
                        y=episodes_df['view_count'],
                        mode='lines+markers',
                        name='Vistas',
                        line=dict(color=COLORS.get('primary', '#6366F1'), width=3),
                        marker=dict(size=10),
                    ))

                    if 'engagement_rate' in episodes_df.columns:
                        fig.add_trace(go.Bar(
                            x=episodes_df['episode_number'],
                            y=episodes_df['engagement_rate'],
                            name='Engagement %',
                            marker_color=COLORS.get('secondary', '#EC4899'),
                            opacity=0.4,
                            yaxis='y2',
                        ))

                    fig.update_layout(
                        title=f"Rendimiento por episodio — {series_name}",
                        xaxis_title="Episodio",
                        yaxis_title="Vistas",
                        yaxis2=dict(
                            title="Engagement %",
                            overlaying='y',
                            side='right',
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color=COLORS.get('text_primary', '#F8FAFC')),
                        height=350,
                        margin=dict(l=40, r=40, t=50, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Tabla de episodios
                    display_cols = ['episode_number', 'title', 'view_count', 'engagement_rate']
                    display_cols = [c for c in display_cols if c in episodes_df.columns]
                    st.dataframe(
                        episodes_df[display_cols].rename(columns={
                            'episode_number': 'Episodio',
                            'title': 'Título',
                            'view_count': 'Vistas',
                            'engagement_rate': 'Engagement %',
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

            # Recomendación AI
            ai_rec = series_row.get('ai_recommendation', '')
            if ai_rec and str(ai_rec).strip():
                st.markdown(f"""
                <div style="background:{COLORS.get('bg_secondary', '#1E293B')};
                            border-left:4px solid {COLORS.get('primary', '#6366F1')};
                            padding:1rem; border-radius:0.5rem; margin-top:0.5rem;">
                    <strong>🤖 Recomendación de formato:</strong><br>
                    {_esc(str(ai_rec))}
                </div>
                """, unsafe_allow_html=True)

    # ── Resumen de tendencias ─────────────────────────────────────
    ui_section_divider("Resumen de tendencias")

    if 'trend' in series_df.columns:
        import plotly.express as px
        trend_counts = series_df['trend'].value_counts().reset_index()
        trend_counts.columns = ['Tendencia', 'Cantidad']
        trend_map = {'growing': 'En crecimiento', 'declining': 'En declive', 'stable': 'Estable'}
        trend_counts['Tendencia'] = trend_counts['Tendencia'].map(
            lambda t: trend_map.get(t, t)
        )
        color_map = {
            'En crecimiento': COLORS.get('success', '#10B981'),
            'En declive': COLORS.get('danger', '#EF4444'),
            'Estable': COLORS.get('warning', '#F59E0B'),
        }
        fig2 = px.pie(
            trend_counts, names='Tendencia', values='Cantidad',
            color='Tendencia', color_discrete_map=color_map,
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS.get('text_primary', '#F8FAFC')),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)
```

**Step 3: Add to navigation radio**

Add `PAGE_SERIES` after `PAGE_CANNIBALIZATION` in the navigation list (in the "Detectores ML" group):

```python
PAGE_CANNIBALIZATION,
PAGE_SERIES,
PAGE_TRENDS,
```

**Step 4: Add to page router**

Add elif before the `PAGE_TRENDS` route:

```python
elif page == PAGE_SERIES:
    show_series_analysis(df, selected_channel_id)
```

**Step 5: Run syntax check**

Run: `.venv/Scripts/python.exe -c "import py_compile; py_compile.compile('dashboard.py', doraise=True)"`
Expected: No output (success)

**Step 6: Commit**

```bash
git add dashboard.py
git commit -m "feat(17.x): add Series y Formatos dashboard page"
```

---

### Task 6: Final verification & push

**Step 1: Run full import test**

```bash
cd C:\apps\AppiA\youtube-ai-agent-recomendation
.venv/Scripts/python.exe -c "
import sys, os
sys.path.append(os.path.join(os.path.dirname('.'), 'src'))
from series_detector import SeriesDetector
from database import YouTubeDatabase
print('All imports OK')
sd = SeriesDetector()
print(f'SeriesDetector threshold={sd.similarity_threshold}')
print('Done')
"
```

Expected: "All imports OK" + threshold output + "Done"

**Step 2: Push all commits**

```bash
git push
```

**Step 3: Update memory**

Update MEMORY.md with:
- V2 completed: add `series detection (17.1)`, `format recommender (17.2)`
- Database: 15 → 17 tables (add video_series, series_episodes)
- Dashboard: 18 → 19 pages
- New module: `src/series_detector.py`

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-03-02-series-management.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?
