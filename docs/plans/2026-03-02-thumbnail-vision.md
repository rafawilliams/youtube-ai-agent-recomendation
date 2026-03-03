# Thumbnail Vision Analysis (8.1 + 8.2) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Analyze YouTube video thumbnails using Claude Vision to score clickability and compare A/B options before publishing.

**Architecture:** New module `src/thumbnail_analyzer.py` with two classes: `ThumbnailScorer` (8.1) scores existing thumbnails via Claude Vision multimodal API, `ThumbnailComparator` (8.2) compares 2-3 uploaded thumbnails. Results stored in two new DB tables. Pipeline step analyzes new thumbnails automatically. Dashboard adds two pages: gallery view and A/B comparator.

**Tech Stack:** Anthropic SDK (multimodal/Vision), MariaDB, Streamlit, Plotly

---

## Task 1: Database — `thumbnail_scores` table

**Files:**
- Modify: `src/database.py`

**Step 1: Add `thumbnail_scores` table in `_create_tables()`**

After the `evergreen_scores` CREATE TABLE block (around line 344), add:

```python
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thumbnail_scores (
                id INT AUTO_INCREMENT PRIMARY KEY,
                video_id VARCHAR(255) NOT NULL,
                channel_id VARCHAR(255) NOT NULL,
                text_clarity FLOAT,
                facial_expression FLOAT,
                color_contrast FLOAT,
                composition FLOAT,
                clickability FLOAT,
                has_text TINYINT(1) DEFAULT 0,
                has_face TINYINT(1) DEFAULT 0,
                dominant_colors TEXT,
                improvement_suggestions TEXT,
                analyzed_at VARCHAR(50) NOT NULL,
                UNIQUE KEY uq_ts_video (video_id),
                FOREIGN KEY (video_id) REFERENCES videos(video_id),
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thumbnail_comparisons (
                id INT AUTO_INCREMENT PRIMARY KEY,
                channel_id VARCHAR(255) NOT NULL,
                comparison_name VARCHAR(500),
                thumbnails_json TEXT NOT NULL,
                winner VARCHAR(10),
                analysis TEXT,
                created_at VARCHAR(50) NOT NULL,
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
```

**Step 2: Add CRUD methods at the end of the class**

Add these methods before the `__enter__`/`__exit__` context manager methods:

```python
    # ------------------------------------------------------------------
    # Thumbnail scores (8.1)
    # ------------------------------------------------------------------

    def save_thumbnail_score(self, data: dict):
        """Guarda o actualiza el score de thumbnail de un video."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        try:
            cursor.execute("""
                INSERT INTO thumbnail_scores
                (video_id, channel_id, text_clarity, facial_expression,
                 color_contrast, composition, clickability,
                 has_text, has_face, dominant_colors,
                 improvement_suggestions, analyzed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    text_clarity = VALUES(text_clarity),
                    facial_expression = VALUES(facial_expression),
                    color_contrast = VALUES(color_contrast),
                    composition = VALUES(composition),
                    clickability = VALUES(clickability),
                    has_text = VALUES(has_text),
                    has_face = VALUES(has_face),
                    dominant_colors = VALUES(dominant_colors),
                    improvement_suggestions = VALUES(improvement_suggestions),
                    analyzed_at = VALUES(analyzed_at)
            """, (
                data['video_id'],
                data['channel_id'],
                data.get('text_clarity'),
                data.get('facial_expression'),
                data.get('color_contrast'),
                data.get('composition'),
                data.get('clickability'),
                1 if data.get('has_text') else 0,
                1 if data.get('has_face') else 0,
                json.dumps(data.get('dominant_colors', []), ensure_ascii=False),
                json.dumps(data.get('improvement_suggestions', []), ensure_ascii=False),
                now,
            ))
            self.conn.commit()
        except Exception as e:
            log.error("Error guardando thumbnail score para %s: %s",
                      data.get('video_id'), e)
            self.conn.rollback()

    def get_thumbnail_scores(self, channel_id: str) -> pd.DataFrame:
        """Obtiene todos los scores de thumbnails para un canal."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT ts.*, v.title, m.view_count
            FROM thumbnail_scores ts
            JOIN videos v ON ts.video_id = v.video_id
            LEFT JOIN (
                SELECT video_id, view_count
                FROM video_metrics
                WHERE (video_id, recorded_at) IN (
                    SELECT video_id, MAX(recorded_at)
                    FROM video_metrics GROUP BY video_id
                )
            ) m ON ts.video_id = m.video_id
            WHERE ts.channel_id = %s
            ORDER BY ts.clickability DESC
        """, (channel_id,))
        rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_videos_without_thumbnail_score(self, channel_id: str) -> pd.DataFrame:
        """Obtiene videos que aún no tienen score de thumbnail."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT v.video_id, v.title
            FROM videos v
            WHERE v.channel_id = %s
              AND v.video_id NOT IN (
                  SELECT video_id FROM thumbnail_scores
              )
            ORDER BY v.published_at DESC
        """, (channel_id,))
        rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Thumbnail comparisons (8.2)
    # ------------------------------------------------------------------

    def save_thumbnail_comparison(self, data: dict):
        """Guarda una comparación A/B de thumbnails."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        try:
            cursor.execute("""
                INSERT INTO thumbnail_comparisons
                (channel_id, comparison_name, thumbnails_json,
                 winner, analysis, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                data['channel_id'],
                data.get('comparison_name', ''),
                json.dumps(data.get('thumbnails', []), ensure_ascii=False),
                data.get('winner', ''),
                data.get('analysis', ''),
                now,
            ))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            log.error("Error guardando comparación de thumbnails: %s", e)
            self.conn.rollback()
            return None

    def get_thumbnail_comparisons(self, channel_id: str) -> pd.DataFrame:
        """Obtiene historial de comparaciones A/B para un canal."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM thumbnail_comparisons
            WHERE channel_id = %s
            ORDER BY created_at DESC
            LIMIT 20
        """, (channel_id,))
        rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
```

**Step 3: Commit**

```bash
git add src/database.py
git commit -m "feat(8.x): add thumbnail_scores and thumbnail_comparisons tables with CRUD"
```

---

## Task 2: Core module — `src/thumbnail_analyzer.py`

**Files:**
- Create: `src/thumbnail_analyzer.py`

**Step 1: Create the ThumbnailScorer class (8.1)**

```python
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
```

**Step 2: Commit**

```bash
git add src/thumbnail_analyzer.py
git commit -m "feat(8.x): add ThumbnailScorer and ThumbnailComparator with Claude Vision"
```

---

## Task 3: Pipeline integration — `main.py`

**Files:**
- Modify: `main.py`

**Step 1: Add import at the top (after the `revenue_analyzer` import, line 20)**

```python
from thumbnail_analyzer import ThumbnailScorer
```

**Step 2: Add the pipeline step function (after `_step_revenue_analysis`, before `_step4_analytics`)**

```python
def _step_thumbnail_analysis(anthropic_api_key: str):
    """Analiza thumbnails de videos sin score usando Claude Vision (Mejora 8.1)."""
    log.info("PASO THUMBNAILS: Analizando miniaturas con Claude Vision...")

    scorer = ThumbnailScorer(anthropic_api_key)

    with YouTubeDatabase() as db:
        own_channels = db.get_own_channels()

    if own_channels.empty:
        log.info("  Sin canales propios para analizar")
        return

    total_analyzed = 0
    for _, ch in own_channels.iterrows():
        channel_id = ch['channel_id']
        with YouTubeDatabase() as db:
            results = scorer.analyze_channel(channel_id, db)

        if not results:
            continue

        avg_click = sum(r.get('clickability', 0) for r in results) / len(results)
        log.info("  🖼 %s: %d thumbnails analizadas (clickability prom: %.1f)",
                 ch.get('channel_name', channel_id), len(results), avg_click)
        total_analyzed += len(results)

    log.info("  Thumbnails analizadas: %d", total_analyzed)
```

**Step 3: Wire the step into `main()` (after `_step_revenue_analysis()`, before `_step3_analyze`)**

In the `main()` function, after line 558 (`_step_revenue_analysis()`), add:

```python
        # Análisis de thumbnails con Claude Vision (Mejora 8.x)
        _step_thumbnail_analysis(anthropic_api_key)
```

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat(8.1): wire thumbnail analysis step into pipeline"
```

---

## Task 4: Dashboard — Galería de Miniaturas page (8.1)

**Files:**
- Modify: `dashboard.py`

**Step 1: Add page constant (near existing PAGE_ constants, around line 53)**

After `PAGE_REVENUE`:

```python
PAGE_THUMBNAILS    = "🖼 Galería de Miniaturas"
PAGE_THUMBNAIL_AB  = "🔀 Comparador A/B"
```

**Step 2: Add both pages to the sidebar navigation (after `PAGE_REVENUE` in the radio list, line 5539)**

After `PAGE_REVENUE,`:

```python
            PAGE_THUMBNAILS,
            PAGE_THUMBNAIL_AB,
```

**Step 3: Add `show_thumbnail_gallery` function (before `main()`)**

```python
def show_thumbnail_gallery(df: pd.DataFrame, channel_id: str):
    """Galería de miniaturas con scores de Claude Vision (8.1)."""
    import plotly.graph_objects as go

    COLORS = _get_colors()
    ui_page_header("🖼", "Galería de Miniaturas",
                   "Análisis de miniaturas con Claude Vision — scores de clickability y sugerencias")

    if df.empty or not channel_id:
        st.warning("No hay datos disponibles. Ejecuta el pipeline primero.")
        return

    with YouTubeDatabase() as db:
        scores_df = db.get_thumbnail_scores(channel_id)

    if scores_df.empty:
        st.info("Aún no se han analizado miniaturas para este canal. "
                "Ejecuta el pipeline para analizar automáticamente.")

        # Botón para análisis manual
        if st.button("🔍 Analizar miniaturas ahora", key="btn_analyze_thumbnails"):
            api_key = os.getenv('ANTHROPIC_API_KEY', '')
            if not api_key:
                st.error("ANTHROPIC_API_KEY no configurada.")
                return
            from thumbnail_analyzer import ThumbnailScorer
            scorer = ThumbnailScorer(api_key)
            with st.spinner("Analizando miniaturas con Claude Vision..."):
                with YouTubeDatabase() as db2:
                    results = scorer.analyze_channel(channel_id, db2, max_per_run=20)
            if results:
                st.success(f"✓ {len(results)} miniaturas analizadas")
                st.rerun()
            else:
                st.warning("No se encontraron miniaturas pendientes o hubo un error.")
        return

    # ── KPIs ──────────────────────────────────────────────────────────
    avg_click = float(scores_df['clickability'].mean()) if 'clickability' in scores_df.columns else 0
    best_click = float(scores_df['clickability'].max()) if 'clickability' in scores_df.columns else 0
    worst_click = float(scores_df['clickability'].min()) if 'clickability' in scores_df.columns else 0
    total_analyzed = len(scores_df)

    cols = st.columns(4)
    with cols[0]:
        st.markdown(ui_metric_card("🖼", "Analizadas", str(total_analyzed)),
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(ui_metric_card("📊", "Clickability prom.",
                                   f"{avg_click:.1f}/10"),
                    unsafe_allow_html=True)
    with cols[2]:
        st.markdown(ui_metric_card("🏆", "Mejor score",
                                   f"{best_click:.1f}/10"),
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(ui_metric_card("⚠", "Peor score",
                                   f"{worst_click:.1f}/10"),
                    unsafe_allow_html=True)

    ui_section_divider()

    # ── Distribución de scores ────────────────────────────────────────
    ui_section_header("📊", "Distribución de Clickability")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores_df['clickability'],
        nbinsx=10,
        marker_color=COLORS.get('primary', '#6366F1'),
        opacity=0.8,
    ))
    fig.update_layout(
        xaxis_title="Clickability Score",
        yaxis_title="Cantidad de videos",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS.get('text_primary', '#F8FAFC')),
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    ui_section_divider()

    # ── Radar por dimensión ───────────────────────────────────────────
    ui_section_header("🎯", "Scores promedio por dimensión")

    dimensions = ['text_clarity', 'color_contrast', 'composition', 'clickability']
    dim_labels = ['Claridad de Texto', 'Contraste de Color', 'Composición', 'Clickability']

    # Agregar facial_expression solo si hay datos
    if 'facial_expression' in scores_df.columns:
        face_scores = scores_df['facial_expression'].dropna()
        if not face_scores.empty:
            dimensions.insert(1, 'facial_expression')
            dim_labels.insert(1, 'Expresión Facial')

    avgs = []
    for dim in dimensions:
        val = scores_df[dim].dropna().mean() if dim in scores_df.columns else 0
        avgs.append(round(float(val), 1))

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=avgs + [avgs[0]],
        theta=dim_labels + [dim_labels[0]],
        fill='toself',
        fillcolor=COLORS.get('primary', '#6366F1') + '33',
        line=dict(color=COLORS.get('primary', '#6366F1'), width=2),
        name='Promedio',
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 10], dtick=2,
                            gridcolor=COLORS.get('border', '#334155')),
            angularaxis=dict(gridcolor=COLORS.get('border', '#334155')),
            bgcolor='rgba(0,0,0,0)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=COLORS.get('text_primary', '#F8FAFC')),
        height=350,
        margin=dict(l=60, r=60, t=30, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    ui_section_divider()

    # ── Galería de miniaturas ─────────────────────────────────────────
    ui_section_header("🖼", "Galería ordenada por Clickability")

    sort_order = st.radio(
        "Ordenar por", ["Mejor → Peor", "Peor → Mejor"],
        horizontal=True, key="thumb_sort",
    )
    ascending = sort_order == "Peor → Mejor"
    sorted_df = scores_df.sort_values('clickability', ascending=ascending)

    # Mostrar en grid de 3 columnas
    rows_to_show = min(len(sorted_df), 30)
    for row_start in range(0, rows_to_show, 3):
        grid_cols = st.columns(3)
        for col_idx in range(3):
            idx = row_start + col_idx
            if idx >= rows_to_show:
                break
            row = sorted_df.iloc[idx]
            video_id = row['video_id']
            thumb_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
            title = _esc(str(row.get('title', video_id))[:50])
            click_score = float(row.get('clickability', 0))
            views = int(row.get('view_count', 0)) if pd.notna(row.get('view_count')) else 0

            # Color por score
            if click_score >= 7:
                badge_color = COLORS.get('success', '#10B981')
            elif click_score >= 4:
                badge_color = COLORS.get('warning', '#F59E0B')
            else:
                badge_color = COLORS.get('danger', '#EF4444')

            with grid_cols[col_idx]:
                st.markdown(f"""
                <div style="background: {COLORS.get('bg_secondary', '#1E293B')};
                            border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem;
                            border: 1px solid {COLORS.get('border', '#334155')};">
                    <img src="{thumb_url}" style="width: 100%; border-radius: 6px;
                         aspect-ratio: 16/9; object-fit: cover;" />
                    <div style="padding: 0.4rem 0.2rem 0;">
                        <div style="font-size: 0.8rem; font-weight: 600;
                                    color: {COLORS.get('text_primary', '#F8FAFC')};
                                    white-space: nowrap; overflow: hidden;
                                    text-overflow: ellipsis;">{title}</div>
                        <div style="display: flex; justify-content: space-between;
                                    align-items: center; margin-top: 4px;">
                            <span style="background: {badge_color}; color: #fff;
                                         padding: 2px 8px; border-radius: 12px;
                                         font-size: 0.75rem; font-weight: 700;">
                                {click_score:.1f}/10
                            </span>
                            <span style="font-size: 0.7rem;
                                         color: {COLORS.get('text_muted', '#94A3B8')};">
                                {views:,} vistas
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    ui_section_divider()

    # ── Detalle expandible por video ──────────────────────────────────
    ui_section_header("🔍", "Detalle y Sugerencias de Mejora")

    # Filtrar solo los peores para sugerencias
    worst_df = scores_df.nsmallest(10, 'clickability')

    for _, row in worst_df.iterrows():
        title = str(row.get('title', row['video_id']))[:60]
        click_score = float(row.get('clickability', 0))

        with st.expander(f"⚠ {title} — {click_score:.1f}/10"):
            detail_cols = st.columns([1, 2])
            with detail_cols[0]:
                vid = row['video_id']
                st.image(f"https://img.youtube.com/vi/{vid}/hqdefault.jpg",
                         use_container_width=True)

            with detail_cols[1]:
                scores_text = (
                    f"- **Claridad de texto:** {row.get('text_clarity', '-')}/10\n"
                    f"- **Contraste de color:** {row.get('color_contrast', '-')}/10\n"
                    f"- **Composición:** {row.get('composition', '-')}/10\n"
                    f"- **Clickability:** {row.get('clickability', '-')}/10\n"
                )
                if pd.notna(row.get('facial_expression')):
                    scores_text += f"- **Expresión facial:** {row['facial_expression']}/10\n"

                st.markdown(scores_text)

                # Sugerencias
                suggestions = []
                raw_sug = row.get('improvement_suggestions', '[]')
                if isinstance(raw_sug, str):
                    try:
                        suggestions = json.loads(raw_sug)
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(raw_sug, list):
                    suggestions = raw_sug

                if suggestions:
                    st.markdown("**💡 Sugerencias de mejora:**")
                    for sug in suggestions:
                        st.markdown(f"- {sug}")
```

**Step 4: Add router entries (in `main()`, after the `PAGE_REVENUE` elif block)**

After the `show_revenue_analysis` elif:

```python
    elif page == PAGE_THUMBNAILS:
        show_thumbnail_gallery(df, selected_channel_id)

    elif page == PAGE_THUMBNAIL_AB:
        show_thumbnail_comparator(df, selected_channel_id)
```

**Step 5: Commit**

```bash
git add dashboard.py
git commit -m "feat(8.1): add thumbnail gallery page with scores, radar, and suggestions"
```

---

## Task 5: Dashboard — Comparador A/B page (8.2)

**Files:**
- Modify: `dashboard.py`

**Step 1: Add `show_thumbnail_comparator` function (after `show_thumbnail_gallery`)**

```python
def show_thumbnail_comparator(df: pd.DataFrame, channel_id: str):
    """Comparador A/B de miniaturas con Claude Vision (8.2)."""
    import base64

    COLORS = _get_colors()
    ui_page_header("🔀", "Comparador A/B de Miniaturas",
                   "Sube 2-3 opciones de miniatura y Claude Vision predice cuál tendrá mayor CTR")

    if not channel_id:
        st.warning("Selecciona un canal en la barra lateral.")
        return

    # ── Upload de imágenes ────────────────────────────────────────────
    ui_section_header("📤", "Sube tus miniaturas", "Sube 2 o 3 imágenes para comparar")

    comparison_name = st.text_input(
        "Nombre de la comparación (opcional)",
        placeholder="Ej: Thumbnail para video de recetas",
        key="thumb_ab_name",
    )

    upload_cols = st.columns(3)
    uploaded_files = []
    labels = ["Miniatura A", "Miniatura B", "Miniatura C (opcional)"]

    for i, col in enumerate(upload_cols):
        with col:
            f = st.file_uploader(
                labels[i],
                type=["png", "jpg", "jpeg", "webp"],
                key=f"thumb_upload_{i}",
            )
            if f:
                uploaded_files.append((chr(65 + i), f))  # ('A', file), ('B', file)...
                st.image(f, caption=f"Miniatura {chr(65 + i)}", use_container_width=True)

    if len(uploaded_files) < 2:
        st.info("Sube al menos 2 imágenes para comparar.")
    else:
        if st.button("🔍 Comparar miniaturas", key="btn_compare_thumbs",
                      type="primary"):
            api_key = os.getenv('ANTHROPIC_API_KEY', '')
            if not api_key:
                st.error("ANTHROPIC_API_KEY no configurada.")
                return

            from thumbnail_analyzer import ThumbnailComparator
            comparator = ThumbnailComparator(api_key)

            # Preparar imágenes como base64
            images = []
            for label, f in uploaded_files:
                f.seek(0)
                data = base64.b64encode(f.read()).decode('utf-8')
                # Determinar media type
                ext = f.name.rsplit('.', 1)[-1].lower() if '.' in f.name else 'png'
                media_types = {'png': 'image/png', 'jpg': 'image/jpeg',
                               'jpeg': 'image/jpeg', 'webp': 'image/webp'}
                mt = media_types.get(ext, 'image/png')
                images.append({
                    "source": {
                        "type": "base64",
                        "media_type": mt,
                        "data": data,
                    }
                })

            with st.spinner("Analizando con Claude Vision..."):
                result = comparator.compare(images)

            if not result:
                st.error("No se pudo completar el análisis. Intenta de nuevo.")
                return

            # Guardar en BD
            with YouTubeDatabase() as db:
                db.save_thumbnail_comparison({
                    'channel_id': channel_id,
                    'comparison_name': comparison_name or 'Sin nombre',
                    'thumbnails': [l for l, _ in uploaded_files],
                    'winner': result.get('winner', '?'),
                    'analysis': json.dumps(result, ensure_ascii=False),
                })

            # ── Mostrar resultado ─────────────────────────────────────
            st.success(f"🏆 **Ganadora: Miniatura {result.get('winner', '?')}**")

            ui_section_divider()

            # Scores por miniatura
            scores = result.get('scores', {})
            score_cols = st.columns(len(uploaded_files))
            for idx, (label, f) in enumerate(uploaded_files):
                with score_cols[idx]:
                    is_winner = (label == result.get('winner'))
                    border = COLORS.get('success', '#10B981') if is_winner else COLORS.get('border', '#334155')
                    winner_badge = ' 🏆' if is_winner else ''

                    st.markdown(f"""
                    <div style="background: {COLORS.get('bg_secondary', '#1E293B')};
                                border: 2px solid {border}; border-radius: 8px;
                                padding: 0.8rem; text-align: center;">
                        <div style="font-weight: 700; font-size: 1.1rem;
                                    color: {COLORS.get('text_primary', '#F8FAFC')};">
                            Miniatura {label}{winner_badge}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    s = scores.get(label, {})
                    for metric, metric_label in [
                        ('clickability', 'Clickability'),
                        ('mobile_legibility', 'Legibilidad móvil'),
                        ('visual_hierarchy', 'Jerarquía visual'),
                        ('engagement_potential', 'Potencial de engagement'),
                    ]:
                        val = s.get(metric, 0)
                        bar_pct = val * 10
                        bar_color = COLORS.get('success', '#10B981') if val >= 7 else (
                            COLORS.get('warning', '#F59E0B') if val >= 4 else COLORS.get('danger', '#EF4444')
                        )
                        st.markdown(f"""
                        <div style="margin: 4px 0;">
                            <div style="display: flex; justify-content: space-between;
                                        font-size: 0.75rem;
                                        color: {COLORS.get('text_muted', '#94A3B8')};">
                                <span>{metric_label}</span><span>{val}/10</span>
                            </div>
                            <div style="background: {COLORS.get('bg_tertiary', '#0F172A')};
                                        border-radius: 4px; height: 6px; margin-top: 2px;">
                                <div style="background: {bar_color}; width: {bar_pct}%;
                                            height: 100%; border-radius: 4px;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            ui_section_divider()

            # Razonamiento
            reasoning = result.get('reasoning', '')
            if reasoning:
                ui_section_header("💡", "Razonamiento")
                st.markdown(reasoning)

            suggestions = result.get('suggestions', [])
            if suggestions:
                ui_section_header("📝", "Sugerencias")
                for sug in suggestions:
                    st.markdown(f"- {sug}")

    # ── Historial de comparaciones ────────────────────────────────────
    ui_section_divider()
    ui_section_header("📜", "Historial de Comparaciones")

    with YouTubeDatabase() as db:
        history_df = db.get_thumbnail_comparisons(channel_id)

    if history_df.empty:
        st.info("Sin comparaciones previas.")
    else:
        for _, row in history_df.iterrows():
            name = row.get('comparison_name', 'Sin nombre')
            winner = row.get('winner', '?')
            created = str(row.get('created_at', ''))[:16]

            with st.expander(f"🏆 {name} — Ganadora: {winner} ({created})"):
                analysis = {}
                raw = row.get('analysis', '{}')
                if isinstance(raw, str):
                    try:
                        analysis = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if analysis.get('reasoning'):
                    st.markdown(analysis['reasoning'])
```

**Step 2: Add sidebar nav group label for the new section**

In the sidebar navigation, right before `PAGE_THUMBNAILS` in the radio list, add a new nav group label. Find the `# ── Detectores ML` comment and after the `PAGE_REVENUE,` entry, add the group. The two new pages go under a logical grouping. Since they belong with the AI/Vision tools, place them after `PAGE_REVENUE` in the radio list (already done in Task 4 Step 2).

**Step 3: Commit**

```bash
git add dashboard.py
git commit -m "feat(8.2): add A/B thumbnail comparator page with upload, scoring, and history"
```

---

## Task 6: Environment config and final wiring

**Files:**
- Modify: `.env.example`

**Step 1: Add thumbnail config to `.env.example`**

After the `ESTIMATED_CPM` block:

```ini
# Thumbnails — análisis con Claude Vision (Mejora 8.x)
# Máximo de thumbnails a analizar por ejecución del pipeline
MAX_THUMBNAILS_PER_RUN=50
```

**Step 2: Update `main.py` to respect `MAX_THUMBNAILS_PER_RUN`**

In `_step_thumbnail_analysis`, read the env var:

```python
    max_per_run = int(os.getenv('MAX_THUMBNAILS_PER_RUN', '50'))
```

And pass it to `scorer.analyze_channel(channel_id, db, max_per_run=max_per_run)`.

**Step 3: Commit**

```bash
git add .env.example main.py
git commit -m "feat(8.x): add MAX_THUMBNAILS_PER_RUN config, update .env.example"
```

---

## Task 7: Final review and squash commit

**Step 1: Verify all imports resolve**

Check `dashboard.py` imports `json` (already imported) and has the `YouTubeDatabase` and `_esc` helpers available (they are).

**Step 2: Run the dashboard to verify no syntax errors**

```bash
cd /c/apps/AppiA/youtube-ai-agent-recomendation
python -c "import py_compile; py_compile.compile('src/thumbnail_analyzer.py', doraise=True)"
python -c "import py_compile; py_compile.compile('dashboard.py', doraise=True)"
python -c "import py_compile; py_compile.compile('main.py', doraise=True)"
python -c "import py_compile; py_compile.compile('src/database.py', doraise=True)"
```

**Step 3: Final commit if any fixes needed**

```bash
git add -A
git commit -m "feat(8.x): thumbnail Vision analysis — gallery, scoring, A/B comparator"
```

---

## Summary of all changes

| File | Action | What |
|---|---|---|
| `src/thumbnail_analyzer.py` | Create | ThumbnailScorer + ThumbnailComparator classes |
| `src/database.py` | Modify | 2 new tables + 5 CRUD methods |
| `main.py` | Modify | New pipeline step `_step_thumbnail_analysis` |
| `dashboard.py` | Modify | 2 new pages: Gallery (8.1) + Comparador A/B (8.2) |
| `.env.example` | Modify | Add `MAX_THUMBNAILS_PER_RUN` |
