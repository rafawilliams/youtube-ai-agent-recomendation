# Alertas de Contenido Competidor (7.2) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When a competitor video gets >2x their average views in the first ~48h, send a Telegram alert with Claude AI analysis of why it took off and how to replicate the success.

**Architecture:** New DB table `competitor_alerts` stores detected alerts to avoid duplicates. New method in `AIAnalyzer` generates per-video "why it went viral" analysis. New method in `TelegramNotifier` formats and sends the alert. New pipeline step `_step_competitor_alerts()` in `main.py` ties it all together. Dashboard gets a new section showing historical alerts.

**Tech Stack:** Python, MariaDB (pymysql), Anthropic Claude API, Telegram Bot API, Streamlit, pandas

---

### Task 1: Create `competitor_alerts` table in database.py

**Files:**
- Modify: `src/database.py:266-289` (inside `_create_tables`, before the indexes block)

**Step 1: Add table creation SQL**

After the `channel_health_reports` CREATE TABLE (line 276) and before the indexes block (line 278), add:

```python
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS competitor_alerts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                video_id VARCHAR(255) NOT NULL,
                channel_id VARCHAR(255) NOT NULL,
                channel_name VARCHAR(500),
                video_title VARCHAR(1000),
                view_count BIGINT NOT NULL,
                competitor_avg_views FLOAT NOT NULL,
                ratio FLOAT NOT NULL,
                ai_analysis TEXT,
                notified TINYINT(1) NOT NULL DEFAULT 0,
                created_at VARCHAR(50) NOT NULL,
                UNIQUE KEY uq_ca_video (video_id),
                FOREIGN KEY (video_id) REFERENCES videos(video_id),
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
```

**Step 2: Run the app to verify table creation**

Run: `cd /c/apps/AppiA/youtube-ai-agent-recomendation && python -c "from dotenv import load_dotenv; load_dotenv(); from src.database import YouTubeDatabase; db = YouTubeDatabase(); print('OK'); db.close()"`
Expected: `OK` (no errors)

**Step 3: Commit**

```bash
git add src/database.py
git commit -m "feat(7.2): add competitor_alerts table"
```

---

### Task 2: Add DB methods for competitor alerts

**Files:**
- Modify: `src/database.py` (add after the competitor section, ~line 1351)

**Step 1: Add `get_recent_competitor_videos` method**

Insert before the `close()` method:

```python
    # ------------------------------------------------------------------
    # Alertas de Competidores (Mejora 7.2)
    # ------------------------------------------------------------------

    def get_recent_competitor_videos(self, days: int = 7) -> pd.DataFrame:
        """Retorna videos de competidores publicados en los últimos N días,
        junto con su vista actual y el promedio de vistas del canal."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                v.video_id,
                v.channel_id,
                v.title,
                v.published_at,
                v.video_type,
                c.channel_name,
                m.view_count,
                m.like_count,
                m.engagement_rate,
                m.recorded_at AS metric_recorded_at,
                ch_avg.avg_views AS competitor_avg_views
            FROM videos v
            JOIN channels c ON v.channel_id = c.channel_id AND c.is_competitor = 1
            LEFT JOIN (
                SELECT video_id, view_count, like_count, engagement_rate, recorded_at,
                       ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY recorded_at DESC) AS rn
                FROM video_metrics
            ) m ON v.video_id = m.video_id AND m.rn = 1
            LEFT JOIN (
                SELECT v2.channel_id, AVG(m2.view_count) AS avg_views
                FROM videos v2
                JOIN (
                    SELECT video_id, view_count,
                           ROW_NUMBER() OVER (PARTITION BY video_id ORDER BY recorded_at DESC) AS rn
                    FROM video_metrics
                ) m2 ON v2.video_id = m2.video_id AND m2.rn = 1
                JOIN channels c2 ON v2.channel_id = c2.channel_id AND c2.is_competitor = 1
                GROUP BY v2.channel_id
            ) ch_avg ON v.channel_id = ch_avg.channel_id
            WHERE v.published_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
              AND c.is_competitor = 1
            ORDER BY m.view_count DESC
        """, (days,))
        rows = cursor.fetchall()
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            df['published_at'] = pd.to_datetime(df['published_at'])
            if 'metric_recorded_at' in df.columns:
                df['metric_recorded_at'] = pd.to_datetime(df['metric_recorded_at'])
        return df

    def save_competitor_alert(self, alert: dict):
        """Guarda una alerta de competidor. Ignora duplicados (UNIQUE en video_id)."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO competitor_alerts
                (video_id, channel_id, channel_name, video_title, view_count,
                 competitor_avg_views, ratio, ai_analysis, notified, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                alert['video_id'], alert['channel_id'], alert['channel_name'],
                alert['video_title'], alert['view_count'],
                alert['competitor_avg_views'], alert['ratio'],
                alert.get('ai_analysis', ''), alert.get('notified', 0),
                datetime.now().isoformat(),
            ))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            if 'Duplicate' in str(e):
                log.debug("Alerta ya existe para video %s", alert['video_id'])
                return False
            raise

    def get_competitor_alerts(self, limit: int = 20) -> pd.DataFrame:
        """Retorna las últimas alertas de competidores."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, video_id, channel_id, channel_name, video_title,
                   view_count, competitor_avg_views, ratio, ai_analysis,
                   notified, created_at
            FROM competitor_alerts
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        rows = cursor.fetchall()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def is_alert_already_sent(self, video_id: str) -> bool:
        """Verifica si ya se envió una alerta para este video."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT 1 FROM competitor_alerts WHERE video_id = %s LIMIT 1",
            (video_id,),
        )
        return cursor.fetchone() is not None
```

**Step 2: Verify syntax**

Run: `cd /c/apps/AppiA/youtube-ai-agent-recomendation && python -c "from dotenv import load_dotenv; load_dotenv(); from src.database import YouTubeDatabase; db = YouTubeDatabase(); print(type(db.get_competitor_alerts())); db.close()"`
Expected: `<class 'pandas.core.frame.DataFrame'>`

**Step 3: Commit**

```bash
git add src/database.py
git commit -m "feat(7.2): add DB methods for competitor alerts"
```

---

### Task 3: Add `analyze_viral_competitor_video` to AIAnalyzer

**Files:**
- Modify: `src/ai_analyzer.py` (add after `analyze_competitor_gaps`, ~line 1399)

**Step 1: Add the new method**

```python
    # ------------------------------------------------------------------
    # Alertas de Competidores (Mejora 7.2)
    # ------------------------------------------------------------------

    def analyze_viral_competitor_video(
        self,
        video_title: str,
        channel_name: str,
        view_count: int,
        competitor_avg_views: float,
        ratio: float,
        video_type: str = '',
        language: str = 'es',
    ) -> str:
        """
        Analiza por qué un video de competidor despegó y cómo replicar el éxito.

        Args:
            video_title: Título del video viral
            channel_name: Nombre del canal competidor
            view_count: Vistas actuales del video
            competitor_avg_views: Promedio de vistas del competidor
            ratio: Multiplicador sobre el promedio (view_count / avg)
            video_type: 'Short' o 'Video'
            language: Código de idioma

        Returns:
            Texto markdown con análisis y recomendaciones.
        """
        lang_name = self._LANGUAGE_NAMES.get(language, language)

        prompt = f"""You are a YouTube strategist analyzing a competitor's viral video.

COMPETITOR VIDEO:
- Title: "{video_title}"
- Channel: {channel_name}
- Views: {view_count:,}
- Channel average views: {competitor_avg_views:,.0f}
- Performance: {ratio:.1f}x above average
- Type: {video_type or 'Unknown'}

TASK: Analyze WHY this video went viral and HOW to replicate its success.
Write EVERYTHING in **{lang_name}**.

STRUCTURE (use markdown):

## Por qué despegó
2-3 bullet points analyzing title, topic, and format factors that explain the
performance spike. Be specific about the title structure, emotional hooks,
trending topics, or format choices.

## Cómo replicar el éxito
3 concrete video ideas inspired by this viral video. For each:
- Suggested title
- Format (Short / Video Largo)
- What angle to take that differentiates from the competitor

## Acción inmediata
ONE specific action to take THIS WEEK to capitalize on the trend.

RULES:
- Be data-driven and specific (reference the actual title)
- Suggest SPECIFIC titles, not generic ideas
- Max 300 words total
- Write in {lang_name} only"""

        return self._call_claude(prompt, max_tokens=1200)
```

**Step 2: Verify syntax**

Run: `cd /c/apps/AppiA/youtube-ai-agent-recomendation && python -c "import sys; sys.path.append('src'); from ai_analyzer import AIAnalyzer; print('Import OK')"`
Expected: `Import OK`

**Step 3: Commit**

```bash
git add src/ai_analyzer.py
git commit -m "feat(7.2): add analyze_viral_competitor_video to AIAnalyzer"
```

---

### Task 4: Add `notify_competitor_alert` to TelegramNotifier

**Files:**
- Modify: `src/telegram_notifier.py` (add after `notify_pipeline_complete`, ~line 109)

**Step 1: Add the new method**

```python
    def notify_competitor_alert(
        self,
        channel_name: str,
        video_title: str,
        view_count: int,
        competitor_avg_views: float,
        ratio: float,
        ai_analysis: str,
    ) -> bool:
        """
        Envía alerta cuando un video de competidor supera 2x su promedio.

        Args:
            channel_name: Nombre del canal competidor
            video_title: Título del video viral
            view_count: Vistas actuales
            competitor_avg_views: Promedio de vistas del competidor
            ratio: Multiplicador sobre el promedio
            ai_analysis: Análisis de Claude (markdown)
        """
        channel_safe = _md_escape(channel_name)
        title_safe = _md_escape(video_title[:80])

        # Truncar análisis para caber en Telegram (max ~4000 chars)
        analysis_short = ai_analysis[:2500] if ai_analysis else '(sin análisis)'

        lines = [
            f"🚨 *Alerta Competidor — {channel_safe}*",
            "",
            f"📺 _{title_safe}_",
            f"👀 Vistas: {view_count:,} ({ratio:.1f}x su promedio de {competitor_avg_views:,.0f})",
            "",
            "🤖 *Análisis:*",
            analysis_short,
        ]

        return self.send_message('\n'.join(lines))
```

**Step 2: Verify syntax**

Run: `cd /c/apps/AppiA/youtube-ai-agent-recomendation && python -c "import sys; sys.path.append('src'); from telegram_notifier import TelegramNotifier; t = TelegramNotifier(); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/telegram_notifier.py
git commit -m "feat(7.2): add notify_competitor_alert to TelegramNotifier"
```

---

### Task 5: Add `_step_competitor_alerts()` to main.py

**Files:**
- Modify: `main.py` (add new function after `_step_competitors` ~line 247, and call it in `main()`)

**Step 1: Add the new pipeline step function**

After `_step_competitors` (line 247), add:

```python
def _step_competitor_alerts(analyzer, notifier):
    """Detecta videos de competidores con rendimiento excepcional y envía alertas."""
    log.info("PASO ALERTAS: Verificando videos virales de competidores...")

    with YouTubeDatabase() as db:
        recent_df = db.get_recent_competitor_videos(days=7)

    if recent_df.empty:
        log.info("  Sin videos recientes de competidores")
        return

    alert_count = 0

    for _, row in recent_df.iterrows():
        video_id = row['video_id']
        view_count = int(row['view_count']) if pd.notna(row['view_count']) else 0
        avg_views = float(row['competitor_avg_views']) if pd.notna(row['competitor_avg_views']) else 0

        if avg_views <= 0 or view_count <= 0:
            continue

        ratio = view_count / avg_views

        if ratio < 2.0:
            continue

        # Verificar que no se haya enviado ya esta alerta
        with YouTubeDatabase() as db:
            if db.is_alert_already_sent(video_id):
                continue

        channel_name = row.get('channel_name', '')
        video_title = row.get('title', '')
        video_type = row.get('video_type', '')

        log.info("  🚨 Video viral detectado: \"%s\" de %s (%.1fx promedio)",
                 video_title[:60], channel_name, ratio)

        # Análisis con Claude
        ai_analysis = analyzer.analyze_viral_competitor_video(
            video_title=video_title,
            channel_name=channel_name,
            view_count=view_count,
            competitor_avg_views=avg_views,
            ratio=ratio,
            video_type=video_type,
        )

        # Guardar alerta en BD
        alert_data = {
            'video_id': video_id,
            'channel_id': row['channel_id'],
            'channel_name': channel_name,
            'video_title': video_title,
            'view_count': view_count,
            'competitor_avg_views': avg_views,
            'ratio': ratio,
            'ai_analysis': ai_analysis,
            'notified': 0,
        }

        with YouTubeDatabase() as db:
            db.save_competitor_alert(alert_data)

        # Notificar por Telegram
        if notifier and notifier.is_enabled():
            sent = notifier.notify_competitor_alert(
                channel_name=channel_name,
                video_title=video_title,
                view_count=view_count,
                competitor_avg_views=avg_views,
                ratio=ratio,
                ai_analysis=ai_analysis,
            )
            if sent:
                log.info("  📨 Alerta enviada por Telegram")
                with YouTubeDatabase() as db:
                    db.conn.cursor().execute(
                        "UPDATE competitor_alerts SET notified = 1 WHERE video_id = %s",
                        (video_id,),
                    )
                    db.conn.commit()

        alert_count += 1

    if alert_count:
        log.info("  🚨 %d alerta(s) de competidores generadas", alert_count)
    else:
        log.info("  Sin videos virales de competidores detectados")
```

**Step 2: Add `import pandas as pd` at the top of main.py**

After the existing imports (line 17), add:

```python
import pandas as pd
```

**Step 3: Wire the step into `main()`**

In the `main()` function, after `_step_competitors(extractor, max_videos)` (line 320) and after the `notifier` and `analyzer` are created (line 328), add the call. The modified section of `main()` should be:

```python
        # Extraer datos de competidores (si están configurados)
        _step_competitors(extractor, max_videos)

        notifier = TelegramNotifier()
        if notifier.is_enabled():
            log.info("Telegram: notificaciones activadas")
        else:
            log.info("Telegram: no configurado (opcional — ver .env)")

        analyzer = AIAnalyzer(anthropic_api_key)

        # Alertas de videos virales de competidores (Mejora 7.2)
        _step_competitor_alerts(analyzer, notifier)

        _step3_analyze(analyzer, videos_df, channel_ids, notifier=notifier)
```

Note: the `analyzer` creation is moved BEFORE `_step_competitor_alerts` and `_step3_analyze` now receives `analyzer` directly instead of creating it. Since `_step3_analyze` already receives `analyzer` as param, this just reorders the creation.

**Step 4: Verify syntax**

Run: `cd /c/apps/AppiA/youtube-ai-agent-recomendation && python -c "import main; print('Import OK')"`
Expected: `Import OK`

**Step 5: Commit**

```bash
git add main.py
git commit -m "feat(7.2): add _step_competitor_alerts pipeline step"
```

---

### Task 6: Add "Alertas de Competidores" section to dashboard

**Files:**
- Modify: `dashboard.py` (add new section inside `show_competitor_analysis`, after the Content Gaps IA section ~line 4160)

**Step 1: Add alerts section to the competitor analysis page**

After the Content Gaps section (line 4160), before the closing of `show_competitor_analysis`, add:

```python
    # ── Sección 7: Alertas de Videos Virales (7.2) ────────────────────
    ui_section_divider("Alertas de Videos Virales")
    st.subheader("🚨 Videos de Competidores con Rendimiento Excepcional")
    st.caption("Videos que superaron 2x el promedio de vistas de su canal")

    try:
        db_alerts = YouTubeDatabase()
        alerts_df = db_alerts.get_competitor_alerts(limit=20)
        db_alerts.close()
    except Exception as e:
        st.error(f"Error al cargar alertas: {e}")
        alerts_df = pd.DataFrame()

    if alerts_df.empty:
        st.info(
            "**Sin alertas todavía.** Las alertas se generan automáticamente cuando "
            "`python main.py` detecta un video de competidor con >2x su promedio de vistas."
        )
    else:
        for _, alert_row in alerts_df.iterrows():
            ratio_val = float(alert_row['ratio']) if pd.notna(alert_row.get('ratio')) else 0
            views_val = int(alert_row['view_count']) if pd.notna(alert_row.get('view_count')) else 0
            avg_val = float(alert_row['competitor_avg_views']) if pd.notna(alert_row.get('competitor_avg_views')) else 0
            a_title = _esc(str(alert_row.get('video_title', ''))[:80])
            a_channel = _esc(str(alert_row.get('channel_name', '')))
            a_date = str(alert_row.get('created_at', ''))[:10]
            notified_icon = "📨" if alert_row.get('notified') else "🔕"

            # Color del badge por ratio
            if ratio_val >= 5:
                badge_color = "#ef4444"
            elif ratio_val >= 3:
                badge_color = "#f97316"
            else:
                badge_color = "#eab308"

            st.markdown(
                f"<div style='background:{COLORS['bg_secondary']};border-radius:8px;"
                f"padding:1rem;margin-bottom:0.75rem;"
                f"border-left:4px solid {badge_color}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:start'>"
                f"<div>"
                f"<div style='font-weight:600;color:{COLORS['text_primary']};font-size:0.95rem'>"
                f"{a_title}</div>"
                f"<div style='font-size:0.8rem;color:{COLORS['text_secondary']};margin-top:2px'>"
                f"🕵 {a_channel} · {a_date} {notified_icon}</div>"
                f"</div>"
                f"<div style='text-align:right;min-width:140px'>"
                f"<div style='font-weight:700;color:{badge_color};font-size:1.1rem'>"
                f"{ratio_val:.1f}x</div>"
                f"<div style='font-size:0.8rem;color:{COLORS['text_secondary']}'>"
                f"{views_val:,} vistas (prom: {avg_val:,.0f})</div>"
                f"</div></div></div>",
                unsafe_allow_html=True,
            )

            # Expandir análisis de Claude
            ai_text = alert_row.get('ai_analysis', '')
            if ai_text and str(ai_text).strip():
                with st.expander(f"🤖 Ver análisis — {a_title[:40]}..."):
                    st.markdown(str(ai_text))
```

**Step 2: Verify dashboard loads**

Run: `cd /c/apps/AppiA/youtube-ai-agent-recomendation && python -c "import ast; ast.parse(open('dashboard.py').read()); print('Syntax OK')"`
Expected: `Syntax OK`

**Step 3: Commit**

```bash
git add dashboard.py
git commit -m "feat(7.2): add competitor alerts section to dashboard"
```

---

### Task 7: End-to-end verification

**Step 1: Verify full import chain**

Run: `cd /c/apps/AppiA/youtube-ai-agent-recomendation && python -c "from dotenv import load_dotenv; load_dotenv(); import sys; sys.path.append('src'); from database import YouTubeDatabase; from ai_analyzer import AIAnalyzer; from telegram_notifier import TelegramNotifier; import main; print('All imports OK')"`
Expected: `All imports OK`

**Step 2: Verify DB table exists and queries run**

Run: `cd /c/apps/AppiA/youtube-ai-agent-recomendation && python -c "from dotenv import load_dotenv; load_dotenv(); import sys; sys.path.append('src'); from database import YouTubeDatabase; db = YouTubeDatabase(); print('Recent:', len(db.get_recent_competitor_videos(7))); print('Alerts:', len(db.get_competitor_alerts())); db.close()"`
Expected: shows counts (possibly 0 each — that's fine)

**Step 3: Verify dashboard syntax**

Run: `cd /c/apps/AppiA/youtube-ai-agent-recomendation && python -c "import ast; ast.parse(open('dashboard.py').read()); print('Dashboard OK')"`
Expected: `Dashboard OK`

**Step 4: Final commit tag**

```bash
git add -A
git commit -m "feat(7.2): complete competitor alerts — detection, AI analysis, Telegram, dashboard"
```
