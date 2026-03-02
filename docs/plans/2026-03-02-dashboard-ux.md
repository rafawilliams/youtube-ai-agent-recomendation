# Dashboard UX Improvements (15.1–15.4) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add four UX features to the Streamlit dashboard: customizable widget layout on Overview, presentation mode, temporal comparison with deltas, and dark/light theme toggle.

**Architecture:** All changes live in `dashboard.py`. Theme state (dark/light) and widget layout are stored in `st.session_state` and persisted to `config/dashboard_prefs.json`. Presentation mode uses conditional CSS injection + `st.set_page_config` sidebar state. Temporal comparison is a standalone section added to the Overview page with dual date pickers and delta cards.

**Tech Stack:** Python, Streamlit, Plotly, pandas, JSON (config persistence)

---

### Task 1: Add preferences persistence layer (config/dashboard_prefs.json)

**Files:**
- Modify: `dashboard.py` (add helper functions after `load_data`, ~line 774)

**Step 1: Add load/save prefs functions**

After `load_data()` (line 774), add:

```python
# ─── Dashboard Preferences (Mejora 15.x) ──────────────────────────────────

PREFS_PATH = os.path.join(os.path.dirname(__file__), 'config', 'dashboard_prefs.json')

_DEFAULT_PREFS = {
    'theme': 'dark',
    'widget_order': ['kpis', 'distribution', 'performance', 'export'],
}


def _load_prefs() -> dict:
    """Carga preferencias del dashboard desde config/dashboard_prefs.json."""
    try:
        if os.path.exists(PREFS_PATH):
            with open(PREFS_PATH, 'r', encoding='utf-8') as f:
                return {**_DEFAULT_PREFS, **json.load(f)}
    except Exception:
        pass
    return dict(_DEFAULT_PREFS)


def _save_prefs(prefs: dict):
    """Guarda preferencias del dashboard a config/dashboard_prefs.json."""
    os.makedirs(os.path.dirname(PREFS_PATH), exist_ok=True)
    with open(PREFS_PATH, 'w', encoding='utf-8') as f:
        json.dump(prefs, f, indent=2, ensure_ascii=False)
```

**Step 2: Initialize prefs in session_state**

In `main()`, right after `st.sidebar.markdown` brand block (~line 4615), add:

```python
    # ── Cargar preferencias (15.x) ─────────────────────────────────────
    if 'prefs' not in st.session_state:
        st.session_state.prefs = _load_prefs()
```

**Step 3: Verify syntax**

Run: `.venv/Scripts/python.exe -c "import ast; ast.parse(open('dashboard.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add dashboard.py
git commit -m "feat(15.x): add dashboard preferences persistence layer"
```

---

### Task 2: Dark/Light Mode Toggle (15.4)

This is the foundational task — theme toggle affects all other features.

**Files:**
- Modify: `dashboard.py` (CSS root variables, sidebar toggle, theme-aware COLORS)

**Step 1: Add light theme CSS variables**

Replace the current static `:root { ... }` CSS block (lines 343–366) with a dynamic version. Change the CSS injection from a static string to a function-generated one. Right after `_save_prefs`, add:

```python
LIGHT_COLORS = {
    "primary":        "#6366F1",
    "primary_light":  "#818CF8",
    "primary_dark":   "#4F46E5",
    "secondary":      "#EC4899",
    "secondary_light":"#F472B6",
    "success":        "#10B981",
    "warning":        "#F59E0B",
    "danger":         "#EF4444",
    "info":           "#3B82F6",
    "bg_primary":     "#F8FAFC",
    "bg_secondary":   "#FFFFFF",
    "bg_tertiary":    "#E2E8F0",
    "text_primary":   "#0F172A",
    "text_secondary": "#475569",
    "text_muted":     "#94A3B8",
    "border":         "#CBD5E1",
}
```

**Step 2: Make COLORS dynamic**

Replace the static `COLORS = { ... }` dict definition (lines 55-78) with:

```python
# ─── Design System — Colores unificados ─────────────────────────────────────
_DARK_COLORS = {
    "primary":        "#6366F1",
    "primary_light":  "#818CF8",
    "primary_dark":   "#4F46E5",
    "secondary":      "#EC4899",
    "secondary_light":"#F472B6",
    "success":        "#10B981",
    "warning":        "#F59E0B",
    "danger":         "#EF4444",
    "info":           "#3B82F6",
    "bg_primary":     "#0F172A",
    "bg_secondary":   "#1E293B",
    "bg_tertiary":    "#334155",
    "text_primary":   "#F8FAFC",
    "text_secondary": "#94A3B8",
    "text_muted":     "#64748B",
    "border":         "#334155",
}

_LIGHT_COLORS = {
    "primary":        "#6366F1",
    "primary_light":  "#818CF8",
    "primary_dark":   "#4F46E5",
    "secondary":      "#EC4899",
    "secondary_light":"#F472B6",
    "success":        "#10B981",
    "warning":        "#F59E0B",
    "danger":         "#EF4444",
    "info":           "#3B82F6",
    "bg_primary":     "#F8FAFC",
    "bg_secondary":   "#FFFFFF",
    "bg_tertiary":    "#E2E8F0",
    "text_primary":   "#0F172A",
    "text_secondary": "#475569",
    "text_muted":     "#94A3B8",
    "border":         "#CBD5E1",
}


def _get_colors() -> dict:
    """Retorna el dict de colores según el tema activo en session_state."""
    theme = st.session_state.get('prefs', {}).get('theme', 'dark')
    return _LIGHT_COLORS if theme == 'light' else _DARK_COLORS


# Backwards-compatible: COLORS se usa en toda la app. Se inicializa con dark
# y se reasigna en main() después de cargar las preferencias.
COLORS = dict(_DARK_COLORS)
```

**Step 3: Generate CSS with dynamic variables**

Replace the static `st.markdown("""<style>...""")` block (lines 337–694) by wrapping it in a function. Move the CSS injection out of module-level and into `main()`.

Add this function after the `_get_colors()` definition:

```python
def _inject_theme_css():
    """Inyecta CSS con variables del tema actual."""
    c = _get_colors()
    # Plotly template font color
    _yt_template.layout.font.color = c['text_primary']

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {{
    --primary: {c['primary']};
    --primary-light: {c['primary_light']};
    --primary-dark: {c['primary_dark']};
    --secondary: {c['secondary']};
    --success: {c['success']};
    --warning: {c['warning']};
    --danger: {c['danger']};
    --info: {c['info']};
    --bg-primary: {c['bg_primary']};
    --bg-secondary: {c['bg_secondary']};
    --bg-tertiary: {c['bg_tertiary']};
    --text-primary: {c['text_primary']};
    --text-secondary: {c['text_secondary']};
    --text-muted: {c['text_muted']};
    --border: {c['border']};
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.3), 0 2px 4px -2px rgba(0,0,0,0.3);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.4), 0 4px 6px -4px rgba(0,0,0,0.3);
    --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}}
</style>
""", unsafe_allow_html=True)
```

Note: Keep the rest of the CSS (typography, sidebar, cards, etc.) in the original static block — only the `:root` variable block becomes dynamic. So the original CSS block should have its `:root { ... }` section **removed** and replaced with a comment like `/* :root injected dynamically by _inject_theme_css() */`.

**Step 4: Add theme toggle to sidebar**

In `main()`, after the prefs initialization and before the navigation radio, add:

```python
    # ── Sidebar: Theme toggle (15.4) ───────────────────────────────────
    theme_label = "☀ Claro" if st.session_state.prefs['theme'] == 'dark' else "🌙 Oscuro"
    if st.sidebar.button(theme_label, key="btn_theme_toggle", help="Cambiar tema claro/oscuro"):
        new_theme = 'light' if st.session_state.prefs['theme'] == 'dark' else 'dark'
        st.session_state.prefs['theme'] = new_theme
        _save_prefs(st.session_state.prefs)
        st.rerun()
```

**Step 5: Update COLORS global in main()**

In `main()`, after prefs initialization:

```python
    # Actualizar colores globales según tema
    global COLORS
    COLORS = _get_colors()
    _inject_theme_css()
```

**Step 6: Verify syntax**

Run: `.venv/Scripts/python.exe -c "import ast; ast.parse(open('dashboard.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

**Step 7: Commit**

```bash
git add dashboard.py
git commit -m "feat(15.4): add dark/light mode toggle with persistent preference"
```

---

### Task 3: Presentation Mode (15.2)

**Files:**
- Modify: `dashboard.py` (sidebar toggle + CSS for presentation mode)

**Step 1: Add presentation mode toggle in sidebar**

In `main()`, after the theme toggle button, add:

```python
    # ── Sidebar: Presentation mode (15.2) ──────────────────────────────
    pres_mode = st.session_state.get('presentation_mode', False)
    pres_label = "🖥 Salir de Presentación" if pres_mode else "🖥 Modo Presentación"
    if st.sidebar.button(pres_label, key="btn_presentation", help="Modo limpio para reuniones"):
        st.session_state.presentation_mode = not pres_mode
        st.rerun()
```

**Step 2: Inject presentation mode CSS**

In `_inject_theme_css()`, after the `:root` CSS block, conditionally add presentation CSS:

```python
    # Presentation mode CSS (15.2)
    if st.session_state.get('presentation_mode', False):
        st.markdown("""
<style>
/* ═══ Presentation Mode ═══ */
section[data-testid="stSidebar"] { display: none !important; }
.main .block-container {
    max-width: 1800px !important;
    padding-top: 1rem !important;
}
h1 { font-size: 2.5rem !important; }
h2 { font-size: 2rem !important; }
h3 { font-size: 1.5rem !important; }
.metric-card .metric-value { font-size: 2.5rem !important; }
.metric-card .metric-label { font-size: 0.9rem !important; }
[data-testid="stMetricValue"] { font-size: 2.2rem !important; }
.page-title-bar .page-title { font-size: 2rem !important; }
.page-title-bar .page-icon { font-size: 2.5rem; width: 64px; height: 64px; }
/* Hide export/action buttons in presentation */
.stDownloadButton, [data-testid="stForm"] { display: none !important; }
</style>
""", unsafe_allow_html=True)
```

**Step 3: Verify syntax**

Run: `.venv/Scripts/python.exe -c "import ast; ast.parse(open('dashboard.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add dashboard.py
git commit -m "feat(15.2): add presentation mode — hides sidebar, larger fonts"
```

---

### Task 4: Customizable Widget Layout on Overview (15.1)

**Files:**
- Modify: `dashboard.py` — refactor `show_overview()` into named widget functions + selectbox ordering

**Step 1: Extract overview sections into widget functions**

Right before `show_overview()`, add four widget functions:

```python
# ─── Overview Widgets (Mejora 15.1) ────────────────────────────────────────

_WIDGET_REGISTRY = {}


def _register_widget(key, label):
    """Decorator para registrar widgets del overview."""
    def decorator(fn):
        _WIDGET_REGISTRY[key] = {'fn': fn, 'label': label}
        return fn
    return decorator


@_register_widget('kpis', '📊 KPIs principales')
def _widget_kpis(df, channel_name, subscriber_count):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        subs_display = f"{subscriber_count:,}" if subscriber_count else "No disponible"
        st.metric("👥 Suscriptores", subs_display)
    with col2:
        st.metric("Total Videos", f"{len(df)}")
    with col3:
        st.metric("Vistas Totales", f"{df['view_count'].sum():,.0f}")
    with col4:
        st.metric("Promedio por Video", f"{df['view_count'].mean():,.0f}")
    with col5:
        st.metric("Engagement Rate", f"{df['engagement_rate'].mean():.2f}%")


@_register_widget('distribution', '🎬 Distribución de Contenido')
def _widget_distribution(df, channel_name, subscriber_count):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎬 Distribución de Contenido")
        type_counts = df['video_type'].value_counts()
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Shorts vs Videos Largos",
            color_discrete_sequence=VIDEO_TYPE_SEQUENCE,
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.subheader("📈 Performance por Tipo")
        perf = df.groupby('video_type').agg(
            {'view_count': 'mean', 'engagement_rate': 'mean'}
        ).reset_index()
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name='Vistas Promedio',
            x=perf['video_type'], y=perf['view_count'],
            marker_color=COLORS['primary'],
        ))
        fig_bar.update_layout(title="Vistas Promedio por Tipo", yaxis_title="Vistas", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)


@_register_widget('performance', '🏆 Top Videos')
def _widget_performance(df, channel_name, subscriber_count):
    st.subheader("🏆 Top 10 Videos con Mejor Performance")
    top10 = df.nlargest(10, 'view_count')[
        ['title', 'video_type', 'view_count', 'engagement_rate', 'published_at']
    ].copy()
    top10['published_at'] = top10['published_at'].dt.strftime('%Y-%m-%d')
    st.dataframe(
        top10.rename(columns={
            'title': 'Título', 'video_type': 'Tipo', 'view_count': 'Vistas',
            'engagement_rate': 'Engagement %', 'published_at': 'Publicado',
        }),
        use_container_width=True, hide_index=True,
    )


@_register_widget('export', '📥 Exportar datos')
def _widget_export(df, channel_name, subscriber_count):
    st.subheader("📥 Exportar datos")
    export_cols = ['title', 'video_type', 'published_at', 'view_count',
                   'like_count', 'comment_count', 'engagement_rate', 'duration_seconds',
                   'tags', 'description']
    export_df = df[[c for c in export_cols if c in df.columns]].copy()
    if 'published_at' in export_df.columns:
        export_df['published_at'] = export_df['published_at'].dt.strftime('%Y-%m-%d %H:%M')

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        export_df.to_excel(writer, sheet_name='Videos', index=False)
        summary = pd.DataFrame({
            'Métrica': ['Canal', 'Total videos', 'Vistas totales', 'Vistas promedio', 'Engagement %'],
            'Valor': [channel_name, len(df), int(df['view_count'].sum()),
                      int(df['view_count'].mean()), round(float(df['engagement_rate'].mean()), 2)]
        })
        summary.to_excel(writer, sheet_name='Resumen', index=False)

    st.download_button(
        label="📥 Descargar Excel (.xlsx)", data=buffer.getvalue(),
        file_name=f"{channel_name}_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if st.button("📊 Exportar a Google Sheets", key="btn_gsheets_metrics"):
        try:
            with st.spinner("Creando Google Sheet..."):
                gsheets = GoogleSheetsExporter()
                result = gsheets.export_video_metrics(df, channel_name=channel_name)
            st.success("✅ Google Sheet creado")
            st.markdown(f"[📊 Abrir Google Sheet]({result['spreadsheet_url']})")
        except FileNotFoundError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
```

**Step 2: Rewrite `show_overview()` to use widget registry + configurable order**

Replace the entire `show_overview()` function body with:

```python
def show_overview(df):
    """Muestra el resumen general del canal con widgets personalizables (15.1)."""
    ui_page_header("📊", "Resumen General", "Vista rápida de las métricas principales del canal")

    if df.empty:
        st.warning("No hay datos disponibles. Ejecuta primero el script main.py para extraer datos.")
        return

    channel_name = df['channel_title'].iloc[0] if pd.notna(df['channel_title'].iloc[0]) else "Canal"
    raw_subs = df['subscriber_count'].iloc[0] if 'subscriber_count' in df.columns else None
    subscriber_count = int(raw_subs) if raw_subs is not None and pd.notna(raw_subs) else None

    st.subheader(f"📺 {channel_name}")

    # Widget order from prefs
    prefs = st.session_state.get('prefs', _DEFAULT_PREFS)
    widget_order = prefs.get('widget_order', _DEFAULT_PREFS['widget_order'])
    # Ensure all registered widgets are included
    all_keys = list(_WIDGET_REGISTRY.keys())
    widget_order = [k for k in widget_order if k in _WIDGET_REGISTRY] + \
                   [k for k in all_keys if k not in widget_order]

    # Layout customizer (collapsed expander)
    with st.expander("⚙ Personalizar layout", expanded=False):
        st.caption("Selecciona el orden de las secciones del resumen")
        new_order = []
        for i, key in enumerate(widget_order):
            label = _WIDGET_REGISTRY[key]['label']
            options = [_WIDGET_REGISTRY[k]['label'] for k in widget_order]
            selected = st.selectbox(
                f"Posición {i + 1}", options, index=i,
                key=f"widget_pos_{i}",
            )
            # Map label back to key
            sel_key = next(k for k, v in _WIDGET_REGISTRY.items() if v['label'] == selected)
            new_order.append(sel_key)

        if st.button("💾 Guardar layout", key="btn_save_layout"):
            # Deduplicate preserving order
            seen = set()
            deduped = []
            for k in new_order:
                if k not in seen:
                    deduped.append(k)
                    seen.add(k)
            st.session_state.prefs['widget_order'] = deduped
            _save_prefs(st.session_state.prefs)
            st.success("Layout guardado")
            st.rerun()

    ui_section_divider()

    # Render widgets in order
    for key in widget_order:
        if key in _WIDGET_REGISTRY:
            _WIDGET_REGISTRY[key]['fn'](df, channel_name, subscriber_count)
            ui_section_divider()
```

**Step 3: Verify syntax**

Run: `.venv/Scripts/python.exe -c "import ast; ast.parse(open('dashboard.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add dashboard.py
git commit -m "feat(15.1): customizable widget layout on Overview page"
```

---

### Task 5: Temporal Comparison (15.3)

**Files:**
- Modify: `dashboard.py` — add a new page + page constant + navigation entry

**Step 1: Add page constant**

After `PAGE_CANNIBALIZATION` (line 50), add:

```python
PAGE_TEMPORAL = "📆 Comparador Temporal"
```

**Step 2: Add the page to navigation radio**

In `main()`, inside the `st.sidebar.radio` list, add `PAGE_TEMPORAL` after `PAGE_CADENCE`:

```python
            PAGE_CADENCE,
            PAGE_TEMPORAL,
            PAGE_COMPETITORS,
```

**Step 3: Add route in page router**

In the page router section of `main()`, add before `elif page == "🎯 Recomendaciones"`:

```python
    elif page == PAGE_TEMPORAL:
        show_temporal_comparison(df)
```

**Step 4: Add the `show_temporal_comparison` function**

Add before the `main()` function:

```python
# ══════════════════════════════════════════════════════════════════════
# Comparador Temporal (Mejora 15.3)
# ══════════════════════════════════════════════════════════════════════

def show_temporal_comparison(df: pd.DataFrame):
    """Compara métricas entre dos rangos de fecha con deltas visuales."""
    ui_page_header(
        "📆", "Comparador Temporal",
        "Compara métricas entre dos períodos para identificar tendencias"
    )

    if df.empty:
        st.warning("No hay datos disponibles.")
        return

    # ── Presets rápidos ────────────────────────────────────────────────
    today = datetime.now().date()
    presets = {
        "Este mes vs anterior": (
            today.replace(day=1),
            today,
            (today.replace(day=1) - timedelta(days=1)).replace(day=1),
            today.replace(day=1) - timedelta(days=1),
        ),
        "Últimos 30d vs 30d anteriores": (
            today - timedelta(days=30), today,
            today - timedelta(days=60), today - timedelta(days=31),
        ),
        "Últimos 90d vs 90d anteriores": (
            today - timedelta(days=90), today,
            today - timedelta(days=180), today - timedelta(days=91),
        ),
        "Personalizado": None,
    }

    preset = st.selectbox("⏱ Período", list(presets.keys()), key="temporal_preset")

    if preset == "Personalizado":
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Período A (actual)**")
            a_start = st.date_input("Desde", today - timedelta(days=30), key="ta_start")
            a_end = st.date_input("Hasta", today, key="ta_end")
        with col_b:
            st.markdown("**Período B (comparar)**")
            b_start = st.date_input("Desde", today - timedelta(days=60), key="tb_start")
            b_end = st.date_input("Hasta", today - timedelta(days=31), key="tb_end")
    else:
        a_start, a_end, b_start, b_end = presets[preset]

    ui_section_divider()

    # ── Filtrar datos ──────────────────────────────────────────────────
    df_ts = df.copy()
    df_ts['pub_date'] = df_ts['published_at'].dt.date

    period_a = df_ts[(df_ts['pub_date'] >= a_start) & (df_ts['pub_date'] <= a_end)]
    period_b = df_ts[(df_ts['pub_date'] >= b_start) & (df_ts['pub_date'] <= b_end)]

    if period_a.empty and period_b.empty:
        st.info("No hay videos en ninguno de los dos períodos seleccionados.")
        return

    # ── KPIs comparativos ──────────────────────────────────────────────
    st.subheader("📊 Comparación de Métricas")

    def _calc_stats(period_df):
        if period_df.empty:
            return {'videos': 0, 'views': 0, 'avg_views': 0, 'engagement': 0, 'total_likes': 0}
        return {
            'videos': len(period_df),
            'views': int(period_df['view_count'].sum()),
            'avg_views': float(period_df['view_count'].mean()),
            'engagement': float(period_df['engagement_rate'].mean()),
            'total_likes': int(period_df['like_count'].sum()) if 'like_count' in period_df.columns else 0,
        }

    stats_a = _calc_stats(period_a)
    stats_b = _calc_stats(period_b)

    metrics = [
        ("🎬", "Videos publicados", stats_a['videos'], stats_b['videos'], ""),
        ("👀", "Vistas totales", stats_a['views'], stats_b['views'], ","),
        ("📊", "Vistas promedio", stats_a['avg_views'], stats_b['avg_views'], ",.0f"),
        ("💬", "Engagement %", stats_a['engagement'], stats_b['engagement'], ".2f"),
        ("👍", "Likes totales", stats_a['total_likes'], stats_b['total_likes'], ","),
    ]

    cols = st.columns(len(metrics))
    for col, (icon, label, val_a, val_b, fmt) in zip(cols, metrics):
        with col:
            if val_b and val_b != 0:
                delta_pct = ((val_a - val_b) / val_b) * 100
                delta_str = f"{'▲' if delta_pct >= 0 else '▼'} {abs(delta_pct):.1f}%"
                delta_type = "positive" if delta_pct >= 0 else "negative"
            else:
                delta_str = "—"
                delta_type = "neutral"

            display_val = f"{val_a:{fmt}}" if fmt else str(val_a)
            st.markdown(
                ui_metric_card(icon, label, display_val, delta_str, delta_type),
                unsafe_allow_html=True,
            )

    # ── Leyenda de períodos ────────────────────────────────────────────
    st.caption(
        f"**Período A:** {a_start} → {a_end} ({stats_a['videos']} videos) · "
        f"**Período B:** {b_start} → {b_end} ({stats_b['videos']} videos)"
    )

    ui_section_divider()

    # ── Gráfico comparativo ────────────────────────────────────────────
    st.subheader("📈 Evolución Comparada")

    if not period_a.empty or not period_b.empty:
        chart_data = []
        for _, row in period_a.iterrows():
            chart_data.append({
                'Fecha': row['published_at'],
                'Vistas': row['view_count'],
                'Período': f'A ({a_start} → {a_end})',
            })
        for _, row in period_b.iterrows():
            chart_data.append({
                'Fecha': row['published_at'],
                'Vistas': row['view_count'],
                'Período': f'B ({b_start} → {b_end})',
            })

        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            fig = px.scatter(
                chart_df, x='Fecha', y='Vistas', color='Período',
                color_discrete_sequence=[COLORS['primary'], COLORS['secondary']],
                title="Vistas por video — Período A vs B",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # ── Tabla detalle ──────────────────────────────────────────────────
    ui_section_divider()
    st.subheader("📋 Detalle por Tipo de Video")

    detail_rows = []
    for label_p, p_df in [("Período A", period_a), ("Período B", period_b)]:
        if p_df.empty:
            continue
        for vtype in p_df['video_type'].unique():
            vt_df = p_df[p_df['video_type'] == vtype]
            detail_rows.append({
                'Período': label_p,
                'Tipo': vtype,
                'Videos': len(vt_df),
                'Vistas prom.': int(vt_df['view_count'].mean()),
                'Engagement %': round(float(vt_df['engagement_rate'].mean()), 2),
            })

    if detail_rows:
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
```

**Step 5: Verify syntax**

Run: `.venv/Scripts/python.exe -c "import ast; ast.parse(open('dashboard.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add dashboard.py
git commit -m "feat(15.3): add temporal comparison page with dual date ranges and delta cards"
```

---

### Task 6: End-to-end verification

**Step 1: Verify full import chain**

Run: `.venv/Scripts/python.exe -c "from dotenv import load_dotenv; load_dotenv(); import ast; ast.parse(open('dashboard.py', encoding='utf-8').read()); print('Syntax OK')" && .venv/Scripts/python.exe -c "import dashboard; print('Import OK')"`

**Step 2: Verify config dir created on save**

Run: `.venv/Scripts/python.exe -c "import sys; sys.path.append('src'); from dotenv import load_dotenv; load_dotenv(); import json, os; path='config/dashboard_prefs.json'; os.makedirs('config', exist_ok=True); json.dump({'theme':'dark','widget_order':['kpis','distribution','performance','export']}, open(path,'w')); print(json.load(open(path))); os.remove(path); print('Prefs OK')"`

**Step 3: Final commit**

```bash
git add dashboard.py .streamlit/config.toml
git commit -m "feat(15.1-15.4): complete dashboard UX improvements"
```
