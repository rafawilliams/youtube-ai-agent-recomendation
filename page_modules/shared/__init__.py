"""Utilidades compartidas del dashboard."""
from page_modules.shared.constants import *  # noqa: F401,F403
from page_modules.shared.theme import _get_colors, _inject_theme_css
from page_modules.shared.ui import _esc, ui_page_header, ui_section_header, ui_section_divider, ui_metric_card
from page_modules.shared.data import load_data, _load_prefs, _save_prefs
from page_modules.shared.predictors import (
    _get_cached_predictor, _get_virality_predictor, _get_view_predictor,
    _get_retention_predictor, _detect_channel_language,
)
