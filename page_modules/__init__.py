"""
Módulos de páginas del dashboard.

Cada página se implementa en su propio archivo para mantener el código organizado.
Las utilidades compartidas están en pages/shared/.
"""
from page_modules.overview import show_overview
from page_modules.performance import show_performance_analysis
from page_modules.predictions import show_virality_prediction, show_view_prediction
from page_modules.history import show_metrics_history
from page_modules.recommendations import show_recommendations, show_generate_new_recommendation
from page_modules.analytics import show_advanced_analytics, show_trends_analysis
from page_modules.content import show_content_analysis, show_channel_comparison
from page_modules.planning import show_weekly_plan
from page_modules.health import show_channel_health
from page_modules.cadence import show_cadence_analysis
from page_modules.competitors import show_competitor_analysis
from page_modules.ml_predictions import (
    show_retention_prediction, show_late_bloomer_analysis,
    show_cannibalization_analysis,
)
from page_modules.temporal import show_temporal_comparison
from page_modules.series import show_series_analysis
from page_modules.revenue import show_revenue_analysis
from page_modules.thumbnails import show_thumbnail_gallery, show_thumbnail_comparator
from page_modules.comments import show_comments_analysis
