"""
Carga de datos y preferencias del dashboard.
"""
import json
import logging
import os

import pandas as pd
import pymysql.err
import streamlit as st

from database import YouTubeDatabase

log = logging.getLogger(__name__)

PREFS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'dashboard_prefs.json')

_DEFAULT_PREFS = {
    'theme': 'dark',
    'widget_order': ['kpis', 'distribution', 'performance', 'export'],
}


@st.cache_data(ttl=300)
def load_data():
    """Carga los datos de la base de datos (cacheado 5 min)."""
    try:
        db = YouTubeDatabase()
        videos_df = db.get_all_videos()
        db.close()
        return videos_df
    except pymysql.err.Error as e:
        st.error(f"Error de base de datos cargando videos: {e}")
        log.error("Error cargando datos: %s", e)
        return pd.DataFrame()


def _load_prefs() -> dict:
    """Carga preferencias del dashboard desde config/dashboard_prefs.json."""
    try:
        if os.path.exists(PREFS_PATH):
            with open(PREFS_PATH, 'r', encoding='utf-8') as f:
                return {**_DEFAULT_PREFS, **json.load(f)}
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Error cargando preferencias, usando defaults: %s", e)
    return dict(_DEFAULT_PREFS)


def _save_prefs(prefs: dict):
    """Guarda preferencias del dashboard a config/dashboard_prefs.json."""
    os.makedirs(os.path.dirname(PREFS_PATH), exist_ok=True)
    with open(PREFS_PATH, 'w', encoding='utf-8') as f:
        json.dump(prefs, f, indent=2, ensure_ascii=False)
