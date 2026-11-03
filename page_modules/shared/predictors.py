"""
Factory de predictores ML con cache en disco.
"""
import json
import logging
import os

import pandas as pd
import streamlit as st

from virality_predictor import ViralityPredictor
from virality_predictor import MODEL_VERSION as VIRALITY_MODEL_VERSION
from view_predictor import ViewPredictor
from view_predictor import MODEL_VERSION as VIEW_MODEL_VERSION
from retention_predictor import RetentionPredictor
from retention_predictor import MODEL_VERSION as RETENTION_MODEL_VERSION
from page_modules.shared.constants import MODELS_DIR

log = logging.getLogger(__name__)


def _get_cached_predictor(predictor_cls, prefix: str, model_version: str,
                          train_df: pd.DataFrame, channel_id: str,
                          count_col: str = 'view_count', min_samples: int = 10):
    """Factory genérico: carga modelo de cache o entrena uno nuevo."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_id = channel_id.replace('/', '_')
    model_path = os.path.join(MODELS_DIR, f'{prefix}_{safe_id}.pkl')
    meta_path = os.path.join(MODELS_DIR, f'{prefix}_{safe_id}.meta.json')

    # Intentar cache
    if os.path.exists(model_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get('model_version') == model_version:
                predictor = predictor_cls.load(model_path)
                log.debug("Modelo %s cargado de cache para %s", prefix, channel_id)
                return predictor, True
        except Exception:
            log.debug("Cache inválido para %s/%s, reentrenando", prefix, channel_id)

    # Entrenar nuevo
    predictor = predictor_cls(min_samples=min_samples)
    train_result = predictor.train(train_df)

    if train_result.get('trained'):
        predictor.save(model_path)
        with open(meta_path, 'w') as f:
            json.dump({'model_version': model_version, **train_result}, f, indent=2)
        log.debug("Modelo %s entrenado y guardado para %s", prefix, channel_id)

    return predictor, False


def _get_virality_predictor(df: pd.DataFrame, channel_id: str):
    """Wrapper: ViralityPredictor con cache."""
    return _get_cached_predictor(
        ViralityPredictor, 'virality', VIRALITY_MODEL_VERSION, df, channel_id)


def _get_view_predictor(df: pd.DataFrame, channel_id: str):
    """Wrapper: ViewPredictor con cache."""
    return _get_cached_predictor(
        ViewPredictor, 'views', VIEW_MODEL_VERSION, df, channel_id)


def _get_retention_predictor(df: pd.DataFrame, analytics_df: pd.DataFrame, channel_id: str):
    """Wrapper: RetentionPredictor con cache (merge con analytics)."""
    merged = df.merge(
        analytics_df[['video_id', 'avg_view_percentage']], on='video_id', how='left'
    )
    return _get_cached_predictor(
        RetentionPredictor, 'retention', RETENTION_MODEL_VERSION, merged, channel_id,
        count_col='avg_view_percentage', min_samples=5)


def _detect_channel_language(db, channel_id: str) -> str:
    """Detecta el idioma predominante de un canal a partir de los títulos.

    Retorna 'en' para inglés, 'es' para español (por defecto).
    """
    try:
        cursor = db.conn.cursor()
        cursor.execute(
            "SELECT title FROM videos WHERE channel_id = %s ORDER BY published_at DESC LIMIT 20",
            (channel_id,),
        )
        rows = cursor.fetchall()
        if not rows:
            return 'es'

        titles_text = ' '.join(r['title'] for r in rows if r.get('title')).lower()
        words = titles_text.split()

        en_words = {'the', 'and', 'for', 'that', 'with', 'from', 'this', 'was',
                    'are', 'but', 'not', 'you', 'she', 'her', 'his', 'they',
                    'will', 'have', 'has', 'had', 'been', 'one', 'when', 'who',
                    'what', 'why', 'how', 'my', 'your', 'it', 'of', 'in', 'to',
                    'is', 'on', 'at', 'an', 'we', 'our', 'no', 'if', 'he'}
        es_words = {'de', 'que', 'en', 'los', 'las', 'por', 'para', 'como',
                    'con', 'del', 'una', 'uno', 'ese', 'esta', 'esto', 'mas',
                    'pero', 'sin', 'sobre', 'entre', 'hasta', 'desde', 'puede',
                    'tiene', 'fue', 'son', 'hay', 'ser', 'porque', 'cuando',
                    'todo', 'sus', 'nos', 'cual', 'donde', 'quien', 'ya'}
        pt_words = {'de', 'que', 'em', 'os', 'as', 'por', 'para', 'como',
                    'com', 'uma', 'nao', 'mais', 'muito', 'voce', 'isso',
                    'seu', 'sua', 'ele', 'ela', 'nos', 'tem', 'foi', 'sao'}

        en_count = sum(1 for w in words if w in en_words)
        es_count = sum(1 for w in words if w in es_words)
        pt_count = sum(1 for w in words if w in pt_words)

        scores = {'en': en_count, 'es': es_count, 'pt': pt_count}
        best = max(scores, key=scores.get)
        if scores[best] < 3:
            return 'es'
        return best
    except Exception:
        return 'es'
