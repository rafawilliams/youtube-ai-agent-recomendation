"""
Predictor de retención de audiencia (Mejora 12.1).

Predice avg_view_percentage (% del video visto) usando Random Forest.
Requiere datos de YouTube Analytics API (OAuth) para entrenar.

Extiende el patrón de ViralityPredictor/ViewPredictor con 3 features
adicionales específicas de retención:
  - has_hook_title   → título promete algo al inicio
  - is_tutorial      → contenido educativo (mayor retención esperada)
  - is_entertainment → contenido entretenimiento (menor retención en largos)
"""
import re
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

log = logging.getLogger(__name__)

MODEL_VERSION = "1"

PANAMA_TZ = pytz.timezone('America/Panama')

FEATURE_NAMES = [
    'hour', 'weekday_num', 'is_short', 'duration_seconds',
    'tags_count', 'title_length', 'title_has_number', 'title_has_question',
    'description_length', 'days_since_last_upload', 'channel_age_days',
    # Features específicas de retención
    'has_hook_title',
    'is_tutorial',
    'is_entertainment',
]

# Keywords para detección de tipos de contenido
_HOOK_PATTERNS = re.compile(
    r'(c[oó]mo|secreto|truco|tip[s\b]|hack|^\d+\s)',
    re.IGNORECASE,
)
_TUTORIAL_KEYWORDS = {
    'tutorial', 'curso', 'aprende', 'clase', 'lección', 'leccion',
    'cómo', 'como hacer', 'paso a paso', 'guía', 'guia',
    'how to', 'learn', 'guide', 'step by step',
}
_ENTERTAINMENT_KEYWORDS = {
    'vlog', 'reaccion', 'reacción', 'challenge', 'reto', 'prank',
    'broma', 'divertido', 'funny', 'comedy', 'sketch', 'react',
}


class RetentionPredictor:
    """
    Predice retención de audiencia (avg_view_percentage) usando Random Forest.
    Entrena con datos del canal que incluyan métricas de YouTube Analytics.
    """

    def __init__(self, min_samples: int = 10):
        self.min_samples = min_samples
        self._model = RandomForestRegressor(n_estimators=100, random_state=42)
        self._trained = False
        self._feature_importance: dict = {}
        self._train_metrics: dict = {}

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae las 14 features para predicción de retención."""
        feat = pd.DataFrame()

        # ── Temporales (idénticas a ViralityPredictor) ─────────────────
        dt = pd.to_datetime(df['published_at'], utc=True).dt.tz_convert(PANAMA_TZ)
        feat['hour'] = dt.dt.hour
        feat['weekday_num'] = dt.dt.weekday

        # ── Formato del video ──────────────────────────────────────────
        feat['is_short'] = df['is_short'].astype(int)
        feat['duration_seconds'] = pd.to_numeric(
            df['duration_seconds'], errors='coerce'
        ).fillna(0)

        # ── Contenido ─────────────────────────────────────────────────
        feat['tags_count'] = df['tags'].apply(
            lambda t: len(str(t).split(',')) if pd.notna(t) and str(t).strip() else 0
        )
        feat['title_length'] = df['title'].apply(
            lambda t: len(str(t)) if pd.notna(t) else 0
        )
        feat['title_has_number'] = df['title'].apply(
            lambda t: int(bool(re.search(r'\d', str(t)))) if pd.notna(t) else 0
        )
        feat['title_has_question'] = df['title'].apply(
            lambda t: int('?' in str(t)) if pd.notna(t) else 0
        )

        if 'description' in df.columns:
            feat['description_length'] = df['description'].apply(
                lambda d: len(str(d)) if pd.notna(d) and str(d).strip() else 0
            )
        else:
            feat['description_length'] = 0

        # ── Temporales de canal ────────────────────────────────────────
        dt_utc = pd.to_datetime(df['published_at'], utc=True)
        df_tmp = pd.DataFrame({'dt': dt_utc}, index=df.index).sort_values('dt')
        df_tmp['days_since_last'] = (
            df_tmp['dt'].diff().dt.total_seconds().div(86400).fillna(30)
        )
        feat['days_since_last_upload'] = df_tmp['days_since_last'].reindex(df.index).fillna(30)

        min_dt = dt_utc.min()
        feat['channel_age_days'] = (dt_utc - min_dt).dt.days

        # ── Features específicas de retención (NUEVAS) ─────────────────
        feat['has_hook_title'] = df['title'].apply(
            lambda t: int(bool(_HOOK_PATTERNS.search(str(t)))) if pd.notna(t) else 0
        )

        def _is_tutorial(row) -> int:
            text = (str(row.get('title', '')) + ' ' + str(row.get('tags', ''))).lower()
            return int(any(kw in text for kw in _TUTORIAL_KEYWORDS))

        def _is_entertainment(row) -> int:
            text = (str(row.get('title', '')) + ' ' + str(row.get('tags', ''))).lower()
            return int(any(kw in text for kw in _ENTERTAINMENT_KEYWORDS))

        feat['is_tutorial'] = df.apply(_is_tutorial, axis=1)
        feat['is_entertainment'] = df.apply(_is_entertainment, axis=1)

        return feat[FEATURE_NAMES]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, videos_df: pd.DataFrame) -> dict:
        """
        Entrena el modelo con videos que tengan avg_view_percentage.

        Args:
            videos_df: DataFrame con columnas de videos + avg_view_percentage
                       (join previo de videos con video_analytics).

        Returns:
            dict con {trained, samples, cv_splits, cv_mae, cv_r2, feature_importance}
        """
        df = videos_df.copy()

        # Filtrar videos con datos de retención válidos
        df['avg_view_percentage'] = pd.to_numeric(
            df.get('avg_view_percentage', pd.Series(dtype=float)), errors='coerce'
        )
        df = df[df['avg_view_percentage'] > 0].copy()

        if len(df) < self.min_samples:
            return {
                'trained': False,
                'reason': (
                    f'Se necesitan al menos {self.min_samples} videos con datos de '
                    f'retención (Analytics API). Disponibles: {len(df)}'
                ),
            }

        # Ordenar cronológicamente
        df = df.sort_values('published_at').reset_index(drop=True)

        X = self._extract_features(df)
        y = df['avg_view_percentage'].values  # 0-100, ya bounded

        # Validación cruzada temporal
        n_splits = min(5, max(2, len(df) // 5))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_maes, fold_r2s = [], []
        for train_idx, test_idx in tscv.split(X):
            x_tr, x_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            fold_model = RandomForestRegressor(n_estimators=100, random_state=42)
            fold_model.fit(x_tr, y_tr)
            y_pred = np.clip(fold_model.predict(x_te), 0, 100)

            fold_maes.append(float(mean_absolute_error(y_te, y_pred)))
            if len(y_te) > 1:
                fold_r2s.append(float(r2_score(y_te, y_pred)))

        cv_mae = float(np.mean(fold_maes))
        cv_r2 = float(np.mean(fold_r2s)) if fold_r2s else 0.0

        # Reentrenar con TODOS los datos
        self._model.fit(X, y)
        self._trained = True

        importances = self._model.feature_importances_
        self._feature_importance = {
            name: round(float(imp), 4)
            for name, imp in zip(FEATURE_NAMES, importances)
        }

        self._train_metrics = {
            'trained': True,
            'samples': len(df),
            'cv_splits': n_splits,
            'cv_mae': round(cv_mae, 2),
            'cv_r2': round(cv_r2, 3),
            'feature_importance': self._feature_importance,
        }
        return self._train_metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, videos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice retención para un DataFrame de videos.

        Returns:
            DataFrame original con columna 'predicted_retention' (0-100%).
        """
        result = videos_df.copy()

        if not self._trained:
            result['predicted_retention'] = 50.0
            return result

        X = self._extract_features(videos_df)
        preds = np.clip(self._model.predict(X), 0, 100)
        result['predicted_retention'] = np.round(preds, 1)
        return result

    def predict_single(
        self,
        hour: int = 12,
        weekday: int = 2,
        is_short: bool = False,
        duration_seconds: int = 600,
        tags_count: int = 10,
        title_length: int = 60,
        title_has_number: int = 0,
        title_has_question: int = 0,
        description_length: int = 500,
        days_since_last_upload: int = 7,
        channel_age_days: int = 365,
        has_hook_title: int = 0,
        is_tutorial: int = 0,
        is_entertainment: int = 0,
    ) -> float:
        """
        Predice retención para un escenario hipotético.

        Returns:
            Retención estimada (0-100%).
        """
        if not self._trained:
            return 50.0

        X = pd.DataFrame([{
            'hour': hour,
            'weekday_num': weekday,
            'is_short': int(is_short),
            'duration_seconds': duration_seconds,
            'tags_count': tags_count,
            'title_length': title_length,
            'title_has_number': title_has_number,
            'title_has_question': title_has_question,
            'description_length': description_length,
            'days_since_last_upload': days_since_last_upload,
            'channel_age_days': channel_age_days,
            'has_hook_title': has_hook_title,
            'is_tutorial': is_tutorial,
            'is_entertainment': is_entertainment,
        }])

        pred = self._model.predict(X)[0]
        return round(float(np.clip(pred, 0, 100)), 1)

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def is_trained(self) -> bool:
        return self._trained

    def get_train_metrics(self) -> dict:
        return self._train_metrics

    def get_feature_importance(self) -> dict:
        return self._feature_importance

    @staticmethod
    def score_color(retention: float) -> str:
        """Color según el nivel de retención."""
        if retention >= 50:
            return '#10B981'  # emerald — excelente
        if retention >= 30:
            return '#F59E0B'  # amber — aceptable
        return '#EF4444'      # red — baja

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Guarda el modelo entrenado."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'RetentionPredictor':
        """Carga un modelo guardado."""
        return joblib.load(path)
