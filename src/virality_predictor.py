"""
Modelo de ML para predicción de viralidad de videos de YouTube.
Score de 1-10 basado en características pre-publicación.
"""
import re
import logging
import pandas as pd
import numpy as np
import pytz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from typing import Optional

log = logging.getLogger(__name__)

PANAMA_TZ = pytz.timezone('America/Panama')

# Incrementar al agregar/quitar features (invalida modelos cacheados)
MODEL_VERSION = "2"

FEATURE_NAMES = [
    'hour',
    'weekday_num',
    'is_short',
    'duration_seconds',
    'tags_count',
    'title_length',
    'title_has_number',
    'title_has_question',
    'description_length',
    'days_since_last_upload',
    'channel_age_days',
]

WEEKDAY_LABELS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']


class ViralityPredictor:
    """
    Predice el potencial viral de un video usando Random Forest.
    Entrena con datos históricos del canal y puntúa 1-10.
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
        """Extrae las features de ML a partir del DataFrame de videos."""
        feat = pd.DataFrame()

        dt = pd.to_datetime(df['published_at'], utc=True).dt.tz_convert(PANAMA_TZ)
        feat['hour'] = dt.dt.hour
        feat['weekday_num'] = dt.dt.weekday

        feat['is_short'] = df['is_short'].astype(int)
        feat['duration_seconds'] = pd.to_numeric(df['duration_seconds'], errors='coerce').fillna(0)

        feat['tags_count'] = df['tags'].apply(
            lambda t: len(str(t).split(',')) if pd.notna(t) and str(t).strip() else 0
        )
        feat['title_length'] = df['title'].apply(lambda t: len(str(t)) if pd.notna(t) else 0)

        # Nuevas features — contenido del título
        feat['title_has_number'] = df['title'].apply(
            lambda t: int(bool(re.search(r'\d', str(t)))) if pd.notna(t) else 0
        )
        feat['title_has_question'] = df['title'].apply(
            lambda t: int('?' in str(t)) if pd.notna(t) else 0
        )

        # Longitud de la descripción
        if 'description' in df.columns:
            feat['description_length'] = df['description'].apply(
                lambda d: len(str(d)) if pd.notna(d) and str(d).strip() else 0
            )
        else:
            feat['description_length'] = 0

        # Días desde la subida anterior (frecuencia de publicación)
        dt_utc = pd.to_datetime(df['published_at'], utc=True)
        df_tmp = pd.DataFrame({'dt': dt_utc}, index=df.index).sort_values('dt')
        df_tmp['days_since_last'] = (
            df_tmp['dt'].diff().dt.total_seconds().div(86400).fillna(30)
        )
        feat['days_since_last_upload'] = df_tmp['days_since_last'].reindex(df.index).fillna(30)

        # Edad del canal en días (días desde el video más antiguo del dataset)
        min_dt = dt_utc.min()
        feat['channel_age_days'] = (dt_utc - min_dt).dt.days

        return feat[FEATURE_NAMES]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, videos_df: pd.DataFrame) -> dict:
        """
        Entrena el modelo con los videos del canal usando validación cruzada temporal.

        Usa TimeSeriesSplit para respetar el orden cronológico: entrena siempre
        con datos pasados y evalúa con datos futuros.

        Returns:
            dict con {trained, samples, cv_splits, cv_mae_percentile, feature_importance}
        """
        df = videos_df.copy()
        df = df[pd.to_numeric(df['view_count'], errors='coerce').fillna(0) > 0].copy()

        if len(df) < self.min_samples:
            return {
                'trained': False,
                'reason': f'Se necesitan al menos {self.min_samples} videos con vistas. Disponibles: {len(df)}'
            }

        # Ordenar cronológicamente — requisito de TimeSeriesSplit
        df = df.sort_values('published_at').reset_index(drop=True)

        X = self._extract_features(df)
        views = pd.to_numeric(df['view_count'], errors='coerce').fillna(0)

        # Target: percentil dentro del canal (0–100) — relativo, sin outliers
        y = views.rank(pct=True) * 100

        # Validación cruzada temporal: al menos 2 folds, máx 5, ≥5 muestras por fold
        n_splits = min(5, max(2, len(df) // 5))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_maes = []
        for train_idx, test_idx in tscv.split(X):
            x_tr, x_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            fold_model = RandomForestRegressor(n_estimators=100, random_state=42)
            fold_model.fit(x_tr, y_tr)
            fold_maes.append(float(mean_absolute_error(y_te, fold_model.predict(x_te))))

        cv_mae_percentile = float(np.mean(fold_maes))

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
            'cv_mae_percentile': round(cv_mae_percentile, 1),
            'feature_importance': self._feature_importance,
        }
        return self._train_metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, videos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el virality_score (1–10) para cada video del DataFrame.
        Si el modelo no está entrenado, devuelve score 5.0 para todos.
        """
        result = videos_df.copy()

        if not self._trained:
            result['virality_score'] = 5.0
            return result

        X = self._extract_features(result)
        percentiles = self._model.predict(X)
        scores = np.clip(percentiles / 10, 1.0, 10.0)
        result['virality_score'] = np.round(scores, 1)
        return result

    def predict_single(
        self,
        hour: int,
        weekday: int,
        is_short: bool,
        duration_seconds: int,
        tags_count: int,
        title_length: int,
        title_has_number: int = 0,
        title_has_question: int = 0,
        description_length: int = 500,
        days_since_last_upload: int = 7,
        channel_age_days: int = 365,
    ) -> float:
        """
        Predice el virality_score para un único video hipotético.

        Args:
            hour: Hora de publicación (0–23)
            weekday: Día de la semana (0=lunes … 6=domingo)
            is_short: True si es Short
            duration_seconds: Duración en segundos
            tags_count: Cantidad de tags
            title_length: Longitud del título en caracteres
            title_has_number: 1 si el título contiene un número
            title_has_question: 1 si el título contiene '?'
            description_length: Longitud de la descripción en caracteres
            days_since_last_upload: Días desde la última subida al canal
            channel_age_days: Antigüedad del canal en días

        Returns:
            Score de viralidad entre 1.0 y 10.0
        """
        if not self._trained:
            return 5.0

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
        }])[FEATURE_NAMES]

        percentile = self._model.predict(X)[0]
        return float(np.clip(round(percentile / 10, 1), 1.0, 10.0))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def is_trained(self) -> bool:
        return self._trained

    def get_train_metrics(self) -> dict:
        return self._train_metrics

    def get_feature_importance(self) -> dict:
        return self._feature_importance

    def score_color(self, score: float) -> str:
        """Devuelve un color hex según el score."""
        if score >= 8:
            return '#10B981'   # emerald (success)
        elif score >= 5:
            return '#F59E0B'   # amber (warning)
        else:
            return '#EF4444'   # red (danger)

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serializa el predictor completo (modelo + métricas) a disco."""
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'ViralityPredictor':
        """Carga un predictor previamente serializado desde disco."""
        import joblib
        return joblib.load(path)
