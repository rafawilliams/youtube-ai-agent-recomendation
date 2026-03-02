"""
Modelo de ML para predicción de vistas de videos de YouTube.
Predice el número esperado de vistas con rango de confianza (percentil 25-75).
"""
import re
import logging
import pandas as pd
import numpy as np
import pytz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
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


class ViewPredictor:
    """
    Predice el número de vistas esperadas para un video usando Random Forest.
    Usa log-transform en el target para manejar la distribución sesgada.
    Entrega predicción central + rango de confianza (percentil 25-75 de árboles).
    """

    def __init__(self, min_samples: int = 10):
        self.min_samples = min_samples
        self._model = RandomForestRegressor(n_estimators=100, random_state=42)
        self._trained = False
        self._feature_importance: dict = {}
        self._train_metrics: dict = {}
        self._channel_avg_views: float = 0.0

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extrae features de ML. Las horas y días se calculan en timezone Panamá."""
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
            dict con {trained, samples, cv_splits, cv_mae, cv_mape, feature_importance, channel_avg_views}
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
        self._channel_avg_views = float(views.mean())

        # Log-transform para manejar distribución sesgada
        y_log = np.log1p(views)

        # Validación cruzada temporal: al menos 2 folds, máx 5, ≥5 muestras por fold
        n_splits = min(5, max(2, len(df) // 5))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_maes, fold_mapes = [], []
        for train_idx, test_idx in tscv.split(X):
            x_tr, x_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y_log.iloc[train_idx], y_log.iloc[test_idx]

            fold_model = RandomForestRegressor(n_estimators=100, random_state=42)
            fold_model.fit(x_tr, y_tr)

            y_pred = np.expm1(fold_model.predict(x_te))
            y_true = np.expm1(y_te)

            fold_maes.append(float(mean_absolute_error(y_true, y_pred)))
            mask = y_true > 0
            if mask.any():
                fold_mapes.append(
                    float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
                )

        cv_mae  = float(np.mean(fold_maes))
        cv_mape = float(np.mean(fold_mapes)) if fold_mapes else 0.0

        # Reentrenar con TODOS los datos
        self._model.fit(X, y_log)
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
            'cv_mae': round(cv_mae, 0),
            'cv_mape': round(cv_mape, 1),
            'feature_importance': self._feature_importance,
            'channel_avg_views': round(self._channel_avg_views, 0),
        }
        return self._train_metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, videos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predice vistas para cada video del DataFrame.
        Agrega columnas: predicted_views, predicted_low, predicted_high.
        Si el modelo no está entrenado, usa el promedio del canal como fallback.
        """
        result = videos_df.copy()

        if not self._trained:
            result['predicted_views'] = int(self._channel_avg_views) if self._channel_avg_views else 0
            result['predicted_low'] = result['predicted_views']
            result['predicted_high'] = result['predicted_views']
            return result

        X = self._extract_features(result)

        # Predicciones de cada árbol individual para el rango de confianza
        tree_preds = np.array([
            np.expm1(tree.predict(X))
            for tree in self._model.estimators_
        ])  # shape: (n_estimators, n_samples)

        result['predicted_views'] = np.round(np.expm1(self._model.predict(X))).astype(int)
        result['predicted_low'] = np.round(np.percentile(tree_preds, 25, axis=0)).astype(int)
        result['predicted_high'] = np.round(np.percentile(tree_preds, 75, axis=0)).astype(int)

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
    ) -> dict:
        """
        Predice vistas para un único video hipotético.

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
            {'predicted': int, 'low': int, 'high': int}
        """
        if not self._trained:
            avg = int(self._channel_avg_views) if self._channel_avg_views else 0
            return {'predicted': avg, 'low': avg, 'high': avg}

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

        tree_preds = np.array([
            np.expm1(tree.predict(X))[0]
            for tree in self._model.estimators_
        ])

        return {
            'predicted': int(round(np.expm1(self._model.predict(X)[0]))),
            'low': int(round(np.percentile(tree_preds, 25))),
            'high': int(round(np.percentile(tree_preds, 75))),
        }

    def get_publishing_heatmap(
        self,
        is_short: bool,
        duration_seconds: int = 300,
        tags_count: int = 10,
        title_length: int = 60,
        title_has_number: int = 0,
        title_has_question: int = 0,
        description_length: int = 500,
        days_since_last_upload: int = 7,
        channel_age_days: int = 365,
    ) -> pd.DataFrame:
        """
        Genera una matriz 7×24 con las vistas predichas para cada combinación
        día (fila) × hora (columna) — útil para el heatmap del dashboard.

        Returns:
            DataFrame con índice = WEEKDAY_LABELS, columnas = horas 0-23
        """
        data = {}
        for hour in range(24):
            col = []
            for weekday in range(7):
                result = self.predict_single(
                    hour=hour,
                    weekday=weekday,
                    is_short=is_short,
                    duration_seconds=duration_seconds,
                    tags_count=tags_count,
                    title_length=title_length,
                    title_has_number=title_has_number,
                    title_has_question=title_has_question,
                    description_length=description_length,
                    days_since_last_upload=days_since_last_upload,
                    channel_age_days=channel_age_days,
                )
                col.append(result['predicted'])
            data[hour] = col

        return pd.DataFrame(data, index=WEEKDAY_LABELS)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def is_trained(self) -> bool:
        return self._trained

    def get_feature_importance(self) -> dict:
        return self._feature_importance

    def get_train_metrics(self) -> dict:
        return self._train_metrics

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serializa el predictor completo (modelo + métricas) a disco."""
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'ViewPredictor':
        """Carga un predictor previamente serializado desde disco."""
        import joblib
        return joblib.load(path)
