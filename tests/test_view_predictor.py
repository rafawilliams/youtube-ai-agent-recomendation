"""
Tests para src/view_predictor.py — ViewPredictor.

Cubre: train con <10 y 20+ videos, predict sin/con entrenamiento,
predict_single, columnas predicted_views/predicted_low/predicted_high.
"""
import pytest
import pandas as pd
import numpy as np
from view_predictor import ViewPredictor


# ===========================================================================
# Training
# ===========================================================================

class TestViewTrain:
    """Tests para ViewPredictor.train()."""

    def test_train_insufficient_samples(self, sample_videos_df):
        """Con <10 videos, train devuelve trained=False."""
        small_df = sample_videos_df.head(5).copy()
        predictor = ViewPredictor(min_samples=10)
        result = predictor.train(small_df)
        assert result["trained"] is False
        assert "reason" in result

    def test_train_sufficient_samples(self, training_videos_df):
        """Con 25 videos, train devuelve trained=True con channel_avg_views."""
        predictor = ViewPredictor(min_samples=10)
        result = predictor.train(training_videos_df)
        assert result["trained"] is True
        assert "channel_avg_views" in result
        assert result["channel_avg_views"] > 0
        assert "feature_importance" in result
        assert len(result["feature_importance"]) == 11

    def test_train_sets_trained_flag(self, training_videos_df):
        """Tras entrenar, is_trained() retorna True."""
        predictor = ViewPredictor()
        predictor.train(training_videos_df)
        assert predictor.is_trained() is True

    def test_train_returns_cv_metrics(self, training_videos_df):
        """El resultado contiene cv_mae y cv_mape."""
        predictor = ViewPredictor()
        result = predictor.train(training_videos_df)
        assert "cv_mae" in result
        assert "cv_mape" in result
        assert result["cv_splits"] >= 2


# ===========================================================================
# Prediction
# ===========================================================================

class TestViewPredict:
    """Tests para ViewPredictor.predict() y predict_single()."""

    def test_predict_without_training_uses_channel_avg(self, sample_videos_df):
        """Sin entrenar, predict usa channel_avg_views (0 por defecto)."""
        predictor = ViewPredictor()
        result = predictor.predict(sample_videos_df)
        assert "predicted_views" in result.columns
        assert "predicted_low" in result.columns
        assert "predicted_high" in result.columns
        # Sin entrenamiento, _channel_avg_views es 0.0
        assert (result["predicted_views"] == 0).all()

    def test_predict_after_training_adds_columns(self, training_videos_df):
        """Tras entrenar, predict agrega predicted_views, predicted_low, predicted_high."""
        predictor = ViewPredictor()
        predictor.train(training_videos_df)
        result = predictor.predict(training_videos_df)
        assert "predicted_views" in result.columns
        assert "predicted_low" in result.columns
        assert "predicted_high" in result.columns
        # Las predicciones deben ser positivas
        assert (result["predicted_views"] >= 0).all()
        # predicted_low <= predicted_views <= predicted_high (en general)
        assert (result["predicted_low"] <= result["predicted_high"]).all()

    def test_predict_single_without_training(self):
        """Sin entrenar, predict_single devuelve avg=0 en todas las claves."""
        predictor = ViewPredictor()
        result = predictor.predict_single(
            hour=14, weekday=2, is_short=False,
            duration_seconds=600, tags_count=10, title_length=50,
        )
        assert isinstance(result, dict)
        assert "predicted" in result
        assert "low" in result
        assert "high" in result
        assert result["predicted"] == 0

    def test_predict_single_after_training(self, training_videos_df):
        """Tras entrenar, predict_single devuelve dict con predicted, low, high."""
        predictor = ViewPredictor()
        predictor.train(training_videos_df)
        result = predictor.predict_single(
            hour=18, weekday=3, is_short=True,
            duration_seconds=45, tags_count=8, title_length=40,
            title_has_number=1, title_has_question=0,
            description_length=300, days_since_last_upload=3,
            channel_age_days=200,
        )
        assert result["predicted"] > 0
        assert result["low"] <= result["predicted"]
        assert result["high"] >= result["predicted"]
