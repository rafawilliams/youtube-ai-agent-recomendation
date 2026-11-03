"""
Tests para src/retention_predictor.py — RetentionPredictor.

Cubre: train con <10 y 20+ videos, predict sin/con entrenamiento,
predict_single, rango predicted_retention 0-100.
"""
import pytest
import pandas as pd
import numpy as np
from retention_predictor import RetentionPredictor


# ===========================================================================
# Training
# ===========================================================================

class TestRetentionTrain:
    """Tests para RetentionPredictor.train()."""

    def test_train_insufficient_samples(self, sample_videos_df):
        """Con <10 videos sin avg_view_percentage, train devuelve trained=False."""
        # sample_videos_df no tiene avg_view_percentage
        predictor = RetentionPredictor(min_samples=10)
        result = predictor.train(sample_videos_df)
        assert result["trained"] is False

    def test_train_insufficient_samples_explicit(self):
        """Con pocos videos que tienen avg_view_percentage, trained=False."""
        np.random.seed(42)
        base_date = pd.Timestamp("2025-10-01", tz="UTC")
        df = pd.DataFrame([{
            "video_id": f"v{i}", "title": f"Video {i}",
            "published_at": base_date + pd.Timedelta(days=i),
            "is_short": False, "duration_seconds": 600,
            "tags": "a,b,c", "description": "desc",
            "avg_view_percentage": 45.0,
        } for i in range(5)])
        predictor = RetentionPredictor(min_samples=10)
        result = predictor.train(df)
        assert result["trained"] is False

    def test_train_sufficient_samples(self, training_videos_df):
        """Con 25 videos que tienen avg_view_percentage, trained=True."""
        predictor = RetentionPredictor(min_samples=10)
        result = predictor.train(training_videos_df)
        assert result["trained"] is True
        assert "feature_importance" in result
        assert len(result["feature_importance"]) == 14  # 14 features
        assert "cv_mae" in result
        assert "cv_r2" in result


# ===========================================================================
# Prediction
# ===========================================================================

class TestRetentionPredict:
    """Tests para RetentionPredictor.predict() y predict_single()."""

    def test_predict_without_training_returns_50(self, sample_videos_df):
        """Sin entrenar, predict devuelve predicted_retention=50.0 para todos."""
        predictor = RetentionPredictor()
        result = predictor.predict(sample_videos_df)
        assert "predicted_retention" in result.columns
        assert (result["predicted_retention"] == 50.0).all()

    def test_predict_after_training_between_0_and_100(self, training_videos_df):
        """Tras entrenar, predicted_retention está entre 0 y 100."""
        predictor = RetentionPredictor()
        predictor.train(training_videos_df)
        result = predictor.predict(training_videos_df)
        assert "predicted_retention" in result.columns
        assert result["predicted_retention"].min() >= 0
        assert result["predicted_retention"].max() <= 100

    def test_predict_single_without_training(self):
        """Sin entrenar, predict_single devuelve 50.0."""
        predictor = RetentionPredictor()
        result = predictor.predict_single()
        assert result == 50.0

    def test_predict_single_after_training(self, training_videos_df):
        """Tras entrenar, predict_single devuelve valor entre 0 y 100."""
        predictor = RetentionPredictor()
        predictor.train(training_videos_df)
        result = predictor.predict_single(
            hour=15, weekday=4, is_short=False,
            duration_seconds=900, tags_count=12, title_length=55,
            title_has_number=0, title_has_question=1,
            description_length=800, days_since_last_upload=5,
            channel_age_days=300, has_hook_title=1,
            is_tutorial=1, is_entertainment=0,
        )
        assert 0 <= result <= 100
