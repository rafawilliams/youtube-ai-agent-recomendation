"""
Tests para src/virality_predictor.py — ViralityPredictor.

Cubre: train con <10 y 20+ videos, predict sin/con entrenamiento,
predict_single sin/con entrenamiento, rango de scores 1-10.
"""
import pytest
import pandas as pd
import numpy as np
from virality_predictor import ViralityPredictor


# ===========================================================================
# Training
# ===========================================================================

class TestViralityTrain:
    """Tests para ViralityPredictor.train()."""

    def test_train_insufficient_samples(self, sample_videos_df):
        """Con <10 videos (view_count > 0), train devuelve trained=False."""
        small_df = sample_videos_df.head(5).copy()
        predictor = ViralityPredictor(min_samples=10)
        result = predictor.train(small_df)
        assert result["trained"] is False
        assert "reason" in result

    def test_train_sufficient_samples(self, training_videos_df):
        """Con 25 videos, train devuelve trained=True con feature_importance."""
        predictor = ViralityPredictor(min_samples=10)
        result = predictor.train(training_videos_df)
        assert result["trained"] is True
        assert "feature_importance" in result
        assert "samples" in result
        assert result["samples"] >= 10
        assert len(result["feature_importance"]) == 11  # 11 features

    def test_train_sets_trained_flag(self, training_videos_df):
        """Tras entrenar, is_trained() retorna True."""
        predictor = ViralityPredictor()
        predictor.train(training_videos_df)
        assert predictor.is_trained() is True

    def test_train_returns_cv_metrics(self, training_videos_df):
        """El resultado contiene cv_splits y cv_mae_percentile."""
        predictor = ViralityPredictor()
        result = predictor.train(training_videos_df)
        assert "cv_splits" in result
        assert "cv_mae_percentile" in result
        assert result["cv_splits"] >= 2


# ===========================================================================
# Prediction
# ===========================================================================

class TestViralityPredict:
    """Tests para ViralityPredictor.predict() y predict_single()."""

    def test_predict_without_training_returns_5(self, sample_videos_df):
        """Sin entrenar, predict devuelve score 5.0 para todos."""
        predictor = ViralityPredictor()
        result = predictor.predict(sample_videos_df)
        assert "virality_score" in result.columns
        assert (result["virality_score"] == 5.0).all()

    def test_predict_after_training_scores_in_range(self, training_videos_df):
        """Tras entrenar, los scores están en [1, 10]."""
        predictor = ViralityPredictor()
        predictor.train(training_videos_df)
        result = predictor.predict(training_videos_df)
        assert "virality_score" in result.columns
        assert result["virality_score"].min() >= 1.0
        assert result["virality_score"].max() <= 10.0

    def test_predict_single_without_training(self):
        """Sin entrenar, predict_single devuelve 5.0."""
        predictor = ViralityPredictor()
        score = predictor.predict_single(
            hour=14, weekday=2, is_short=False,
            duration_seconds=600, tags_count=10, title_length=50,
        )
        assert score == 5.0

    def test_predict_single_after_training(self, training_videos_df):
        """Tras entrenar, predict_single devuelve score entre 1.0 y 10.0."""
        predictor = ViralityPredictor()
        predictor.train(training_videos_df)
        score = predictor.predict_single(
            hour=18, weekday=3, is_short=True,
            duration_seconds=45, tags_count=8, title_length=40,
            title_has_number=1, title_has_question=0,
            description_length=300, days_since_last_upload=3,
            channel_age_days=200,
        )
        assert 1.0 <= score <= 10.0
