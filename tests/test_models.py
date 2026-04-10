"""
Unit Tests — ML Models
"""

import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.models.autoencoder import LSTMAutoencoder
from src.models.lstm_predictor import LSTMPredictor
from src.models.xgboost_rul import XGBoostRUL


@pytest.fixture
def sample_sequences():
    """Generate sample sequences for testing."""
    np.random.seed(42)
    n_samples = 100
    seq_len = 30
    n_features = 14
    X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
    y_rul = np.random.uniform(0, 125, n_samples).astype(np.float32)
    y_binary = (y_rul <= 30).astype(np.float32)
    return X, y_rul, y_binary


@pytest.fixture
def flat_features():
    """Generate flat feature matrix for XGBoost."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.uniform(0, 125, n_samples).astype(np.float32)
    return X, y


class TestLSTMAutoencoder:
    def test_forward_pass_shape(self, sample_sequences):
        X, _, _ = sample_sequences
        model = LSTMAutoencoder(input_dim=X.shape[2], seq_len=X.shape[1])
        x_tensor = torch.FloatTensor(X[:5])
        output = model(x_tensor)
        assert output.shape == x_tensor.shape

    def test_anomaly_score_shape(self, sample_sequences):
        X, _, _ = sample_sequences
        model = LSTMAutoencoder(input_dim=X.shape[2], seq_len=X.shape[1])
        scores = model.compute_anomaly_score(torch.FloatTensor(X[:10]))
        assert scores.shape == (10,)
        assert np.all(scores >= 0)

    def test_threshold_setting(self, sample_sequences):
        X, _, _ = sample_sequences
        model = LSTMAutoencoder(input_dim=X.shape[2], seq_len=X.shape[1])
        scores = model.compute_anomaly_score(torch.FloatTensor(X))
        threshold = model.set_threshold(scores)
        assert threshold > 0
        assert model.threshold == threshold

    def test_detect_anomalies(self, sample_sequences):
        X, _, _ = sample_sequences
        model = LSTMAutoencoder(input_dim=X.shape[2], seq_len=X.shape[1])
        scores = model.compute_anomaly_score(torch.FloatTensor(X))
        model.set_threshold(scores)
        scores, is_anomaly = model.detect_anomalies(torch.FloatTensor(X[:10]))
        assert is_anomaly.shape == (10,)
        assert is_anomaly.dtype == bool

    def test_threshold_uses_val_not_train(self, sample_sequences):
        """Trainer must derive threshold from val scores when X_val is provided."""
        X, _, _ = sample_sequences
        X_train, X_val = X[:80], X[80:]
        model = LSTMAutoencoder(input_dim=X.shape[2], seq_len=X.shape[1])
        from src.models.autoencoder import AutoencoderTrainer
        trainer = AutoencoderTrainer(model, epochs=1, batch_size=32)
        trainer.train(X_train, X_val=X_val)
        val_scores = model.compute_anomaly_score(torch.FloatTensor(X_val))
        # Threshold must be within reasonable range of val reconstruction errors
        assert model.threshold >= val_scores.min()


class TestLSTMPredictor:
    def test_forward_pass_output(self, sample_sequences):
        X, _, _ = sample_sequences
        model = LSTMPredictor(input_dim=X.shape[2])
        x_tensor = torch.FloatTensor(X[:5])
        logits, attn = model(x_tensor)

        assert logits.shape == (5,)
        assert attn.shape == (5, X.shape[1])
        # Attention weights should sum to ~1 (softmax)
        attn_sums = attn.sum(dim=1)
        assert torch.allclose(attn_sums, torch.ones(5), atol=1e-5)

    def test_predict_proba(self, sample_sequences):
        X, _, _ = sample_sequences
        model = LSTMPredictor(input_dim=X.shape[2])
        proba, attn = model.predict_proba(torch.FloatTensor(X[:10]))

        assert proba.shape == (10,)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert attn.shape == (10, X.shape[1])


class TestXGBoostRUL:
    def test_train_and_predict(self, flat_features):
        X, y = flat_features
        model = XGBoostRUL(params={"n_estimators": 10, "max_depth": 3})
        model.train(X[:150], y[:150])

        predictions = model.predict(X[150:])
        assert len(predictions) == 50
        assert np.all(predictions >= 0)
        assert np.all(predictions <= config.MAX_RUL)

    def test_evaluate(self, flat_features):
        X, y = flat_features
        model = XGBoostRUL(params={"n_estimators": 10, "max_depth": 3})
        model.train(X[:150], y[:150])

        metrics = model.evaluate(X[150:], y[150:])
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] >= 0

    def test_evaluate_includes_nasa_score(self, flat_features):
        X, y = flat_features
        model = XGBoostRUL(params={"n_estimators": 10, "max_depth": 3})
        model.train(X[:150], y[:150])
        metrics = model.evaluate(X[150:], y[150:])
        assert "nasa_score" in metrics
        assert metrics["nasa_score"] >= 0

    def test_nasa_score_penalises_late_more(self):
        from src.models.xgboost_rul import nasa_score
        y_true = np.array([50.0])
        # y_pred=60 > y_true=50 → d>0 → late detection (model overestimates remaining life)
        score_late = nasa_score(y_true, np.array([60.0]))
        # y_pred=40 < y_true=50 → d<0 → early detection (model underestimates remaining life)
        score_early = nasa_score(y_true, np.array([40.0]))
        assert score_late > score_early

    def test_walk_forward_cv_unit_aware(self, flat_features):
        X, y = flat_features
        model = XGBoostRUL(params={"n_estimators": 10, "max_depth": 3})
        model.train(X[:150], y[:150])
        # 200 rows, 10 units of 20 rows each
        unit_ids = np.repeat(np.arange(10), 20)
        results = model.walk_forward_cv(X, y, n_splits=3, unit_ids=unit_ids)
        assert len(results) > 0
        assert "rmse" in results.columns

    def test_feature_importance(self, flat_features):
        X, y = flat_features
        model = XGBoostRUL(params={"n_estimators": 10, "max_depth": 3})
        model.train(X, y)

        assert model.feature_importance is not None
        assert len(model.feature_importance) == X.shape[1]


class TestPredictorEvaluate:
    def test_evaluate_finds_best_threshold(self):
        """_evaluate must sweep thresholds and return optimal_threshold key."""
        from src.models.lstm_predictor import PredictorTrainer
        from torch.utils.data import DataLoader, TensorDataset
        X = np.random.randn(50, 30, 14).astype(np.float32)
        y = np.zeros(50, dtype=np.float32)
        y[:5] = 1.0
        model = LSTMPredictor(input_dim=14)
        trainer = PredictorTrainer(model, epochs=1, batch_size=32)
        loader = DataLoader(TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)),
                            batch_size=32)
        metrics = trainer._evaluate(loader, y)
        assert "optimal_threshold" in metrics
        assert 0.0 < metrics["optimal_threshold"] < 1.0


class TestFeatureEngineering:
    def test_engineer_features_no_cycle_norm(self):
        """engineer_features must not produce cycle_norm or cycle_squared (leaky features)."""
        import pandas as pd
        from src.data.feature_engineering import FeatureEngineer
        df = pd.DataFrame({
            "unit_id": [1] * 10 + [2] * 10,
            "cycle": list(range(1, 11)) * 2,
            "op_setting_1": np.random.rand(20),
            "op_setting_2": np.random.rand(20),
            "sensor_2": np.random.rand(20),
            "sensor_3": np.random.rand(20),
            "sensor_4": np.random.rand(20),
            "RUL": list(range(9, -1, -1)) * 2,
        })
        fe = FeatureEngineer()
        df_out = fe.engineer_features(df, fit=True)
        assert "cycle_norm" not in df_out.columns, "cycle_norm is target leakage and must be excluded"
        assert "cycle_squared" not in df_out.columns, "cycle_squared is target leakage and must be excluded"


class TestReproducibility:
    def test_training_is_deterministic(self, sample_sequences):
        """Two forward passes with same seed must produce identical outputs."""
        X, _, _ = sample_sequences

        def get_output():
            torch.manual_seed(config.RANDOM_SEED)
            model = LSTMAutoencoder(input_dim=X.shape[2], seq_len=X.shape[1])
            x = torch.FloatTensor(X[:5])
            return model(x).detach().numpy()

        out1 = get_output()
        out2 = get_output()
        np.testing.assert_array_equal(out1, out2)
