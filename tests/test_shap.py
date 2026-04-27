"""
Unit Tests — SHAP Explainability
==================================
Tests for explainer setup, SHAP value computation, sensor ranking,
aggregated ranking, and save/load persistence.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.explainability.shap_analysis import SHAPExplainer


@pytest.fixture
def trained_xgboost():
    """Train a tiny XGBoost model for SHAP tests.

    Returns the XGBoostRUL *wrapper* because SHAPExplainer expects it
    (it accesses ``self.model.model`` internally to reach the XGBRegressor).
    """
    from src.models.xgboost_rul import XGBoostRUL
    np.random.seed(42)
    X = np.random.randn(100, 20).astype(np.float32)
    y = np.random.uniform(0, 125, 100).astype(np.float32)
    model = XGBoostRUL(params={"n_estimators": 10, "max_depth": 3})
    model.train(X[:80], y[:80])
    return model, X, y


@pytest.fixture
def feature_df():
    """Create a DataFrame with sensor-named columns."""
    np.random.seed(42)
    n = 50
    cols = {
        "sensor_11_roll20_max": np.random.rand(n),
        "sensor_11_roll5_mean": np.random.rand(n),
        "sensor_13_roll10_max": np.random.rand(n),
        "sensor_4_roll20_max": np.random.rand(n),
        "sensor_9_roll5_mean": np.random.rand(n),
        "sensor_9_div_sensor_7": np.random.rand(n),
        "op_regime": np.random.randint(0, 3, n).astype(float),
    }
    return pd.DataFrame(cols)


# ============================================================
# TreeExplainer (XGBoost)
# ============================================================

class TestXGBoostExplainer:
    def test_setup_explainer(self, trained_xgboost):
        model, X, _ = trained_xgboost
        explainer = SHAPExplainer(model, model_type="xgboost")
        explainer.setup_explainer()
        assert explainer.explainer is not None

    def test_compute_shap_values(self, trained_xgboost):
        model, X, _ = trained_xgboost
        explainer = SHAPExplainer(model, model_type="xgboost")
        sv = explainer.compute_shap_values(X, max_samples=30)
        assert sv is not None
        # SHAP values shape should match input features
        if isinstance(sv, list):
            assert sv[0].shape[1] == X.shape[1]
        else:
            assert sv.shape[1] == X.shape[1]

    def test_auto_setup_for_xgboost(self, trained_xgboost):
        """compute_shap_values should auto-setup explainer for XGBoost."""
        model, X, _ = trained_xgboost
        explainer = SHAPExplainer(model, model_type="xgboost")
        assert explainer.explainer is None  # not set up yet
        explainer.compute_shap_values(X, max_samples=20)
        assert explainer.explainer is not None  # auto-setup happened


# ============================================================
# Sensor ranking
# ============================================================

class TestSensorRanking:
    def test_ranking_returns_dataframe(self, trained_xgboost):
        model, X, _ = trained_xgboost
        explainer = SHAPExplainer(model, model_type="xgboost")
        explainer.compute_shap_values(X, max_samples=30)
        ranking = explainer.get_sensor_ranking()
        assert isinstance(ranking, pd.DataFrame)
        assert "feature" in ranking.columns
        assert "mean_abs_shap" in ranking.columns
        assert len(ranking) == X.shape[1]

    def test_ranking_is_sorted_descending(self, trained_xgboost):
        model, X, _ = trained_xgboost
        explainer = SHAPExplainer(model, model_type="xgboost")
        explainer.compute_shap_values(X, max_samples=30)
        ranking = explainer.get_sensor_ranking()
        values = ranking["mean_abs_shap"].values
        assert np.all(values[:-1] >= values[1:]), "Ranking should be descending"


class TestAggregatedSensorRanking:
    def test_aggregation_groups_by_base_sensor(self, trained_xgboost):
        model, _, _ = trained_xgboost
        explainer = SHAPExplainer(model, model_type="xgboost")

        # Use the named DataFrame as input
        np.random.seed(42)
        df = pd.DataFrame({
            "sensor_11_roll20_max": np.random.rand(50),
            "sensor_11_roll5_mean": np.random.rand(50),
            "sensor_13_roll10_max": np.random.rand(50),
        })

        # We need a model that matches this feature count — train a new tiny one
        from src.models.xgboost_rul import XGBoostRUL
        y = np.random.uniform(0, 125, 50).astype(np.float32)
        m2 = XGBoostRUL(params={"n_estimators": 5, "max_depth": 2})
        m2.train(df.values[:40], y[:40])

        ex = SHAPExplainer(m2, model_type="xgboost")
        ex.compute_shap_values(df, max_samples=20)
        agg = ex.get_sensor_ranking_aggregated()

        assert "sensor" in agg.columns
        assert "total_shap_importance" in agg.columns
        # sensor_11 features should be aggregated into one row
        sensor_11_rows = agg[agg["sensor"] == "sensor_11"]
        assert len(sensor_11_rows) == 1


# ============================================================
# Save / Load persistence
# ============================================================

class TestSHAPPersistence:
    def test_save_and_load_round_trip(self, trained_xgboost, tmp_path):
        model, X, _ = trained_xgboost
        explainer = SHAPExplainer(model, model_type="xgboost")
        explainer.compute_shap_values(X, max_samples=20)

        filepath = str(tmp_path / "test_shap.pkl")
        explainer.save_shap_values(filepath)
        assert os.path.exists(filepath)

        # Load into a fresh explainer
        explainer2 = SHAPExplainer(model.model, model_type="xgboost")
        explainer2.load_shap_values(filepath)
        assert explainer2.shap_values is not None
        assert explainer2.feature_names == explainer.feature_names

    def test_save_without_compute_raises(self, trained_xgboost):
        model, _, _ = trained_xgboost
        explainer = SHAPExplainer(model, model_type="xgboost")
        with pytest.raises(RuntimeError, match="No SHAP values"):
            explainer.save_shap_values()


# ============================================================
# Error handling
# ============================================================

class TestSHAPErrors:
    def test_ranking_without_compute_raises(self, trained_xgboost):
        model, _, _ = trained_xgboost
        explainer = SHAPExplainer(model, model_type="xgboost")
        with pytest.raises(RuntimeError):
            explainer.get_sensor_ranking()

    def test_plot_without_compute_raises(self, trained_xgboost):
        model, _, _ = trained_xgboost
        explainer = SHAPExplainer(model, model_type="xgboost")
        with pytest.raises(RuntimeError):
            explainer.plot_global_importance()
