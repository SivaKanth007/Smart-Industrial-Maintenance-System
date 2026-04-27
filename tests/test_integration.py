"""
Integration Test — Full Pipeline Health Check
===============================================
Runs the entire training → inference pipeline on a TINY dataset (5 units,
~50 cycles each) to confirm all components wire together without errors.

Target runtime: < 60 seconds on CPU.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _make_mini_cmapss(n_units=15, min_cycles=40, max_cycles=60):
    """Generate a tiny C-MAPSS-like dataset for integration testing."""
    np.random.seed(42)
    rows = []
    for uid in range(1, n_units + 1):
        n_cycles = np.random.randint(min_cycles, max_cycles + 1)
        for c in range(1, n_cycles + 1):
            row = {"unit_id": uid, "cycle": c}
            for i in range(1, 4):
                row[f"op_setting_{i}"] = np.random.normal(0, 1)
            for i in range(1, 22):
                base = 50 + (c / n_cycles) * 10  # mild degradation trend
                row[f"sensor_{i}"] = base + np.random.normal(0, 2)
            # Make constant sensors truly constant (like real C-MAPSS)
            for s in config.SENSORS_TO_DROP:
                row[s] = 1.0
            row["op_setting_3"] = 100.0
            rows.append(row)

    df = pd.DataFrame(rows)
    max_c = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"] = (max_c - df["cycle"]).clip(upper=config.MAX_RUL)
    return df


class TestFullPipelineIntegration:
    """
    End-to-end smoke test: preprocess → train all models → inference.

    Uses a 5-unit dataset so it finishes in ~30 seconds on CPU.
    """

    @pytest.fixture(autouse=True)
    def mini_data(self):
        self.df = _make_mini_cmapss(n_units=15)

    # ------------------------------------------------------------------
    # Step 1: Preprocessing
    # ------------------------------------------------------------------

    def test_step1_preprocess(self):
        from src.data.preprocess import DataPreprocessor

        pp = DataPreprocessor()
        result = pp.fit_transform(self.df, augment=True)

        # Must produce all 3 splits
        for split in ["train", "val", "test"]:
            assert split in result, f"Missing split: {split}"
            assert result[split]["X"].ndim == 3
            assert result[split]["y_rul"].ndim == 1
            assert result[split]["y_binary"].ndim == 1

        # No NaN anywhere
        for split in result.values():
            assert not np.isnan(split["X"]).any(), "NaN in features"
            assert not np.isnan(split["y_rul"]).any(), "NaN in RUL"

        # Store for later tests
        self._data = result
        self._pp = pp
        return result

    # ------------------------------------------------------------------
    # Step 2: Autoencoder
    # ------------------------------------------------------------------

    def test_step2_autoencoder(self):
        data = self.test_step1_preprocess()
        from src.models.autoencoder import LSTMAutoencoder, AutoencoderTrainer

        X_train = data["train"]["X"]
        X_val = data["val"]["X"]
        n_features = X_train.shape[2]

        model = LSTMAutoencoder(
            input_dim=n_features,
            seq_len=X_train.shape[1],
            hidden_dim=16,
            latent_dim=8,
            num_layers=1,
        )
        trainer = AutoencoderTrainer(model, epochs=2, batch_size=32)

        # Filter healthy samples (RUL > 30)
        healthy_mask = data["train"]["y_rul"] > 30
        X_healthy = X_train[healthy_mask] if healthy_mask.sum() > 10 else X_train
        trainer.train(X_healthy, X_val=X_val)

        # Model should have a threshold set
        assert model.threshold > 0

        # Anomaly detection should return scores and booleans
        scores, is_anomaly = model.detect_anomalies(
            torch.FloatTensor(X_val[:5])
        )
        assert scores.shape[0] == 5
        assert is_anomaly.dtype == bool

    # ------------------------------------------------------------------
    # Step 3: LSTM Predictor
    # ------------------------------------------------------------------

    def test_step3_predictor(self):
        data = self.test_step1_preprocess()
        from src.models.lstm_predictor import LSTMPredictor, PredictorTrainer

        X_train = data["train"]["X"]
        y_train = data["train"]["y_binary"]
        n_features = X_train.shape[2]

        model = LSTMPredictor(
            input_dim=n_features,
            hidden_dim=16,
            num_layers=1,
            dropout=0.1,
        )
        trainer = PredictorTrainer(model, epochs=2, batch_size=32)
        trainer.train(X_train, y_train)

        # Predict probabilities
        proba, attn = model.predict_proba(torch.FloatTensor(X_train[:5]))
        assert proba.shape == (5,)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    # ------------------------------------------------------------------
    # Step 4: XGBoost
    # ------------------------------------------------------------------

    def test_step4_xgboost(self):
        from src.data.feature_engineering import FeatureEngineer
        from src.models.xgboost_rul import XGBoostRUL

        fe = FeatureEngineer()
        df_eng = fe.engineer_features(self.df.copy(), fit=True)

        # Need to split, drop non-feature columns
        from src.data.preprocess import DataPreprocessor
        pp = DataPreprocessor()
        train, val, test = pp.temporal_split(df_eng)

        feature_cols = [c for c in train.columns
                        if c not in ["unit_id", "cycle", "RUL"]
                        and not c.startswith("op_setting")]
        X_train = train[feature_cols].values
        y_train = train["RUL"].values

        model = XGBoostRUL(params={"n_estimators": 10, "max_depth": 3})
        model.train(X_train, y_train)

        preds = model.predict(val[feature_cols].values)
        assert len(preds) == len(val)
        assert np.all(preds >= 0)

    # ------------------------------------------------------------------
    # Step 5: Monte Carlo Simulation
    # ------------------------------------------------------------------

    def test_step5_simulation(self):
        from src.evaluation.simulation import MaintenanceSimulator

        sim = MaintenanceSimulator(n_machines=5, n_periods=30, seed=42)
        df, summary = sim.run_comparison(n_simulations=3)

        # Must have all 4 policies
        assert df["policy"].nunique() == 4
        assert len(df) == 4 * 3  # 4 policies × 3 simulations

        # Optimized should cost less than reactive on average
        reactive_avg = df[df["policy"] == "Reactive"]["total_cost"].mean()
        optimized_avg = df[df["policy"] == "Optimized (Risk-Based)"]["total_cost"].mean()
        assert optimized_avg <= reactive_avg

    # ------------------------------------------------------------------
    # Step 6: MILP Scheduler
    # ------------------------------------------------------------------

    def test_step6_scheduler(self):
        from src.optimization.milp_scheduler import MaintenanceScheduler

        risks = {f"unit_{i}": np.random.uniform(0.1, 0.99) for i in range(5)}
        scheduler = MaintenanceScheduler(n_crews=2)
        result = scheduler.create_schedule(risks, n_time_slots=5)

        assert result["status"] == "Optimal"
        assert len(result["schedule"]) == 5

    # ------------------------------------------------------------------
    # Step 7: End-to-end wiring check
    # ------------------------------------------------------------------

    def test_step7_full_chain_no_crash(self):
        """
        The most important test: does the full chain run without error?
        This catches import issues, shape mismatches, and wiring bugs.
        """
        data = self.test_step1_preprocess()
        # If we got here, preprocessing works
        assert data is not None

        # Quick model instantiation + forward pass
        n_features = data["train"]["X"].shape[2]

        from src.models.autoencoder import LSTMAutoencoder
        ae = LSTMAutoencoder(input_dim=n_features,
                             seq_len=data["train"]["X"].shape[1],
                             hidden_dim=16, latent_dim=8, num_layers=1)
        ae_out = ae(torch.FloatTensor(data["train"]["X"][:2]))
        assert ae_out.shape[2] == n_features

        from src.models.lstm_predictor import LSTMPredictor
        pred = LSTMPredictor(input_dim=n_features, hidden_dim=16,
                             num_layers=1, dropout=0.1)
        logits, attn = pred(torch.FloatTensor(data["train"]["X"][:2]))
        assert logits.shape == (2,)

        from src.evaluation.simulation import MaintenanceSimulator
        sim = MaintenanceSimulator(n_machines=3, n_periods=20, seed=42)
        df, _ = sim.run_comparison(n_simulations=2)
        assert df["policy"].nunique() == 4
