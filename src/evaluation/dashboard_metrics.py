"""
Dashboard Metrics Persistence
==============================
Save, load, and collect training metrics so the Streamlit dashboard
displays live results instead of hardcoded values.

Usage
-----
After training (in train_all.py, notebooks, or any script):
    from src.evaluation.dashboard_metrics import collect_and_save_metrics
    collect_and_save_metrics()

The dashboard loads metrics via ``load_dashboard_metrics()``.
"""

import json
import os
from datetime import datetime

import numpy as np
import torch

import config

METRICS_PATH = os.path.join(config.MODELS_DIR, "dashboard_metrics.json")


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_dashboard_metrics(metrics: dict):
    """Save metrics dict to JSON."""
    metrics["timestamp"] = datetime.now().isoformat()
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    print(f"[METRICS] Dashboard metrics saved to {METRICS_PATH}")


def load_dashboard_metrics() -> dict | None:
    """Load metrics JSON. Returns None if file missing."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Collect from saved model checkpoints + quick evaluation
# ---------------------------------------------------------------------------

def collect_and_save_metrics():
    """
    Extract metrics from saved model checkpoints, run quick evaluations
    on test data, and save a comprehensive dashboard_metrics.json.

    Call this at the end of any training run (script or notebook) to ensure
    the dashboard reflects the latest results.
    """
    metrics = {}

    metrics["autoencoder"] = _collect_autoencoder_metrics()
    metrics["predictor"] = _collect_predictor_metrics()
    metrics["xgboost"] = _collect_xgboost_metrics()
    metrics["survival"] = _collect_survival_metrics()
    metrics["simulation"] = _collect_simulation_metrics()

    save_dashboard_metrics(metrics)
    return metrics


# ---------------------------------------------------------------------------
# Per-model metric collectors
# ---------------------------------------------------------------------------

def _collect_autoencoder_metrics() -> dict:
    """Extract AE metrics from checkpoint + run test anomaly rate."""
    ae_path = os.path.join(config.MODELS_DIR, "autoencoder.pt")
    if not os.path.exists(ae_path):
        return {}

    ckpt = torch.load(ae_path, map_location="cpu", weights_only=False)
    train_hist = ckpt.get("train_history", [])
    val_hist = ckpt.get("val_history", [])

    info = {
        "input_dim": ckpt.get("input_dim"),
        "hidden_dim": ckpt.get("hidden_dim", config.AE_HIDDEN_DIM),
        "latent_dim": ckpt.get("latent_dim", config.AE_LATENT_DIM),
        "num_layers": ckpt.get("num_layers", config.AE_NUM_LAYERS),
        "seq_len": ckpt.get("seq_len", config.SEQUENCE_LENGTH),
        "epochs": len(train_hist),
        "train_loss_final": train_hist[-1] if train_hist else None,
        "val_loss_best": min(val_hist) if val_hist else None,
        "anomaly_threshold": ckpt.get("threshold"),
        "anomaly_threshold_sigma": config.AE_ANOMALY_THRESHOLD_SIGMA,
    }

    # Compute test anomaly rate from saved model + test data
    test_path = os.path.join(config.PROCESSED_DATA_DIR, "test_data.npz")
    if os.path.exists(test_path):
        try:
            from src.models.autoencoder import load_autoencoder
            ae = load_autoencoder(ae_path)
            test_data = np.load(test_path)
            X_test = test_data["X"]
            scores = ae.compute_anomaly_score(torch.FloatTensor(X_test))
            info["test_anomaly_rate"] = float(np.mean(scores > ae.threshold))
            info["test_mean_score"] = float(scores.mean())
            info["test_max_score"] = float(scores.max())
        except Exception as e:
            print(f"[METRICS] Warning: could not compute AE test stats: {e}")

    # Training sample count from train data
    train_path = os.path.join(config.PROCESSED_DATA_DIR, "train_data.npz")
    if os.path.exists(train_path):
        try:
            train_data = np.load(train_path)
            y_rul = train_data["y_rul"]
            healthy_mask = y_rul > config.MAX_RUL * 0.5
            info["training_samples"] = int(healthy_mask.sum())
            info["total_train_samples"] = int(len(y_rul))
        except Exception:
            pass

    return info


def _collect_predictor_metrics() -> dict:
    """Extract LSTM predictor metrics from checkpoint."""
    pred_path = os.path.join(config.MODELS_DIR, "lstm_predictor.pt")
    if not os.path.exists(pred_path):
        return {}

    ckpt = torch.load(pred_path, map_location="cpu", weights_only=False)
    val_hist = ckpt.get("val_history", [])

    info = {
        "input_dim": ckpt.get("input_dim"),
        "hidden_dim": ckpt.get("hidden_dim", config.PRED_HIDDEN_DIM),
        "num_layers": ckpt.get("num_layers", config.PRED_NUM_LAYERS),
        "epochs": len(ckpt.get("train_history", [])),
    }

    # val_history is list of dicts: {f1, precision, recall, auc, optimal_threshold}
    if val_hist:
        # Find best by AUC
        best = max(val_hist, key=lambda m: m.get("auc", 0))
        info["f1"] = best.get("f1")
        info["precision"] = best.get("precision")
        info["recall"] = best.get("recall")
        info["auc"] = best.get("auc")
        info["optimal_threshold"] = best.get("optimal_threshold")

    # Training sample count
    train_path = os.path.join(config.PROCESSED_DATA_DIR, "train_data.npz")
    if os.path.exists(train_path):
        try:
            train_data = np.load(train_path)
            info["training_samples"] = int(len(train_data["X"]))
        except Exception:
            pass

    return info


def _collect_xgboost_metrics() -> dict:
    """Extract XGBoost metrics from checkpoint + run quick evaluation."""
    import joblib

    xgb_path = os.path.join(config.MODELS_DIR, "xgboost_rul.pkl")
    if not os.path.exists(xgb_path):
        # Try alternate name
        xgb_path = os.path.join(config.MODELS_DIR, "xgboost_model.pkl")
        if not os.path.exists(xgb_path):
            return {}

    state = joblib.load(xgb_path)
    model = state.get("model")
    params = state.get("params", {})
    feat_imp = state.get("feature_importance")

    info = {
        "n_estimators": params.get("n_estimators"),
        "max_depth": params.get("max_depth"),
        "learning_rate": params.get("learning_rate"),
    }

    if state.get("feature_names"):
        info["n_features"] = len(state["feature_names"])

    # Feature importance top 15
    if feat_imp is not None:
        top15 = feat_imp.head(15)
        info["feature_importance_top15"] = [
            {"feature": row["feature"], "importance": float(row["importance"])}
            for _, row in top15.iterrows()
        ]

    # Run quick evaluation on test data
    if model is not None:
        try:
            from src.data.feature_engineering import FeatureEngineer
            from src.data.download import load_cmapss_all_subsets

            # Load and engineer features
            df_train = load_cmapss_all_subsets()
            fe = FeatureEngineer()
            df_for_fe = df_train.drop(columns=["subset"], errors="ignore").copy()
            df_engineered = fe.engineer_features(df_for_fe)

            exclude_cols = ["unit_id", "cycle", "RUL"]
            feature_cols = [c for c in df_engineered.columns if c not in exclude_cols]

            # Use same split as training
            unit_ids = df_engineered["unit_id"].unique()
            np.random.seed(config.RANDOM_SEED)
            np.random.shuffle(unit_ids)
            n = len(unit_ids)
            n_train = int(n * config.TRAIN_RATIO)
            n_val = int(n * config.VAL_RATIO)
            val_units = unit_ids[n_train:n_train + n_val]
            test_units = unit_ids[n_train + n_val:]

            # Evaluate on val set (same as train_all.py)
            X_val = df_engineered[df_engineered["unit_id"].isin(val_units)][feature_cols]
            y_val = df_engineered[df_engineered["unit_id"].isin(val_units)]["RUL"]

            if len(X_val) > 0:
                from src.models.xgboost_rul import XGBoostRUL
                xgb_inst = XGBoostRUL(params=params)
                xgb_inst.model = model
                xgb_inst.feature_names = state.get("feature_names")
                eval_metrics = xgb_inst.evaluate(X_val, y_val.values)
                info.update({
                    "rmse": eval_metrics["rmse"],
                    "mae": eval_metrics["mae"],
                    "r2": eval_metrics["r2"],
                    "within_10_pct": eval_metrics["within_10_pct"],
                    "within_20_pct": eval_metrics["within_20_pct"],
                    "nasa_score": eval_metrics["nasa_score"],
                })
                info["training_samples"] = int(
                    df_engineered[df_engineered["unit_id"].isin(unit_ids[:n_train])].shape[0]
                )
        except Exception as e:
            print(f"[METRICS] Warning: could not compute XGBoost eval metrics: {e}")

    return info


def _collect_survival_metrics() -> dict:
    """Extract survival model metrics from checkpoint + evaluate."""
    import joblib

    # Prefer survival_model.pkl (has selected_feature_cols), fallback to bayesian_survival.pkl
    surv_path = os.path.join(config.MODELS_DIR, "survival_model.pkl")
    if not os.path.exists(surv_path):
        surv_path = os.path.join(config.MODELS_DIR, "bayesian_survival.pkl")
        if not os.path.exists(surv_path):
            return {}

    state = joblib.load(surv_path)
    model = state.get("model")

    info = {}
    if model is not None:
        try:
            info["aic"] = float(model.AIC_)
        except Exception:
            pass
        try:
            info["log_likelihood"] = float(model.log_likelihood_)
        except Exception:
            pass
        try:
            info["n_covariates"] = len(state.get("selected_feature_cols", []))
        except Exception:
            pass

    # Run quick evaluation
    if state.get("fitted"):
        try:
            from src.data.download import load_cmapss_all_subsets
            from src.models.bayesian_survival import BayesianSurvival

            surv_inst = BayesianSurvival(confidence_levels=state.get("confidence_levels"))
            surv_inst.model = model
            surv_inst.km_fitter = state.get("km_fitter")
            surv_inst.fitted = True
            surv_inst.selected_feature_cols = state.get("selected_feature_cols")

            df_train = load_cmapss_all_subsets()
            survival_features = config.ACTIVE_SENSORS + ["cycle"]
            survival_cols = [c for c in survival_features if c in df_train.columns] + ["RUL"]

            unit_ids = df_train["unit_id"].unique()
            np.random.seed(config.RANDOM_SEED)
            np.random.shuffle(unit_ids)
            n = len(unit_ids)
            n_train = int(n * config.TRAIN_RATIO)
            n_val = int(n * config.VAL_RATIO)
            val_units = unit_ids[n_train:n_train + n_val]

            df_val = df_train[df_train["unit_id"].isin(val_units)][
                ["unit_id"] + survival_cols
            ]
            eval_metrics = surv_inst.evaluate(df_val)
            info["concordance_index"] = eval_metrics["concordance_index"]
            info["rmse_failures"] = eval_metrics["rmse_failures"]
            info["n_events"] = eval_metrics["n_events"]
            info["n_censored"] = eval_metrics["n_censored"]
        except Exception as e:
            print(f"[METRICS] Warning: could not compute survival eval metrics: {e}")

    return info


def _collect_simulation_metrics() -> dict:
    """Load simulation metrics if previously saved, else run fresh."""
    sim_path = os.path.join(config.MODELS_DIR, "simulation_metrics.json")
    if os.path.exists(sim_path):
        with open(sim_path) as f:
            return json.load(f)

    # Run simulation
    try:
        from src.evaluation.simulation import MaintenanceSimulator
        simulator = MaintenanceSimulator(n_machines=50, n_periods=100)
        sim_df, sim_summary = simulator.run_comparison(n_simulations=50)
        return _build_simulation_metrics(sim_df)
    except Exception as e:
        print(f"[METRICS] Warning: could not run simulation: {e}")
        return {}


def save_simulation_metrics(sim_df):
    """Save simulation results from a DataFrame to JSON."""
    metrics = _build_simulation_metrics(sim_df)
    sim_path = os.path.join(config.MODELS_DIR, "simulation_metrics.json")
    with open(sim_path, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    print(f"[METRICS] Simulation metrics saved to {sim_path}")
    return metrics


def _build_simulation_metrics(sim_df) -> dict:
    """Build simulation metrics dict from DataFrame."""
    policies = {}
    for policy_name, canonical in [
        ("Reactive", "reactive"),
        ("Scheduled (every 30)", "scheduled"),
        ("Optimized (Risk-Based)", "optimized"),
    ]:
        pdf = sim_df[sim_df["policy"] == policy_name]
        if len(pdf) > 0:
            policies[canonical] = {
                "avg_total_cost": float(pdf["total_cost"].mean()),
                "avg_downtime_hours": float(pdf["total_downtime_hours"].mean()),
                "availability_pct": float(pdf["availability_pct"].mean()),
                "avg_failures": float(pdf["n_failures"].mean()),
                "preventive_actions": float(pdf["n_preventive"].mean()),
            }

    reactive = policies.get("reactive", {})
    optimized = policies.get("optimized", {})

    cost_red = 0
    downtime_red = 0
    failure_red = 0
    avail_gain = 0
    if reactive.get("avg_total_cost", 0) > 0:
        cost_red = (1 - optimized.get("avg_total_cost", 0) / reactive["avg_total_cost"]) * 100
    if reactive.get("avg_downtime_hours", 0) > 0:
        downtime_red = (1 - optimized.get("avg_downtime_hours", 0) / reactive["avg_downtime_hours"]) * 100
    if reactive.get("avg_failures", 0) > 0:
        failure_red = (1 - optimized.get("avg_failures", 0) / reactive["avg_failures"]) * 100
    avail_gain = optimized.get("availability_pct", 0) - reactive.get("availability_pct", 0)

    return {
        "n_simulations": 50,
        "n_machines": 50,
        "n_periods": 100,
        "policies": policies,
        "cost_reduction_pct": round(cost_red, 1),
        "downtime_reduction_pct": round(downtime_red, 1),
        "failure_reduction_pct": round(failure_red, 1),
        "availability_gain_pp": round(avail_gain, 1),
    }


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
