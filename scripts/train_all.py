"""
End-to-End Training Script
============================
Orchestrates the complete pipeline: data → preprocessing → model training → saving.
"""

import os
import sys
import time
import numpy as np
import torch

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.data.download import download_cmapss, load_cmapss_train, load_cmapss_all_subsets
from src.data.preprocess import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.data.synthetic_generator import SyntheticDataGenerator
from src.models.autoencoder import LSTMAutoencoder, AutoencoderTrainer
from src.models.lstm_predictor import LSTMPredictor, PredictorTrainer
from src.models.xgboost_rul import XGBoostRUL
from src.models.bayesian_survival import BayesianSurvival
from src.evaluation.simulation import MaintenanceSimulator
from src.evaluation.dashboard_metrics import (
    save_dashboard_metrics, save_simulation_metrics,
)


def main():
    start_time = time.time()
    print("=" * 70)
    print("  SMART INDUSTRIAL MAINTENANCE SYSTEM — TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Device: {config.DEVICE}")
    print(f"  Random Seed: {config.RANDOM_SEED}")
    print()

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # =========================================================================
    # Step 1: Download Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA DOWNLOAD")
    print("=" * 70)

    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    existing_data = [f for f in os.listdir(config.RAW_DATA_DIR)
                     if os.path.isfile(os.path.join(config.RAW_DATA_DIR, f))
                     and (f.endswith('.txt') or f.endswith('.csv'))]
    if not existing_data:
        print("[TRAIN] Downloading C-MAPSS dataset...")
        download_cmapss()
    else:
        print(f"[TRAIN] Data already downloaded ({len(existing_data)} files), skipping...")

    # Load all C-MAPSS subsets (FD001-FD004)
    print(f"[TRAIN] Loading subsets: {config.CMAPSS_SUBSETS}")
    df_train = load_cmapss_all_subsets()

    # =========================================================================
    # Step 2: Generate Synthetic Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: SYNTHETIC DATA GENERATION")
    print("=" * 70)

    generator = SyntheticDataGenerator()
    logs, context, schedule = generator.generate_all(df_train)

    # =========================================================================
    # Step 3: Feature Engineering (for XGBoost & Bayesian)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 70)

    fe = FeatureEngineer()
    # Drop subset column if present (from multi-subset loading) before engineering
    df_for_fe = df_train.drop(columns=["subset"], errors="ignore").copy()
    df_engineered = fe.engineer_features(df_for_fe)

    # =========================================================================
    # Step 4: Preprocessing (for LSTM models)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: DATA PREPROCESSING")
    print("=" * 70)

    # Drop subset column before preprocessing (not a feature)
    df_for_preprocess = df_train.drop(columns=["subset"], errors="ignore")

    preprocessor = DataPreprocessor()
    data = preprocessor.fit_transform(df_for_preprocess, augment=config.SYNTHETIC_AUGMENT)
    preprocessor.save()

    # Save processed data
    for split_name, split_data in data.items():
        np.savez_compressed(
            os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}_data.npz"),
            X=split_data["X"],
            y_rul=split_data["y_rul"],
            y_binary=split_data["y_binary"],
            unit_ids=split_data["unit_ids"],
        )

    X_train = data["train"]["X"]
    y_train_rul = data["train"]["y_rul"]
    y_train_binary = data["train"]["y_binary"]
    X_val = data["val"]["X"]
    y_val_rul = data["val"]["y_rul"]
    y_val_binary = data["val"]["y_binary"]

    n_features = X_train.shape[2]
    print(f"\n[TRAIN] Feature dimension: {n_features}")

    # Metrics dict — populated incrementally, saved at end
    dashboard_metrics = {}

    # =========================================================================
    # Step 5: Train LSTM Autoencoder
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: LSTM AUTOENCODER TRAINING")
    print("=" * 70)

    autoencoder = LSTMAutoencoder(input_dim=n_features)
    ae_trainer = AutoencoderTrainer(autoencoder)

    # Train only on "healthy" data (high RUL)
    healthy_mask = y_train_rul > config.MAX_RUL * 0.5
    X_healthy = X_train[healthy_mask]
    X_val_ae = X_val[y_val_rul > config.MAX_RUL * 0.5] if len(X_val) > 0 else None

    print(f"[TRAIN] Training autoencoder on {len(X_healthy)} healthy samples")
    ae_trainer.train(X_healthy, X_val_ae)
    ae_trainer.save_model()

    # Compute anomaly scores on full data
    test_scores = autoencoder.compute_anomaly_score(
        torch.FloatTensor(data["test"]["X"])
    )
    test_anomaly_rate = float(np.mean(test_scores > autoencoder.threshold))
    print(f"[TRAIN] Test anomaly scores: mean={test_scores.mean():.6f}, "
          f"max={test_scores.max():.6f}, anomaly_rate={test_anomaly_rate:.2%}")

    dashboard_metrics["autoencoder"] = {
        "input_dim": n_features,
        "hidden_dim": autoencoder.hidden_dim,
        "latent_dim": autoencoder.latent_dim,
        "num_layers": autoencoder.num_layers,
        "seq_len": autoencoder.seq_len,
        "epochs": len(ae_trainer.train_history),
        "train_loss_final": ae_trainer.train_history[-1] if ae_trainer.train_history else None,
        "val_loss_best": min(ae_trainer.val_history) if ae_trainer.val_history else None,
        "anomaly_threshold": float(autoencoder.threshold),
        "anomaly_threshold_sigma": config.AE_ANOMALY_THRESHOLD_SIGMA,
        "test_anomaly_rate": test_anomaly_rate,
        "test_mean_score": float(test_scores.mean()),
        "test_max_score": float(test_scores.max()),
        "training_samples": int(len(X_healthy)),
        "total_train_samples": int(len(X_train)),
    }

    # =========================================================================
    # Step 6: Train LSTM Failure Predictor
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: LSTM FAILURE PREDICTOR TRAINING")
    print("=" * 70)

    predictor = LSTMPredictor(input_dim=n_features)
    pred_trainer = PredictorTrainer(predictor)
    pred_trainer.train(X_train, y_train_binary, X_val, y_val_binary)
    pred_trainer.save_model()

    pred_val_hist = pred_trainer.val_history
    pred_best = max(pred_val_hist, key=lambda m: m.get("auc", 0)) if pred_val_hist else {}
    dashboard_metrics["predictor"] = {
        "input_dim": n_features,
        "hidden_dim": predictor.hidden_dim,
        "num_layers": predictor.num_layers,
        "epochs": len(pred_trainer.train_history),
        "f1": pred_best.get("f1"),
        "precision": pred_best.get("precision"),
        "recall": pred_best.get("recall"),
        "auc": pred_best.get("auc"),
        "optimal_threshold": pred_best.get("optimal_threshold"),
        "training_samples": int(len(X_train)),
        "pos_rate": float(y_train_binary.sum() / len(y_train_binary)),
    }

    # =========================================================================
    # Step 7: Train XGBoost RUL Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: XGBOOST RUL TRAINING")
    print("=" * 70)

    # Get flat features for XGBoost
    exclude_cols = ["unit_id", "cycle", "RUL"]
    feature_cols = [c for c in df_engineered.columns if c not in exclude_cols]

    # Temporal split for XGBoost (matching preprocessor split)
    unit_ids = df_engineered["unit_id"].unique()
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(unit_ids)
    n = len(unit_ids)
    n_train = int(n * config.TRAIN_RATIO)
    n_val = int(n * config.VAL_RATIO)
    train_units = unit_ids[:n_train]
    val_units = unit_ids[n_train:n_train + n_val]

    X_train_xgb = df_engineered[df_engineered["unit_id"].isin(train_units)][feature_cols]
    y_train_xgb = df_engineered[df_engineered["unit_id"].isin(train_units)]["RUL"]
    X_val_xgb = df_engineered[df_engineered["unit_id"].isin(val_units)][feature_cols]
    y_val_xgb = df_engineered[df_engineered["unit_id"].isin(val_units)]["RUL"]

    xgb_model = XGBoostRUL()
    xgb_model.train(X_train_xgb, y_train_xgb.values, X_val_xgb, y_val_xgb.values,
                     feature_names=feature_cols)
    xgb_eval = xgb_model.evaluate(X_val_xgb, y_val_xgb.values)
    xgb_model.save()

    xgb_top15 = []
    if xgb_model.feature_importance is not None:
        for _, row in xgb_model.feature_importance.head(15).iterrows():
            xgb_top15.append({"feature": row["feature"], "importance": float(row["importance"])})
    dashboard_metrics["xgboost"] = {
        "rmse": xgb_eval["rmse"],
        "mae": xgb_eval["mae"],
        "r2": xgb_eval["r2"],
        "within_10_pct": xgb_eval["within_10_pct"],
        "within_20_pct": xgb_eval["within_20_pct"],
        "nasa_score": xgb_eval["nasa_score"],
        "n_features": len(feature_cols),
        "n_estimators": config.XGB_PARAMS.get("n_estimators"),
        "max_depth": config.XGB_PARAMS.get("max_depth"),
        "learning_rate": config.XGB_PARAMS.get("learning_rate"),
        "training_samples": int(len(X_train_xgb)),
        "feature_importance_top15": xgb_top15,
    }

    # =========================================================================
    # Step 8: Bayesian Survival Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: BAYESIAN SURVIVAL ANALYSIS")
    print("=" * 70)

    # Use a subset of features for survival analysis
    survival_features = config.ACTIVE_SENSORS + ["cycle"]
    survival_cols = [c for c in survival_features if c in df_train.columns] + ["RUL"]

    df_survival_train = df_train[df_train["unit_id"].isin(train_units)][
        ["unit_id"] + survival_cols
    ]

    survival_model = BayesianSurvival()
    survival_model.fit(df_survival_train)

    df_survival_val = df_train[df_train["unit_id"].isin(val_units)][
        ["unit_id"] + survival_cols
    ]
    surv_eval = survival_model.evaluate(df_survival_val)
    survival_model.save()

    surv_metrics = {
        "concordance_index": surv_eval["concordance_index"],
        "rmse_failures": surv_eval["rmse_failures"],
        "n_events": surv_eval["n_events"],
        "n_censored": surv_eval["n_censored"],
        "n_covariates": len(survival_model.selected_feature_cols or []),
    }
    try:
        surv_metrics["aic"] = float(survival_model.model.AIC_)
        surv_metrics["log_likelihood"] = float(survival_model.model.log_likelihood_)
    except Exception:
        pass
    dashboard_metrics["survival"] = surv_metrics

    # =========================================================================
    # Step 9: Run Simulation
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 9: MAINTENANCE POLICY SIMULATION")
    print("=" * 70)

    simulator = MaintenanceSimulator(n_machines=50, n_periods=100)
    sim_df, sim_summary = simulator.run_comparison(n_simulations=50)
    sim_plot_path = os.path.join(config.MODELS_DIR, "..", "simulation_comparison.png")
    simulator.plot_comparison(sim_df, save_path=sim_plot_path)

    sim_metrics = save_simulation_metrics(sim_df)
    dashboard_metrics["simulation"] = sim_metrics

    # =========================================================================
    # Save Dashboard Metrics
    # =========================================================================
    save_dashboard_metrics(dashboard_metrics)

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Models saved to: {config.MODELS_DIR}")
    print(f"  Processed data: {config.PROCESSED_DATA_DIR}")
    print(f"  Synthetic data: {config.SYNTHETIC_DATA_DIR}")
    print()
    print("  Saved models:")
    for f in os.listdir(config.MODELS_DIR):
        size = os.path.getsize(os.path.join(config.MODELS_DIR, f)) / 1e6
        print(f"    - {f} ({size:.2f} MB)")
    print()
    print("  Running inference pipeline to generate recommendations...")
    # Run inference so the dashboard has up-to-date recommendations
    _run_inference()

    print()
    print("  Next: Run 'streamlit run dashboard/app.py' for the interactive dashboard")


def _run_inference():
    """Run inference + screenshot capture after training completes."""
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        from run_pipeline import run_pipeline
        run_pipeline()
    except Exception as exc:
        print(f"  [WARN] Inference step failed: {exc}")


if __name__ == "__main__":
    main()
