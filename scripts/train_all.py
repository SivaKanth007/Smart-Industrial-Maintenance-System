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
    print(f"[TRAIN] Test anomaly scores: mean={test_scores.mean():.6f}, "
          f"max={test_scores.max():.6f}, anomaly_rate={np.mean(test_scores > autoencoder.threshold):.2%}")

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
    xgb_model.evaluate(X_val_xgb, y_val_xgb.values)
    xgb_model.save()

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
    survival_model.evaluate(df_survival_val)
    survival_model.save()

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
