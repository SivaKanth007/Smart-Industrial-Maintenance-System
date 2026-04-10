"""
Full Inference Pipeline
========================
Loads trained models and runs the complete inference pipeline:
data → anomaly detection → risk scoring → MILP optimization → recommendations.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.data.preprocess import DataPreprocessor


def _capture_screenshots():
    """Update assets/ screenshots after results change. Silently skips if playwright is missing."""
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
        from capture_screenshots import capture
        capture()
    except Exception:
        pass  # Never block the pipeline due to screenshot errors
from src.models.autoencoder import load_autoencoder
from src.models.lstm_predictor import load_predictor
from src.models.xgboost_rul import XGBoostRUL
from src.models.bayesian_survival import BayesianSurvival
from src.optimization.milp_scheduler import MaintenanceScheduler


def run_pipeline(test_mode=False):
    """
    Run the full inference pipeline (C-MAPSS).

    Parameters
    ----------
    test_mode : bool
        If True, uses a small subset for quick testing.
    """
    print("=" * 70)
    print("  MAINTENANCE DECISION SUPPORT — INFERENCE PIPELINE (C-MAPSS)")
    print("=" * 70)

    # =========================================================================
    # Load Models
    # =========================================================================
    print("\n[PIPELINE] Loading trained models...")

    autoencoder = load_autoencoder()
    predictor = load_predictor()
    xgb_model = XGBoostRUL.load()
    survival_model = BayesianSurvival.load()

    # =========================================================================
    # Load Test Data
    # =========================================================================
    print("\n[PIPELINE] Loading test data...")

    test_data = np.load(os.path.join(config.PROCESSED_DATA_DIR, "test_data.npz"))
    X_test = test_data["X"]
    y_rul = test_data["y_rul"]
    unit_ids = test_data["unit_ids"]

    if test_mode:
        # Use only first 50 samples
        X_test = X_test[:50]
        y_rul = y_rul[:50]
        unit_ids = unit_ids[:50]

    print(f"[PIPELINE] Test data: {X_test.shape[0]} sequences, "
          f"{len(np.unique(unit_ids))} unique units")

    # =========================================================================
    # Stage 1: Anomaly Detection
    # =========================================================================
    print("\n[PIPELINE] Stage 1: Anomaly Detection")

    anomaly_scores, is_anomaly = autoencoder.detect_anomalies(
        torch.FloatTensor(X_test)
    )
    print(f"  Anomalies detected: {is_anomaly.sum()}/{len(is_anomaly)} "
          f"({is_anomaly.mean():.1%})")

    # =========================================================================
    # Stage 2: Failure Risk Prediction
    # =========================================================================
    print("\n[PIPELINE] Stage 2: Failure Risk Prediction")

    failure_proba, attention_weights = predictor.predict_proba(
        torch.FloatTensor(X_test)
    )
    print(f"  High risk (>0.7): {(failure_proba > 0.7).sum()}")
    print(f"  Medium risk (0.4-0.7): {((failure_proba > 0.4) & (failure_proba <= 0.7)).sum()}")
    print(f"  Low risk (<0.4): {(failure_proba <= 0.4).sum()}")

    # =========================================================================
    # Stage 3: Per-Unit Risk Aggregation
    # =========================================================================
    print("\n[PIPELINE] Stage 3: Per-Unit Risk Aggregation")

    # For each unique unit, take the latest (most recent) prediction
    unit_risks = {}
    for uid in np.unique(unit_ids):
        mask = unit_ids == uid
        latest_idx = np.where(mask)[0][-1]  # Last sequence for this unit
        unit_risks[int(uid)] = float(failure_proba[latest_idx])

    print(f"  Units assessed: {len(unit_risks)}")

    # =========================================================================
    # Stage 4: MILP Optimization
    # =========================================================================
    print("\n[PIPELINE] Stage 4: Maintenance Scheduling (MILP)")

    scheduler = MaintenanceScheduler()
    result = scheduler.create_schedule(
        machine_risks=unit_risks,
        machine_names={uid: f"Engine-{uid:03d}" for uid in unit_risks},
    )

    # =========================================================================
    # Stage 5: Generate Recommendations
    # =========================================================================
    print("\n[PIPELINE] Stage 5: Generating Recommendations")

    schedule = result["schedule"]

    recommendations = []
    for _, row in schedule.iterrows():
        rec = {
            "machine": row["machine_name"],
            "risk_score": row["failure_risk"],
            "risk_level": row["risk_level"],
            "action": "Immediate maintenance" if row["risk_level"] == "Service Immediately"
                      else "Schedule maintenance" if row["risk_level"] == "Schedule Soon"
                      else "Continue monitoring",
            "scheduled_slot": row["scheduled_slot"] if row["is_scheduled"] else "N/A",
            "is_anomalous": bool(is_anomaly[
                np.where(unit_ids == row["machine_id"])[0][-1]
            ]) if row["machine_id"] in unit_ids else False,
        }
        recommendations.append(rec)

    rec_df = pd.DataFrame(recommendations)

    print("\n" + "=" * 70)
    print("MAINTENANCE RECOMMENDATIONS")
    print("=" * 70)
    print(rec_df.to_string(index=False))

    # Save recommendations
    rec_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "recommendations.csv"), index=False)
    print(f"\n[PIPELINE] Recommendations saved to {config.PROCESSED_DATA_DIR}/recommendations.csv")

    # Auto-update dashboard screenshots
    _capture_screenshots()

    return {
        "anomaly_scores": anomaly_scores,
        "failure_proba": failure_proba,
        "attention_weights": attention_weights,
        "schedule": schedule,
        "recommendations": rec_df,
    }


def run_ims_pipeline(experiment=2, test_mode=False):
    """
    Run the full inference pipeline for IMS bearing data.

    Parameters
    ----------
    experiment : int
        IMS experiment number (1, 2, or 3).
    test_mode : bool
        If True, uses a small subset for quick testing.
    """
    print("=" * 70)
    print("  MAINTENANCE DECISION SUPPORT — INFERENCE PIPELINE (IMS)")
    print("=" * 70)

    # =========================================================================
    # Load IMS Models
    # =========================================================================
    print("\n[IMS PIPELINE] Loading IMS-trained models...")

    autoencoder = load_autoencoder(os.path.join(config.MODELS_DIR, "ims_autoencoder.pt"))
    predictor = load_predictor(os.path.join(config.MODELS_DIR, "ims_predictor.pt"))
    xgb_model = XGBoostRUL.load(os.path.join(config.MODELS_DIR, "ims_xgboost.pkl"))

    # =========================================================================
    # Load IMS Test Data
    # =========================================================================
    print("\n[IMS PIPELINE] Loading IMS test data...")

    test_data = np.load(os.path.join(config.IMS_PROCESSED_DIR, "ims_test_data.npz"))
    X_test = test_data["X"]
    y_rul = test_data["y_rul"]

    if test_mode:
        X_test = X_test[:50]
        y_rul = y_rul[:50]

    print(f"[IMS PIPELINE] Test data: {X_test.shape[0]} sequences")

    # =========================================================================
    # Stage 1: Bearing Anomaly Detection
    # =========================================================================
    print("\n[IMS PIPELINE] Stage 1: Bearing Anomaly Detection")

    anomaly_scores, is_anomaly = autoencoder.detect_anomalies(
        torch.FloatTensor(X_test)
    )
    print(f"  Anomalies detected: {is_anomaly.sum()}/{len(is_anomaly)} "
          f"({is_anomaly.mean():.1%})")

    # =========================================================================
    # Stage 2: Bearing Failure Risk
    # =========================================================================
    print("\n[IMS PIPELINE] Stage 2: Bearing Failure Risk Prediction")

    failure_proba, attention_weights = predictor.predict_proba(
        torch.FloatTensor(X_test)
    )
    print(f"  High risk (>0.7): {(failure_proba > 0.7).sum()}")
    print(f"  Medium risk (0.4-0.7): {((failure_proba > 0.4) & (failure_proba <= 0.7)).sum()}")
    print(f"  Low risk (<0.4): {(failure_proba <= 0.4).sum()}")

    # =========================================================================
    # Stage 3: MILP Optimization (4 bearings)
    # =========================================================================
    print("\n[IMS PIPELINE] Stage 3: Bearing Maintenance Scheduling")

    exp_info = config.IMS_EXPERIMENTS[experiment]
    bearing_risks = {}
    for b in range(1, exp_info["bearings"] + 1):
        # Use the max failure probability as bearing risk
        bearing_risks[b] = float(failure_proba[-1])  # Latest prediction

    scheduler = MaintenanceScheduler()
    result = scheduler.create_schedule(
        machine_risks=bearing_risks,
        machine_names={b: f"Bearing-{b}" for b in bearing_risks},
    )

    schedule = result["schedule"]

    # =========================================================================
    # Recommendations
    # =========================================================================
    print("\n" + "=" * 70)
    print("BEARING MAINTENANCE RECOMMENDATIONS")
    print("=" * 70)

    recommendations = []
    for _, row in schedule.iterrows():
        rec = {
            "bearing": row["machine_name"],
            "risk_score": row["failure_risk"],
            "risk_level": row["risk_level"],
            "action": "Immediate maintenance" if row["risk_level"] == "Service Immediately"
                      else "Schedule maintenance" if row["risk_level"] == "Schedule Soon"
                      else "Continue monitoring",
            "failure_mode": exp_info["failure_modes"][0] if row["machine_id"] in exp_info["failed_bearings"] else "none",
            "scheduled_slot": row["scheduled_slot"] if row["is_scheduled"] else "N/A",
        }
        recommendations.append(rec)

    rec_df = pd.DataFrame(recommendations)
    print(rec_df.to_string(index=False))

    # Save
    rec_df.to_csv(os.path.join(config.IMS_PROCESSED_DIR, "ims_recommendations.csv"), index=False)
    print(f"\n[IMS PIPELINE] Recommendations saved to {config.IMS_PROCESSED_DIR}/ims_recommendations.csv")

    return {
        "anomaly_scores": anomaly_scores,
        "failure_proba": failure_proba,
        "attention_weights": attention_weights,
        "schedule": schedule,
        "recommendations": rec_df,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference pipeline")
    parser.add_argument("--dataset", type=str, default="cmapss",
                        choices=["cmapss", "ims"],
                        help="Dataset to run inference on (default: cmapss)")
    parser.add_argument("--experiment", type=int, default=2,
                        help="IMS experiment number (only used with --dataset ims)")
    parser.add_argument("--test-mode", action="store_true",
                        help="Run with a small subset for testing")
    args = parser.parse_args()

    if args.dataset == "ims":
        results = run_ims_pipeline(experiment=args.experiment, test_mode=args.test_mode)
    else:
        results = run_pipeline(test_mode=args.test_mode)
