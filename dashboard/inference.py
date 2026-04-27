"""
Shared Inference Layer for the Dashboard
=========================================
All pages import from here so models are loaded ONCE per Streamlit session
and re-used across pages. Wrapped with @st.cache_resource for thread safety.

Public API
----------
- get_preprocessor()
- get_autoencoder()        -> (model, threshold)
- get_predictor()          -> model
- get_xgboost()            -> model
- get_survival()           -> model
- score_anomaly(window)    -> {score, threshold, is_anomaly, recon}
- predict_failure(window, horizon=None) -> {proba, attention}
- predict_rul(features)    -> rul_cycles
- survival_curve(features, times=None) -> DataFrame (cols = unit indices)
- pick_random_engine_window(data, unit_id=None) -> (window, unit_id, meta)
- inject_drift(window, sensor_idx, magnitude) -> drifted_window
"""

from __future__ import annotations

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.models.autoencoder import load_autoencoder
from src.models.lstm_predictor import load_predictor
from src.models.xgboost_rul import XGBoostRUL
from src.models.bayesian_survival import BayesianSurvival
from src.data.preprocess import DataPreprocessor


# ---------------------------------------------------------------------------
# Cached loaders — Streamlit guarantees these run at most once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading preprocessor…")
def get_preprocessor():
    path = os.path.join(config.MODELS_DIR, "preprocessor.pkl")
    if not os.path.exists(path):
        return None
    pre = DataPreprocessor()
    try:
        pre.load(path)
        return pre
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading autoencoder…")
def get_autoencoder():
    path = os.path.join(config.MODELS_DIR, "autoencoder.pt")
    if not os.path.exists(path):
        return None, None
    try:
        model = load_autoencoder(path)
        threshold = getattr(model, "threshold", None)
        return model, threshold
    except Exception as e:
        st.warning(f"Autoencoder load failed: {e}")
        return None, None


@st.cache_resource(show_spinner="Loading failure predictor…")
def get_predictor():
    path = os.path.join(config.MODELS_DIR, "lstm_predictor.pt")
    if not os.path.exists(path):
        return None
    try:
        return load_predictor(path)
    except Exception as e:
        st.warning(f"Predictor load failed: {e}")
        return None


@st.cache_resource(show_spinner="Loading XGBoost RUL model…")
def get_xgboost():
    path = os.path.join(config.MODELS_DIR, "xgboost_rul.pkl")
    if not os.path.exists(path):
        return None
    try:
        return XGBoostRUL.load(path)
    except Exception as e:
        st.warning(f"XGBoost load failed: {e}")
        return None


@st.cache_resource(show_spinner="Loading survival model…")
def get_survival():
    path = os.path.join(config.MODELS_DIR, "survival_model.pkl")
    if not os.path.exists(path):
        # Some pipelines name it bayesian_survival.pkl
        alt = os.path.join(config.MODELS_DIR, "bayesian_survival.pkl")
        if os.path.exists(alt):
            path = alt
        else:
            return None
    try:
        return BayesianSurvival.load(path)
    except Exception as e:
        st.warning(f"Survival load failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Inference wrappers — small, pure functions
# ---------------------------------------------------------------------------
def _to_batch(window: np.ndarray) -> torch.Tensor:
    """Convert a single (seq_len, feat) window into (1, seq_len, feat) tensor."""
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return torch.from_numpy(arr)


def score_anomaly(window: np.ndarray) -> dict:
    """Run autoencoder on a single window and return rich diagnostics."""
    model, threshold = get_autoencoder()
    if model is None:
        return {"available": False}

    x = _to_batch(window).to(config.DEVICE)
    model.eval()
    with torch.no_grad():
        recon = model(x).cpu().numpy()[0]
    inp = x.cpu().numpy()[0]
    per_feature_mse = ((inp - recon) ** 2).mean(axis=0)  # (n_features,)
    score = float(per_feature_mse.mean())
    thr = float(threshold) if threshold is not None else float("nan")
    return {
        "available": True,
        "score": score,
        "threshold": thr,
        "is_anomaly": bool(score > thr) if threshold is not None else None,
        "recon": recon,
        "input": inp,
        "per_feature_mse": per_feature_mse,
    }


def predict_failure(window: np.ndarray) -> dict:
    """Run LSTM predictor on a single window."""
    model = get_predictor()
    if model is None:
        return {"available": False}
    x = _to_batch(window)
    proba, attn = model.predict_proba(x)
    return {
        "available": True,
        "proba": float(proba[0]) if len(proba) else float("nan"),
        "attention": attn[0] if len(attn) else None,
    }


def predict_rul(features) -> dict:
    """Run XGBoost on a flat feature vector / matrix."""
    model = get_xgboost()
    if model is None:
        return {"available": False}
    X = features
    if isinstance(X, pd.Series):
        X = X.to_frame().T
    if isinstance(X, np.ndarray) and X.ndim == 1:
        X = X.reshape(1, -1)
    rul = model.predict(X)
    return {
        "available": True,
        "rul": float(rul[0]) if len(rul) == 1 else rul,
        "feature_names": model.feature_names,
    }


def survival_curve(df_features: pd.DataFrame, times=None):
    """Return survival probabilities at requested time points."""
    model = get_survival()
    if model is None:
        return None
    if times is None:
        times = np.arange(1, config.MAX_RUL + 1, 5)
    try:
        return model.predict_survival(df_features, times=times)
    except Exception as e:
        st.warning(f"Survival prediction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Helpers for synthetic / what-if scenarios
# ---------------------------------------------------------------------------
def pick_random_engine_window(data: dict, unit_id: int | None = None,
                              prefer_critical: bool = False, rng=None):
    """
    Pull a single (seq_len, feat) window out of the loaded dashboard data.

    Returns (window, unit_id, meta_dict).
    """
    rng = rng or np.random.default_rng()
    fleet = data.get("all")
    if fleet is None or len(fleet["X"]) == 0:
        return None, None, {}

    X = fleet["X"]
    units = fleet["unit_ids"]
    rul = fleet["y_rul"]
    binary = fleet["y_binary"]

    if unit_id is not None:
        mask = units == unit_id
        if not mask.any():
            return None, None, {}
        idx_pool = np.where(mask)[0]
    elif prefer_critical:
        idx_pool = np.where(binary == 1)[0]
        if len(idx_pool) == 0:
            idx_pool = np.arange(len(X))
    else:
        idx_pool = np.arange(len(X))

    pick = int(rng.choice(idx_pool))
    return X[pick], int(units[pick]), {
        "rul_true": float(rul[pick]),
        "is_critical": bool(binary[pick] == 1),
        "row_index": pick,
    }


def inject_drift(window: np.ndarray, sensor_idx: int, magnitude: float,
                 mode: str = "ramp") -> np.ndarray:
    """
    Inject an artificial fault into one feature of a window for what-if demos.

    mode='ramp'   : linear ramp from 0 → magnitude across the sequence
    mode='step'   : step change at the midpoint
    mode='spike'  : narrow spike near the end
    mode='noise'  : gaussian noise of given std
    """
    out = window.copy().astype(np.float32)
    seq_len, n_feat = out.shape
    if not (0 <= sensor_idx < n_feat):
        return out
    if mode == "ramp":
        out[:, sensor_idx] += np.linspace(0, magnitude, seq_len, dtype=np.float32)
    elif mode == "step":
        out[seq_len // 2:, sensor_idx] += magnitude
    elif mode == "spike":
        spike = np.zeros(seq_len, dtype=np.float32)
        spike[-3:] = magnitude
        out[:, sensor_idx] += spike
    elif mode == "noise":
        rng = np.random.default_rng(0)
        # numpy requires scale >= 0 — interpret magnitude as noise std
        out[:, sensor_idx] += rng.normal(
            0, abs(magnitude), seq_len).astype(np.float32)
    return out


def model_availability() -> dict:
    """Quick health-check used in headers — does NOT trigger heavy loads."""
    return {
        "preprocessor": os.path.exists(os.path.join(config.MODELS_DIR, "preprocessor.pkl")),
        "autoencoder":  os.path.exists(os.path.join(config.MODELS_DIR, "autoencoder.pt")),
        "predictor":    os.path.exists(os.path.join(config.MODELS_DIR, "lstm_predictor.pt")),
        "xgboost":      os.path.exists(os.path.join(config.MODELS_DIR, "xgboost_rul.pkl")),
        "survival":     (os.path.exists(os.path.join(config.MODELS_DIR, "survival_model.pkl"))
                         or os.path.exists(os.path.join(config.MODELS_DIR, "bayesian_survival.pkl"))),
    }


# ---------------------------------------------------------------------------
# Engineered tabular features (for XGBoost RUL & Survival pages)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Engineering tabular features (one-time)…")
def get_engineered_test_features() -> pd.DataFrame | None:
    """
    Run the same feature engineering pipeline used at training, and return a
    DataFrame aligned with what the XGBoost / Survival models expect. Cached
    so it only runs once per session.
    """
    from src.data.download import load_cmapss_train, load_cmapss_test
    from src.data.feature_engineering import FeatureEngineer
    try:
        train_df = load_cmapss_train()
        test_df, rul_true = load_cmapss_test()
    except Exception as e:
        st.warning(f"Couldn't load raw C-MAPSS for feature engineering: {e}")
        return None

    # The C-MAPSS test set has no RUL column — rul_true gives the RUL at each
    # engine's LAST observed cycle. Reconstruct per-cycle RUL.
    if "RUL" not in test_df.columns:
        max_cycle = test_df.groupby("unit_id")["cycle"].transform("max")
        # rul_true is indexed 0..N-1 keyed by unit_id 1..N
        rul_map = {uid: float(rul_true.iloc[uid - 1])
                   for uid in test_df["unit_id"].unique()
                   if (uid - 1) < len(rul_true)}
        test_df["RUL"] = (max_cycle - test_df["cycle"]
                          + test_df["unit_id"].map(rul_map)).astype(float)
        # Cap at the same MAX_RUL used in training
        test_df["RUL"] = test_df["RUL"].clip(upper=config.MAX_RUL)

    fe = FeatureEngineer()
    fe.engineer_features(train_df, fit=True)         # fit the regime clusterer
    test_eng = fe.engineer_features(test_df, fit=False)

    # The interaction-feature step picks the "top-5 sensors" by variance, which
    # may differ between train and test runs → some column names won't match
    # the trained models. Add any missing columns (as 0 = neutral) but DO NOT
    # drop extras — the survival model needs columns the XGB model doesn't,
    # and vice versa. Each consumer selects what it needs.
    try:
        xgb = get_xgboost()
        if xgb is not None and xgb.feature_names is not None:
            for c in xgb.feature_names:
                if c not in test_eng.columns:
                    test_eng[c] = 0.0
    except Exception:
        pass
    try:
        surv = get_survival()
        if surv is not None and surv.selected_feature_cols is not None:
            # Use small Gaussian noise (NOT zeros) — survival's
            # prepare_survival_data() drops zero-variance columns, which
            # would silently re-introduce the missing-column error.
            rng = np.random.default_rng(0)
            n_rows = len(test_eng)
            for c in surv.selected_feature_cols:
                if c not in test_eng.columns:
                    test_eng[c] = rng.normal(0.0, 1e-3, n_rows).astype(np.float32)
    except Exception:
        pass

    return test_eng

