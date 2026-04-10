"""
XGBoost RUL (Remaining Useful Life) Estimator
===============================================
Gradient boosted regression for RUL prediction using engineered features.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

import config


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA C-MAPSS asymmetric scoring function.

    Penalises late predictions (underestimated RUL, d >= 0) more than early
    predictions (overestimated RUL, d < 0), matching the published benchmark.

    d = y_pred - y_true
    s_i = exp(-d/13) - 1  if d < 0  (early prediction)
    s_i = exp(d/10)  - 1  if d >= 0 (late prediction)
    Score = sum(s_i)  — lower is better
    """
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


class XGBoostRUL:
    """
    XGBoost-based Remaining Useful Life regression model.

    Uses engineered tabular features (rolling stats, trends, interactions)
    for accurate RUL prediction with built-in feature importance.
    """

    def __init__(self, params=None):
        self.params = params or config.XGB_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """
        Train XGBoost RUL model.

        Parameters
        ----------
        X_train : np.ndarray or pd.DataFrame — flat feature matrix
        y_train : np.ndarray — RUL target values
        X_val, y_val : optional validation data
        feature_names : list[str], optional
        """
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_train_arr = X_train.values
        else:
            self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
            X_train_arr = X_train

        print(f"[XGBOOST] Training with {X_train_arr.shape[1]} features, "
              f"{X_train_arr.shape[0]} samples")

        # Setup evaluation
        eval_set = [(X_train_arr, y_train)]
        if X_val is not None and y_val is not None:
            X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            eval_set.append((X_val_arr, y_val))

        # Train model (with CUDA fallback)
        self.model = xgb.XGBRegressor(**self.params)
        try:
            self.model.fit(
                X_train_arr, y_train,
                eval_set=eval_set,
                verbose=10,
            )
        except Exception as e:
            if "cuda" in str(e).lower() or "gpu" in str(e).lower() or "device" in str(e).lower():
                print(f"[XGBOOST] GPU training failed ({e}), falling back to CPU...")
                cpu_params = {k: v for k, v in self.params.items() if k not in ("device", "tree_method")}
                self.params = cpu_params
                self.model = xgb.XGBRegressor(**cpu_params)
                self.model.fit(
                    X_train_arr, y_train,
                    eval_set=eval_set,
                    verbose=10,
                )
            else:
                raise

        # Feature importance
        importance = self.model.feature_importances_
        self.feature_importance = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        print(f"\n[XGBOOST] Top 10 features:")
        print(self.feature_importance.head(10).to_string(index=False))

        return self

    def predict(self, X):
        """Predict RUL for input features."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        predictions = self.model.predict(X_arr)
        # Clip predictions to valid range
        return np.clip(predictions, 0, config.MAX_RUL)

    def evaluate(self, X, y_true):
        """
        Evaluate model performance.

        Returns
        -------
        dict with RMSE, MAE, R², and score within tolerance metrics.
        """
        y_pred = self.predict(X)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Custom industry metric: % predictions within ±10 and ±20 cycles
        within_10 = np.mean(np.abs(y_true - y_pred) <= 10) * 100
        within_20 = np.mean(np.abs(y_true - y_pred) <= 20) * 100

        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "within_10_pct": within_10,
            "within_20_pct": within_20,
            "nasa_score": nasa_score(y_true, y_pred),
        }

        print(f"\n[XGBOOST] Evaluation Results:")
        print(f"  RMSE:        {rmse:.2f} cycles")
        print(f"  MAE:         {mae:.2f} cycles")
        print(f"  R²:          {r2:.4f}")
        print(f"  NASA Score:  {metrics['nasa_score']:.2f}  (lower = better)")
        print(f"  Within ±10 cycles: {within_10:.1f}%")
        print(f"  Within ±20 cycles: {within_20:.1f}%")

        return metrics

    def walk_forward_cv(self, X, y, n_splits=5, unit_ids=None):
        """
        Walk-forward time-series cross-validation.

        When `unit_ids` is provided, splits are made at unit boundaries so that
        no window from the same engine appears in both train and test folds.
        When omitted, falls back to positional splitting with a warning.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
        y : np.ndarray
        n_splits : int
        unit_ids : np.ndarray or None
            1-D array of unit IDs parallel to X/y rows. Pass this to avoid
            fold leakage when rows from the same unit span a fold boundary.
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        if unit_ids is not None:
            unique_units = np.unique(unit_ids)
            fold_size = max(1, len(unique_units) // (n_splits + 1))
            results = []
            for i in range(n_splits):
                train_units = unique_units[: fold_size * (i + 2)]
                test_units = unique_units[fold_size * (i + 2): fold_size * (i + 3)]
                if len(test_units) == 0:
                    break
                train_mask = np.isin(unit_ids, train_units)
                test_mask = np.isin(unit_ids, test_units)
                X_tr, y_tr = X_arr[train_mask], y[train_mask]
                X_te, y_te = X_arr[test_mask], y[test_mask]
                model = xgb.XGBRegressor(**self.params)
                model.fit(X_tr, y_tr, verbose=0)
                y_pred = np.clip(model.predict(X_te), 0, config.MAX_RUL)
                rmse = np.sqrt(mean_squared_error(y_te, y_pred))
                mae = mean_absolute_error(y_te, y_pred)
                results.append({
                    "fold": i + 1, "rmse": rmse, "mae": mae,
                    "n_train": len(X_tr), "n_test": len(X_te),
                })
        else:
            print("[XGBOOST] WARNING: walk_forward_cv called without unit_ids — "
                  "splitting on row index. Windows from the same engine may appear "
                  "in both train and test folds. Pass unit_ids to avoid this.")
            n = len(X_arr)
            fold_size = n // (n_splits + 1)
            results = []
            for i in range(n_splits):
                train_end = fold_size * (i + 2)
                test_start = train_end
                test_end = min(test_start + fold_size, n)
                if test_end <= test_start:
                    break
                X_tr, y_tr = X_arr[:train_end], y[:train_end]
                X_te, y_te = X_arr[test_start:test_end], y[test_start:test_end]
                model = xgb.XGBRegressor(**self.params)
                model.fit(X_tr, y_tr, verbose=0)
                y_pred = np.clip(model.predict(X_te), 0, config.MAX_RUL)
                rmse = np.sqrt(mean_squared_error(y_te, y_pred))
                mae = mean_absolute_error(y_te, y_pred)
                results.append({
                    "fold": i + 1, "rmse": rmse, "mae": mae,
                    "n_train": len(X_tr), "n_test": len(X_te),
                })

        df_results = pd.DataFrame(results)
        print(f"\n[XGBOOST] Walk-Forward CV Results ({n_splits} folds):")
        print(df_results.to_string(index=False))
        print(f"\n  Mean RMSE: {df_results['rmse'].mean():.2f} ± {df_results['rmse'].std():.2f}")
        print(f"  Mean MAE:  {df_results['mae'].mean():.2f} ± {df_results['mae'].std():.2f}")
        return df_results

    def save(self, filepath=None):
        """Save trained model."""
        filepath = filepath or os.path.join(config.MODELS_DIR, "xgboost_rul.pkl")
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance,
            "params": self.params,
        }, filepath)
        print(f"[XGBOOST] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath=None):
        """Load a trained model."""
        filepath = filepath or os.path.join(config.MODELS_DIR, "xgboost_rul.pkl")
        state = joblib.load(filepath)
        instance = cls(params=state["params"])
        instance.model = state["model"]
        instance.feature_names = state["feature_names"]
        instance.feature_importance = state["feature_importance"]
        print(f"[XGBOOST] Model loaded from {filepath}")
        return instance
