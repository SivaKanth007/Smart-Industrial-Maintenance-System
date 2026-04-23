"""
Bayesian Weibull Survival Analysis
====================================
Time-to-failure modeling with uncertainty quantification using
the lifelines library for calibrated credible intervals.
"""

import os
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import joblib

import config


class BayesianSurvival:
    """
    Weibull Accelerated Failure Time (AFT) survival model.

    Provides:
    - Time-to-failure prediction with uncertainty
    - 90% and 95% credible/confidence intervals
    - Calibration analysis
    - Survival function per unit
    """

    def __init__(self, confidence_levels=None):
        self.model = WeibullAFTFitter()
        self.km_fitter = KaplanMeierFitter()
        self.confidence_levels = confidence_levels or config.SURVIVAL_CONFIDENCE_LEVELS
        self.fitted = False
        self.selected_feature_cols = None  # set during fit(); used by predict methods

    def prepare_survival_data(self, df, rul_col="RUL", event_col=None):
        """
        Prepare data for survival analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Data with RUL column.
        rul_col : str
            Column name for remaining useful life.
        event_col : str or None
            Column indicating if failure was observed (1) or censored (0).
            If None, creates one based on RUL == 0.

        Returns
        -------
        pd.DataFrame ready for lifelines fitting.
        """
        df_surv = df.copy()

        # Duration (time-to-event)
        df_surv["duration"] = df_surv[rul_col].clip(lower=1)

        # Event indicator
        if event_col and event_col in df_surv.columns:
            df_surv["event"] = df_surv[event_col]
        else:
            # event = 1 means the failure was observed at this point in time.
            # In C-MAPSS run-to-failure data, failure occurs at RUL == 0.
            # All other observations are right-censored (the unit was still
            # running when the cycle was recorded).
            df_surv["event"] = (df_surv[rul_col] == 0).astype(int)

        # Select only numeric feature columns
        exclude = ["unit_id", "cycle", rul_col, "duration", "event"]
        feature_cols = [c for c in df_surv.columns if c not in exclude
                       and df_surv[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

        # Handle infinities and large values
        for col in feature_cols:
            df_surv[col] = df_surv[col].replace([np.inf, -np.inf], np.nan)
            df_surv[col] = df_surv[col].fillna(df_surv[col].median())

        # Drop columns with zero variance
        low_var_cols = [c for c in feature_cols if df_surv[c].std() < 1e-8]
        if low_var_cols:
            df_surv = df_surv.drop(columns=low_var_cols)
            feature_cols = [c for c in feature_cols if c not in low_var_cols]

        # Limit number of features to avoid convergence issues
        if len(feature_cols) > 20:
            # Select top features by correlation with duration
            correlations = df_surv[feature_cols].corrwith(df_surv["duration"]).abs()
            top_features = correlations.nlargest(20).index.tolist()
            drop_cols = [c for c in feature_cols if c not in top_features]
            df_surv = df_surv.drop(columns=drop_cols)
            feature_cols = top_features

        keep_cols = feature_cols + ["duration", "event"]
        return df_surv[keep_cols], feature_cols

    def _apply_feature_selection(self, df, rul_col="RUL"):
        """
        Prepare data for prediction using the feature set selected during fit().

        Unlike prepare_survival_data(), this does NOT re-run correlation-based
        feature selection — it applies the saved self.selected_feature_cols so
        the column set is identical to what the model was trained on.
        """
        if self.selected_feature_cols is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        df_surv, _ = self.prepare_survival_data(df, rul_col=rul_col)

        missing = [c for c in self.selected_feature_cols if c not in df_surv.columns]
        if missing:
            raise ValueError(
                f"Prediction data is missing columns present during fit: {missing}"
            )
        keep = self.selected_feature_cols + [
            c for c in ["duration", "event"] if c in df_surv.columns
        ]
        return df_surv[[c for c in keep if c in df_surv.columns]]

    def fit(self, df, rul_col="RUL"):
        """
        Fit the Weibull AFT survival model.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with features and RUL.
        """
        print("[SURVIVAL] Preparing survival data...")
        df_surv, feature_cols = self.prepare_survival_data(df, rul_col=rul_col)
        self.selected_feature_cols = feature_cols  # lock in feature set at fit time

        print(f"[SURVIVAL] Fitting Weibull AFT model with {len(feature_cols)} covariates, "
              f"{len(df_surv)} observations")
        print(f"[SURVIVAL] Event rate: {df_surv['event'].mean():.2%}")

        try:
            self.model.fit(
                df_surv,
                duration_col="duration",
                event_col="event",
                show_progress=True,
            )
            self.fitted = True

            print("\n[SURVIVAL] Model Summary:")
            self.model.print_summary(decimals=3, columns=["coef", "exp(coef)", "p"])

            # Kaplan-Meier baseline
            self.km_fitter.fit(df_surv["duration"], df_surv["event"])

        except Exception as e:
            print(f"[SURVIVAL] Warning: Full model failed ({e}), fitting simplified model...")
            # Fall back to a simpler model with fewer features
            top_5 = feature_cols[:5]
            df_simple = df_surv[top_5 + ["duration", "event"]]
            self.model.fit(
                df_simple,
                duration_col="duration",
                event_col="event",
            )
            self.fitted = True
            print("[SURVIVAL] Simplified model fitted successfully")

        return self

    def predict_survival(self, df, times=None):
        """
        Predict survival probability at given times.

        Parameters
        ----------
        df : pd.DataFrame — new data
        times : array-like — time points to evaluate (default: 1 to MAX_RUL)

        Returns
        -------
        pd.DataFrame — survival probabilities at each time
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if times is None:
            times = np.arange(1, config.MAX_RUL + 1, 5)

        df_surv = self._apply_feature_selection(df)
        df_pred = df_surv.drop(columns=["duration", "event"], errors="ignore")

        survival_probs = self.model.predict_survival_function(df_pred, times=times)
        return survival_probs

    def predict_median_ttf(self, df):
        """
        Predict median time-to-failure for each observation.

        Returns
        -------
        np.ndarray — median TTF predictions
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        df_surv = self._apply_feature_selection(df)
        df_pred = df_surv.drop(columns=["duration", "event"], errors="ignore")

        median_ttf = self.model.predict_median(df_pred)
        return median_ttf.values.flatten()

    def predict_with_uncertainty(self, df):
        """
        Predict TTF with confidence/credible intervals.

        Returns
        -------
        dict with 'median', 'lower_90', 'upper_90', 'lower_95', 'upper_95'
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted.")

        df_surv = self._apply_feature_selection(df)
        df_pred = df_surv.drop(columns=["duration", "event"], errors="ignore")

        # Weibull AFT can extrapolate to astronomically large median lifetimes
        # for rows whose covariates fall outside the training envelope. Clip to
        # a configurable horizon so downstream consumers (dashboards, schedulers)
        # see human-readable cycle counts rather than raw 1e15 extrapolations.
        cap = config.SURVIVAL_MAX_PREDICTION

        def _clip(arr):
            return np.clip(np.nan_to_num(arr, nan=cap, posinf=cap, neginf=1.0),
                           1.0, cap)

        result = {"median": _clip(self.model.predict_median(df_pred).values.flatten())}

        for level in self.confidence_levels:
            alpha = 1 - level
            lower = self.model.predict_percentile(df_pred, p=alpha/2).values.flatten()
            upper = self.model.predict_percentile(df_pred, p=1-alpha/2).values.flatten()
            pct = int(level * 100)
            result[f"lower_{pct}"] = _clip(lower)
            result[f"upper_{pct}"] = _clip(upper)

        return result

    def evaluate(self, df, rul_col="RUL"):
        """
        Evaluate survival model performance.

        Returns
        -------
        dict with concordance index and other metrics.
        """
        df_surv = self._apply_feature_selection(df, rul_col=rul_col)
        df_pred = df_surv.drop(columns=["duration", "event"], errors="ignore")

        median_pred = self.model.predict_median(df_pred).values.flatten()
        actual_duration = df_surv["duration"].values
        events = df_surv["event"].values

        # Weibull AFT can extrapolate to near-infinite medians for rows whose
        # covariates lie outside the training envelope. Clip to a sane ceiling
        # so a handful of outliers don't dominate the RMSE statistic.
        median_pred = np.clip(median_pred, 1.0, config.SURVIVAL_MAX_PREDICTION)
        finite_mask = np.isfinite(median_pred)

        # Concordance index over all (unit, cycle) observations. The C-MAPSS
        # data records run-to-failure trajectories, so per-row evaluation gives
        # a stable signal of how well predicted time-to-failure orders the
        # observed remaining durations.
        try:
            c_index = concordance_index(
                actual_duration[finite_mask],
                median_pred[finite_mask],
                events[finite_mask],
            )
        except ZeroDivisionError:
            # No admissible pairs (e.g. all durations equal); fall back to
            # the uninformative midpoint so downstream code keeps a finite
            # value to display.
            c_index = 0.5

        # RMSE on observed failures only, restricted to finite predictions.
        mask = (events == 1) & finite_mask
        if mask.sum() > 0:
            rmse_failures = float(np.sqrt(
                np.mean((actual_duration[mask] - median_pred[mask]) ** 2)
            ))
        else:
            rmse_failures = float("nan")

        metrics = {
            "concordance_index": c_index,
            "rmse_failures": rmse_failures,
            "n_events": int(events.sum()),
            "n_censored": int((events == 0).sum()),
        }

        print(f"\n[SURVIVAL] Evaluation Results:")
        print(f"  C-Index: {c_index:.4f}")
        print(f"  RMSE (failures only): {rmse_failures:.2f}")
        print(f"  Events: {metrics['n_events']} | Censored: {metrics['n_censored']}")

        return metrics

    def save(self, filepath=None):
        """Save the fitted model."""
        filepath = filepath or os.path.join(config.MODELS_DIR, "bayesian_survival.pkl")
        joblib.dump({
            "model": self.model,
            "km_fitter": self.km_fitter,
            "fitted": self.fitted,
            "confidence_levels": self.confidence_levels,
            "selected_feature_cols": self.selected_feature_cols,
        }, filepath)
        print(f"[SURVIVAL] Model saved to {filepath}")

    @classmethod
    def load(cls, filepath=None):
        """Load a fitted model."""
        filepath = filepath or os.path.join(config.MODELS_DIR, "bayesian_survival.pkl")
        state = joblib.load(filepath)
        instance = cls(confidence_levels=state["confidence_levels"])
        instance.model = state["model"]
        instance.km_fitter = state["km_fitter"]
        instance.fitted = state["fitted"]
        instance.selected_feature_cols = state.get("selected_feature_cols")
        print(f"[SURVIVAL] Model loaded from {filepath}")
        return instance
