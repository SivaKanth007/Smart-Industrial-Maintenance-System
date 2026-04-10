"""
Feature Engineering Module
===========================
Advanced feature extraction from C-MAPSS sensor data for ML models.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import config


class FeatureEngineer:
    """
    Feature engineering pipeline for industrial sensor data.

    Generates:
    - Rolling statistics (mean, std, min, max) over multiple windows
    - Trend features (linear slope)
    - Sensor cross-correlations
    - Operating regime clusters
    - Lag features
    """

    def __init__(self):
        self.regime_model = None
        self.n_regimes = 3

    def add_rolling_statistics(self, df, windows=None, stats=None):
        """
        Add rolling window statistics per unit.

        Parameters
        ----------
        df : pd.DataFrame
        windows : list[int]
            Window sizes for rolling calculations.
        stats : list[str]
            Statistics to compute ('mean', 'std', 'min', 'max').
        """
        windows = windows or config.ROLLING_WINDOWS
        stats = stats or config.ROLLING_STATS

        sensor_cols = [c for c in df.columns
                       if c.startswith("sensor_") and c not in config.SENSORS_TO_DROP]

        new_cols = {}
        for unit_id, group in df.groupby("unit_id"):
            group = group.sort_values("cycle")
            idx = group.index

            for window in windows:
                rolling = group[sensor_cols].rolling(window=window, min_periods=1)
                for stat in stats:
                    result = getattr(rolling, stat)()
                    for col in sensor_cols:
                        col_name = f"{col}_roll{window}_{stat}"
                        if col_name not in new_cols:
                            new_cols[col_name] = pd.Series(dtype=np.float64, index=df.index)
                        new_cols[col_name].loc[idx] = result[col].values

        for col_name, series in new_cols.items():
            df[col_name] = series

        print(f"[FEATURES] Added {len(new_cols)} rolling features "
              f"({len(windows)} windows × {len(stats)} stats × {len(sensor_cols)} sensors)")
        return df

    def add_trend_features(self, df, window=10):
        """
        Add linear trend (slope) features over recent cycles.

        Slope is computed as the linear regression coefficient over a rolling window.
        Uses vectorized computation instead of per-element np.polyfit for performance.
        """
        sensor_cols = [c for c in df.columns
                       if c.startswith("sensor_") and c not in config.SENSORS_TO_DROP
                       and "roll" not in c]

        new_cols = {}
        for unit_id, group in df.groupby("unit_id"):
            group = group.sort_values("cycle")
            idx = group.index

            for col in sensor_cols:
                values = group[col].values
                n = len(values)
                slopes = np.zeros(n)

                if n >= 2:
                    # Vectorized slope: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
                    # For a rolling window, use cumulative sums for efficiency
                    x_full = np.arange(n, dtype=np.float64)

                    # Compute slopes using rolling window with cumsum
                    for i in range(1, n):
                        start = max(0, i - window + 1)
                        seg_len = i - start + 1
                        x = np.arange(seg_len, dtype=np.float64)
                        y = values[start:i + 1]
                        x_mean = x.mean()
                        y_mean = y.mean()
                        denom = np.sum((x - x_mean) ** 2)
                        if denom > 1e-12:
                            slopes[i] = np.sum((x - x_mean) * (y - y_mean)) / denom

                col_name = f"{col}_trend_{window}"
                if col_name not in new_cols:
                    new_cols[col_name] = pd.Series(dtype=np.float64, index=df.index)
                new_cols[col_name].loc[idx] = slopes

        for col_name, series in new_cols.items():
            df[col_name] = series

        print(f"[FEATURES] Added {len(new_cols)} trend features")
        return df

    def add_operating_regimes(self, df, n_clusters=None, fit=True):
        """
        Cluster operational settings into regimes using K-Means.
        """
        n_clusters = n_clusters or self.n_regimes
        op_cols = [c for c in df.columns
                   if c.startswith("op_setting_") and c not in config.OP_SETTINGS_TO_DROP]

        if not op_cols:
            print("[FEATURES] No operational settings found for regime clustering")
            return df

        if fit:
            # IMPORTANT: Only call with fit=True on the TRAINING split.
            # Fitting on the full dataset (train+val+test) leaks val/test
            # distribution into the regime cluster assignments.
            self.regime_model = KMeans(
                n_clusters=n_clusters,
                random_state=config.RANDOM_SEED,
                n_init=10,
            )
            df["operating_regime"] = self.regime_model.fit_predict(df[op_cols])
        else:
            if self.regime_model is None:
                raise RuntimeError("Regime model not fitted.")
            df["operating_regime"] = self.regime_model.predict(df[op_cols])

        regime_counts = df["operating_regime"].value_counts().sort_index()
        print(f"[FEATURES] Clustered {len(op_cols)} settings into {n_clusters} regimes:")
        for regime, count in regime_counts.items():
            print(f"  Regime {regime}: {count} observations ({count/len(df):.1%})")

        return df

    def add_lag_features(self, df, lags=[1, 5, 10]):
        """Add lagged sensor values as features."""
        sensor_cols = [c for c in df.columns
                       if c.startswith("sensor_") and c not in config.SENSORS_TO_DROP
                       and "roll" not in c and "trend" not in c]

        new_count = 0
        for unit_id, group in df.groupby("unit_id"):
            group = group.sort_values("cycle")
            idx = group.index

            for lag in lags:
                for col in sensor_cols:
                    col_name = f"{col}_lag{lag}"
                    if col_name not in df.columns:
                        df[col_name] = np.nan
                    df.loc[idx, col_name] = group[col].shift(lag).values
                    new_count += 1

        # Fill NaN from lagging with forward fill
        lag_cols = [c for c in df.columns if "_lag" in c]
        df[lag_cols] = df[lag_cols].bfill().fillna(0)

        print(f"[FEATURES] Added {len(lag_cols)} lag features ({len(lags)} lags × {len(sensor_cols)} sensors)")
        return df

    def add_sensor_interactions(self, df, top_n=5):
        """
        Add pairwise interaction features for the top-N most variable sensors.
        """
        sensor_cols = [c for c in df.columns
                       if c.startswith("sensor_") and c not in config.SENSORS_TO_DROP
                       and "roll" not in c and "trend" not in c and "lag" not in c]

        # Select top-N by variance
        variances = df[sensor_cols].var().sort_values(ascending=False)
        top_sensors = variances.head(top_n).index.tolist()

        new_count = 0
        for i, s1 in enumerate(top_sensors):
            for s2 in top_sensors[i+1:]:
                df[f"{s1}_x_{s2}"] = df[s1] * df[s2]
                df[f"{s1}_div_{s2}"] = df[s1] / (df[s2] + 1e-8)
                new_count += 2

        print(f"[FEATURES] Added {new_count} interaction features from top-{top_n} sensors")
        return df

    def add_cycle_features(self, df):
        """
        DEPRECATED — do not call this method.

        cycle_norm = cycle / max_cycle encodes the unit's failure cycle (future
        information) as a feature, causing target leakage. This method is
        retained for reference only and raises an error if called.
        """
        raise RuntimeError(
            "add_cycle_features() is disabled: cycle_norm and cycle_squared encode "
            "the unit's failure cycle (future information) and cause target leakage. "
            "Remove this call from your pipeline."
        )

    def engineer_features(self, df, fit=True):
        """
        Run the complete feature engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
        fit : bool
            If True, fit regime model. If False, use previously fitted model.

        Returns
        -------
        pd.DataFrame with engineered features
        """
        print("=" * 60)
        print("Running Feature Engineering Pipeline")
        print("=" * 60)

        initial_cols = len(df.columns)

        # NOTE: add_cycle_features() is intentionally excluded — cycle_norm
        # encodes the failure cycle (future information) and causes target leakage.
        df = self.add_rolling_statistics(df)
        df = self.add_trend_features(df)
        df = self.add_operating_regimes(df, fit=fit)
        df = self.add_lag_features(df)
        df = self.add_sensor_interactions(df)

        final_cols = len(df.columns)
        print(f"\n[FEATURES] Total features: {initial_cols} -> {final_cols} "
              f"(+{final_cols - initial_cols} engineered)")

        return df

    def get_flat_features(self, df):
        """
        Get flat feature matrix for tabular models (e.g., XGBoost).

        Returns
        -------
        X : pd.DataFrame (features only)
        y_rul : pd.Series
        """
        exclude = ["unit_id", "cycle", "RUL"]
        feature_cols = [c for c in df.columns if c not in exclude]
        return df[feature_cols], df["RUL"]


if __name__ == "__main__":
    from download import load_cmapss_train

    df = load_cmapss_train()
    fe = FeatureEngineer()
    df_engineered = fe.engineer_features(df)

    print(f"\nFinal dataframe shape: {df_engineered.shape}")
    print(f"Feature columns: {len([c for c in df_engineered.columns if c not in ['unit_id', 'cycle', 'RUL']])}")
    print(f"\nSample:")
    print(df_engineered.head())
