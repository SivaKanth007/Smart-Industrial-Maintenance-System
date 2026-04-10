"""
Synthetic C-MAPSS Sensor Data Generator
==========================================
Generates realistic synthetic turbofan engine degradation trajectories
using statistics learned from real C-MAPSS data. Designed to augment
training data only — val/test remain real.

Approach:
1. Learn per-sensor statistics (mean, std) at different lifecycle phases
   (early, mid, late) from real data.
2. Generate synthetic units with randomized lifetimes.
3. Apply degradation curves (exponential, linear, polynomial) per sensor.
4. Add correlated Gaussian noise matching real sensor covariance.
"""

import numpy as np
import pandas as pd

import config


class SyntheticCMAPSSGenerator:
    """
    Generates synthetic C-MAPSS sensor degradation trajectories
    from real data statistics.
    """

    def __init__(self, noise_level=None, seed=None):
        self.noise_level = noise_level or config.SYNTHETIC_NOISE_LEVEL
        self.rng = np.random.default_rng(seed or config.RANDOM_SEED + 1000)
        self.sensor_stats = None
        self.op_stats = None
        self.sensor_cov = None
        self.sensor_cols = []
        self.op_cols = []

    def fit(self, df_real):
        """
        Learn sensor statistics from real C-MAPSS data.

        Splits each unit's lifecycle into 3 phases (early/mid/late)
        and computes per-phase statistics.

        Parameters
        ----------
        df_real : pd.DataFrame
            Real C-MAPSS training data with unit_id, cycle, sensors, RUL.
        """
        self.sensor_cols = [c for c in df_real.columns
                            if c.startswith("sensor_") and c not in config.SENSORS_TO_DROP]
        self.op_cols = [c for c in df_real.columns
                        if c.startswith("op_setting_") and c not in config.OP_SETTINGS_TO_DROP]

        # Per-phase statistics (early = healthy, mid = degrading, late = near-failure)
        phase_dfs = {"early": [], "mid": [], "late": []}

        for _, group in df_real.groupby("unit_id"):
            n = len(group)
            third = n // 3
            phase_dfs["early"].append(group.iloc[:third])
            phase_dfs["mid"].append(group.iloc[third:2*third])
            phase_dfs["late"].append(group.iloc[2*third:])

        self.sensor_stats = {}
        for phase, dfs in phase_dfs.items():
            combined = pd.concat(dfs)
            self.sensor_stats[phase] = {
                "mean": combined[self.sensor_cols].mean().values,
                "std": combined[self.sensor_cols].std().values,
            }

        # Learn sensor covariance from full dataset for correlated noise
        self.sensor_cov = df_real[self.sensor_cols].cov().values
        # Regularize covariance to ensure positive definite
        self.sensor_cov += np.eye(len(self.sensor_cols)) * 1e-6

        # Operational settings statistics
        self.op_stats = {
            "mean": df_real[self.op_cols].mean().values,
            "std": df_real[self.op_cols].std().values,
        }

        # Lifecycle length statistics
        cycle_counts = df_real.groupby("unit_id")["cycle"].max()
        self.lifetime_mean = cycle_counts.mean()
        self.lifetime_std = cycle_counts.std()
        self.lifetime_min = cycle_counts.min()
        self.lifetime_max = cycle_counts.max()

        print(f"[SYNTHETIC CMAPSS] Fitted on {df_real['unit_id'].nunique()} real units")
        print(f"[SYNTHETIC CMAPSS] Sensors: {len(self.sensor_cols)}, "
              f"Lifetime: {self.lifetime_mean:.0f} ± {self.lifetime_std:.0f} cycles")

    def generate(self, n_units, start_unit_id=1):
        """
        Generate synthetic C-MAPSS units.

        Parameters
        ----------
        n_units : int
            Number of synthetic units to generate.
        start_unit_id : int
            Starting unit ID (to avoid collisions with real data).

        Returns
        -------
        pd.DataFrame
            Synthetic data with same columns as real C-MAPSS data.
        """
        if self.sensor_stats is None:
            raise RuntimeError("Call fit() first with real data.")

        rows = []
        degradation_models = config.SYNTHETIC_DEGRADATION_MODELS

        for i in range(n_units):
            unit_id = start_unit_id + i

            # Random lifetime within real data range
            n_cycles = int(np.clip(
                self.rng.normal(self.lifetime_mean, self.lifetime_std),
                self.lifetime_min,
                self.lifetime_max * 1.2,
            ))
            n_cycles = max(50, n_cycles)

            # Choose degradation model for this unit
            deg_model = self.rng.choice(degradation_models)

            # Generate degradation curve [0, 1] over lifecycle
            t = np.linspace(0, 1, n_cycles)
            if deg_model == "exponential":
                degradation = (np.exp(3 * t) - 1) / (np.exp(3) - 1)
            elif deg_model == "linear":
                degradation = t
            else:  # polynomial
                power = self.rng.uniform(1.5, 3.0)
                degradation = t ** power

            # Generate sensor values: interpolate from early→late stats + noise
            early_mean = self.sensor_stats["early"]["mean"]
            late_mean = self.sensor_stats["late"]["mean"]
            early_std = self.sensor_stats["early"]["std"]

            # Per-cycle sensor values
            for cycle_idx in range(n_cycles):
                d = degradation[cycle_idx]
                cycle = cycle_idx + 1

                # Interpolate sensor means based on degradation
                sensor_means = early_mean + d * (late_mean - early_mean)

                # Add correlated noise
                noise = self.rng.multivariate_normal(
                    np.zeros(len(self.sensor_cols)),
                    self.sensor_cov * self.noise_level,
                )

                sensor_values = sensor_means + noise

                # Operational settings (sampled from real distribution)
                op_values = self.rng.normal(self.op_stats["mean"], self.op_stats["std"])

                row = {"unit_id": unit_id, "cycle": cycle}
                for j, col in enumerate(self.op_cols):
                    row[col] = op_values[j]
                for j, col in enumerate(self.sensor_cols):
                    row[col] = sensor_values[j]

                rows.append(row)

        df_synthetic = pd.DataFrame(rows)

        # Add dropped sensor columns as constants (so schema matches)
        for col in config.SENSORS_TO_DROP:
            if col not in df_synthetic.columns:
                df_synthetic[col] = 0.0
        for col in config.OP_SETTINGS_TO_DROP:
            if col not in df_synthetic.columns:
                df_synthetic[col] = 0.0

        # Compute RUL
        max_cycles = df_synthetic.groupby("unit_id")["cycle"].max().reset_index()
        max_cycles.columns = ["unit_id", "max_cycle"]
        df_synthetic = df_synthetic.merge(max_cycles, on="unit_id")
        df_synthetic["RUL"] = df_synthetic["max_cycle"] - df_synthetic["cycle"]
        df_synthetic.drop("max_cycle", axis=1, inplace=True)
        df_synthetic["RUL"] = df_synthetic["RUL"].clip(upper=config.MAX_RUL)

        # Reorder columns to match real data
        expected_cols = ["unit_id", "cycle"]
        expected_cols += [f"op_setting_{i}" for i in range(1, 4)]
        expected_cols += [f"sensor_{i}" for i in range(1, 22)]
        expected_cols += ["RUL"]
        available = [c for c in expected_cols if c in df_synthetic.columns]
        df_synthetic = df_synthetic[available]

        print(f"[SYNTHETIC CMAPSS] Generated {n_units} synthetic units "
              f"({len(df_synthetic)} rows, degradation models: "
              f"{', '.join(degradation_models)})")

        return df_synthetic

    def generate_for_augmentation(self, df_train):
        """
        Generate synthetic data sized to hit the target ratio.

        Computes how many synthetic units are needed so that
        synthetic data = SYNTHETIC_TARGET_RATIO of total training data.

        Parameters
        ----------
        df_train : pd.DataFrame
            Real training data (already split).

        Returns
        -------
        pd.DataFrame
            Synthetic data ready to append to training set.
        """
        self.fit(df_train)

        real_rows = len(df_train)
        target_ratio = config.SYNTHETIC_TARGET_RATIO
        # synthetic / (real + synthetic) = target_ratio
        # synthetic = target_ratio * real / (1 - target_ratio)
        target_synthetic_rows = int(target_ratio * real_rows / (1 - target_ratio))

        # Estimate rows per unit from real data
        avg_rows_per_unit = real_rows / df_train["unit_id"].nunique()
        n_units = max(1, int(target_synthetic_rows / avg_rows_per_unit))

        # Start unit IDs after real data
        start_id = int(df_train["unit_id"].max()) + 1

        df_synthetic = self.generate(n_units, start_unit_id=start_id)

        actual_ratio = len(df_synthetic) / (real_rows + len(df_synthetic))
        print(f"[SYNTHETIC CMAPSS] Augmentation: {real_rows} real + "
              f"{len(df_synthetic)} synthetic = {real_rows + len(df_synthetic)} total "
              f"(synthetic ratio: {actual_ratio:.1%}, target: {target_ratio:.0%})")

        return df_synthetic
