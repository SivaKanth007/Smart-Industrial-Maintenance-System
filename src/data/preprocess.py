"""
Data Preprocessing Pipeline
============================
Cleans, normalizes, and windows the C-MAPSS sensor data for model consumption.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

import config


class DataPreprocessor:
    """
    End-to-end preprocessor for C-MAPSS turbofan sensor data.

    Steps:
    1. Drop constant/near-constant sensors
    2. Min-max normalization
    3. Handle missing values
    4. Create sliding windows (30-cycle sequences)
    5. Compute piecewise-linear RUL labels
    6. Temporal train/val/test split
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        self.fitted = False

    def drop_constant_sensors(self, df, variance_threshold=0.001):
        """Remove sensors with near-zero variance."""
        sensor_cols = [c for c in df.columns if c.startswith("sensor_") or c.startswith("op_setting_")]

        # Drop known constant columns from config
        cols_to_drop = config.SENSORS_TO_DROP + config.OP_SETTINGS_TO_DROP
        cols_to_drop = [c for c in cols_to_drop if c in df.columns]

        # Also detect any additional near-constant columns
        for col in sensor_cols:
            if col not in cols_to_drop and df[col].std() < variance_threshold:
                cols_to_drop.append(col)

        df_clean = df.drop(columns=cols_to_drop, errors='ignore')
        dropped = [c for c in cols_to_drop if c in df.columns]
        print(f"[PREPROCESS] Dropped {len(dropped)} constant columns: {dropped}")

        return df_clean

    def normalize(self, df, fit=True):
        """
        Min-max normalize sensor and operational features.

        Parameters
        ----------
        df : pd.DataFrame
        fit : bool
            If True, fit the scaler on this data. If False, transform only.
        """
        # Identify feature columns (exclude unit_id, cycle, RUL)
        exclude = ["unit_id", "cycle", "RUL"]
        self.feature_columns = [c for c in df.columns if c not in exclude]

        df_out = df.copy()

        if fit:
            df_out[self.feature_columns] = self.scaler.fit_transform(
                df[self.feature_columns]
            )
            self.fitted = True
            print(f"[PREPROCESS] Fitted scaler on {len(self.feature_columns)} features")
        else:
            if not self.fitted:
                raise RuntimeError("Scaler not fitted. Call normalize with fit=True first.")
            df_out[self.feature_columns] = self.scaler.transform(
                df[self.feature_columns]
            )

        return df_out

    def handle_missing_values(self, df):
        """Handle missing values via forward-fill then backward-fill."""
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            df = df.groupby("unit_id", group_keys=False).apply(
                lambda g: g.ffill().bfill()
            )
            missing_after = df.isnull().sum().sum()
            print(f"[PREPROCESS] Filled {missing_before - missing_after} missing values")
        else:
            print("[PREPROCESS] No missing values detected")
        return df

    def create_sequences(self, df, sequence_length=None):
        """
        Create sliding window sequences for each unit.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed data with feature columns + RUL.
        sequence_length : int
            Window length (default from config).

        Returns
        -------
        X : np.ndarray, shape (N, sequence_length, n_features)
        y : np.ndarray, shape (N,)  — RUL at end of each window
        unit_ids : np.ndarray — unit ID for each sequence
        """
        seq_len = sequence_length or config.SEQUENCE_LENGTH

        sequences = []
        labels = []
        units = []

        for unit_id, group in df.groupby("unit_id"):
            group = group.sort_values("cycle")
            features = group[self.feature_columns].values
            rul = group["RUL"].values

            # Create sliding windows
            if len(features) < seq_len:
                # Pad shorter sequences
                pad_len = seq_len - len(features)
                features_padded = np.vstack([
                    np.zeros((pad_len, features.shape[1])),
                    features
                ])
                sequences.append(features_padded)
                labels.append(rul[-1])
                units.append(unit_id)
            else:
                for i in range(len(features) - seq_len + 1):
                    sequences.append(features[i:i + seq_len])
                    labels.append(rul[i + seq_len - 1])
                    units.append(unit_id)

        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        unit_ids = np.array(units)

        print(f"[PREPROCESS] Created {len(X)} sequences of shape {X.shape[1:]}")
        return X, y, unit_ids

    def temporal_split(self, df):
        """
        Temporal train/val/test split by unit ID.

        Splits units into train/val/test groups preserving temporal ordering.
        """
        unit_ids = df["unit_id"].unique()
        np.random.seed(config.RANDOM_SEED)
        np.random.shuffle(unit_ids)

        n = len(unit_ids)
        n_train = int(n * config.TRAIN_RATIO)
        n_val = int(n * config.VAL_RATIO)

        train_units = unit_ids[:n_train]
        val_units = unit_ids[n_train:n_train + n_val]
        test_units = unit_ids[n_train + n_val:]

        df_train = df[df["unit_id"].isin(train_units)].copy()
        df_val = df[df["unit_id"].isin(val_units)].copy()
        df_test = df[df["unit_id"].isin(test_units)].copy()

        print(f"[PREPROCESS] Split: train={len(train_units)} units ({len(df_train)} rows), "
              f"val={len(val_units)} units ({len(df_val)} rows), "
              f"test={len(test_units)} units ({len(df_test)} rows)")

        return df_train, df_val, df_test

    def create_binary_labels(self, rul_values, horizon=None):
        """
        Create binary labels: 1 if failure within horizon cycles, else 0.
        """
        h = horizon or config.PRED_FAILURE_HORIZON
        return (rul_values <= h).astype(np.float32)

    def fit_transform(self, df, augment=None):
        """
        Full preprocessing pipeline (fit on training data).

        Parameters
        ----------
        df : pd.DataFrame
            Raw C-MAPSS data (all subsets combined).
        augment : bool or None
            If True, generate synthetic data and inject into training set.
            Defaults to config.SYNTHETIC_AUGMENT.

        Returns
        -------
        dict with keys: 'train', 'val', 'test', each containing (X, y_rul, y_binary, unit_ids)
        """
        augment = augment if augment is not None else config.SYNTHETIC_AUGMENT

        print("=" * 60)
        print("Running Full Preprocessing Pipeline")
        print("=" * 60)

        # Step 1: Drop constant sensors
        df = self.drop_constant_sensors(df)

        # Step 2: Handle missing values
        df = self.handle_missing_values(df)

        # Step 3: Temporal split (BEFORE normalization to prevent leakage)
        df_train, df_val, df_test = self.temporal_split(df)

        # Step 3.5: Synthetic augmentation (training set only)
        if augment:
            from src.data.synthetic_cmapss import SyntheticCMAPSSGenerator
            print(f"\n[PREPROCESS] Augmenting training data with synthetic units "
                  f"(target ratio: {config.SYNTHETIC_TARGET_RATIO:.0%})...")

            generator = SyntheticCMAPSSGenerator()
            df_synthetic = generator.generate_for_augmentation(df_train)

            # Drop constant sensors from synthetic data to match schema
            cols_to_drop = config.SENSORS_TO_DROP + config.OP_SETTINGS_TO_DROP
            cols_to_drop = [c for c in cols_to_drop if c in df_synthetic.columns]
            df_synthetic = df_synthetic.drop(columns=cols_to_drop, errors='ignore')

            # Ensure column alignment
            missing_cols = set(df_train.columns) - set(df_synthetic.columns)
            for col in missing_cols:
                df_synthetic[col] = 0.0
            df_synthetic = df_synthetic[df_train.columns]

            real_train_count = len(df_train)

            # Step 4a: Fit scaler on REAL training data only (before synthetic merge)
            # so that synthetic value ranges don't bias the normalization.
            df_train = self.normalize(df_train, fit=True)
            df_synthetic = self.normalize(df_synthetic, fit=False)

            df_train = pd.concat([df_train, df_synthetic], ignore_index=True)
            print(f"[PREPROCESS] Training set: {real_train_count} real + "
                  f"{len(df_synthetic)} synthetic = {len(df_train)} total")

            # Transform val/test with the real-data-fitted scaler
            df_val = self.normalize(df_val, fit=False)
            df_test = self.normalize(df_test, fit=False)
        else:
            # No augmentation — fit scaler on train, transform val/test
            df_train = self.normalize(df_train, fit=True)
            df_val = self.normalize(df_val, fit=False)
            df_test = self.normalize(df_test, fit=False)

        # Step 5: Create sequences
        result = {}
        for name, data in [("train", df_train), ("val", df_val), ("test", df_test)]:
            X, y_rul, unit_ids = self.create_sequences(data)
            y_binary = self.create_binary_labels(y_rul)
            result[name] = {
                "X": X,
                "y_rul": y_rul,
                "y_binary": y_binary,
                "unit_ids": unit_ids,
            }
            print(f"[PREPROCESS] {name}: X={X.shape}, y_rul range=[{y_rul.min():.0f}, {y_rul.max():.0f}], "
                  f"failure_rate={y_binary.mean():.2%}")

        # Validate that training set has failure samples
        train_failure_rate = result["train"]["y_binary"].mean()
        if train_failure_rate == 0:
            print("[PREPROCESS] WARNING: Training set has 0% failure rate! "
                  "Model cannot learn failure patterns. Check data split.")
        elif train_failure_rate < 0.01:
            print(f"[PREPROCESS] WARNING: Training failure rate is very low "
                  f"({train_failure_rate:.2%}). Model may not generalize.")

        return result

    def save(self, filepath=None):
        """Save the fitted preprocessor (scaler + config)."""
        filepath = filepath or os.path.join(config.MODELS_DIR, "preprocessor.pkl")
        joblib.dump({
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "fitted": self.fitted,
        }, filepath)
        print(f"[PREPROCESS] Saved preprocessor to {filepath}")

    def load(self, filepath=None):
        """Load a fitted preprocessor."""
        filepath = filepath or os.path.join(config.MODELS_DIR, "preprocessor.pkl")
        state = joblib.load(filepath)
        self.scaler = state["scaler"]
        self.feature_columns = state["feature_columns"]
        self.fitted = state["fitted"]
        print(f"[PREPROCESS] Loaded preprocessor from {filepath}")


if __name__ == "__main__":
    from download import load_cmapss_train

    df = load_cmapss_train()
    preprocessor = DataPreprocessor()
    data = preprocessor.fit_transform(df)
    preprocessor.save()

    print("\n" + "=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    for split_name, split_data in data.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Sequences: {split_data['X'].shape[0]}")
        print(f"  Window: {split_data['X'].shape[1]} cycles × {split_data['X'].shape[2]} features")
        print(f"  RUL: [{split_data['y_rul'].min():.0f}, {split_data['y_rul'].max():.0f}]")
        print(f"  Failure rate: {split_data['y_binary'].mean():.2%}")

    # Save processed data
    for split_name, split_data in data.items():
        np.savez_compressed(
            os.path.join(config.PROCESSED_DATA_DIR, f"{split_name}_data.npz"),
            X=split_data["X"],
            y_rul=split_data["y_rul"],
            y_binary=split_data["y_binary"],
            unit_ids=split_data["unit_ids"],
        )
    print(f"\n[PREPROCESS] Saved processed data to {config.PROCESSED_DATA_DIR}")
