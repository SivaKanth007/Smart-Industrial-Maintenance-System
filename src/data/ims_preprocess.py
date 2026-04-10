"""
IMS Bearing Vibration Preprocessing
======================================
Extracts health features from raw 20kHz vibration snapshots and prepares
data for ML model consumption.

Pipeline:
1. Parse raw snapshots → time-domain signals
2. Extract time-domain features (RMS, kurtosis, crest factor, peak-to-peak)
3. Extract frequency-domain features (FFT dominant freq, spectral energy bands)
4. Compute degradation indicators (rolling RMS trend)
5. Assign pseudo-RUL labels (based on known failure point)
6. Create sliding windows for LSTM input
7. Normalize and split
"""

import os
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import MinMaxScaler
import joblib

import config


class IMSFeatureExtractor:
    """
    Extracts health-indicator features from raw vibration signals.

    For each 1-second vibration snapshot per channel, computes:
    - Time-domain: RMS, peak, peak-to-peak, crest factor, kurtosis, skewness, std
    - Frequency-domain: dominant freq, spectral centroid, energy bands, spectral kurtosis
    """

    def __init__(self, sampling_rate=None, n_fft_bands=None):
        self.sampling_rate = sampling_rate or config.IMS_SAMPLING_RATE
        self.n_fft_bands = n_fft_bands or config.IMS_FFT_BANDS

    def extract_time_features(self, signal):
        """
        Extract time-domain features from a vibration signal.

        Parameters
        ----------
        signal : np.ndarray, shape (n_points,)

        Returns
        -------
        dict of feature_name: value
        """
        rms = np.sqrt(np.mean(signal ** 2))
        peak = np.max(np.abs(signal))
        peak_to_peak = np.max(signal) - np.min(signal)
        crest_factor = peak / (rms + 1e-10)
        kurtosis = sp_stats.kurtosis(signal, fisher=True)
        skewness = sp_stats.skew(signal)
        std = np.std(signal)

        return {
            "rms": rms,
            "peak": peak,
            "peak_to_peak": peak_to_peak,
            "crest_factor": crest_factor,
            "kurtosis": kurtosis,
            "skewness": skewness,
            "std": std,
        }

    def extract_freq_features(self, signal):
        """
        Extract frequency-domain features from a vibration signal.

        Parameters
        ----------
        signal : np.ndarray, shape (n_points,)

        Returns
        -------
        dict of feature_name: value
        """
        n = len(signal)
        # Compute FFT
        fft_vals = np.abs(rfft(signal))
        fft_freqs = rfftfreq(n, d=1.0 / self.sampling_rate)

        # Avoid DC component
        fft_vals = fft_vals[1:]
        fft_freqs = fft_freqs[1:]

        total_energy = np.sum(fft_vals ** 2)

        # Dominant frequency
        dominant_idx = np.argmax(fft_vals)
        dominant_freq = fft_freqs[dominant_idx]

        # Spectral centroid
        spectral_centroid = (
            np.sum(fft_freqs * fft_vals) / (np.sum(fft_vals) + 1e-10)
        )

        # Energy in frequency bands
        max_freq = fft_freqs[-1]
        band_edges = np.linspace(0, max_freq, self.n_fft_bands + 1)
        band_energies = {}
        for i in range(self.n_fft_bands):
            mask = (fft_freqs >= band_edges[i]) & (fft_freqs < band_edges[i + 1])
            band_energy = np.sum(fft_vals[mask] ** 2) / (total_energy + 1e-10)
            band_energies[f"band_{i+1}_energy"] = band_energy

        # Spectral kurtosis
        spectral_kurtosis = sp_stats.kurtosis(fft_vals, fisher=True)

        result = {
            "dominant_freq": dominant_freq,
            "spectral_centroid": spectral_centroid,
            "spectral_kurtosis": spectral_kurtosis,
        }
        result.update(band_energies)
        return result

    def extract_snapshot_features(self, snapshot_data, channel_names):
        """
        Extract all features from one snapshot (all channels).

        Parameters
        ----------
        snapshot_data : dict
            Keys are channel names, values are signal arrays.
        channel_names : list[str]

        Returns
        -------
        dict of feature_name: value (flat, all channels)
        """
        features = {}
        for ch_name in channel_names:
            if ch_name not in snapshot_data:
                continue
            signal = snapshot_data[ch_name]
            if not isinstance(signal, np.ndarray) or len(signal) < 10:
                continue

            # Time-domain features
            time_feats = self.extract_time_features(signal)
            for feat_name, val in time_feats.items():
                features[f"{ch_name}_{feat_name}"] = val

            # Frequency-domain features
            freq_feats = self.extract_freq_features(signal)
            for feat_name, val in freq_feats.items():
                features[f"{ch_name}_{feat_name}"] = val

        return features


class IMSPreprocessor:
    """
    End-to-end preprocessor for IMS bearing vibration data.

    Steps:
    1. Extract features from raw snapshots
    2. Dynamic variance-based sensor filtering (no hard-coded drops)
    3. Compute pseudo-RUL labels
    4. Min-max normalization
    5. Create sliding windows
    6. Temporal train/val/test split
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_columns = []
        self.fitted = False
        self.extractor = IMSFeatureExtractor()

    def extract_features_from_snapshots(self, snapshots, channel_names, exp_info):
        """
        Extract features from raw snapshot data.

        Parameters
        ----------
        snapshots : list[dict]
            Each dict has 'file_index', 'filename', and channel arrays.
        channel_names : list[str]
        exp_info : dict
            Experiment metadata.

        Returns
        -------
        pd.DataFrame with features + pseudo-RUL
        """
        print(f"[IMS PREPROCESS] Extracting features from {len(snapshots)} snapshots...")

        rows = []
        for snap in snapshots:
            features = self.extractor.extract_snapshot_features(snap, channel_names)
            features["file_index"] = snap["file_index"]
            rows.append(features)

        df = pd.DataFrame(rows)
        df = df.sort_values("file_index").reset_index(drop=True)

        # Assign pseudo-RUL
        total = len(df)
        df["RUL"] = total - df["file_index"] - 1
        df["RUL"] = df["RUL"].clip(upper=config.IMS_MAX_RUL)

        # Add bearing_id column (using experiment's first failed bearing as target)
        df["unit_id"] = 1  # Single experiment = single unit

        n_features = len([c for c in df.columns if c not in ["file_index", "RUL", "unit_id"]])
        print(f"[IMS PREPROCESS] Extracted {n_features} features per snapshot")
        print(f"[IMS PREPROCESS] RUL range: [{df['RUL'].min()}, {df['RUL'].max()}]")

        return df

    def add_rolling_features(self, df, windows=None):
        """Add rolling RMS trend features for degradation tracking."""
        windows = windows or config.IMS_ROLLING_WINDOWS

        rms_cols = [c for c in df.columns if c.endswith("_rms")]
        new_cols = {}

        for col in rms_cols:
            for window in windows:
                roll_mean = df[col].rolling(window=window, min_periods=1).mean()
                roll_std = df[col].rolling(window=window, min_periods=1).std().fillna(0)
                new_cols[f"{col}_roll{window}_mean"] = roll_mean
                new_cols[f"{col}_roll{window}_std"] = roll_std

        for col_name, series in new_cols.items():
            df[col_name] = series

        print(f"[IMS PREPROCESS] Added {len(new_cols)} rolling trend features")
        return df

    def drop_low_variance(self, df, variance_threshold=0.001):
        """
        Dynamic variance-based feature filtering.
        No hard-coded sensor drops — adapts to any dataset.
        """
        exclude = ["file_index", "unit_id", "RUL"]
        feature_cols = [c for c in df.columns if c not in exclude]

        to_drop = []
        for col in feature_cols:
            if df[col].std() < variance_threshold:
                to_drop.append(col)

        if to_drop:
            df = df.drop(columns=to_drop)
            print(f"[IMS PREPROCESS] Dropped {len(to_drop)} low-variance features: "
                  f"{to_drop[:5]}{'...' if len(to_drop) > 5 else ''}")
        else:
            print("[IMS PREPROCESS] No low-variance features to drop")

        return df

    def normalize(self, df, fit=True):
        """Min-max normalize features."""
        exclude = ["file_index", "unit_id", "RUL"]
        self.feature_columns = [c for c in df.columns if c not in exclude]

        df_out = df.copy()
        if fit:
            df_out[self.feature_columns] = self.scaler.fit_transform(
                df[self.feature_columns]
            )
            self.fitted = True
            print(f"[IMS PREPROCESS] Fitted scaler on {len(self.feature_columns)} features")
        else:
            df_out[self.feature_columns] = self.scaler.transform(
                df[self.feature_columns]
            )
        return df_out

    def create_sequences(self, df, sequence_length=None):
        """
        Create sliding window sequences for LSTM models.

        Returns
        -------
        X : np.ndarray, shape (N, seq_len, n_features)
        y : np.ndarray, shape (N,) — RUL at end of each window
        """
        seq_len = sequence_length or config.IMS_SEQUENCE_LENGTH
        features = df[self.feature_columns].values
        rul = df["RUL"].values

        sequences = []
        labels = []

        if len(features) < seq_len:
            # Pad shorter sequences
            pad_len = seq_len - len(features)
            features_padded = np.vstack([
                np.zeros((pad_len, features.shape[1])),
                features
            ])
            sequences.append(features_padded)
            labels.append(rul[-1])
        else:
            for i in range(len(features) - seq_len + 1):
                sequences.append(features[i:i + seq_len])
                labels.append(rul[i + seq_len - 1])

        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        print(f"[IMS PREPROCESS] Created {len(X)} sequences of shape {X.shape[1:]}")
        return X, y

    def create_binary_labels(self, rul_values, horizon=30):
        """Create binary labels: 1 if failure within horizon snapshots."""
        return (rul_values <= horizon).astype(np.float32)

    def fit_transform(self, snapshots, channel_names, exp_info):
        """
        Full preprocessing pipeline.

        Returns
        -------
        dict with 'train', 'val', 'test' splits
        """
        print("=" * 60)
        print("Running IMS Preprocessing Pipeline")
        print("=" * 60)

        # Step 1: Extract features
        df = self.extract_features_from_snapshots(snapshots, channel_names, exp_info)

        # Step 2: Add rolling features
        df = self.add_rolling_features(df)

        # Step 3: Dynamic variance-based filtering
        df = self.drop_low_variance(df)

        # Step 4: Handle missing values
        df = df.fillna(0)

        # Step 5: Interleaved split for single time-series
        # For a single time-series that only fails once at the end, chronological
        # splitting starves the training set of failures. Interleaved splitting
        # ensures all phases of degradation are represented in train/val/test.
        n = len(df)
        indices = np.arange(n)
        test_idx = indices[::7]  # ~14% for test
        remaining = np.array([i for i in indices if i not in test_idx])
        val_idx = remaining[::6]  # ~14% for val
        train_idx = np.array([i for i in remaining if i not in val_idx])  # ~72% for train

        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()
        df_test = df.iloc[test_idx].copy()

        print(f"[IMS PREPROCESS] Interleaved Split: train={len(df_train)}, "
              f"val={len(df_val)}, test={len(df_test)}")

        # Step 6: Normalize (fit on train)
        df_train = self.normalize(df_train, fit=True)
        df_val = self.normalize(df_val, fit=False)
        df_test = self.normalize(df_test, fit=False)

        # Step 7: Create full normalized df for sequence creation
        full_df = pd.concat([df_train, df_val, df_test]).sort_values("file_index").reset_index(drop=True)

        # Create all sequences from full timeline
        all_X, all_y_rul = self.create_sequences(full_df)
        all_y_binary = self.create_binary_labels(all_y_rul)

        # Sequence end indices (corresponding to file_index)
        seq_len = config.IMS_SEQUENCE_LENGTH
        all_indices = np.arange(seq_len - 1, len(full_df))

        # Split sequences based on original split indices
        train_mask = np.isin(all_indices, train_idx)
        val_mask = np.isin(all_indices, val_idx)
        test_mask = np.isin(all_indices, test_idx)

        result = {
            "train": {
                "X": all_X[train_mask],
                "y_rul": all_y_rul[train_mask],
                "y_binary": all_y_binary[train_mask],
            },
            "val": {
                "X": all_X[val_mask],
                "y_rul": all_y_rul[val_mask],
                "y_binary": all_y_binary[val_mask],
            },
            "test": {
                "X": all_X[test_mask],
                "y_rul": all_y_rul[test_mask],
                "y_binary": all_y_binary[test_mask],
            },
        }

        for name in ["train", "val", "test"]:
            d = result[name]
            print(f"[IMS PREPROCESS] {name}: X={d['X'].shape}, "
                  f"RUL=[{d['y_rul'].min():.0f}, {d['y_rul'].max():.0f}], "
                  f"failure_rate={d['y_binary'].mean():.2%}")

        # Validate that training set has failure samples
        train_failure_rate = result["train"]["y_binary"].mean()
        if train_failure_rate == 0:
            print("[IMS PREPROCESS] WARNING: Training set has 0% failure rate! "
                  "Model cannot learn failure patterns. Check data split.")
        elif train_failure_rate < 0.01:
            print(f"[IMS PREPROCESS] WARNING: Training failure rate is very low "
                  f"({train_failure_rate:.2%}). Model may not generalize.")

        # Return split DataFrames too (for XGBoost/Survival which need flat features)
        splits_df = {"train": df_train, "val": df_val, "test": df_test}
        return result, df, splits_df

    def save(self, filepath=None):
        """Save fitted preprocessor."""
        filepath = filepath or os.path.join(config.MODELS_DIR, "ims_preprocessor.pkl")
        joblib.dump({
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "fitted": self.fitted,
        }, filepath)
        print(f"[IMS PREPROCESS] Saved preprocessor to {filepath}")

    def load(self, filepath=None):
        """Load a fitted preprocessor."""
        filepath = filepath or os.path.join(config.MODELS_DIR, "ims_preprocessor.pkl")
        state = joblib.load(filepath)
        self.scaler = state["scaler"]
        self.feature_columns = state["feature_columns"]
        self.fitted = state["fitted"]
        print(f"[IMS PREPROCESS] Loaded preprocessor from {filepath}")


if __name__ == "__main__":
    from ims_download import load_ims_experiment, download_ims

    download_ims()
    snapshots, ch_names, info = load_ims_experiment(experiment=2)

    preprocessor = IMSPreprocessor()
    data, df_features, _ = preprocessor.fit_transform(snapshots, ch_names, info)
    preprocessor.save()

    print("\n" + "=" * 60)
    print("IMS Preprocessing Summary")
    print("=" * 60)
    for split_name, split_data in data.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Sequences: {split_data['X'].shape[0]}")
        print(f"  Window: {split_data['X'].shape[1]} snapshots × {split_data['X'].shape[2]} features")
        print(f"  RUL: [{split_data['y_rul'].min():.0f}, {split_data['y_rul'].max():.0f}]")
        print(f"  Failure rate: {split_data['y_binary'].mean():.2%}")
