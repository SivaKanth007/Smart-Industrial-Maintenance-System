"""
IMS Bearing Dataset Tests
============================
Unit tests for IMS data loading, feature extraction, preprocessing, and model compatibility.
"""

import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import config


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def synthetic_signal():
    """Generate a synthetic vibration signal."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 1024)
    signal = np.sin(2 * np.pi * 100 * t) + rng.normal(0, 0.1, 1024)
    return signal


@pytest.fixture
def synthetic_snapshots():
    """Generate synthetic IMS-like snapshot data."""
    rng = np.random.default_rng(42)
    n_snapshots = 60
    channel_names = ["bearing1", "bearing2", "bearing3", "bearing4"]
    exp_info = config.IMS_EXPERIMENTS[2]

    snapshots = []
    for i in range(n_snapshots):
        snap = {"file_index": i, "filename": f"snap_{i:04d}.csv"}
        degradation = (i / n_snapshots) ** 2

        for ch_name in channel_names:
            t = np.linspace(0, 1, 1024)
            signal = rng.normal(0, 0.5, 1024) + np.sin(2 * np.pi * 100 * t) * 0.3

            # Add degradation to bearing1 (failed bearing)
            if ch_name == "bearing1":
                signal *= (1 + degradation * 3)
                n_impulses = int(degradation * 10)
                for _ in range(n_impulses):
                    pos = rng.integers(0, 1024)
                    signal[pos] += rng.normal(0, degradation * 5)

            snap[ch_name] = signal
        snapshots.append(snap)

    return snapshots, channel_names, exp_info


# ============================================================
# Test Feature Extraction
# ============================================================

class TestIMSFeatureExtraction:
    def test_time_features(self, synthetic_signal):
        from src.data.ims_preprocess import IMSFeatureExtractor
        extractor = IMSFeatureExtractor()
        features = extractor.extract_time_features(synthetic_signal)

        assert "rms" in features
        assert "peak" in features
        assert "kurtosis" in features
        assert "crest_factor" in features
        assert features["rms"] > 0
        assert features["peak"] > 0
        assert features["peak_to_peak"] > 0

    def test_freq_features(self, synthetic_signal):
        from src.data.ims_preprocess import IMSFeatureExtractor
        extractor = IMSFeatureExtractor()
        features = extractor.extract_freq_features(synthetic_signal)

        assert "dominant_freq" in features
        assert "spectral_centroid" in features
        assert "spectral_kurtosis" in features
        # Check frequency bands exist
        for i in range(1, config.IMS_FFT_BANDS + 1):
            assert f"band_{i}_energy" in features

    def test_snapshot_features(self, synthetic_snapshots):
        from src.data.ims_preprocess import IMSFeatureExtractor
        snapshots, channel_names, _ = synthetic_snapshots
        extractor = IMSFeatureExtractor()

        features = extractor.extract_snapshot_features(snapshots[0], channel_names)
        # Should have features for each channel
        n_time = 7   # rms, peak, peak_to_peak, crest_factor, kurtosis, skewness, std
        n_freq = 3 + config.IMS_FFT_BANDS  # dominant_freq, centroid, spec_kurt + bands
        expected_per_ch = n_time + n_freq
        assert len(features) == len(channel_names) * expected_per_ch


# ============================================================
# Test Preprocessor
# ============================================================

class TestIMSPreprocessor:
    def test_feature_extraction(self, synthetic_snapshots):
        from src.data.ims_preprocess import IMSPreprocessor
        snapshots, ch_names, exp_info = synthetic_snapshots

        preprocessor = IMSPreprocessor()
        df = preprocessor.extract_features_from_snapshots(snapshots, ch_names, exp_info)

        assert len(df) == len(snapshots)
        assert "RUL" in df.columns
        assert "file_index" in df.columns
        assert df["RUL"].max() <= config.IMS_MAX_RUL

    def test_pseudo_rul_labels(self, synthetic_snapshots):
        from src.data.ims_preprocess import IMSPreprocessor
        snapshots, ch_names, exp_info = synthetic_snapshots

        preprocessor = IMSPreprocessor()
        df = preprocessor.extract_features_from_snapshots(snapshots, ch_names, exp_info)

        # RUL should decrease over time
        assert df["RUL"].iloc[0] > df["RUL"].iloc[-1]
        # Last sample should have RUL = 0
        assert df["RUL"].iloc[-1] == 0

    def test_binary_labels(self, synthetic_snapshots):
        from src.data.ims_preprocess import IMSPreprocessor
        preprocessor = IMSPreprocessor()

        rul = np.array([100, 50, 30, 15, 5, 0], dtype=np.float32)
        labels = preprocessor.create_binary_labels(rul, horizon=30)

        assert labels[0] == 0  # RUL=100, healthy
        assert labels[2] == 1  # RUL=30, at threshold
        assert labels[-1] == 1  # RUL=0, failed

    def test_full_pipeline(self, synthetic_snapshots):
        from src.data.ims_preprocess import IMSPreprocessor
        snapshots, ch_names, exp_info = synthetic_snapshots

        preprocessor = IMSPreprocessor()
        data, df, _ = preprocessor.fit_transform(snapshots, ch_names, exp_info)

        assert "train" in data
        assert "val" in data
        assert "test" in data
        assert data["train"]["X"].ndim == 3  # (N, seq_len, features)
        assert data["train"]["y_rul"].ndim == 1
        assert data["train"]["y_binary"].ndim == 1

    def test_dynamic_variance_filtering(self, synthetic_snapshots):
        from src.data.ims_preprocess import IMSPreprocessor
        snapshots, ch_names, exp_info = synthetic_snapshots

        preprocessor = IMSPreprocessor()
        df = preprocessor.extract_features_from_snapshots(snapshots, ch_names, exp_info)

        # Add a constant column
        df["constant_feature"] = 5.0
        n_before = len(df.columns)
        df = preprocessor.drop_low_variance(df)
        n_after = len(df.columns)

        assert n_after < n_before  # constant column should be dropped
        assert "constant_feature" not in df.columns


# ============================================================
# Test Model Compatibility
# ============================================================

class TestIMSModelCompatibility:
    def test_autoencoder_accepts_ims_features(self, synthetic_snapshots):
        import torch
        from src.models.autoencoder import LSTMAutoencoder
        snapshots, ch_names, exp_info = synthetic_snapshots
        from src.data.ims_preprocess import IMSPreprocessor

        preprocessor = IMSPreprocessor()
        data, _, _ = preprocessor.fit_transform(snapshots, ch_names, exp_info)

        if len(data["train"]["X"]) > 0:
            X = data["train"]["X"]
            n_features = X.shape[2]
            model = LSTMAutoencoder(input_dim=n_features, seq_len=X.shape[1])
            x_tensor = torch.FloatTensor(X[:3])
            output = model(x_tensor)
            assert output.shape == x_tensor.shape

    def test_predictor_accepts_ims_features(self, synthetic_snapshots):
        import torch
        from src.models.lstm_predictor import LSTMPredictor
        from src.data.ims_preprocess import IMSPreprocessor

        snapshots, ch_names, exp_info = synthetic_snapshots
        preprocessor = IMSPreprocessor()
        data, _, _ = preprocessor.fit_transform(snapshots, ch_names, exp_info)

        if len(data["train"]["X"]) > 0:
            X = data["train"]["X"]
            n_features = X.shape[2]
            n_samples = min(3, X.shape[0])
            model = LSTMPredictor(input_dim=n_features)
            x_tensor = torch.FloatTensor(X[:n_samples])
            logits, attn = model(x_tensor)
            assert logits.shape == (n_samples,)


# ============================================================
# Test IMS Download (Fallback)
# ============================================================

class TestIMSDownload:
    def test_synthetic_generation(self, tmp_path):
        from src.data.ims_download import _generate_fallback_ims_data
        _generate_fallback_ims_data(str(tmp_path))

        # Check that experiment folders were created
        for exp_id, exp_info in config.IMS_EXPERIMENTS.items():
            folder = exp_info["folder"]
            exp_path = tmp_path / folder
            assert exp_path.is_dir()
            files = list(exp_path.iterdir())
            assert len(files) > 0
