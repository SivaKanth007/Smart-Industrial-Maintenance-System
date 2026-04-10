"""
Unit Tests — Data Preprocessing
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.data.preprocess import DataPreprocessor
from src.data.synthetic_cmapss import SyntheticCMAPSSGenerator


@pytest.fixture
def sample_data():
    """Create minimal C-MAPSS-like data for testing."""
    np.random.seed(42)
    rows = []
    for unit_id in range(1, 11):
        n_cycles = np.random.randint(50, 150)
        for cycle in range(1, n_cycles + 1):
            row = {"unit_id": unit_id, "cycle": cycle}
            for i in range(1, 4):
                row[f"op_setting_{i}"] = np.random.normal(0, 1)
            for i in range(1, 22):
                row[f"sensor_{i}"] = np.random.normal(50, 10)
            # Make some sensors constant
            row["sensor_1"] = 1.0
            row["sensor_5"] = 2.0
            rows.append(row)

    df = pd.DataFrame(rows)
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")
    df["RUL"] = (max_cycles - df["cycle"]).clip(upper=config.MAX_RUL)
    return df


@pytest.fixture
def preprocessor():
    return DataPreprocessor()


class TestDropConstantSensors:
    def test_removes_constant_columns(self, preprocessor, sample_data):
        result = preprocessor.drop_constant_sensors(sample_data)
        assert "sensor_1" not in result.columns
        assert "sensor_5" not in result.columns

    def test_keeps_variable_columns(self, preprocessor, sample_data):
        result = preprocessor.drop_constant_sensors(sample_data)
        assert "sensor_2" in result.columns
        assert "sensor_7" in result.columns


class TestNormalization:
    def test_values_in_range(self, preprocessor, sample_data):
        sample_data = preprocessor.drop_constant_sensors(sample_data)
        result = preprocessor.normalize(sample_data, fit=True)
        for col in preprocessor.feature_columns:
            assert result[col].min() >= -0.01  # allow tiny float error
            assert result[col].max() <= 1.01

    def test_transform_without_fit_raises(self, preprocessor, sample_data):
        sample_data = preprocessor.drop_constant_sensors(sample_data)
        with pytest.raises(RuntimeError):
            preprocessor.normalize(sample_data, fit=False)


class TestCreateSequences:
    def test_output_shape(self, preprocessor, sample_data):
        sample_data = preprocessor.drop_constant_sensors(sample_data)
        sample_data = preprocessor.normalize(sample_data, fit=True)
        X, y, unit_ids = preprocessor.create_sequences(sample_data, sequence_length=10)

        assert X.ndim == 3
        assert X.shape[1] == 10  # sequence length
        assert X.shape[2] == len(preprocessor.feature_columns)

    def test_rul_values_valid(self, preprocessor, sample_data):
        sample_data = preprocessor.drop_constant_sensors(sample_data)
        sample_data = preprocessor.normalize(sample_data, fit=True)
        _, y, _ = preprocessor.create_sequences(sample_data)

        assert y.min() >= 0
        assert y.max() <= config.MAX_RUL


class TestTemporalSplit:
    def test_no_unit_overlap(self, preprocessor, sample_data):
        train, val, test = preprocessor.temporal_split(sample_data)
        train_units = set(train["unit_id"].unique())
        val_units = set(val["unit_id"].unique())
        test_units = set(test["unit_id"].unique())

        assert train_units.isdisjoint(val_units)
        assert train_units.isdisjoint(test_units)
        assert val_units.isdisjoint(test_units)

    def test_all_units_covered(self, preprocessor, sample_data):
        train, val, test = preprocessor.temporal_split(sample_data)
        all_units = set(sample_data["unit_id"].unique())
        split_units = (set(train["unit_id"].unique()) |
                       set(val["unit_id"].unique()) |
                       set(test["unit_id"].unique()))
        assert all_units == split_units


class TestBinaryLabels:
    def test_binary_values(self, preprocessor):
        rul = np.array([100, 50, 30, 20, 10, 5, 0])
        labels = preprocessor.create_binary_labels(rul, horizon=30)
        expected = np.array([0, 0, 1, 1, 1, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(labels, expected)


class TestFullPipeline:
    def test_fit_transform(self, preprocessor, sample_data):
        result = preprocessor.fit_transform(sample_data, augment=False)

        assert "train" in result
        assert "val" in result
        assert "test" in result

        for split in result.values():
            assert "X" in split
            assert "y_rul" in split
            assert "y_binary" in split
            assert "unit_ids" in split
            assert split["X"].ndim == 3

    def test_fit_transform_with_augmentation(self, preprocessor, sample_data):
        result_no_aug = preprocessor.fit_transform(sample_data, augment=False)
        preprocessor2 = DataPreprocessor()
        result_aug = preprocessor2.fit_transform(sample_data, augment=True)

        # Augmented training set should be larger
        assert result_aug["train"]["X"].shape[0] > result_no_aug["train"]["X"].shape[0]
        # Val/test should be same size (no augmentation)
        assert result_aug["val"]["X"].shape[0] == result_no_aug["val"]["X"].shape[0]
        assert result_aug["test"]["X"].shape[0] == result_no_aug["test"]["X"].shape[0]


class TestEqualSplit:
    def test_split_ratios_roughly_equal(self, preprocessor, sample_data):
        train, val, test = preprocessor.temporal_split(sample_data)
        total_units = sample_data["unit_id"].nunique()
        train_units = train["unit_id"].nunique()
        val_units = val["unit_id"].nunique()
        test_units = test["unit_id"].nunique()

        # With 10 units and 70/15/15 split: train=7, val≈1-2, test≈1-2
        # At minimum each split must have at least 1 unit and all units accounted for
        assert train_units >= 1
        assert val_units >= 1
        assert test_units >= 1
        assert train_units + val_units + test_units == total_units
        # Train should be the largest split (~70%)
        assert train_units > val_units
        assert train_units > test_units


class TestSyntheticCMAPSSGenerator:
    def test_fit_and_generate(self, sample_data):
        gen = SyntheticCMAPSSGenerator(seed=42)
        gen.fit(sample_data)
        df_syn = gen.generate(n_units=5, start_unit_id=100)

        assert df_syn["unit_id"].nunique() == 5
        assert "RUL" in df_syn.columns
        assert "cycle" in df_syn.columns
        assert df_syn["RUL"].max() <= config.MAX_RUL
        assert len(df_syn) > 0

    def test_generate_for_augmentation(self, sample_data):
        gen = SyntheticCMAPSSGenerator(seed=42)
        df_syn = gen.generate_for_augmentation(sample_data)

        # Synthetic data should exist
        assert len(df_syn) > 0
        # Unit IDs should not overlap with real data
        real_max = sample_data["unit_id"].max()
        assert df_syn["unit_id"].min() > real_max

    def test_synthetic_schema_matches_real(self, sample_data):
        gen = SyntheticCMAPSSGenerator(seed=42)
        gen.fit(sample_data)
        df_syn = gen.generate(n_units=3, start_unit_id=100)

        # Check key columns exist
        for col in ["unit_id", "cycle", "RUL"]:
            assert col in df_syn.columns
        # Check at least some sensor columns
        sensor_cols = [c for c in df_syn.columns if c.startswith("sensor_")]
        assert len(sensor_cols) >= 10
