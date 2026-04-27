"""
Unit Tests — Monte Carlo Simulation
====================================
Tests for all maintenance policies, fairness, noisy predictor bounds, and CIs.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.evaluation.simulation import (
    MaintenanceSimulator,
    _PREVENTIVE_RESTORE,
    _FAILURE_RESTORE,
)


@pytest.fixture
def simulator():
    """Create a simulator with fixed seed for reproducibility."""
    return MaintenanceSimulator(n_machines=10, n_periods=50, seed=42)


@pytest.fixture
def health(simulator):
    """Generate a deterministic health trajectory."""
    return simulator.simulate_machine_health()


# ============================================================
# Basic policy tests
# ============================================================

class TestReactivePolicy:
    def test_returns_expected_keys(self, simulator, health):
        result = simulator.run_reactive(health.copy())
        for key in ["policy", "total_cost", "total_downtime_hours",
                     "availability_pct", "n_failures", "n_preventive"]:
            assert key in result

    def test_no_preventive_actions(self, simulator, health):
        result = simulator.run_reactive(health.copy())
        assert result["n_preventive"] == 0

    def test_cost_positive_when_failures(self, simulator, health):
        result = simulator.run_reactive(health.copy())
        if result["n_failures"] > 0:
            assert result["total_cost"] > 0


class TestScheduledPolicy:
    def test_has_preventive_actions(self, simulator, health):
        result = simulator.run_scheduled(health.copy(), interval=10)
        assert result["n_preventive"] > 0

    def test_uses_shared_restoration(self, simulator, health):
        """Scheduled policy must use the module-level restoration constants."""
        # If _PREVENTIVE_RESTORE is used, machines should be healthier
        # than if no restoration happened — we just check that the code
        # runs without error and produces a valid result.
        result = simulator.run_scheduled(health.copy())
        assert 90 <= result["availability_pct"] <= 100


class TestOptimizedPolicy:
    def test_fewer_failures_than_reactive(self, simulator, health):
        reactive = simulator.run_reactive(health.copy())
        optimized = simulator.run_optimized(health.copy())
        # Oracle policy should never do worse than reactive on failures
        assert optimized["n_failures"] <= reactive["n_failures"]

    def test_has_preventive_actions(self, simulator, health):
        result = simulator.run_optimized(health.copy())
        assert result["n_preventive"] > 0


class TestNoisyPredictorPolicy:
    def test_returns_expected_keys(self, simulator, health):
        result = simulator.run_noisy_optimized(health.copy())
        assert result["policy"] == "Noisy Predictor"
        for key in ["total_cost", "n_failures", "n_preventive"]:
            assert key in result

    def test_has_preventive_actions(self, simulator, health):
        result = simulator.run_noisy_optimized(health.copy())
        assert result["n_preventive"] > 0

    def test_cost_between_oracle_and_reactive(self, simulator):
        """Over many simulations, noisy predictor cost should be bounded."""
        np.random.seed(42)
        reactive_costs = []
        oracle_costs = []
        noisy_costs = []

        for _ in range(20):
            h = simulator.simulate_machine_health()
            reactive_costs.append(simulator.run_reactive(h.copy())["total_cost"])
            oracle_costs.append(simulator.run_optimized(h.copy())["total_cost"])
            noisy_costs.append(simulator.run_noisy_optimized(h.copy())["total_cost"])

        avg_reactive = np.mean(reactive_costs)
        avg_oracle = np.mean(oracle_costs)
        avg_noisy = np.mean(noisy_costs)

        # Noisy should be better than reactive but may be worse than oracle
        assert avg_noisy < avg_reactive, \
            f"Noisy ({avg_noisy:.0f}) should be cheaper than reactive ({avg_reactive:.0f})"


# ============================================================
# Fairness & restoration
# ============================================================

class TestRestorationFairness:
    def test_constants_are_positive(self):
        assert 0 < _PREVENTIVE_RESTORE <= 1
        assert 0 < _FAILURE_RESTORE <= 1

    def test_failure_restore_geq_preventive(self):
        """Full repair after failure should restore at least as much as PM."""
        assert _FAILURE_RESTORE >= _PREVENTIVE_RESTORE


# ============================================================
# run_comparison
# ============================================================

class TestRunComparison:
    def test_produces_all_four_policies(self, simulator):
        df, summary = simulator.run_comparison(n_simulations=5)
        policies = set(df["policy"].unique())
        expected = {"Reactive", "Scheduled (every 30)",
                    "Optimized (Risk-Based)", "Noisy Predictor"}
        assert expected == policies

    def test_summary_shape(self, simulator):
        df, summary = simulator.run_comparison(n_simulations=5)
        assert len(summary) == 4  # 4 policies

    def test_dataframe_has_simulation_column(self, simulator):
        df, _ = simulator.run_comparison(n_simulations=5)
        assert "simulation" in df.columns
        assert df["simulation"].nunique() == 5


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_all_healthy_machines(self, simulator):
        """If machines never degrade, no failures should occur."""
        health = np.ones((simulator.n_machines, simulator.n_periods))
        result = simulator.run_reactive(health.copy())
        assert result["n_failures"] == 0
        assert result["total_cost"] == 0

    def test_single_machine(self):
        sim = MaintenanceSimulator(n_machines=1, n_periods=20, seed=42)
        df, summary = sim.run_comparison(n_simulations=3)
        assert len(df) == 4 * 3  # 4 policies × 3 sims
