"""
Monte Carlo Simulation & Evaluation
=====================================
Evaluates maintenance policies through simulation and compares
reactive, scheduled, and optimized approaches.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

import config

# ---------------------------------------------------------------------------
# Shared constants — every policy uses the SAME restoration model so the
# comparison is fair (policies differ only by *when* they trigger, not by
# *how much* health is restored).
# ---------------------------------------------------------------------------
_PREVENTIVE_RESTORE = 0.6   # health bump after a planned service
_FAILURE_RESTORE    = 0.9   # health bump after a breakdown repair


class MaintenanceSimulator:
    """
    Monte Carlo simulator for comparing maintenance strategies.

    Policies:
    1. Reactive: Fix only after failure
    2. Scheduled: Fixed-interval preventive maintenance
    3. Optimized (Oracle): Risk-based with perfect health knowledge (upper bound)
    4. Noisy Predictor: Risk-based with realistic prediction noise (estimated real performance)

    Metrics:
    - Total cost (downtime + maintenance)
    - Total downtime hours
    - Equipment availability (%)
    - Number of failures
    """

    def __init__(self, n_machines=50, n_periods=100, seed=None):
        self.n_machines = n_machines
        self.n_periods = n_periods
        self.rng = np.random.default_rng(seed or config.RANDOM_SEED)

        # Cost parameters
        self.failure_cost = config.DOWNTIME_COST_PER_HOUR * 16  # avg 16hr downtime
        self.preventive_cost = config.MAINTENANCE_COST_BASE
        self.downtime_per_failure = 16  # hours
        self.downtime_per_preventive = 4  # hours

    def simulate_machine_health(self):
        """
        Simulate degradation trajectories for all machines.

        Returns
        -------
        health : np.ndarray, shape (n_machines, n_periods)
            Health score [0, 1], 0 = failed
        """
        health = np.ones((self.n_machines, self.n_periods))

        for m in range(self.n_machines):
            # Weibull-distributed failure time
            shape = self.rng.uniform(1.5, 3.0)
            scale = self.rng.uniform(40, 80)
            failure_time = int(self.rng.weibull(shape) * scale)
            failure_time = min(failure_time, self.n_periods)

            # Gradual degradation
            for t in range(self.n_periods):
                if t < failure_time:
                    # Exponential degradation curve
                    health[m, t] = max(0, 1 - (t / failure_time) ** 2)
                else:
                    health[m, t] = 0  # Failed

        return health

    def run_reactive(self, health):
        """Reactive policy: fix only after failure."""
        total_cost = 0
        total_downtime = 0
        failures = 0

        for m in range(self.n_machines):
            for t in range(1, self.n_periods):
                if health[m, t] == 0 and health[m, t-1] > 0:
                    total_cost += self.failure_cost
                    total_downtime += self.downtime_per_failure
                    failures += 1

        availability = (1 - total_downtime / (self.n_machines * self.n_periods * 24)) * 100
        return {
            "policy": "Reactive",
            "total_cost": total_cost,
            "total_downtime_hours": total_downtime,
            "availability_pct": min(100, availability),
            "n_failures": failures,
            "n_preventive": 0,
        }

    def run_scheduled(self, health, interval=30):
        """Scheduled policy: fixed-interval preventive maintenance."""
        total_cost = 0
        total_downtime = 0
        failures = 0
        n_preventive = 0

        for m in range(self.n_machines):
            last_maintenance = 0
            for t in range(1, self.n_periods):
                # Scheduled preventive maintenance
                if t - last_maintenance >= interval:
                    total_cost += self.preventive_cost
                    total_downtime += self.downtime_per_preventive
                    last_maintenance = t
                    n_preventive += 1
                    health[m, t:] = np.minimum(
                        health[m, t:] + _PREVENTIVE_RESTORE, 1
                    )

                # Still check for failures
                if health[m, t] <= 0.05:
                    total_cost += self.failure_cost
                    total_downtime += self.downtime_per_failure
                    failures += 1
                    health[m, t:] = np.minimum(health[m, t:] + _FAILURE_RESTORE, 1)

        availability = (1 - total_downtime / (self.n_machines * self.n_periods * 24)) * 100
        return {
            "policy": "Scheduled (every 30)",
            "total_cost": total_cost,
            "total_downtime_hours": total_downtime,
            "availability_pct": min(100, availability),
            "n_failures": failures,
            "n_preventive": n_preventive,
        }

    def run_optimized(self, health, risk_threshold=0.4):
        """
        Optimized policy: risk-based predictive maintenance.
        Maintenance is triggered when the simulated health score falls below
        ``risk_threshold``. The simulation uses the simulator's true health
        trajectory as a stand-in for the predictor's risk estimate, so this
        policy represents the upper bound of cost / downtime savings that a
        well-calibrated risk-based scheduler can achieve. Replacing the
        oracle health signal with noisy predictor outputs would give the
        realised performance of the trained model.
        """
        total_cost = 0
        total_downtime = 0
        failures = 0
        n_preventive = 0

        for m in range(self.n_machines):
            for t in range(1, self.n_periods):
                health_score = health[m, t]

                # Predictive maintenance when risk is high
                if health_score < risk_threshold and health_score > 0.05:
                    total_cost += self.preventive_cost
                    total_downtime += self.downtime_per_preventive
                    n_preventive += 1
                    # Same restoration model as scheduled policy
                    health[m, t:] = np.minimum(
                        health[m, t:] + _PREVENTIVE_RESTORE, 1
                    )

                elif health_score <= 0.05:
                    # Failure occurred despite predictions
                    total_cost += self.failure_cost
                    total_downtime += self.downtime_per_failure
                    failures += 1
                    health[m, t:] = np.minimum(health[m, t:] + _FAILURE_RESTORE, 1)

        availability = (1 - total_downtime / (self.n_machines * self.n_periods * 24)) * 100
        return {
            "policy": "Optimized (Risk-Based)",
            "total_cost": total_cost,
            "total_downtime_hours": total_downtime,
            "availability_pct": min(100, availability),
            "n_failures": failures,
            "n_preventive": n_preventive,
        }

    def run_noisy_optimized(self, health, risk_threshold=0.4, noise_sigma=0.15):
        """
        Noisy-predictor policy: risk-based maintenance with imperfect predictions.

        Same logic as ``run_optimized`` but adds Gaussian noise to the health
        signal before making decisions. This simulates the real-world scenario
        where the LSTM predictor has limited accuracy (F1 ≈ 0.87).

        Parameters
        ----------
        health : np.ndarray, shape (n_machines, n_periods)
        risk_threshold : float
            Maintenance triggered when *observed* health < threshold.
        noise_sigma : float
            Std-dev of the Gaussian noise added to the true health.
            Default 0.15 approximates the uncertainty of a strong but
            imperfect binary predictor (F1 ≈ 0.87).
        """
        total_cost = 0
        total_downtime = 0
        failures = 0
        n_preventive = 0

        for m in range(self.n_machines):
            for t in range(1, self.n_periods):
                true_health = health[m, t]

                # Simulate noisy prediction: the model doesn't know exact health
                observed_health = true_health + self.rng.normal(0, noise_sigma)
                observed_health = np.clip(observed_health, 0, 1)

                # Decide based on noisy observation
                if observed_health < risk_threshold and true_health > 0.05:
                    total_cost += self.preventive_cost
                    total_downtime += self.downtime_per_preventive
                    n_preventive += 1
                    health[m, t:] = np.minimum(
                        health[m, t:] + _PREVENTIVE_RESTORE, 1
                    )

                elif true_health <= 0.05:
                    # Actual failure — noise doesn't prevent this
                    total_cost += self.failure_cost
                    total_downtime += self.downtime_per_failure
                    failures += 1
                    health[m, t:] = np.minimum(health[m, t:] + _FAILURE_RESTORE, 1)

        availability = (1 - total_downtime / (self.n_machines * self.n_periods * 24)) * 100
        return {
            "policy": "Noisy Predictor",
            "total_cost": total_cost,
            "total_downtime_hours": total_downtime,
            "availability_pct": min(100, availability),
            "n_failures": failures,
            "n_preventive": n_preventive,
        }

    def run_comparison(self, n_simulations=100):
        """
        Run Monte Carlo comparison of all three policies.

        Returns
        -------
        pd.DataFrame — aggregated results across simulations
        """
        print("=" * 70)
        print(f"Monte Carlo Simulation ({n_simulations} runs)")
        print(f"  Machines: {self.n_machines} | Periods: {self.n_periods}")
        print("=" * 70)

        all_results = []

        for sim in range(n_simulations):
            health_base = self.simulate_machine_health()

            # Run each policy on an independent copy of the same health trajectory
            reactive = self.run_reactive(health_base.copy())
            scheduled = self.run_scheduled(health_base.copy())
            optimized = self.run_optimized(health_base.copy())
            noisy     = self.run_noisy_optimized(health_base.copy())

            for result in [reactive, scheduled, optimized, noisy]:
                result["simulation"] = sim
                all_results.append(result)

            if (sim + 1) % 20 == 0:
                print(f"  Completed {sim + 1}/{n_simulations} simulations")

        df = pd.DataFrame(all_results)

        # Aggregate
        summary = df.groupby("policy").agg({
            "total_cost": ["mean", "std"],
            "total_downtime_hours": ["mean", "std"],
            "availability_pct": ["mean"],
            "n_failures": ["mean"],
            "n_preventive": ["mean"],
        }).round(2)

        print(f"\n{'-' * 70}")
        print("SIMULATION RESULTS (averaged over simulations)")
        print(f"{'-' * 70}")
        print(summary.to_string())

        # Improvement stats
        reactive_cost = df[df["policy"] == "Reactive"]["total_cost"].mean()
        optimized_cost = df[df["policy"] == "Optimized (Risk-Based)"]["total_cost"].mean()
        cost_reduction = (1 - optimized_cost / reactive_cost) * 100

        reactive_downtime = df[df["policy"] == "Reactive"]["total_downtime_hours"].mean()
        optimized_downtime = df[df["policy"] == "Optimized (Risk-Based)"]["total_downtime_hours"].mean()
        downtime_reduction = (1 - optimized_downtime / reactive_downtime) * 100

        # Per-policy 95% confidence intervals
        print("\n  95% Confidence Intervals (cost):")
        for policy_name in ["Reactive", "Scheduled (every 30)",
                            "Optimized (Risk-Based)", "Noisy Predictor"]:
            costs = df[df["policy"] == policy_name]["total_cost"]
            if len(costs) > 1:
                ci = sp_stats.t.interval(
                    0.95, df=len(costs) - 1,
                    loc=costs.mean(), scale=costs.sem(),
                )
                print(f"  {policy_name}: ${ci[0]:,.0f} – ${ci[1]:,.0f}")

        # Noisy predictor vs reactive (realistic savings)
        noisy_cost = df[df["policy"] == "Noisy Predictor"]["total_cost"].mean()
        noisy_downtime = df[df["policy"] == "Noisy Predictor"]["total_downtime_hours"].mean()
        noisy_cost_red = (1 - noisy_cost / reactive_cost) * 100 if reactive_cost > 0 else 0
        noisy_dt_red = (1 - noisy_downtime / reactive_downtime) * 100 if reactive_downtime > 0 else 0

        print(f"\n[RESULTS] Optimized (Oracle) vs Reactive:")
        print(f"  Cost reduction:     {cost_reduction:.1f}%  (theoretical upper bound)")
        print(f"  Downtime reduction: {downtime_reduction:.1f}%")
        print(f"\n[RESULTS] Noisy Predictor vs Reactive:")
        print(f"  Cost reduction:     {noisy_cost_red:.1f}%  (realistic estimate)")
        print(f"  Downtime reduction: {noisy_dt_red:.1f}%")

        return df, summary

    def plot_comparison(self, df, save_path=None):
        """
        Plot comparison of maintenance policies.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Maintenance Policy Comparison — Monte Carlo Simulation",
                     fontsize=16, fontweight='bold', y=1.02)

        colors = {"Reactive": "#FF6B6B", "Scheduled (every 30)": "#FFA726",
                  "Optimized (Risk-Based)": "#66BB6A", "Noisy Predictor": "#42A5F5"}

        # 1. Total Cost
        ax = axes[0, 0]
        for policy in colors:
            data = df[df["policy"] == policy]["total_cost"]
            if len(data) > 0:
                ax.hist(data, alpha=0.5, label=policy, color=colors[policy], bins=20)
        ax.set_xlabel("Total Cost ($)")
        ax.set_title("Cost Distribution")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 2. Downtime
        ax = axes[0, 1]
        policy_names = [p for p in colors if len(df[df["policy"] == p]) > 0]
        means = [df[df["policy"] == p]["total_downtime_hours"].mean() for p in policy_names]
        stds = [df[df["policy"] == p]["total_downtime_hours"].std() for p in policy_names]
        bars = ax.bar(range(len(policy_names)), means, yerr=stds,
                      color=[colors[p] for p in policy_names], capsize=5)
        ax.set_xticks(range(len(policy_names)))
        ax.set_xticklabels(["Reactive", "Scheduled", "Oracle", "Noisy"], fontsize=9)
        ax.set_ylabel("Downtime (hours)")
        ax.set_title("Average Downtime")
        ax.grid(True, alpha=0.3)

        # 3. Failures
        ax = axes[1, 0]
        means = [df[df["policy"] == p]["n_failures"].mean() for p in policy_names]
        bars = ax.bar(range(len(policy_names)), means,
                      color=[colors[p] for p in policy_names])
        ax.set_xticks(range(len(policy_names)))
        ax.set_xticklabels(["Reactive", "Scheduled", "Oracle", "Noisy"], fontsize=9)
        ax.set_ylabel("Number of Failures")
        ax.set_title("Average Failures per Simulation")
        ax.grid(True, alpha=0.3)

        # 4. Availability
        ax = axes[1, 1]
        means = [df[df["policy"] == p]["availability_pct"].mean() for p in policy_names]
        bars = ax.bar(range(len(policy_names)), means,
                      color=[colors[p] for p in policy_names])
        ax.set_xticks(range(len(policy_names)))
        ax.set_xticklabels(["Reactive", "Scheduled", "Oracle", "Noisy"], fontsize=9)
        ax.set_ylabel("Availability (%)")
        ax.set_title("Equipment Availability")
        ax.set_ylim(90, 101)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SIM] Saved comparison plot to {save_path}")
        plt.close()

        return fig


if __name__ == "__main__":
    sim = MaintenanceSimulator(n_machines=50, n_periods=100)
    df, summary = sim.run_comparison(n_simulations=50)
    sim.plot_comparison(df, save_path=os.path.join(config.MODELS_DIR, "..", "comparison.png"))
