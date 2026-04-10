"""
Synthetic Data Generator
=========================
Generates realistic maintenance logs and operational context data
to complement the C-MAPSS sensor telemetry.
"""

import os
import numpy as np
import pandas as pd

import config


class SyntheticDataGenerator:
    """
    Generates synthetic maintenance records and operational context
    aligned with C-MAPSS unit timelines.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed or config.RANDOM_SEED)

    # -----------------------------------------------------------------
    # Failure Types
    # -----------------------------------------------------------------
    FAILURE_TYPES = {
        "bearing_fault":    {"probability": 0.35, "cost_range": (8000, 25000),  "downtime_hours": (8, 24)},
        "blade_erosion":    {"probability": 0.20, "cost_range": (15000, 50000), "downtime_hours": (12, 36)},
        "overheating":      {"probability": 0.25, "cost_range": (5000, 20000),  "downtime_hours": (4, 16)},
        "vibration_imbal":  {"probability": 0.12, "cost_range": (10000, 30000), "downtime_hours": (6, 20)},
        "sensor_drift":     {"probability": 0.08, "cost_range": (2000, 8000),   "downtime_hours": (2, 8)},
    }

    MAINTENANCE_ACTIONS = {
        "inspection":       {"cost_range": (500, 1500),   "duration_hours": (1, 3)},
        "lubrication":      {"cost_range": (300, 800),    "duration_hours": (0.5, 2)},
        "part_replacement": {"cost_range": (2000, 15000), "duration_hours": (4, 12)},
        "calibration":      {"cost_range": (800, 2500),   "duration_hours": (1, 4)},
        "full_overhaul":    {"cost_range": (20000, 60000),"duration_hours": (16, 48)},
    }

    def generate_maintenance_logs(self, df_train, n_logs_per_unit=None):
        """
        Generate realistic maintenance logs aligned with unit lifecycles.

        Parameters
        ----------
        df_train : pd.DataFrame
            C-MAPSS training data with unit_id, cycle, RUL columns.
        n_logs_per_unit : int or None
            Average number of maintenance events per unit.

        Returns
        -------
        pd.DataFrame
            Maintenance log with columns:
            unit_id, cycle, action_type, failure_type, cost, downtime_hours,
            was_planned, technician_id
        """
        logs = []
        units = df_train.groupby("unit_id")

        for unit_id, group in units:
            max_cycle = group["cycle"].max()

            # More maintenance near end of life
            if n_logs_per_unit is None:
                n_events = self.rng.poisson(lam=max(3, max_cycle // 50))
            else:
                n_events = self.rng.poisson(lam=n_logs_per_unit)

            # Generate maintenance events
            for _ in range(n_events):
                cycle = int(self.rng.uniform(1, max_cycle))
                rul_at_cycle = max(0, max_cycle - cycle)

                # Probability of failure event increases as RUL decreases
                failure_prob = max(0.05, 1 - (rul_at_cycle / max_cycle))

                if self.rng.random() < failure_prob * 0.3:
                    # Failure event
                    failure_type = self._sample_failure_type()
                    ft_info = self.FAILURE_TYPES[failure_type]
                    action = "part_replacement" if self.rng.random() > 0.3 else "full_overhaul"

                    cost = self.rng.uniform(*ft_info["cost_range"])
                    downtime = self.rng.uniform(*ft_info["downtime_hours"])
                    was_planned = False
                else:
                    # Planned maintenance
                    failure_type = "none"
                    action = self.rng.choice(["inspection", "lubrication", "calibration"])
                    act_info = self.MAINTENANCE_ACTIONS[action]

                    cost = self.rng.uniform(*act_info["cost_range"])
                    downtime = self.rng.uniform(*act_info["duration_hours"])
                    was_planned = True

                logs.append({
                    "unit_id": unit_id,
                    "cycle": cycle,
                    "action_type": action,
                    "failure_type": failure_type,
                    "cost_usd": round(cost, 2),
                    "downtime_hours": round(downtime, 1),
                    "was_planned": was_planned,
                    "technician_id": f"TECH-{self.rng.integers(1, 16):03d}",
                })

        # Final failure event at end of each unit's life
        for unit_id, group in units:
            max_cycle = group["cycle"].max()
            failure_type = self._sample_failure_type()
            ft_info = self.FAILURE_TYPES[failure_type]

            logs.append({
                "unit_id": unit_id,
                "cycle": max_cycle,
                "action_type": "full_overhaul",
                "failure_type": failure_type,
                "cost_usd": round(self.rng.uniform(*ft_info["cost_range"]) * 1.5, 2),
                "downtime_hours": round(self.rng.uniform(*ft_info["downtime_hours"]) * 1.5, 1),
                "was_planned": False,
                "technician_id": f"TECH-{self.rng.integers(1, 16):03d}",
            })

        df_logs = pd.DataFrame(logs).sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        print(f"[SYNTHETIC] Generated {len(df_logs)} maintenance logs for "
              f"{df_logs['unit_id'].nunique()} units")
        print(f"  Planned: {df_logs['was_planned'].sum()} | "
              f"Unplanned: {(~df_logs['was_planned']).sum()}")
        print(f"  Total cost: ${df_logs['cost_usd'].sum():,.0f}")

        return df_logs

    def generate_operational_context(self, df_train):
        """
        Generate operational context data for each unit.

        Returns
        -------
        pd.DataFrame with columns:
            unit_id, machine_type, install_date, production_line,
            max_operating_temp, rated_speed, crew_shift, priority_level
        """
        machine_types = ["Turbofan-A", "Turbofan-B", "Turbofan-C"]
        production_lines = [f"LINE-{i}" for i in range(1, 6)]
        shifts = ["morning", "afternoon", "night"]
        priority_levels = ["low", "medium", "high", "critical"]

        units = df_train["unit_id"].unique()
        records = []

        for uid in units:
            unit_data = df_train[df_train["unit_id"] == uid]
            max_cycle = unit_data["cycle"].max()

            records.append({
                "unit_id": uid,
                "machine_type": self.rng.choice(machine_types),
                "total_cycles": max_cycle,
                "production_line": self.rng.choice(production_lines),
                "max_operating_temp_c": round(self.rng.normal(550, 30), 1),
                "rated_speed_rpm": int(self.rng.normal(9000, 500)),
                "crew_shift": self.rng.choice(shifts),
                "priority_level": self.rng.choice(
                    priority_levels,
                    p=[0.2, 0.4, 0.3, 0.1]
                ),
                "last_overhaul_cycle": int(self.rng.uniform(0, max_cycle * 0.3)),
            })

        df_context = pd.DataFrame(records)
        print(f"[SYNTHETIC] Generated operational context for {len(df_context)} units")
        return df_context

    def generate_crew_schedule(self, n_slots=None):
        """
        Generate crew availability schedule.

        Returns
        -------
        pd.DataFrame with time_slot, available_crews, shift_type
        """
        n_slots = n_slots or config.SCHEDULING_HORIZON
        schedule = []

        for slot in range(n_slots):
            shift = ["morning", "afternoon", "night"][slot % 3]
            base_crews = config.MAX_CONCURRENT_CREWS

            # Vary availability by shift
            if shift == "night":
                available = max(1, base_crews - self.rng.integers(0, 2))
            elif shift == "morning":
                available = base_crews
            else:
                available = max(2, base_crews - self.rng.integers(0, 1))

            schedule.append({
                "time_slot": slot,
                "shift_type": shift,
                "available_crews": available,
                "overtime_available": self.rng.random() < 0.3,
            })

        return pd.DataFrame(schedule)

    def _sample_failure_type(self):
        """Sample a failure type based on probabilities."""
        types = list(self.FAILURE_TYPES.keys())
        probs = [self.FAILURE_TYPES[t]["probability"] for t in types]
        return self.rng.choice(types, p=probs)

    def generate_all(self, df_train, save_dir=None):
        """Generate and save all synthetic data."""
        save_dir = save_dir or config.SYNTHETIC_DATA_DIR

        print("=" * 60)
        print("Generating Synthetic Industrial Data")
        print("=" * 60)

        # Generate all datasets
        logs = self.generate_maintenance_logs(df_train)
        context = self.generate_operational_context(df_train)
        schedule = self.generate_crew_schedule()

        # Save to CSV
        logs.to_csv(os.path.join(save_dir, "maintenance_logs.csv"), index=False)
        context.to_csv(os.path.join(save_dir, "operational_context.csv"), index=False)
        schedule.to_csv(os.path.join(save_dir, "crew_schedule.csv"), index=False)

        print(f"\n[SYNTHETIC] All data saved to {save_dir}")
        return logs, context, schedule


if __name__ == "__main__":
    from download import load_cmapss_train

    df = load_cmapss_train()
    gen = SyntheticDataGenerator()
    logs, context, schedule = gen.generate_all(df)

    print("\n" + "=" * 60)
    print("Maintenance Logs Sample:")
    print(logs.head(10))

    print("\nOperational Context Sample:")
    print(context.head(10))

    print("\nCrew Schedule:")
    print(schedule)
