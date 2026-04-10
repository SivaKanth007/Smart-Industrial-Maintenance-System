"""
MILP Maintenance Scheduler
============================
Mixed-Integer Linear Programming optimization for maintenance scheduling.
Minimizes expected cost subject to crew capacity and safety constraints.
"""

import os
import numpy as np
import pandas as pd
from pulp import (LpProblem, LpMinimize, LpVariable, LpBinary,
                  lpSum, PULP_CBC_CMD, LpStatus, value)

import config


class MaintenanceScheduler:
    """
    MILP-based maintenance scheduling optimizer.

    Objective: Minimize Σ(failure_risk × downtime_cost + maintenance_cost)
    Subject to:
    - Crew capacity: ≤ MAX_CONCURRENT_CREWS concurrent jobs per time slot
    - Each machine scheduled at most once per horizon
    - Critical-risk machines MUST be scheduled
    - Time window constraints
    """

    def __init__(self, n_crews=None, downtime_cost=None,
                 maintenance_cost=None, safety_threshold=None):
        self.n_crews = n_crews or config.MAX_CONCURRENT_CREWS
        self.downtime_cost = downtime_cost or config.DOWNTIME_COST_PER_HOUR
        self.maintenance_cost = maintenance_cost or config.MAINTENANCE_COST_BASE
        self.safety_threshold = safety_threshold or config.SAFETY_RISK_THRESHOLD

    def create_schedule(self, machine_risks, n_time_slots=None,
                         crew_availability=None, machine_names=None):
        """
        Solve the maintenance scheduling optimization problem.

        Parameters
        ----------
        machine_risks : dict or pd.Series
            {machine_id: failure_probability}
        n_time_slots : int
            Number of time slots in scheduling horizon.
        crew_availability : dict or None
            {time_slot: n_available_crews}, defaults to uniform.
        machine_names : dict or None
            {machine_id: display_name}

        Returns
        -------
        dict with:
            - schedule: pd.DataFrame
            - total_cost: float
            - summary: dict
            - status: str
        """
        n_time_slots = n_time_slots or config.SCHEDULING_HORIZON
        machines = list(machine_risks.keys())
        n_machines = len(machines)

        if crew_availability is None:
            crew_availability = {t: self.n_crews for t in range(n_time_slots)}

        if machine_names is None:
            machine_names = {m: f"Machine-{m}" for m in machines}

        print(f"\n[MILP] Scheduling {n_machines} machines across {n_time_slots} time slots")
        print(f"[MILP] Max crews: {self.n_crews}")

        # Handle empty input
        if n_machines == 0:
            empty_schedule = pd.DataFrame(columns=[
                "machine_id", "machine_name", "failure_risk", "risk_level",
                "risk_color", "is_scheduled", "scheduled_slot", "estimated_cost"
            ])
            return {
                "schedule": empty_schedule,
                "total_cost": 0.0,
                "summary": {
                    "total_machines": 0, "scheduled": 0, "not_scheduled": 0,
                    "critical": 0, "elevated": 0, "normal": 0,
                    "total_cost": 0.0, "status": "Optimal",
                },
                "status": "Optimal",
            }

        # =====================================================================
        # Problem Definition
        # =====================================================================
        prob = LpProblem("Maintenance_Scheduling", LpMinimize)

        # Decision variables: x[m][t] = 1 if machine m is maintained at time t
        x = {}
        for m in machines:
            x[m] = {}
            for t in range(n_time_slots):
                x[m][t] = LpVariable(f"x_{m}_{t}", cat=LpBinary)

        # =====================================================================
        # Objective: Minimize expected cost
        # =====================================================================
        # Cost of NOT maintaining = risk × downtime_cost
        # Cost of maintaining = maintenance_cost (fixed)
        # For each machine: if scheduled, pay maintenance_cost; if not, pay risk × downtime_cost

        # Total expected cost
        objective_terms = []
        for m in machines:
            risk = machine_risks[m]

            # Cost if maintained (in any slot)
            for t in range(n_time_slots):
                # Maintenance cost (with urgency factor: earlier slots cost less)
                urgency_factor = 1 + 0.1 * t  # slight preference for earlier maintenance
                maint_cost = self.maintenance_cost * urgency_factor
                objective_terms.append(x[m][t] * maint_cost)

            # Penalty for NOT scheduling high-risk machines
            not_scheduled = 1 - lpSum(x[m][t] for t in range(n_time_slots))

            # Expected cost of inaction
            inaction_cost = risk * self.downtime_cost * 8  # 8-hour shift assumption
            objective_terms.append(not_scheduled * inaction_cost)

        prob += lpSum(objective_terms), "Total_Expected_Cost"

        # =====================================================================
        # Constraints
        # =====================================================================

        # 1. Each machine scheduled at most once
        for m in machines:
            prob += (
                lpSum(x[m][t] for t in range(n_time_slots)) <= 1,
                f"MaxOnce_{m}"
            )

        # 2. Crew capacity per time slot
        for t in range(n_time_slots):
            available = crew_availability.get(t, self.n_crews)
            prob += (
                lpSum(x[m][t] for m in machines) <= available,
                f"CrewCap_{t}"
            )

        # 3. Critical machines MUST be scheduled
        for m in machines:
            if machine_risks[m] >= self.safety_threshold:
                prob += (
                    lpSum(x[m][t] for t in range(n_time_slots)) >= 1,
                    f"MustSchedule_{m}"
                )

        # =====================================================================
        # Solve
        # =====================================================================
        solver = PULP_CBC_CMD(msg=0, timeLimit=60)
        prob.solve(solver)

        status = LpStatus[prob.status]
        total_cost = value(prob.objective) if prob.status == 1 else float("inf")

        print(f"[MILP] Solution status: {status}")
        print(f"[MILP] Total expected cost: ${total_cost:,.2f}")

        # =====================================================================
        # Extract Schedule
        # =====================================================================
        schedule_rows = []
        for m in machines:
            risk = machine_risks[m]
            risk_level = self._get_risk_level(risk)
            scheduled_slot = None
            is_scheduled = False

            for t in range(n_time_slots):
                if value(x[m][t]) == 1:
                    scheduled_slot = t
                    is_scheduled = True
                    break

            schedule_rows.append({
                "machine_id": m,
                "machine_name": machine_names[m],
                "failure_risk": round(risk, 4),
                "risk_level": risk_level["label"],
                "risk_color": risk_level["color"],
                "is_scheduled": is_scheduled,
                "scheduled_slot": scheduled_slot,
                "estimated_cost": round(
                    self.maintenance_cost if is_scheduled
                    else risk * self.downtime_cost * 8, 2
                ),
            })

        schedule_df = pd.DataFrame(schedule_rows).sort_values(
            "failure_risk", ascending=False
        ).reset_index(drop=True)

        # Summary
        summary = {
            "total_machines": n_machines,
            "scheduled": schedule_df["is_scheduled"].sum(),
            "not_scheduled": n_machines - schedule_df["is_scheduled"].sum(),
            "critical": (schedule_df["risk_level"] == "Service Immediately").sum(),
            "elevated": (schedule_df["risk_level"] == "Schedule Soon").sum(),
            "normal": (schedule_df["risk_level"] == "Continue Monitoring").sum(),
            "total_cost": total_cost,
            "status": status,
        }

        self._print_schedule(schedule_df, summary)

        return {
            "schedule": schedule_df,
            "total_cost": total_cost,
            "summary": summary,
            "status": status,
        }

    def _get_risk_level(self, risk):
        """Categorize risk into levels."""
        if risk >= config.RISK_LEVELS["critical"]["threshold"]:
            return config.RISK_LEVELS["critical"]
        elif risk >= config.RISK_LEVELS["elevated"]["threshold"]:
            return config.RISK_LEVELS["elevated"]
        else:
            return config.RISK_LEVELS["normal"]

    def _print_schedule(self, df, summary):
        """Print formatted schedule."""
        print("\n" + "=" * 70)
        print("OPTIMIZED MAINTENANCE SCHEDULE")
        print("=" * 70)

        # Critical
        critical = df[df["risk_level"] == "Service Immediately"]
        if len(critical) > 0:
            print("\n[!!] SERVICE IMMEDIATELY:")
            for _, row in critical.iterrows():
                slot = f"Slot {row['scheduled_slot']}" if row['is_scheduled'] else "UNSCHEDULED!"
                print(f"   {row['machine_name']:20s} | Risk: {row['failure_risk']:.2%} | {slot}")

        # Elevated
        elevated = df[df["risk_level"] == "Schedule Soon"]
        if len(elevated) > 0:
            print("\n[!] SCHEDULE SOON:")
            for _, row in elevated.iterrows():
                slot = f"Slot {row['scheduled_slot']}" if row['is_scheduled'] else "Not scheduled"
                print(f"   {row['machine_name']:20s} | Risk: {row['failure_risk']:.2%} | {slot}")

        # Normal
        normal = df[df["risk_level"] == "Continue Monitoring"]
        if len(normal) > 0:
            print(f"\n[OK] CONTINUE MONITORING: {len(normal)} machines")

        print(f"\n{'-' * 70}")
        print(f"Summary: {summary['scheduled']}/{summary['total_machines']} scheduled | "
              f"Est. cost: ${summary['total_cost']:,.0f}")
        print(f"Risk breakdown: {summary['critical']} critical, "
              f"{summary['elevated']} elevated, {summary['normal']} normal")

    def create_gantt_data(self, schedule_result):
        """
        Convert schedule to Gantt chart format for visualization.

        Returns
        -------
        list of dicts for Plotly Gantt chart
        """
        df = schedule_result["schedule"]
        gantt_data = []

        for _, row in df[df["is_scheduled"]].iterrows():
            slot = row["scheduled_slot"]
            gantt_data.append({
                "Task": row["machine_name"],
                "Start": slot,
                "Finish": slot + 1,
                "Resource": row["risk_level"],
                "Risk": row["failure_risk"],
            })

        return gantt_data


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    n_machines = 15

    machine_risks = {
        i: np.clip(np.random.beta(2, 5), 0, 1)
        for i in range(1, n_machines + 1)
    }
    # Ensure some high-risk machines
    machine_risks[1] = 0.92
    machine_risks[5] = 0.85
    machine_risks[8] = 0.78

    scheduler = MaintenanceScheduler()
    result = scheduler.create_schedule(machine_risks)

    print("\n\nDetailed Schedule:")
    print(result["schedule"].to_string(index=False))
