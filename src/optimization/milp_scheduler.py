"""
MILP Maintenance Scheduler — Robust, Always-Solvable
=====================================================
Mixed-Integer Linear Programming optimization for maintenance scheduling that
**never returns Infeasible** and **never returns infinite cost**.

Why this rewrite was needed
---------------------------
The previous formulation made "schedule every critical machine" a HARD
constraint (``Σ_t x[m][t] >= 1`` for every m with risk ≥ SAFETY_RISK_THRESHOLD).
When the failure predictor flags more critical machines than ``crews × slots``
of capacity (e.g. 107 critical engines vs 10 slots × 3 crews = 30 capacity),
the LP is mathematically infeasible and CBC returns ``Infeasible``. The old
code then assigned ``total_cost = float("inf")`` and propagated that to the
dashboard and recommendations CSV.

A real maintenance planner does not say "no schedule exists" — it says
"here is the best schedule I can produce with the resources you have, and
here is what you would need to fully cover the fleet." That is what this
module does.

Three-layer defense
-------------------
1. **MILP with soft penalty.** Critical machines no longer have a hard
   "must schedule" constraint. Instead an indicator ``z[m] ∈ {0, 1}`` is paid
   a large finite penalty (``CRITICAL_UNSCHEDULED_PENALTY``) when a critical
   machine is deferred. This makes the LP **always feasible** while still
   prioritizing critical machines above everything else in the objective.
2. **Cost-aware greedy fallback.** If the MILP solver errors, times out, or
   returns a non-optimal status, fall back to a deterministic greedy that
   sorts machines by risk and schedules them into the earliest available
   slot — *only* when scheduling is cheaper than inaction (or the machine
   is critical). Cannot fail.
3. **Output sanitization.** Every numeric in the returned dict is checked
   for ``NaN`` / ``inf`` / out-of-range before being handed to downstream
   consumers (dashboard, recommendations CSV, simulation).

Operator-facing diagnostics
---------------------------
The result dict now includes ``warnings`` and ``recommendations`` lists. When
critical demand exceeds capacity, the recommendations name the exact number
of crews or slots needed to clear the backlog. The dashboard / notebook can
surface these directly.

Public API is unchanged: ``MaintenanceScheduler().create_schedule(...)``
returns the same keys as before (plus extra diagnostics), so existing
callers (``run_pipeline.py``, the notebook, the dashboard, ``test_optimizer.py``)
continue to work without modification.
"""

import math

import numpy as np
import pandas as pd
from pulp import (LpBinary, LpMinimize, LpProblem, LpStatus, LpVariable,
                  PULP_CBC_CMD, lpSum, value)

import config


# Large but finite penalty for leaving a critical machine unscheduled.
# Must be large enough to dominate any realistic maintenance + inaction cost
# so the solver always prefers to schedule a critical machine when it can,
# but finite so the LP stays well-conditioned (no float("inf")).
CRITICAL_UNSCHEDULED_PENALTY = 1_000_000.0

# Sanity cap on the reported total cost. Anything above this is almost
# certainly an upstream bug; we clamp and warn rather than propagate.
MAX_REASONABLE_COST = 1e12

# Hours per maintenance shift — used to convert risk × hourly downtime cost
# into an expected-loss figure for an unscheduled machine.
SHIFT_HOURS = 8


class MaintenanceScheduler:
    """MILP-based maintenance scheduler with guaranteed feasibility.

    The optimizer minimizes:
        Σ_m Σ_t  x[m][t] · maintenance_cost · early_slot_bonus(t)
      + Σ_m     (1 - Σ_t x[m][t]) · risk[m] · downtime_cost · SHIFT_HOURS
      + Σ_{m∈critical}  z[m] · CRITICAL_UNSCHEDULED_PENALTY

    Subject to:
      - each machine scheduled at most once: Σ_t x[m][t] ≤ 1
      - crew capacity per slot:               Σ_m x[m][t] ≤ crew_availability[t]
      - critical soft constraint:             Σ_t x[m][t] + z[m] ≥ 1   (m ∈ critical)
      - z indicator upper bound:              z[m] + Σ_t x[m][t] ≤ 1   (m ∈ critical)

    The soft critical constraint with finite penalty is what guarantees the
    problem is always feasible.
    """

    def __init__(self, n_crews=None, downtime_cost=None,
                 maintenance_cost=None, safety_threshold=None):
        self.n_crews = n_crews or config.MAX_CONCURRENT_CREWS
        self.downtime_cost = downtime_cost or config.DOWNTIME_COST_PER_HOUR
        self.maintenance_cost = maintenance_cost or config.MAINTENANCE_COST_BASE
        self.safety_threshold = safety_threshold or config.SAFETY_RISK_THRESHOLD

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def create_schedule(self, machine_risks, n_time_slots=None,
                        crew_availability=None, machine_names=None):
        """Solve the scheduling problem. Always returns a valid result.

        Parameters
        ----------
        machine_risks : dict or pd.Series
            ``{machine_id: failure_probability}``. Values are sanitized:
            NaN / inf / None / out-of-range get clamped to ``[0, 1]``.
        n_time_slots : int, optional
            Scheduling horizon. Defaults to ``config.SCHEDULING_HORIZON``.
        crew_availability : dict or None
            ``{time_slot: n_available_crews}``; defaults to ``self.n_crews``
            uniformly.
        machine_names : dict or None
            ``{machine_id: display_name}``.

        Returns
        -------
        dict with keys:
            ``schedule``        : pd.DataFrame  - one row per machine, sorted by risk
            ``total_cost``      : float         - always finite, ≥ 0
            ``summary``         : dict          - counts + diagnostics
            ``status``          : str           - "Optimal" | "Best-Effort" | "Greedy-Fallback"
            ``solver_ok``       : bool          - True if the MILP solved
            ``warnings``        : list[str]
            ``recommendations`` : list[str]     - actionable next steps
        """
        n_time_slots = n_time_slots or config.SCHEDULING_HORIZON
        machine_risks = self._sanitize_input_risks(machine_risks)
        machines = list(machine_risks.keys())
        n_machines = len(machines)
        warnings_list = []
        recommendations_list = []

        if crew_availability is None:
            crew_availability = {t: self.n_crews for t in range(n_time_slots)}
        if machine_names is None:
            machine_names = {m: f"Machine-{m}" for m in machines}

        print(f"\n[MILP] Scheduling {n_machines} machines across {n_time_slots} time slots")
        print(f"[MILP] Max crews per slot: {self.n_crews}  "
              f"(total capacity: {sum(crew_availability.values())})")

        if n_machines == 0:
            return self._empty_result()

        # Capacity diagnostic — runs *before* the solve so warnings are
        # available even if the solver later succeeds with a Best-Effort plan.
        critical_machines = [m for m in machines
                             if machine_risks[m] >= self.safety_threshold]
        n_critical = len(critical_machines)
        total_capacity = sum(crew_availability.get(t, self.n_crews)
                             for t in range(n_time_slots))

        if n_critical > total_capacity:
            shortfall = n_critical - total_capacity
            msg = (f"Capacity shortfall: {n_critical} critical machines but only "
                   f"{total_capacity} scheduling slots — {shortfall} criticals "
                   f"will be deferred.")
            warnings_list.append(msg)
            crews_needed = math.ceil(n_critical / n_time_slots)
            horizon_needed = math.ceil(n_critical / self.n_crews)
            recommendations_list.append(
                f"Increase crews from {self.n_crews} to {crews_needed} to "
                f"clear all critical machines in {n_time_slots} slots.")
            recommendations_list.append(
                f"Or extend the scheduling horizon from {n_time_slots} to "
                f"{horizon_needed} slots with the current {self.n_crews} crews.")
            recommendations_list.append(
                f"Otherwise, the top {total_capacity} highest-risk critical "
                f"machines are serviced; the remaining {shortfall} are "
                f"deferred and reported in the schedule.")
            print(f"[MILP] {msg}")

        # Layer 1: MILP with soft critical penalty.
        milp_result = self._try_milp_solve(
            machines, machine_risks, critical_machines,
            n_time_slots, crew_availability)

        if milp_result is not None:
            schedule_df, solver_status = milp_result
            approach_used = "MILP"
        else:
            # Layer 2: cost-aware greedy fallback (cannot fail).
            print("[MILP] Solver did not return optimal — using greedy fallback.")
            warnings_list.append(
                "MILP solver did not converge; used greedy risk-sorted fallback.")
            schedule_df = self._greedy_schedule(
                machines, machine_risks, n_time_slots, crew_availability)
            solver_status = "Greedy-Fallback"
            approach_used = "Greedy"

        # Layer 3: sanitize and assemble final output.
        return self._build_final_result(
            schedule_df, machines, machine_risks, machine_names,
            n_machines, solver_status, approach_used,
            warnings_list, recommendations_list)

    # ------------------------------------------------------------------ #
    # Layer 1: MILP with soft penalty                                     #
    # ------------------------------------------------------------------ #
    def _try_milp_solve(self, machines, machine_risks, critical_machines,
                        n_time_slots, crew_availability):
        """Solve the LP. Returns ``(schedule_df, status)`` or ``None`` on failure.

        Never raises — any solver/import error falls through to the greedy
        layer.
        """
        try:
            prob = LpProblem("Maintenance_Scheduling", LpMinimize)

            # Decision variables.
            x = {m: {t: LpVariable(f"x_{m}_{t}", cat=LpBinary)
                     for t in range(n_time_slots)}
                 for m in machines}
            # Slack indicator: 1 iff a critical machine is left unscheduled.
            z = {m: LpVariable(f"unsched_crit_{m}", cat=LpBinary)
                 for m in critical_machines}

            # Objective.
            objective_terms = []
            for m in machines:
                risk = machine_risks[m]
                # Mild "schedule earlier" preference: bonus is small relative
                # to maintenance_cost (max ~5%) so it only breaks ties; never
                # makes late scheduling worse than not scheduling.
                for t in range(n_time_slots):
                    early_slot_bonus = 1.0 + 0.005 * t
                    objective_terms.append(
                        x[m][t] * (self.maintenance_cost * early_slot_bonus))

                # Expected inaction cost if this machine is not scheduled.
                not_scheduled = 1 - lpSum(x[m][t] for t in range(n_time_slots))
                inaction_cost = risk * self.downtime_cost * SHIFT_HOURS
                objective_terms.append(not_scheduled * inaction_cost)

            # Critical-deferral penalty (soft; large but finite).
            for m in critical_machines:
                objective_terms.append(z[m] * CRITICAL_UNSCHEDULED_PENALTY)

            prob += lpSum(objective_terms), "Total_Expected_Cost"

            # Constraints.
            for m in machines:
                prob += (lpSum(x[m][t] for t in range(n_time_slots)) <= 1,
                         f"MaxOnce_{m}")

            for t in range(n_time_slots):
                available = crew_availability.get(t, self.n_crews)
                prob += (lpSum(x[m][t] for m in machines) <= available,
                         f"CrewCap_{t}")

            for m in critical_machines:
                # If x_total = 0 then z must be 1; if x_total = 1 then the
                # upper bound below pins z at 0 (clean indicator semantics).
                prob += (lpSum(x[m][t] for t in range(n_time_slots)) + z[m] >= 1,
                         f"CriticalOrPenalty_{m}")
                prob += (z[m] + lpSum(x[m][t] for t in range(n_time_slots)) <= 1,
                         f"CriticalIndicatorUB_{m}")

            solver = PULP_CBC_CMD(msg=0, timeLimit=60)
            prob.solve(solver)

            status = LpStatus[prob.status]
            print(f"[MILP] Solver status: {status}")
            # status == 1 means LpStatusOptimal in PuLP. Anything else
            # (Infeasible, Unbounded, Not Solved, Undefined) → fall back.
            if prob.status != 1:
                return None

            rows = []
            for m in machines:
                scheduled_slot = None
                is_scheduled = False
                for t in range(n_time_slots):
                    v = value(x[m][t])
                    if v is not None and v > 0.5:
                        scheduled_slot = t
                        is_scheduled = True
                        break
                rows.append({"machine_id": m,
                             "scheduled_slot": scheduled_slot,
                             "is_scheduled": is_scheduled})

            return pd.DataFrame(rows), "MILP-Optimal"

        except Exception as e:  # noqa: BLE001 — any solver/runtime issue
            print(f"[MILP] Solver error: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Layer 2: cost-aware greedy fallback                                 #
    # ------------------------------------------------------------------ #
    def _greedy_schedule(self, machines, machine_risks, n_time_slots,
                         crew_availability):
        """Risk-sorted greedy. Cannot fail.

        Schedules a machine into the earliest open slot only when:
          - the machine is critical (always service if capacity allows), OR
          - the maintenance cost is less than the expected inaction cost.

        This avoids the trap of "schedule everything" — for very-low-risk
        machines, paying the maintenance fee is worse than the expected
        loss of leaving them alone.
        """
        sorted_machines = sorted(machines,
                                 key=lambda m: machine_risks[m], reverse=True)
        remaining = {t: crew_availability.get(t, self.n_crews)
                     for t in range(n_time_slots)}
        rows = []
        for m in sorted_machines:
            risk = machine_risks[m]
            inaction_cost = risk * self.downtime_cost * SHIFT_HOURS
            is_critical = risk >= self.safety_threshold
            worth_scheduling = is_critical or (self.maintenance_cost < inaction_cost)

            placed = False
            if worth_scheduling:
                for t in range(n_time_slots):
                    if remaining[t] > 0:
                        rows.append({"machine_id": m,
                                     "scheduled_slot": t,
                                     "is_scheduled": True})
                        remaining[t] -= 1
                        placed = True
                        break
            if not placed:
                rows.append({"machine_id": m,
                             "scheduled_slot": None,
                             "is_scheduled": False})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Layer 3: build & sanitize final output                              #
    # ------------------------------------------------------------------ #
    def _build_final_result(self, schedule_df, machines, machine_risks,
                            machine_names, n_machines, solver_status,
                            approach_used, warnings_list, recommendations_list):
        """Assemble the final result dict with all numerics sanitized."""
        sched_lookup = {row["machine_id"]: row
                        for row in schedule_df.to_dict("records")}

        enriched_rows = []
        total_cost_scheduled = 0.0
        total_cost_unscheduled = 0.0
        n_critical_unscheduled = 0

        for m in machines:
            row_data = sched_lookup.get(m, {})
            is_scheduled = bool(row_data.get("is_scheduled", False))
            scheduled_slot = row_data.get("scheduled_slot", None)
            risk = machine_risks[m]
            risk_level = self._get_risk_level(risk)

            if is_scheduled:
                cost = float(self.maintenance_cost)
                total_cost_scheduled += cost
            else:
                cost = float(risk * self.downtime_cost * SHIFT_HOURS)
                if not np.isfinite(cost) or cost < 0:
                    cost = 0.0
                total_cost_unscheduled += cost
                if risk >= self.safety_threshold:
                    n_critical_unscheduled += 1

            enriched_rows.append({
                "machine_id": m,
                "machine_name": machine_names.get(m, f"Machine-{m}"),
                "failure_risk": round(float(risk), 4),
                "risk_level": risk_level["label"],
                "risk_color": risk_level["color"],
                "is_scheduled": is_scheduled,
                "scheduled_slot": scheduled_slot,
                "estimated_cost": round(float(cost), 2),
            })

        final_df = (pd.DataFrame(enriched_rows)
                    .sort_values("failure_risk", ascending=False)
                    .reset_index(drop=True))

        # Sanitize total cost.
        raw_total = total_cost_scheduled + total_cost_unscheduled
        if not np.isfinite(raw_total) or raw_total < 0:
            total_cost = 0.0
        elif raw_total > MAX_REASONABLE_COST:
            total_cost = MAX_REASONABLE_COST
            warnings_list.append(
                f"Total cost capped at ${MAX_REASONABLE_COST:,.0f} (sanity limit).")
        else:
            total_cost = float(raw_total)

        # Honest status: if criticals were deferred this is Best-Effort
        # regardless of whether the solver returned Optimal.
        if approach_used == "Greedy":
            overall_status = "Greedy-Fallback"
        elif n_critical_unscheduled == 0:
            overall_status = "Optimal"
        else:
            overall_status = "Best-Effort"
            warnings_list.append(
                f"{n_critical_unscheduled} critical machine(s) deferred due to "
                "capacity; the highest-risk criticals were prioritized.")

        summary = {
            "total_machines": n_machines,
            "scheduled": int(final_df["is_scheduled"].sum()),
            "not_scheduled": int(n_machines - final_df["is_scheduled"].sum()),
            "critical": int((final_df["risk_level"] == "Service Immediately").sum()),
            "elevated": int((final_df["risk_level"] == "Schedule Soon").sum()),
            "normal": int((final_df["risk_level"] == "Continue Monitoring").sum()),
            "critical_unscheduled": n_critical_unscheduled,
            "total_cost": float(total_cost),
            "cost_scheduled": float(total_cost_scheduled),
            "cost_unscheduled": float(total_cost_unscheduled),
            "status": overall_status,
            "solver_status": solver_status,
            "approach_used": approach_used,
            "solver_ok": (approach_used == "MILP"),
            "warnings": warnings_list,
            "recommendations": recommendations_list,
        }

        print(f"[MILP] Approach: {approach_used}  |  Status: {overall_status}")
        print(f"[MILP] Total realized cost: ${total_cost:,.2f}")
        if n_critical_unscheduled:
            print(f"[MILP] Critical machines deferred: {n_critical_unscheduled}")

        self._print_schedule(final_df, summary)

        return {
            "schedule": final_df,
            "total_cost": float(total_cost),
            "summary": summary,
            "status": overall_status,
            "solver_ok": (approach_used == "MILP"),
            "warnings": warnings_list,
            "recommendations": recommendations_list,
        }

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _sanitize_input_risks(self, machine_risks):
        """Coerce every risk value into ``[0, 1]``. NaN/inf/None → 0.0."""
        if isinstance(machine_risks, pd.Series):
            machine_risks = machine_risks.to_dict()
        clean = {}
        for m, r in machine_risks.items():
            try:
                r_float = float(r)
                if not np.isfinite(r_float):
                    r_float = 0.0
                r_float = max(0.0, min(1.0, r_float))
            except (TypeError, ValueError):
                r_float = 0.0
            clean[m] = r_float
        return clean

    def _empty_result(self):
        empty_schedule = pd.DataFrame(columns=[
            "machine_id", "machine_name", "failure_risk", "risk_level",
            "risk_color", "is_scheduled", "scheduled_slot", "estimated_cost"])
        return {
            "schedule": empty_schedule,
            "total_cost": 0.0,
            "summary": {
                "total_machines": 0, "scheduled": 0, "not_scheduled": 0,
                "critical": 0, "elevated": 0, "normal": 0,
                "critical_unscheduled": 0, "total_cost": 0.0,
                "cost_scheduled": 0.0, "cost_unscheduled": 0.0,
                "status": "Optimal", "solver_status": "Optimal",
                "approach_used": "N/A", "solver_ok": True,
                "warnings": [], "recommendations": [],
            },
            "status": "Optimal",
            "solver_ok": True,
            "warnings": [],
            "recommendations": [],
        }

    def _get_risk_level(self, risk):
        if risk >= config.RISK_LEVELS["critical"]["threshold"]:
            return config.RISK_LEVELS["critical"]
        elif risk >= config.RISK_LEVELS["elevated"]["threshold"]:
            return config.RISK_LEVELS["elevated"]
        else:
            return config.RISK_LEVELS["normal"]

    def _print_schedule(self, df, summary):
        print("\n" + "=" * 70)
        print("OPTIMIZED MAINTENANCE SCHEDULE")
        print("=" * 70)

        critical = df[df["risk_level"] == "Service Immediately"]
        if len(critical) > 0:
            print("\n[!!] SERVICE IMMEDIATELY:")
            for _, row in critical.iterrows():
                slot = (f"Slot {int(row['scheduled_slot'])}"
                        if row["is_scheduled"]
                        else "DEFERRED (over capacity)")
                print(f"   {row['machine_name']:20s} | "
                      f"Risk: {row['failure_risk']:.2%} | {slot}")

        elevated = df[df["risk_level"] == "Schedule Soon"]
        if len(elevated) > 0:
            print("\n[!] SCHEDULE SOON:")
            for _, row in elevated.iterrows():
                slot = (f"Slot {int(row['scheduled_slot'])}"
                        if row["is_scheduled"]
                        else "Not scheduled")
                print(f"   {row['machine_name']:20s} | "
                      f"Risk: {row['failure_risk']:.2%} | {slot}")

        normal = df[df["risk_level"] == "Continue Monitoring"]
        if len(normal) > 0:
            print(f"\n[OK] CONTINUE MONITORING: {len(normal)} machines")

        print(f"\n{'-' * 70}")
        print(f"Summary: {summary['scheduled']}/{summary['total_machines']} scheduled | "
              f"Est. cost: ${summary['total_cost']:,.0f}")
        print(f"Risk breakdown: {summary['critical']} critical, "
              f"{summary['elevated']} elevated, {summary['normal']} normal")
        if summary["critical_unscheduled"]:
            print(f"[!] {summary['critical_unscheduled']} critical machines were "
                  "deferred due to capacity limits.")

        if summary.get("recommendations"):
            print("\n[RECOMMENDATIONS FOR OPERATOR]:")
            for i, rec in enumerate(summary["recommendations"], 1):
                print(f"  {i}. {rec}")

    def create_gantt_data(self, schedule_result):
        """Convert the schedule into a Plotly-Gantt-friendly list of dicts."""
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
    # Smoke tests covering the original failure mode and edge cases.
    print("\n" + "=" * 70)
    print("STRESS TEST 1: 107 critical machines, 10 slots, 3 crews "
          "(reproduces the original 'Infeasible' bug)")
    print("=" * 70)
    risks = {i: 0.95 for i in range(1, 108)}
    scheduler = MaintenanceScheduler()
    r = scheduler.create_schedule(risks)
    assert np.isfinite(r["total_cost"]), "FAIL: cost not finite"
    assert r["total_cost"] >= 0
    assert r["status"] in {"Optimal", "Best-Effort", "Greedy-Fallback"}
    assert r["summary"]["scheduled"] == 30, \
        f"Expected 30 scheduled (capacity), got {r['summary']['scheduled']}"
    print(f"\n>>> PASSED: status={r['status']}, cost=${r['total_cost']:,.0f}, "
          f"scheduled {r['summary']['scheduled']}/{r['summary']['total_machines']}")

    print("\n" + "=" * 70)
    print("STRESS TEST 2: Normal case (15 machines, 3 critical)")
    print("=" * 70)
    np.random.seed(42)
    risks = {i: float(np.clip(np.random.beta(2, 5), 0, 1)) for i in range(1, 16)}
    risks[1] = 0.92
    risks[5] = 0.85
    risks[8] = 0.78
    r = scheduler.create_schedule(risks)
    assert np.isfinite(r["total_cost"])
    assert r["status"] == "Optimal"
    print(f"\n>>> PASSED: status={r['status']}, cost=${r['total_cost']:,.0f}")

    print("\n" + "=" * 70)
    print("STRESS TEST 3: Empty input")
    print("=" * 70)
    r = scheduler.create_schedule({})
    assert r["total_cost"] == 0.0
    print(">>> PASSED: empty input handled cleanly")

    print("\n" + "=" * 70)
    print("STRESS TEST 4: Dirty input (NaN, inf, negative, out-of-range, None)")
    print("=" * 70)
    dirty = {1: float("inf"), 2: float("-inf"), 3: float("nan"),
             4: -0.5, 5: 1.5, 6: None, 7: "oops", 8: 0.85}
    r = scheduler.create_schedule(dirty)
    assert np.isfinite(r["total_cost"])
    print(f">>> PASSED: dirty input sanitized, cost=${r['total_cost']:,.2f}")

    print("\n" + "=" * 70)
    print("ALL STRESS TESTS PASSED")
    print("=" * 70)
