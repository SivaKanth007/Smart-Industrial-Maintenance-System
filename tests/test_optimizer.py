"""
Unit Tests — MILP Optimizer
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.optimization.milp_scheduler import MaintenanceScheduler


@pytest.fixture
def scheduler():
    return MaintenanceScheduler(n_crews=3)


@pytest.fixture
def machine_risks():
    """Sample machine risks with various levels."""
    return {
        1: 0.92,   # Critical
        2: 0.85,   # Critical
        3: 0.75,   # Critical
        4: 0.55,   # Elevated
        5: 0.45,   # Elevated
        6: 0.30,   # Normal
        7: 0.20,   # Normal
        8: 0.10,   # Normal
        9: 0.05,   # Normal
        10: 0.78,  # Critical
    }


class TestMILPScheduler:
    def test_schedule_created(self, scheduler, machine_risks):
        result = scheduler.create_schedule(machine_risks, n_time_slots=5)
        assert result["status"] == "Optimal"
        assert "schedule" in result
        assert len(result["schedule"]) == len(machine_risks)

    def test_crew_capacity_respected(self, scheduler, machine_risks):
        result = scheduler.create_schedule(machine_risks, n_time_slots=5)
        schedule = result["schedule"]

        # Count scheduled per slot
        scheduled = schedule[schedule["is_scheduled"]]
        if len(scheduled) > 0:
            slot_counts = scheduled["scheduled_slot"].value_counts()
            for count in slot_counts.values:
                assert count <= 3, f"Crew capacity violated: {count} > 3"

    def test_critical_machines_scheduled(self, scheduler, machine_risks):
        result = scheduler.create_schedule(machine_risks, n_time_slots=5)
        schedule = result["schedule"]

        critical = schedule[schedule["failure_risk"] >= config.SAFETY_RISK_THRESHOLD]
        assert all(critical["is_scheduled"]), \
            "Not all critical machines were scheduled!"

    def test_no_duplicate_scheduling(self, scheduler, machine_risks):
        result = scheduler.create_schedule(machine_risks, n_time_slots=5)
        schedule = result["schedule"]

        # Each machine should appear exactly once
        assert len(schedule) == len(machine_risks)

    def test_risk_levels_correct(self, scheduler, machine_risks):
        result = scheduler.create_schedule(machine_risks, n_time_slots=5)
        schedule = result["schedule"]

        for _, row in schedule.iterrows():
            risk = row["failure_risk"]
            if risk >= 0.7:
                assert row["risk_level"] == "Service Immediately"
            elif risk >= 0.4:
                assert row["risk_level"] == "Schedule Soon"
            else:
                assert row["risk_level"] == "Continue Monitoring"

    def test_gantt_data(self, scheduler, machine_risks):
        result = scheduler.create_schedule(machine_risks, n_time_slots=5)
        gantt = scheduler.create_gantt_data(result)

        assert isinstance(gantt, list)
        if len(gantt) > 0:
            assert "Task" in gantt[0]
            assert "Start" in gantt[0]
            assert "Finish" in gantt[0]

    def test_empty_machines(self, scheduler):
        result = scheduler.create_schedule({}, n_time_slots=5)
        assert result["status"] == "Optimal"
        assert len(result["schedule"]) == 0

    def test_single_machine_high_risk(self, scheduler):
        result = scheduler.create_schedule({1: 0.95}, n_time_slots=3)
        assert result["schedule"]["is_scheduled"].iloc[0] == True
