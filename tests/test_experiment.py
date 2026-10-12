import importlib

import pytest


def api():
    try:
        return importlib.import_module("run_experiment")
    except ModuleNotFoundError:
        pytest.fail("Experiment orchestration is not implemented")


def test_failed_solver_cannot_be_reported_as_sampled_safe():
    assert api().classify(False, 0.01, 0.01) == "solver_failed"


def test_remaining_retained_collision_is_not_a_success():
    assert api().classify(True, 0.01, -0.001) == "remaining_collision"


def test_safe_grid_is_explicitly_only_sampled_safety():
    assert api().classify(True, 0.001, 0.001) == "sampled_collision_free"


def test_duplicate_counterexample_stops_without_another_solve():
    assert api().next_action([0, 0.05], 0.05, -0.01, 1) == "retained_scenario_violation"


def test_five_attempts_is_a_hard_stop():
    assert api().next_action([0], 0.05, -0.01, 5) == "iteration_limit"


def test_control_limit_violation_prevents_success_even_without_collision():
    assert api().classify(True, 0.01, 0.01, controls_valid=False) == "control_limits_violated"


def test_optimizer_runtime_includes_failed_attempt():
    report = {"solver_attempts": [{"runtime_optimizer_s": 1.5},
                                  {"runtime_optimizer_s": 2.25, "converged": False}]}
    assert api().optimizer_runtime(report) == pytest.approx(3.75)


def test_before_after_validation_uses_each_solvers_actual_horizon():
    from falsifier import ControlSchedule
    before = ControlSchedule([0, 8], [[0.5, 0], [0.5, 0]])
    after = ControlSchedule([1e-7, 8 + 1e-7], [[0.5, 0], [0.5, 0]])
    original, updated = api().validate_pair(before, after, biases=[0], count=81)
    assert original["clearance"] == pytest.approx(-0.55)
    assert updated["clearance"] == pytest.approx(-0.55)
