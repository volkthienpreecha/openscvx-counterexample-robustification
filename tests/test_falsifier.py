import importlib

import numpy as np
import pytest


def api():
    try:
        return importlib.import_module("falsifier")
    except ModuleNotFoundError:
        pytest.fail("Independent falsifier is not implemented")


def test_zero_bias_straight_line():
    f = api()
    schedule = f.ControlSchedule([0, 8], [[0.5, 0], [0.5, 0]])
    x = f.rollout(schedule, 0, np.linspace(0, 8, 81))
    np.testing.assert_allclose(x[-1], [4, 0, 0], atol=1e-9)


def test_bias_terminal_heading_offset_is_bT():
    f = api()
    schedule = f.ControlSchedule([0, 2, 8], [[0.5, 0.2], [0.7, -0.1], [0.3, 0]])
    times = np.linspace(0, 8, 101)
    nominal = f.rollout(schedule, 0, times)
    disturbed = f.rollout(schedule, 0.05, times)
    assert disturbed[-1, 2] - nominal[-1, 2] == pytest.approx(0.4, abs=1e-9)


def test_nonuniform_knots_and_segment_extrapolation():
    f = api()
    schedule = f.ControlSchedule([0, 2, 8], [[0.2, 0], [0.6, 0.2], [0.8, -0.2]], fraction_scale=1.25)
    np.testing.assert_allclose(schedule.at(1), [0.45, 0.125])
    np.testing.assert_allclose(schedule.at(2, segment=0), [0.7, 0.25])
    np.testing.assert_allclose(schedule.at(2, segment=1), [0.6, 0.2])


def test_dense_nonlinear_safety_detects_collision_between_nodes():
    f = api()
    schedule = f.ControlSchedule([0, 8], [[0.5, 0], [0.5, 0]])
    nodes = f.rollout(schedule, 0, [0, 8])
    assert np.min(np.linalg.norm(nodes[:, :2] - [2, 0], axis=1) - 0.55) > 0
    worst = f.sweep(schedule, [0], np.linspace(0, 8, 801))
    assert worst["clearance"] == pytest.approx(-0.55, abs=1e-9)
    assert worst["time"] == pytest.approx(4)


def test_invalid_knots_are_rejected():
    with pytest.raises(ValueError):
        api().ControlSchedule([0, 0, 8], np.zeros((3, 2)))


def test_variable_time_dilation_maps_actual_normalized_control():
    f = api()
    # On this segment s(tau)=4+8*tau, so t=4*tau+4*tau^2.
    schedule = f.ControlSchedule([0, 3], [[0, 0], [1, 0]],
                                 dilation=[4, 8], fraction_scale=1, tau_step=0.5)
    assert schedule.at(1.25)[0] == pytest.approx(0.5)
