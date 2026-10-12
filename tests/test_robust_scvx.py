import importlib
from types import SimpleNamespace

import numpy as np
import pytest


def api():
    try:
        return importlib.import_module("robust_scvx")
    except ModuleNotFoundError:
        pytest.fail("Scenario-based OpenSCvx feedback is not implemented")


def test_scenarios_share_original_controls_and_nominal_terminal_only():
    built = api().build_problem([0, 0.05, -0.03])
    assert len(built.controls) == 2
    assert [u.name for u in built.controls] == ["speed", "turn_rate"]
    # Walk the symbolic dynamics; every referenced control is one of the originals.
    for expression in built.dynamics.values():
        pending = [expression]
        while pending:
            node = pending.pop()
            if type(node).__name__ == "Control":
                assert any(node is control for control in built.controls)
            pending.extend(node.children())
    np.testing.assert_allclose(built.states[0].final, [4, 0])
    for state in built.states:
        if "_b" in state.name:
            assert all(str(kind).lower() == "free" for kind in state.final_type)
            np.testing.assert_allclose(state.initial, np.zeros(state.shape))
    assert len(built.biases) == 3


def test_nonconvergent_solver_result_is_rejected():
    with pytest.raises(RuntimeError, match="did not converge"):
        api().require_converged(SimpleNamespace(converged=False))


def test_interpolation_source_is_inspected():
    info = api().interpolation_info()
    assert info["openscvx_version"]
    assert info["segment_fraction_scale"] == pytest.approx(1.25)
    assert "prop_aug_dy" in info["source"]


def test_capture_handles_scalar_solver_history(tmp_path):
    result = SimpleNamespace(x=np.zeros((5, 4)), u=np.ones((5, 3)),
                             t_full=np.array([0, 8]), x_full=np.zeros((2, 4)),
                             u_full=np.ones((2, 3)), converged=True,
                             ctcs_violation=np.array([0.1]),
                             nodes={"speed": np.ones((5, 1))},
                             trajectory={"speed": np.ones((2, 1))},
                             J_vc_history=np.float64(0.01))
    api().save_capture(result, tmp_path / "result.npz")
    with np.load(tmp_path / "result.npz", allow_pickle=False) as saved:
        assert saved["J_vc_history"].item() == pytest.approx(0.01)
        np.testing.assert_allclose(saved["nodes_speed"], 1)


def test_warm_start_preserves_nonuniform_time_and_dilation():
    from falsifier import ControlSchedule
    times = np.array([0, 1, 3, 5, 8], dtype=float)
    positions = np.column_stack([times * 0.5, np.zeros(5)])
    nodes = {"position": positions, "theta": np.zeros((5, 1)),
             "distance": (times * 0.5)[:, None], "speed": np.full((5, 1), 0.5),
             "turn_rate": np.zeros((5, 1)), "time": times[:, None],
             "_time_dilation": np.array([4, 8, 8, 12, 8])[:, None]}
    previous = {"result": SimpleNamespace(nodes=nodes),
                "schedule": ControlSchedule(times, np.column_stack([nodes['speed'], nodes['turn_rate']]), fraction_scale=1.25)}
    built = api().build_problem([0, -0.05], previous)
    new_time = next(s for s in built.problem.symbolic.states if s.name == "time")
    new_dilation = next(u for u in built.problem.symbolic.controls if u.name == "_time_dilation")
    np.testing.assert_allclose(new_time.guess, nodes["time"])
    np.testing.assert_allclose(new_dilation.guess, nodes["_time_dilation"])
