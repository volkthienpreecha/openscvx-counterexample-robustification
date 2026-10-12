"""Extend Fabio's exact baseline using shared-control disturbance scenarios."""
from dataclasses import dataclass
import inspect
from importlib.metadata import version
import json
from pathlib import Path
import runpy
from time import perf_counter

import numpy as np
import openscvx as ox
from openscvx.propagation.propagation import prop_aug_dy, s_to_t
from openscvx.discretization.base import _resolve_foh_mask

from falsifier import ControlSchedule, rollout


@dataclass
class BuiltProblem:
    problem: ox.Problem
    states: list
    controls: list
    dynamics: dict
    biases: list


def interpolation_info():
    source = inspect.getsource(prop_aug_dy)
    if "(tau - tau_init) * N * foh_mask" not in source:
        raise RuntimeError("Installed OpenSCvx interpolation differs; inspect it before running")
    return {"openscvx_version": version("openscvx"), "hold": "FOH in normalized time",
            "segment_fraction_scale": 5 / 4,
            "source": "openscvx.propagation.propagation.prop_aug_dy",
            "expression": "beta=(tau-tau_init)*N; segment duration=1/(N-1)",
            "warning": "In 0.5.2, segment endpoint beta=1.25; u_full uses ordinary physical-time interpolation and differs from applied controls."}


def build_problem(biases, previous=None):
    biases = list(dict.fromkeys([0.0] + [float(b) for b in biases]))
    if any(not np.isfinite(b) or abs(b) > 0.05 + 1e-12 for b in biases):
        raise ValueError("Retained biases must lie in [-0.05, 0.05]")
    # Run module-level construction unchanged; do not enter baseline's plotting main.
    b = runpy.run_path(str(Path(__file__).with_name("baseline_unicycle.py")))
    states = list(b["states"])
    controls = list(b["controls"])
    dynamics = dict(b["dynamics"])
    constraints = list(b["constraints"])
    knots = np.linspace(0, b["T"], b["N"])
    if previous is not None:
        for control in controls:
            control.guess = np.asarray(previous["result"].nodes[control.name]).copy()
        for state in states:
            state.guess = np.asarray(previous["result"].nodes[state.name]).copy()
        knots = previous["schedule"].times
    for i, bias in enumerate(biases[1:], 1):
        p = ox.State(f"position_b{i}", shape=(2,))
        p.min, p.max = b["position"].min.copy(), b["position"].max.copy()
        p.initial = b["START"][:2].copy()
        theta = ox.State(f"theta_b{i}", shape=(1,))
        theta.min, theta.max = b["theta"].min.copy(), b["theta"].max.copy()
        theta.initial = np.array([b["START"][2]])
        if previous is None:
            p.guess = b["position"].guess.copy()
            theta.guess = b["theta"].guess + bias * knots[:, None]
        else:
            x = rollout(previous["schedule"], bias, knots)
            p.guess, theta.guess = x[:, :2], x[:, 2:3]
        p.final = [ox.Free(float(value)) for value in p.guess[-1]]
        theta.final = [ox.Free(float(theta.guess[-1, 0]))]
        states.extend([p, theta])
        dynamics[p.name] = ox.Concat(controls[0][0] * ox.Cos(theta[0]),
                                     controls[0][0] * ox.Sin(theta[0]))
        dynamics[theta.name] = controls[1][0] + bias
        constraints.append(ox.ctcs(b["obs_radius"] <= ox.linalg.Norm(p - b["obs_center"]),
                                   penalty="squared_relu", check_nodally=True))
    if len(biases) == 1 and previous is None:
        problem = b["problem"]
    else:
        problem = ox.Problem(
            dynamics=dynamics, states=states, controls=controls,
            time=ox.Time(initial=0.0, final=b["T"], min=0.0, max=b["T"]),
            constraints=constraints, N=b["N"],
            algorithm={"lam_cost": {"distance": 1.0}, "lam_vc": 3e2, "k_max": 500,
                       "autotuner": {"type": "AugmentedLagrangian", "rho_init": 1e7, "rho_max": 1e11}},
            discretizer={"ode_solver": "Dopri8", "diffrax_kwargs": {
                "max_steps": 50000, "num_substeps": 256}}, float_dtype="float64")
    return BuiltProblem(problem, states, controls, dynamics, biases)


def require_converged(result):
    if not result.converged:
        raise RuntimeError("OpenSCvx did not converge; this is not a successful robustification")


def save_capture(result, output):
    # 0.5.2 save() assumes every history is iterable, but some arrive as scalars.
    data = {name: np.asarray(getattr(result, name)) for name in
            ("x", "u", "t_full", "x_full", "u_full", "converged", "ctcs_violation")}
    for name in ("J_tr_history", "J_vb_history", "J_vc_history", "X", "U",
                 "discretization_history", "VC_history", "TR_history"):
        if hasattr(result, name):
            data[name] = np.asarray(getattr(result, name))
    for group in ("nodes", "trajectory"):
        for name, value in getattr(result, group).items():
            data[f"{group}_{name}"] = np.asarray(value)
    np.savez_compressed(output, **data)


def solve_and_capture(built, output):
    started = perf_counter()
    problem = built.problem
    problem.initialize()
    initialized = perf_counter()
    solved = problem.solve()
    solve_finished = perf_counter()
    result = problem.post_process()
    # Keep solve() status: post_process() independently reconstructs its status.
    result.converged = bool(solved.converged)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_capture(result, output.with_suffix(".npz"))
    times = np.asarray(s_to_t(result.x, result.u, problem.settings, problem.discretizer)).reshape(-1)
    values = np.column_stack([result.nodes["speed"].reshape(-1),
                              result.nodes["turn_rate"].reshape(-1)])
    interpolation = interpolation_info()
    dilation_slice = problem.settings.sim.time_dilation_slice
    dilation = result.u[:, dilation_slice].reshape(-1)
    scale = interpolation["segment_fraction_scale"]
    # Reconstruct physical times of the ACTUALLY applied normalized-time controls.
    masks = _resolve_foh_mask(problem.discretizer.dis_type, result.u.shape[1],
                              problem.settings.sim.u.foh_mask)
    interpolation["resolved_control_foh_mask"] = masks.tolist()
    if masks[dilation_slice.start] == 0:
        # Time is ZOH even though speed and turn_rate are FOH. Each optimized
        # physical interval has constant dilation, so normalized beta maps linearly.
        applied_times = times
        schedule = ControlSchedule(times, values, fraction_scale=scale)
    else:
        duration = (dilation[:-1] + 0.5 * scale * np.diff(dilation)) / (len(times) - 1)
        applied_times = np.r_[times[0], times[0] + np.cumsum(duration)]
        schedule = ControlSchedule(applied_times, values, fraction_scale=scale,
                                   dilation=dilation, tau_step=1 / (len(times) - 1))
    dense_times = np.asarray(result.t_full)
    if abs(applied_times[-1] - 8) > 1e-5:
        raise RuntimeError("Applied time-dilation interpolation changes the horizon; inspect upstream API")
    independent = rollout(schedule, 0, np.clip(dense_times, applied_times[0], applied_times[-1]))
    original_pose = np.column_stack([result.trajectory["position"], result.trajectory["theta"]])
    reproduction_error = np.max(np.abs(independent - original_pose))
    nodal_pose = np.column_stack([result.nodes["position"], result.nodes["theta"]])
    nodal_independent = rollout(schedule, 0, applied_times)
    applied_controls = np.asarray([schedule.at(t) for t in dense_times])
    control_extrema = np.asarray([schedule.at(t, k) for k in range(len(times) - 1)
                                  for t in schedule.times[k:k + 2]])
    published_controls = np.column_stack([result.trajectory["speed"], result.trajectory["turn_rate"]])
    state = problem._state
    summary = {
        "converged": result.converged, "scvx_iterations": int(state.k),
        "path_length_m": schedule.length(),
        "openscvx_propagated_distance_m": float(result.trajectory["distance"][-1, 0]),
        "nominal_terminal_error": (independent[-1] - [4, 0, 0]).tolist(),
        "nominal_terminal_error_norm": float(np.linalg.norm(independent[-1] - [4, 0, 0])),
        "nominal_reproduction_max_abs_error": float(reproduction_error),
        "nominal_nodal_propagation_defect_max_abs": float(np.max(np.abs(nodal_independent - nodal_pose))),
        "ctcs_violation": np.asarray(result.ctcs_violation).tolist(),
        "final_J_tr": float(state.J_tr), "final_J_vb": float(state.J_vb),
        "final_J_vc": float(state.J_vc),
        "max_speed_m_s": float(np.max(control_extrema[:, 0])),
        "min_speed_m_s": float(np.min(control_extrema[:, 0])),
        "max_abs_turn_rate_rad_s": float(np.max(np.abs(control_extrema[:, 1]))),
        "u_full_vs_applied_controls_max_abs_difference": float(np.max(np.abs(applied_controls - published_controls))),
        "optimization_time_grid_s": times.tolist(), "applied_time_grid_s": applied_times.tolist(),
        "control_knots": values.tolist(), "time_dilation_knots": dilation.tolist(),
        "retained_biases_rad_s": built.biases,
        "runtime_initialize_s": initialized - started,
        "runtime_optimizer_s": solve_finished - initialized,
        "runtime_postprocess_s": perf_counter() - solve_finished,
        "runtime_total_s": perf_counter() - started,
        "interpolation": interpolation,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return {"result": result, "schedule": schedule, "summary": summary, "built": built}
