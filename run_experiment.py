"""Run the fixed N=5 experiment; never label failed validation a repair."""
import argparse
import json
import os
from pathlib import Path
import traceback
from time import perf_counter

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("OPENSCVX_CACHE_DIR", str(Path(".cache/openscvx").resolve()))

import matplotlib.pyplot as plt
import numpy as np

from falsifier import rollout, sweep
from robust_scvx import build_problem, interpolation_info, solve_and_capture


def classify(converged, full_clearance, retained_clearance, controls_valid=True):
    if not converged:
        return "solver_failed"
    if min(full_clearance, retained_clearance) < 0:
        return "remaining_collision"
    if not controls_valid:
        return "control_limits_violated"
    return "sampled_collision_free"


def next_action(biases, bias, clearance, attempts):
    if clearance >= 0:
        return "sampled_collision_free"
    if attempts >= 5:
        return "iteration_limit"
    if any(abs(bias - b) < 1e-10 for b in biases):
        return "retained_scenario_violation"
    return "add_scenario"


def json_worst(worst):
    return {k: v for k, v in worst.items() if k != "trajectory"}


def controls_valid(summary):
    return (summary["min_speed_m_s"] >= -1e-7 and summary["max_speed_m_s"] <= 1 + 1e-7
            and summary["max_abs_turn_rate_rad_s"] <= 0.8 + 1e-7)


def optimizer_runtime(report):
    return sum(s["runtime_optimizer_s"] for s in report["solver_attempts"])


def validate_pair(before, after, biases=None, count=8001):
    if biases is None:
        biases = np.linspace(-0.05, 0.05, 501)
    return tuple(sweep(schedule, biases, np.linspace(schedule.times[0], schedule.times[-1], count))
                 for schedule in (before, after))


def figures(records, final_worst, output, status):
    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    output = Path(output)
    titles = ["Falsification of the Nominal OpenSCvx Trajectory",
              "Counterexample-Guided SCvx Robustification",
              "Worst-Case Sampled Clearance Across Robustification Iterations"]
    names = ["01_nominal_falsification", "02_counterexample_robustification", "03_robustness_iterations"]
    def base_axis(title):
        fig, ax = plt.subplots(figsize=(7.8, 4.6), layout="constrained")
        ax.add_patch(plt.Circle((2, 0), 0.55, fc="#cf766e", ec="#a33e36", alpha=0.4))
        ax.scatter([0, 4], [0, 0], c=["#20874c", "#7054a4"], s=30, zorder=4)
        ax.set(xlabel="$p_x$ [m]", ylabel="$p_y$ [m]", title=title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.18)
        return fig, ax
    def plot(ax, x, label, color, style="-"):
        ax.plot(x[:, 0], x[:, 1], style, color=color, lw=1.8, label=label)
        violation = np.linalg.norm(x[:, :2] - [2, 0], axis=1) < 0.55
        if np.any(violation):
            ax.plot(x[violation, 0], x[violation, 1], ".", color="#d02020", ms=3)
    first = records[0]
    fig, ax = base_axis(titles[0])
    plot(ax, first["nominal"], "Nominal (bias=0)", "#1769aa")
    plot(ax, first["worst"]["trajectory"], f"Worst bias={first['worst']['bias']:+.3f} rad/s", "#cf711f", "--")
    if first["worst"]["clearance"] < 0:
        ax.scatter(*first["worst"]["position"], marker="x", color="#bf1616", s=60,
                   label=f"Collision: {first['worst']['clearance'] * 1000:.1f} mm clearance")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output / (names[0] + ".pdf")); fig.savefig(output / (names[0] + ".png"), dpi=160)
    plt.close(fig)
    last = records[-1]
    fig, ax = base_axis(titles[1])
    plot(ax, first["nominal"], "Original nominal", "#8290a0", "--")
    plot(ax, first["worst"]["trajectory"], "Original worst disturbance", "#cf711f", ":")
    plot(ax, last["nominal"], "After scenario feedback: nominal", "#1769aa")
    plot(ax, final_worst["trajectory"], f"Final validation worst: b={final_worst['bias']:+.3f}", "#793d9b", "--")
    ax.text(0.02, 0.03, f"Status: {status}\nRed points mark remaining collisions.",
            transform=ax.transAxes, fontsize=8, bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output / (names[1] + ".pdf")); fig.savefig(output / (names[1] + ".png"), dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    ax.plot([r["iteration"] for r in records], [r["worst"]["clearance"] for r in records], "o-", color="#1769aa")
    ax.axhline(0, color="#bf1616", lw=1, ls="--", label="Zero clearance")
    ax.set(xlabel="Number of scenario re-optimization attempts", ylabel="Worst sampled clearance [m]", title=titles[2])
    ax.set_xticks([r["iteration"] for r in records]); ax.grid(alpha=0.2)
    ax.legend(); fig.savefig(output / (names[2] + ".pdf")); fig.savefig(output / (names[2] + ".png"), dpi=160)
    plt.close(fig)


def run(output="results"):
    started = perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"disturbance_interval_rad_s": [-0.05, 0.05], "N": 5, "T_s": 8,
              "search_bias_count": 101, "validation_bias_count": 501,
              "search_time_count": 4001, "validation_time_count": 8001,
              "formal_safety_certificate": False, "interpolation": interpolation_info(),
              "iterations": [], "solver_attempts": [], "status": "running", "failures": []}
    def save():
        report["runtime_total_s"] = perf_counter() - started
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    records, biases, attempts = [], [0.0], 0
    current = None
    try:
        current = solve_and_capture(build_problem(biases), output / "baseline")
        report["baseline"] = current["summary"]
        report["solver_attempts"].append(current["summary"])
        save()
        if not current["result"].converged:
            raise RuntimeError("Baseline did not converge; robustification was not attempted")
        if current["summary"]["nominal_reproduction_max_abs_error"] > 1e-3:
            raise RuntimeError("Independent nominal propagation does not reproduce OpenSCvx within 1 mm")
        while True:
            schedule = current["schedule"]
            times = np.linspace(schedule.times[0], schedule.times[-1], 4001)
            worst = sweep(schedule, np.linspace(-0.05, 0.05, 101), times)
            nominal = rollout(schedule, 0, times)
            records.append({"iteration": attempts, "nominal": nominal, "worst": worst, "current": current})
            report["iterations"].append({"iteration": attempts, "worst": json_worst(worst),
                                         "solver": current["summary"]})
            save()
            print(f"iteration={attempts}; worst_clearance={worst['clearance']:.9f}; bias={worst['bias']:+.5f}", flush=True)
            action = next_action(biases, worst["bias"], worst["clearance"], attempts)
            if action != "add_scenario":
                report["stop_reason"] = action
                break
            biases.append(worst["bias"])
            attempts += 1
            try:
                candidate = solve_and_capture(build_problem(biases, previous=current), output / f"iteration_{attempts:02d}")
                report["solver_attempts"].append(candidate["summary"])
                if not candidate["result"].converged:
                    report["failed_solver"] = candidate["summary"]
                    raise RuntimeError("Expanded OpenSCvx problem did not converge")
                if candidate["summary"]["nominal_reproduction_max_abs_error"] > 1e-3:
                    raise RuntimeError("Expanded problem independent reproduction exceeds 1 mm")
                current = candidate
            except Exception as exc:
                report["failures"].append({"stage": "reoptimization", "iteration": attempts,
                                            "error": str(exc), "traceback": traceback.format_exc()})
                report["stop_reason"] = "solver_failed"
                break
        final_times = np.linspace(current["schedule"].times[0], current["schedule"].times[-1], 8001)
        before_worst, final_worst = validate_pair(records[0]["current"]["schedule"], current["schedule"])
        retained = []
        for bias in biases:
            worst = sweep(current["schedule"], [bias], final_times)
            retained.append(json_worst(worst))
        report["final"] = current["summary"]
        report["before_validation"] = json_worst(before_worst)
        report["after_validation"] = json_worst(final_worst)
        report["retained_scenario_validation"] = retained
        report["all_retained_scenarios_sampled_safe"] = all(r["clearance"] >= 0 for r in retained)
        report["reoptimization_attempts"] = attempts
        report["retained_biases_rad_s"] = biases
        report["baseline_applied_controls_within_bounds"] = controls_valid(report["baseline"])
        report["final_applied_controls_within_bounds"] = controls_valid(current["summary"])
        report["status"] = classify(not report["failures"], final_worst["clearance"],
                                     min(r["clearance"] for r in retained), controls_valid(current["summary"]))
        report["runtime_falsifier_s"] = sum(r["worst"]["runtime_s"] for r in records) + final_worst["runtime_s"] + before_worst["runtime_s"] + sum(r["runtime_s"] for r in retained)
        report["runtime_optimizer_s"] = optimizer_runtime(report)
        figures(records, final_worst, output, report["status"])
        rows = [("Geometric path length [m]", "path_length_m"),
                ("Signed distance objective [m]", "openscvx_propagated_distance_m"),
                ("Nominal terminal error norm", "nominal_terminal_error_norm"),
                ("Maximum applied speed [m/s]", "max_speed_m_s"),
                ("Minimum applied speed [m/s]", "min_speed_m_s"),
                ("Maximum applied turn rate [rad/s]", "max_abs_turn_rate_rad_s")]
        table = ["| Metric | Before | After |", "|---|---:|---:|"]
        table.extend(f"| {name} | {report['baseline'][key]:.9g} | {report['final'][key]:.9g} |" for name, key in rows)
        table.append(f"| Worst 501-bias clearance [m] | {before_worst['clearance']:.9g} | {final_worst['clearance']:.9g} |")
        (output / "table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    except Exception as exc:
        report["status"] = "failed"
        report["failures"].append({"stage": "experiment", "error": str(exc), "traceback": traceback.format_exc()})
        traceback.print_exc()
    save()
    print(f"status={report['status']}; results={output / 'results.json'}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    report = run(args.output)
    raise SystemExit(0 if report["status"] == "sampled_collision_free" else 1)
