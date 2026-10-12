# OpenSCvx Robustification Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task, with tests before implementation and sequential commits.

**Goal:** Execute and document one honest nominal solve, independent falsification, scenario feedback, OpenSCvx re-solve, and final sampled validation.

**Architecture:** Preserve `baseline_unicycle.py` as supplied. `falsifier.py` independently integrates the optimized controls. `robust_scvx.py` extends the baseline with scenario states sharing the existing controls. `run_experiment.py` records the complete experiment and figures.

**Tech stack:** Python 3.12, OpenSCvx, JAX, SciPy, NumPy, Matplotlib, pytest.

**Spec:** [User-provided experiment specification](experiment-specification.md).

## Global constraints

- N=5, T=8, original bounds, objective, CTCS, augmented-Lagrangian settings, and Dopri8 settings remain intact.
- Steering bias lies in [-0.05, 0.05] rad/s. Search at least 101 biases; final validation at least 501 biases.
- Use the installed OpenSCvx control interpolation and original optimization knots.
- Retained scenarios share speed and turn_rate; nominal terminal constraints apply only to the nominal state.
- Stop on nonconvergence or failure; at most five re-optimization attempts.
- Finite sampled safety is not a proof of robust safety; no invented results.

## Review focus

- Control knots and physical time must reproduce OpenSCvx propagation; test nonuniform knots and terminal heading offset.
- Detect collisions between nodes using nonlinear dense simulation.
- Repeated counterexamples must not trigger endless duplicate solves.
- Solver failures must remain failures in JSON and figure captions.
- Missing external TeX affects plotting only; keep the original baseline and record any environment adaptation.

## Task 1: Preserve and execute the baseline

- [x] Commit supplied baseline and experiment specification.
- [x] Install isolated dependencies; run the baseline unchanged, log any environment failure.
- [x] Inspect installed interpolation, result, and solver APIs; record version and source references.
- [x] Preserve solver status, controls, physical time grid, propagated states, residuals, objective, and timings.

## Task 2: Independent falsification

**Files:** `falsifier.py`, `tests/test_falsifier.py`.

**Interfaces:** `ControlSchedule(times, values)` interpolates verified controls; `rollout(schedule, bias, times)` returns nonlinear pose samples; `sweep(schedule, biases, times)` returns worst clearance, bias, time, position and trajectory.

- [x] Write failing tests for zero bias, bT heading offset, nonuniform interpolation, and collision between nodes.
- [x] Run pytest to verify the missing implementation fails.
- [x] Implement segmented solve_ivp integration with tight tolerances and dense time evaluation.
- [x] Run tests, compare nominal propagation, and commit.

## Task 3: Scenario feedback

**Files:** `robust_scvx.py`, `tests/test_robust_scvx.py`.

**Interfaces:** `build_problem(biases, previous=None)` returns the original problem expanded with biased position/theta states; `solve_and_capture(problem, output)` preserves solver data and exposes optimized controls.

- [x] Write failing tests for shared controls, nominal-only terminal conditions, and refusal of nonconvergent results.
- [x] Create one CTCS safety constraint per scenario; warm-start from independently propagated previous controls.
- [x] Preserve all original settings and the single distance objective.
- [x] Run tests and commit.

## Task 4: Full experiment and publication

**Files:** `run_experiment.py`, `results.json`, `results/`, `README.md`.

- [x] Write failure-reporting and stopping-condition tests before implementing orchestration.
- [x] Run baseline, search, retain violating biases, and re-solve up to five times.
- [x] Validate on at least 501 biases; save quantitative table and three vector PDF figures.
- [x] Inspect the figures and report remaining violations explicitly.
- [x] Run the complete test suite; commit measured results and reproduction instructions.
- [x] Create the authorized public GitHub repository, push all sequential commits, and verify visibility and remote history.
