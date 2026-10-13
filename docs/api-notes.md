# Installed OpenSCvx API and numerical checks

The experiment uses OpenSCvx 0.5.2 and its original default QOCO solver.
Dependencies are pinned in `requirements-lock.txt`. No installed optimizer code
is patched. The supplied `baseline_unicycle.py` is preserved.

## Applied control interpolation

The installed source, rather than the names of the hold types, determines the
control signal simulated by the independent falsifier:

- `openscvx/discretization/linearize_discretize.py`, `_dVdt`: `beta = tau * N * foh_mask`.
- `openscvx/propagation/propagation.py`, `prop_aug_dy`: `beta = (tau - tau_init) * N * foh_mask`.
- The normalized segment duration is `1 / (N - 1)`. At N=5, beta reaches 1.25.
- The resolved hold mask is `[1, 1, 0]`: speed and turn rate are interpolated;
  time dilation is constant on each interval. Optimization times are nonuniform.
- Consequently, applied speed and turn rate extrapolate beyond the next knot
  before resetting at the next segment. Knot bounds alone do not bound these
  applied signals. Independent extrema checks use both sides of each segment.
- `t_to_tau` produces `u_full` with ordinary physical-time interpolation, which
  differs from the signal used by nonlinear propagation in this installed version.

`robust_scvx.py` inspects the installed propagation source and refuses an
unrecognized interpolation implementation. Updating OpenSCvx requires rechecking
both discretization and propagation before removing this guard. The pinned
experiment reproduces the actual implementation; it is not an ideal-FOH
experiment.

## Scenarios, warm starts, and results

- `State.final` must be explicitly set. `ox.Free(...)` specifies a disturbed
  endpoint without imposing the nominal goal.
- A fresh nominal module construction supplies the original `speed` and
  `turn_rate` objects to all scenario dynamics; there is one distance objective.
- `Time.guess` and `Time.time_dilation_guess` preserve the optimized time grid
  during expansion, alongside the speed/turn-rate knots and nominal state nodes.
- Disturbed state guesses come from independent nonlinear propagation of the
  previous control signal at its actual physical knots.
- `solve()` and `post_process()` have separate convergence formatting. The driver
  retains the status from `solve()` and checks it before reporting a repair.
- The upstream `OptimizationResults.save()` failed on a scalar history field.
  `save_capture()` stores numeric arrays and histories directly in compressed NPZ;
  no pickles or object arrays are needed to inspect the saved evidence.
- Original CTCS residuals are retained even when small or negative due to
  numerical integration. Collision decisions use independent nonlinear clearance,
  not those residuals or nodal feasibility.

The unmodified baseline first encountered a Windows console encoding error.
`PYTHONUTF8=1` resolved that environment issue. Its optimization subsequently
converged, but external-LaTeX plot rendering failed because `latex` is absent.
The experiment driver uses the same optimization construction and Matplotlib
mathtext for research figures. Generated figures and manuscript assets are kept
outside the repository; the experiment still regenerates its figures on demand.

## Official references

- [OpenSCvx 0.5.2 Problem API](https://openscvx.github.io/OpenSCvx/v0.5.2/Reference/problem/)
- [Post-processing API](https://openscvx.github.io/OpenSCvx/v0.5.2/Reference/propagation/post_processing/)
- [Discretization API](https://openscvx.github.io/OpenSCvx/v0.5.2/Reference/discretization/)
- [Dawson and Fan: counterexample-guided planning](https://arxiv.org/abs/2203.02038)

Counterexample-guided planning predates this experiment. This repository tests
the feedback mechanism on the supplied baseline and makes no novelty or
computational-superiority claim.
