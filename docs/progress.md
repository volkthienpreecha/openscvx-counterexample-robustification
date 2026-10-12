# Execution ledger: docs/implementation-plan.md

- Plan approved by the user for native execution.
- Original manuscript preserved in commit 927bddb; baseline and specification in b437581.
- Baseline unchanged first attempt failed on Windows cp1252 encoding. Second attempt with PYTHONUTF8=1 converged, then plotting failed because LaTeX is absent.
- Baseline reported clearance -0.005716 m before any disturbance.
- Interpolation inspection: OpenSCvx 0.5.2 uses beta=(tau-tau_init)*N in propagation and beta=tau*N in discretization, with segment duration 1/(N-1). Independent integration must reproduce that actual interpolation, including endpoint extrapolation.
- Ruling: use this new repository's existing checkout and main branch for the authorized initial publication; no concurrent work or existing code requires isolation. Cost if wrong: branch isolation would need to be added later.
- Ruling: preserve the supplied baseline file and bypass its external-TeX plotting in the experiment driver. Solve settings remain intact; experiment figures use Matplotlib mathtext. Cost if wrong: different typography only.
- Task 1: complete. Unchanged baseline was run and numerical results captured; plotting environment limitation documented.
- Task 2: complete. Zero bias, bT heading offset, nonuniform interpolation, between-node collision, and reverse-motion length tests went from RED to GREEN.
- Task 3: complete. Shared controls, free disturbed terminal states, numerical capture and solver failure tests went from RED to GREEN.
- Task 4: experiment complete. One scenario feedback/re-solve, 501-bias final validation, three vector figures, JSON/NPZ data and table produced.
- Fresh final reviewer found two issues: time/time-dilation warm start and failed-attempt runtime accounting. Both fixed with failing tests then a green full suite. Additional validation endpoint regression fixed with a failing test and green suite.
- Final verification: 20 tests passed. Baseline and expanded solve converge; final result is remaining_collision, not successful robustification.
- Ruling: report a retained-scenario validation failure and stop when the same violating bias is already retained. Re-adding it changes no constraint; the upstream interpolation and numerical safety failures are documented instead of hidden. Cost if wrong: another initialization or tighter numerical study could find a better outcome; no robust feasibility is claimed.
- Reviewer declined to judge PDF appearance and later README content. Executor separately rendered all three PDFs, corrected a clipped title, and checked the README against the final JSON measurements.
- No deferred reviewer findings remain. Optional N=11 and comparative experiments were outside the prioritized single-cycle scope and were not run.
- All three final research PDFs were rendered and inspected; the manuscript compiled with Tectonic 0.17.0 and all eight pages were inspected. A repeated reference heading was removed.

- Public repository created and verified: https://github.com/volkthienpreecha/openscvx-counterexample-robustification. GitHub visibility is PUBLIC, default branch is main, and remote history matches the local commits.
