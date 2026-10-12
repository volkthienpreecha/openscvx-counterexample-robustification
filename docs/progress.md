# Execution ledger: docs/implementation-plan.md

- Plan approved by the user for native execution.
- Original manuscript preserved in commit 927bddb; baseline and specification in b437581.
- Baseline unchanged first attempt failed on Windows cp1252 encoding. Second attempt with PYTHONUTF8=1 converged, then plotting failed because LaTeX is absent.
- Baseline reported clearance -0.005716 m before any disturbance.
- Interpolation inspection: OpenSCvx 0.5.2 uses beta=(tau-tau_init)*N in propagation and beta=tau*N in discretization, with segment duration 1/(N-1). Independent integration must reproduce that actual interpolation, including endpoint extrapolation.
- Ruling: use this new repository's existing checkout and main branch for the authorized initial publication; no concurrent work or existing code requires isolation. Cost if wrong: branch isolation would need to be added later.
- Ruling: preserve the supplied baseline file and bypass its external-TeX plotting in the experiment driver. Solve settings remain intact; experiment figures use Matplotlib mathtext. Cost if wrong: different typography only.
