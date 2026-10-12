"""Independent nonlinear validation of a fixed OpenSCvx control schedule."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class ControlSchedule:
    times: np.ndarray
    values: np.ndarray
    fraction_scale: float = 1.0
    dilation: np.ndarray | None = None
    tau_step: float | None = None

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if (self.times.ndim != 1 or len(self.times) < 2
                or not np.all(np.isfinite(self.times))
                or np.any(np.diff(self.times) <= 0)
                or self.values.shape != (len(self.times), 2)
                or not np.all(np.isfinite(self.values))):
            raise ValueError("Controls require finite, strictly increasing time knots and Nx2 values")
        if self.dilation is not None:
            self.dilation = np.asarray(self.dilation, dtype=float)
            if self.dilation.shape != self.times.shape or np.any(self.dilation <= 0):
                raise ValueError("Time dilation must be positive at each knot")
            if self.tau_step is None or self.tau_step <= 0:
                raise ValueError("Normalized segment duration must be positive")

    def at(self, time, segment=None):
        k = (np.clip(np.searchsorted(self.times, time, side="right") - 1,
                     0, len(self.times) - 2) if segment is None else segment)
        elapsed = time - self.times[k]
        if self.dilation is None:
            beta = elapsed / (self.times[k + 1] - self.times[k]) * self.fraction_scale
        else:
            # Invert elapsed=s0*d_tau + 0.5*slope*d_tau^2.
            s0 = self.dilation[k]
            slope = (self.dilation[k + 1] - s0) * self.fraction_scale / self.tau_step
            if abs(slope) < 1e-12:
                d_tau = elapsed / s0
            else:
                radicand = s0 * s0 + 2 * slope * elapsed
                if radicand <= 0:
                    raise ValueError("Nonpositive interpolated time dilation")
                d_tau = 2 * elapsed / (s0 + np.sqrt(radicand))
            beta = d_tau / self.tau_step * self.fraction_scale
        return self.values[k] + beta * (self.values[k + 1] - self.values[k])

    def length(self):
        # Integrate the applied speed, including the installed segment extrapolation.
        total = 0.0
        for k in range(len(self.times) - 1):
            sol = solve_ivp(lambda t, y: [self.at(t, k)[0]], self.times[k:k + 2],
                            [0.0], method="DOP853", rtol=1e-11, atol=1e-12)
            total += sol.y[0, -1]
        return float(total)


def rollout(schedule, bias, times, initial=(0.0, 0.0, 0.0)):
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) == 0 or np.any(np.diff(times) < 0)
            or times[0] < schedule.times[0] - 1e-10
            or times[-1] > schedule.times[-1] + 1e-10):
        raise ValueError("Evaluation times must be sorted and inside the control horizon")
    samples = np.empty((len(times), 3))
    state = np.asarray(initial, dtype=float)
    for k, (left, right) in enumerate(zip(schedule.times[:-1], schedule.times[1:])):
        def rhs(t, x):
            v, w = schedule.at(t, segment=k)
            return [v * np.cos(x[2]), v * np.sin(x[2]), w + bias]
        sol = solve_ivp(rhs, (left, right), state, method="DOP853",
                        rtol=1e-10, atol=1e-12, dense_output=True,
                        max_step=(right - left) / 8)
        if not sol.success:
            raise RuntimeError(sol.message)
        mask = (times >= left - 1e-10) & (times <= right + 1e-10)
        samples[mask] = sol.sol(np.clip(times[mask], left, right)).T
        state = sol.y[:, -1]
    return samples


def sweep(schedule, biases, times, center=(2.0, 0.0), radius=0.55):
    started = perf_counter()
    worst = None
    for bias in biases:
        x = rollout(schedule, float(bias), times)
        clearance = np.linalg.norm(x[:, :2] - center, axis=1) - radius
        i = int(np.argmin(clearance))
        if worst is None or clearance[i] < worst["clearance"]:
            worst = {"clearance": float(clearance[i]), "bias": float(bias),
                     "time": float(times[i]), "position": x[i, :2].tolist(),
                     "trajectory": x}
    if worst is None:
        raise ValueError("At least one disturbance bias is required")
    worst["runtime_s"] = perf_counter() - started
    return worst
