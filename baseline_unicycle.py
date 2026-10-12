"""Fixed-horizon unicycle path planning with a circular obstacle (OpenSCvx)."""
import os
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
os.environ.setdefault("OPENSCVX_CACHE_DIR", "/tmp/openscvx-cache")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import openscvx as ox
from openscvx import Problem

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["serif"],
    "font.sans-serif": ["sans-serif"],
    "font.monospace": ["Courier"],
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

N, T = 5, 8.0
START = np.array([0.0, 0.0, 0.0])
GOAL = np.array([4.0, 0.0, 0.0])
V_MAX, OMEGA_MAX = 1.0, 0.8
OBS_C = np.array([2.0, 0.0])
OBS_R = 0.55

position = ox.State("position", shape=(2,))
position.min = np.array([-0.2, -2.0])
position.max = np.array([4.2, 2.0])
position.initial = START[:2]
position.final = GOAL[:2]

theta = ox.State("theta", shape=(1,))
theta.min = np.array([-np.pi])
theta.max = np.array([np.pi])
theta.initial = np.array([START[2]])
theta.final = np.array([GOAL[2]])

distance = ox.State("distance", shape=(1,))
distance.min = np.array([0.0])
distance.max = np.array([20.0])
distance.initial = np.array([0.0])
distance.final = [ox.Minimize(5.0)]

speed = ox.Control("speed", shape=(1,))
speed.min, speed.max = np.array([0.0]), np.array([V_MAX])

turn_rate = ox.Control("turn_rate", shape=(1,))
turn_rate.min, turn_rate.max = (
    np.array([-OMEGA_MAX]),
    np.array([OMEGA_MAX]),
)

tgrid = np.linspace(0.0, 1.0, N)
dt = T / (N - 1)
xg = 4.0 * tgrid
yg = 1.15 * np.sin(np.pi * tgrid)

dx, dy = np.gradient(xg, dt), np.gradient(yg, dt)
heading = np.unwrap(np.arctan2(dy, dx))
vguess = np.sqrt(dx**2 + dy**2)
wguess = np.gradient(heading, dt)

position.guess = np.column_stack((xg, yg))
theta.guess = heading[:, None]
distance.guess = (np.linspace(0.0, 1.0, N) * 5.0)[:, None]
speed.guess = np.clip(vguess[:, None], 0.0, V_MAX)
turn_rate.guess = np.clip(
    wguess[:, None], -OMEGA_MAX, OMEGA_MAX
)

states = [position, theta, distance]
controls = [speed, turn_rate]

obs_center = ox.Parameter(
    "obs_center", shape=(2,), value=OBS_C
)
obs_radius = ox.Parameter(
    "obs_radius", shape=(), value=OBS_R
)

constraints = [
    ox.ctcs(
        obs_radius <= ox.linalg.Norm(position - obs_center),
        penalty="squared_relu",
        check_nodally=True,
    )
]

dynamics = {
    "position": ox.Concat(
        speed[0] * ox.Cos(theta[0]),
        speed[0] * ox.Sin(theta[0]),
    ),
    "theta": turn_rate[0],
    "distance": speed[0],
}

problem = Problem(
    dynamics=dynamics,
    states=states,
    controls=controls,
    time=ox.Time(
        initial=0.0,
        final=T,
        min=0.0,
        max=T,
    ),
    constraints=constraints,
    N=N,
    algorithm={
        "lam_cost": {"distance": 1.0},
        "lam_vc": 3e2,
        "k_max": 500,
        "autotuner": {
            "type": "AugmentedLagrangian",
            "rho_init": 1e7,
            "rho_max": 1e11,
        },
    },
    discretizer={
        "ode_solver": "Dopri8",
        "diffrax_kwargs": {
            "max_steps": 50000,
            "num_substeps": 256,
        },
    },
    float_dtype="float64",
)

if __name__ == "__main__":
    problem.initialize()
    result = problem.solve()
    result = problem.post_process()

    traj = result.trajectory
    p = np.asarray(traj["position"])
    th = np.asarray(traj["theta"]).reshape(-1)
    v = np.asarray(traj["speed"]).reshape(-1)
    w = np.asarray(traj["turn_rate"]).reshape(-1)
    ts = np.linspace(0.0, T, len(p))

    clearance = np.min(
        np.linalg.norm(p - OBS_C, axis=1) - OBS_R
    )
    final_distance = float(
        np.asarray(traj["distance"])[-1].reshape(-1)[0]
    )

    print(f"converged={result.converged}")
    print(f"cost_distance={final_distance:.6f}")
    print(f"min_obstacle_clearance={clearance:.6f}")
    print(
        f"max_speed={np.max(v):.6f}; "
        f"max_abs_turn_rate={np.max(np.abs(w)):.6f}"
    )
    print(
        f"initial_pose="
        f"{p[0, 0]:.6f},{p[0, 1]:.6f},{th[0]:.6f}"
    )
    print(
        f"final_pose="
        f"{p[-1, 0]:.6f},{p[-1, 1]:.6f},{th[-1]:.6f}"
    )

    fig, ax = plt.subplots(
        figsize=(12.5, 4.2),
        constrained_layout=True,
    )

    circle = plt.Circle(
        OBS_C,
        OBS_R,
        color="#d95f59",
        alpha=0.25,
        ec="#b33b35",
        lw=2,
        label="circular obstacle",
    )
    ax.add_patch(circle)

    ax.plot(
        p[:, 0],
        p[:, 1],
        color="#1769aa",
        lw=2.6,
        label="OpenSCvx trajectory",
    )

    ax.scatter(
        [START[0], GOAL[0]],
        [START[1], GOAL[1]],
        c=["#2a9d62", "#7e57c2"],
        s=65,
        zorder=4,
    )

    ax.annotate(
        "start",
        START[:2],
        xytext=(-2, 10),
        textcoords="offset points",
    )
    ax.annotate(
        "goal",
        GOAL[:2],
        xytext=(-2, 10),
        textcoords="offset points",
    )

    ids = np.linspace(0, len(p) - 1, 9, dtype=int)
    ax.quiver(
        p[ids, 0],
        p[ids, 1],
        np.cos(th[ids]),
        np.sin(th[ids]),
        color="#17324d",
        scale=16,
        width=0.004,
        label="heading",
    )

    ax.set(
        xlabel="$p_x$ [m]",
        ylabel="$p_y$ [m]",
        title="Unicycle minimum-distance path",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
    )
    ax.set_xlim(-0.35, 4.35)
    ax.set_ylim(-0.65, 0.85)

    fig.savefig("unicycle_solution.pdf")
