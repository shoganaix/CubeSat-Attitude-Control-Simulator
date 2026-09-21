"""Plotting helpers for the ADCS simulator (matplotlib, Agg backend)."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cubesat.attitude.quaternion import to_rotation_matrix  # noqa: E402
from cubesat.utils.constants import R_EARTH  # noqa: E402

RAD2DEG = 180.0 / np.pi


def _earth_sphere(ax, radius_m=R_EARTH, n=24):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = radius_m * np.outer(np.cos(u), np.sin(v))
    y = radius_m * np.outer(np.sin(u), np.sin(v))
    z = radius_m * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="tab:blue", alpha=0.25, linewidth=0)


def plot_3d_orbit(positions_eci, title="Orbit", save_path=None):
    """3D view of an ECI trajectory (positions in meters) over an Earth sphere."""
    positions_eci = np.asarray(positions_eci, dtype=float)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    _earth_sphere(ax)
    ax.plot(positions_eci[:, 0], positions_eci[:, 1], positions_eci[:, 2], color="k", lw=1.5)
    ax.scatter(*positions_eci[0], color="tab:red", s=40, label="start")
    ax.scatter(*positions_eci[-1], color="tab:green", s=40, label="end")
    ax.set_xlabel("X ECI [m]")
    ax.set_ylabel("Y ECI [m]")
    ax.set_zlabel("Z ECI [m]")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_attitude(result, save_path=None):
    """4-panel attitude response: error, rates, control and disturbance torques."""
    t = result.t
    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

    axes[0].plot(t, result.errors_deg, lw=1.5)
    axes[0].axhline(1.0, color="r", ls="--", lw=1, label="1 deg threshold")
    axes[0].set_ylabel("error [deg]")
    axes[0].set_title("Attitude error to reference (nadir-pointing)")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    w = np.degrees(result.omegas)
    axes[1].plot(t, w[:, 0], label=r"$\omega_x$")
    axes[1].plot(t, w[:, 1], label=r"$\omega_y$")
    axes[1].plot(t, w[:, 2], label=r"$\omega_z$")
    axes[1].set_ylabel("body rate [deg/s]")
    axes[1].set_title("Angular velocity (body frame)")
    axes[1].grid(True)
    axes[1].legend()

    for i, label in enumerate("xyz"):
        axes[2].plot(t, result.tau_commanded[:, i], label=f"cmd {label}")
    for i, label in enumerate("xyz"):
        axes[2].plot(t, result.tau_actuated[:, i], ls="--", lw=1, label=f"act {label}")
    axes[2].set_ylabel("control torque [N m]")
    axes[2].set_title("Commanded vs actuated torque (per axis)")
    axes[2].grid(True)
    axes[2].legend(ncol=2)

    axes[3].plot(t, np.linalg.norm(result.tau_disturbance, axis=1), label="gravity gradient", lw=1.5)
    axes[3].plot(t, np.linalg.norm(result.tau_commanded, axis=1), label="commanded norm", lw=1.5)
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("torque [N m]")
    axes[3].set_title("Disturbance vs commanded torque magnitude")
    axes[3].grid(True)
    axes[3].legend()

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_orbit_with_attitude(result, save_path=None, axis_len_km=200.0, n_samples=5):
    """Orbit trace plus body-frame axes drawn at sampled instants."""
    positions = result.r_ecis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    _earth_sphere(ax)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="k", lw=1.5)
    ax.scatter(*positions[0], color="k", s=30)

    quotes = np.linspace(0, result.n_steps - 1, n_samples).astype(int)
    for idx in quotes:
        r = positions[idx]
        C = to_rotation_matrix(result.quaternions[idx])
        for axis_i, color in zip(range(3), ("r", "g", "b")):
            end = r + axis_len_km * 1e3 * C[:, axis_i]
            ax.plot(
                [r[0], end[0]], [r[1], end[1]], [r[2], end[2]],
                color=color, lw=1.5,
            )

    ax.set_xlabel("X ECI [m]")
    ax.set_ylabel("Y ECI [m]")
    ax.set_zlabel("Z ECI [m]")
    ax.set_title(f"Orbit with body axes (RGB = X, Y, Z), ref={result.config.reference_mode}")
    ax.set_aspect("equal")
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig