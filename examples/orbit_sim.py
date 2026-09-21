"""Exemplo: orbit propagation over two periods with a 3D plot.

Run: python examples/orbit_sim.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from cubesat.orbit.orbital_elements import ClassicalElements  # noqa: E402
from cubesat.orbit.propagation import Orbit  # noqa: E402
from cubesat.sim.visualization import plot_3d_orbit  # noqa: E402
from cubesat.utils.constants import (  # noqa: E402
    MU_EARTH,
    R_EARTH,
    REFERENCE_ALTITUDE,
    REFERENCE_INCLINATION_DEG,
)


def main():
    elements = ClassicalElements(
        a=R_EARTH + REFERENCE_ALTITUDE,
        e=0.0,
        i=np.deg2rad(REFERENCE_INCLINATION_DEG),
        raan=0.0,
        argp=0.0,
        nu=0.0,
        mu=MU_EARTH,
    )
    orbit = Orbit(elements, mu=MU_EARTH, j2_on=True)
    period = elements.period

    print(f"Altitude       : {REFERENCE_ALTITUDE / 1e3:.1f} km")
    print(f"Inclination    : {REFERENCE_INCLINATION_DEG:.1f} deg")
    print(f"Orbital period : {period:.1f} s ({period / 60:.2f} min)")
    print(f"Mean motion    : {elements.mean_motion:.6f} rad/s")

    times = np.linspace(0.0, 2.0 * period, 500)
    rv = orbit.rv_over_time(times)
    positions = rv[:, 0, :]

    speed = np.linalg.norm(rv[:, 1, :], axis=1)
    print(f"Speed min/mean/max : {speed.min():.2f} / {speed.mean():.2f} / {speed.max():.2f} m/s")
    print(f"Vis-viva circular speed: {np.sqrt(MU_EARTH / elements.a):.2f} m/s")
    print(f"|r| spread over 2 periods: {(np.linalg.norm(positions, axis=1).max() - np.linalg.norm(positions, axis=1).min()):.3f} m")

    out = pathlib.Path("results")
    out.mkdir(exist_ok=True)
    plot_3d_orbit(positions, title="References: 2-body + J2 secular", save_path=out / "orbit.png")
    print(f"\nsaved -> {out / 'orbit.png'}")


if __name__ == "__main__":
    main()