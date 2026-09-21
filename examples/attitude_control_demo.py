"""Attitude control demo: nadir-pointing maneuver on a 550 km orbit.

Run: python examples/attitude_control_demo.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cubesat.sim.simulation import Simulation, SimulationConfig  # noqa: E402
from cubesat.sim.visualization import plot_attitude, plot_orbit_with_attitude  # noqa: E402


def main():
    cfg = SimulationConfig(
        duration=500.0,
        dt=1.0,
        reference_mode="nadir",
        initial_offset_deg=150.0,
    )
    sim = Simulation(cfg)
    result = sim.run()

    settle = result.settling_time(threshold_deg=1.0)
    print("Nadir-pointing control demo")
    print(f"  duration      : {cfg.duration:.0f} s | dt = {cfg.dt} s")
    print(f"  initial error : {result.errors_deg[0]:.1f} deg")
    if settle is not None:
        print(f"  settling (<1 deg) : {settle:.0f} s")
    else:
        print("  settling (<1 deg) : NOT reached")
    print(f"  final error   : {result.errors_deg[-1]:.4f} deg")
    print(f"  final rate    : {max(abs(result.omegas[-1])):.3e} rad/s")
    print(f"  max cmd torque: {max((result.tau_commanded ** 2).sum(1) ** 0.5):.3e} N m")

    out = pathlib.Path("results")
    out.mkdir(exist_ok=True)
    result.to_csv(out / "telemetry.csv")
    plot_attitude(result, save_path=out / "attitude.png")
    plot_orbit_with_attitude(result, save_path=out / "orbit_attitude.png", n_samples=5)
    print(f"saved -> {out / 'telemetry.csv'}, {out / 'attitude.png'}, {out / 'orbit_attitude.png'}")


if __name__ == "__main__":
    main()