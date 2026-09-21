"""End-to-end tests of the closed-loop attitude controller."""

import numpy as np

from cubesat.sim.simulation import Simulation, SimulationConfig


def _run(mode="inertial", duration=500.0, offset=150.0):
    cfg = SimulationConfig(
        duration=duration,
        dt=1.0,
        reference_mode=mode,
        initial_offset_deg=offset,
        seed=11,
    )
    return Simulation(cfg).run()


def test_inertial_attitude_converges():
    result = _run("inertial", duration=500.0)
    settle = result.settling_time(threshold_deg=1.0)
    assert settle is not None, "controller did not settle within 1 deg"
    assert settle < 500.0
    assert result.errors_deg[-1] < 1.0
    assert np.linalg.norm(result.omegas[-1]) < 1.0e-3


def test_nadir_pointing_tracks_reference():
    result = _run("nadir", duration=500.0)
    assert result.errors_deg[-1] < 5.0, "nadir pointing lagged too much"
    # Attitude must remain well-conditioned (unit quaternions throughout).
    assert np.allclose(np.linalg.norm(result.quaternions, axis=1), 1.0, atol=1e-9)


def test_actuator_never_exceeds_clamp():
    for mode in ("inertial", "nadir"):
        result = _run(mode, duration=300.0)
        assert np.all(np.abs(result.tau_actuated) <= result.config.max_torque + 1e-15)


def test_control_reduces_error_monotonically_after_transient():
    result = _run("inertial", duration=500.0)
    # After the transient, the error envelope decays: check the last third is below the first third.
    n = result.n_steps
    lo = int(n // 3)
    assert np.max(result.errors_deg[2 * lo:]) < np.max(result.errors_deg[:lo])


def test_zero_error_stays_at_rest():
    # Starting exactly on the reference at rest: controller must not move it.
    cfg = SimulationConfig(
        duration=200.0,
        dt=1.0,
        reference_mode="inertial",
        initial_offset_deg=0.0,
        initial_omega_body=np.zeros(3),
        enable_gravity_gradient=False,
        seed=3,
    )
    result = Simulation(cfg).run()
    assert result.errors_deg[-1] < 1e-6
    assert np.linalg.norm(result.omegas[-1]) < 1e-9