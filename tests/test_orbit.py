"""Unit tests for orbital elements and Keplerian propagation."""

import numpy as np

from cubesat.orbit.orbital_elements import ClassicalElements, state_to_elements
from cubesat.orbit.propagation import Orbit
from cubesat.utils.constants import MU_EARTH, R_EARTH


def _orbit(e=0.1, i_deg=51.6):
    return ClassicalElements(
        a=R_EARTH + 550.0e3,
        e=e,
        i=np.deg2rad(i_deg),
        raan=np.deg2rad(30.0),
        argp=np.deg2rad(60.0),
        nu=np.deg2rad(40.0),
        mu=MU_EARTH,
    )


def test_circular_speed_matches_vis_viva():
    a = R_EARTH + 550.0e3
    el = ClassicalElements(a, 0.0, 0.0, 0.0, 0.0, 0.0, mu=MU_EARTH)
    r, v = el.to_state()
    assert np.isclose(np.linalg.norm(r), a, rtol=1e-12)
    assert np.isclose(np.linalg.norm(v), np.sqrt(MU_EARTH / a), rtol=1e-12)


def test_radius_constant_over_two_periods():
    el = _orbit(e=0.0)
    orbit = Orbit(el, mu=MU_EARTH, j2_on=False)
    times = np.linspace(0.0, 2.0 * el.period, 400)
    rv = orbit.rv_over_time(times)
    radii = np.linalg.norm(rv[:, 0, :], axis=1)
    assert np.allclose(radii, el.a, rtol=1e-6)


def test_energy_conserved_by_analytic_propagation():
    el = _orbit(e=0.1)
    orbit = Orbit(el, mu=MU_EARTH, j2_on=False)
    r0, v0 = el.to_state()
    r1, v1 = orbit.state_at(2.5 * el.period)
    e0 = 0.5 * np.dot(v0, v0) - MU_EARTH / np.linalg.norm(r0)
    e1 = 0.5 * np.dot(v1, v1) - MU_EARTH / np.linalg.norm(r1)
    assert np.isclose(e1, e0, rtol=1e-9)


def test_state_to_elements_roundtrip():
    el = _orbit(e=0.1)
    r, v = el.to_state()
    back = state_to_elements(r, v, mu=MU_EARTH)
    assert np.isclose(back.a, el.a, rtol=1e-9)
    assert np.isclose(back.e, el.e, rtol=1e-9)
    assert np.isclose(back.i, el.i, rtol=1e-9)
    assert np.isclose(back.raan, el.raan, rtol=1e-9)
    assert np.isclose(back.argp, el.argp, rtol=1e-9)
    assert np.isclose(back.nu, el.nu, rtol=1e-9)


def test_j2_secular_raan_drift_direction():
    # Near-polar prograde orbit: J2 makes RAAN drift *backwards* (negative).
    el = ClassicalElements(
        a=R_EARTH + 800.0e3, e=0.0, i=np.deg2rad(51.6), raan=0.0, argp=0.0, nu=0.0
    )
    r0, v0 = el.to_state()
    drift_per_day = 86400.0
    r1, v1 = __import__("cubesat.orbit.propagation", fromlist=["propagate_state"]).propagate_state(
        r0, v0, drift_per_day, j2_on=True
    )
    back = state_to_elements(r1, v1)
    assert back.raan < el.raan


def test_sun_vector_body_reference_is_normalized():
    from cubesat.attitude.rotations import sun_eci, eci_to_body
    from cubesat.attitude.quaternion import identity

    el = _orbit(e=0.0)
    r, _ = el.to_state()
    s_hat = sun_eci(r, np.array([1.0, 0.0, 0.0]))
    assert np.isclose(np.linalg.norm(s_hat), 1.0)
    s_b = eci_to_body(identity(), s_hat)
    assert np.isclose(np.linalg.norm(s_b), 1.0)