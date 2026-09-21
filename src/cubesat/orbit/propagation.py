"""Orbit propagation.

* ``propagate_state`` : two-body (Keplerian) propagation with optional J2 secular terms.
* ``Orbit``           : helper that evaluates ``r(t), v(t)`` from initial elements.
"""

import numpy as np

from cubesat.orbit.orbital_elements import ClassicalElements, solve_kepler, state_to_elements
from cubesat.utils.constants import R_EARTH, J2, MU_EARTH


def propagate_state(r0, v0, dt, mu=MU_EARTH, j2_on=True, j2=J2, re=R_EARTH):
    """Two-body propagation of an ECI state by ``dt`` seconds.

    Not valid for e very close to 1; use a numerical integrator (e.g. SGP4)
    for long-term / high-eccentricity arcs. J2 adds secular rates on RAAN,
    argument of perigee and mean motion.
    """
    el = state_to_elements(r0, v0, mu)
    n0 = el.mean_motion
    p = el.a * (1.0 - el.e * el.e)
    factor = -(1.5 * j2 * (re / p) ** 2 * n0) if j2_on else 0.0

    raan_dot = factor * np.cos(el.i)
    argp_dot = 0.5 * 3.0 * j2 * (re / p) ** 2 * n0 * (5.0 * np.cos(el.i) ** 2 - 1.0) if j2_on else 0.0
    n_j2 = n0 * (1.0 + 0.5 * 3.0 * j2 * (re / p) ** 2 * np.sqrt(1.0 - el.e ** 2)
                 * (1.0 - 1.5 * np.sin(el.i) ** 2)) if j2_on else n0

    raan = el.raan + raan_dot * dt
    argp = el.argp + argp_dot * dt
    M = el.mean_anomaly + n_j2 * dt
    E = solve_kepler(M, el.e)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + el.e) * np.sin(E / 2.0),
        np.sqrt(1.0 - el.e) * np.cos(E / 2.0),
    )

    els = ClassicalElements(el.a, el.e, el.i, raan, argp, nu, mu=mu)
    return els.to_state()


class Orbit:
    """Compute ``r(t), v(t)`` from initial Keplerian elements."""

    def __init__(self, elements, mu=MU_EARTH, j2_on=True):
        self.elements = elements
        self.mu = float(mu)
        self.j2_on = bool(j2_on)

    def state_at(self, t):
        """ECI position [m] and velocity [m/s] at time ``t`` [s]."""
        r0, v0 = self.elements.to_state()
        return propagate_state(r0, v0, float(t), mu=self.mu, j2_on=self.j2_on)

    def rv_over_time(self, times):
        """Array of (r, v) pairs evaluated at each time in ``times``."""
        return np.array([self.state_at(t) for t in times])