"""Classical orbital elements and conversions to/from ECI state vectors."""

import numpy as np

from cubesat.utils.constants import MU_EARTH


def solve_kepler(M, e, tol=1e-12, max_iter=100):
    """Solve Kepler's equation ``M = E - e sin(E)`` for the eccentric anomaly E."""
    M = np.asarray(M, dtype=float)
    if e < 1e-10:
        return M
    E = M + e * np.sin(M)
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = f / fp
        E -= dE
        if np.all(np.abs(dE) < tol):
            break
    return E


class ClassicalElements:
    """Keplerian orbital elements (all angles in rad, lengths in m)."""

    def __init__(self, a, e, i, raan, argp, nu, mu=MU_EARTH):
        self.a = float(a)
        self.e = float(e)
        self.i = float(i)
        self.raan = float(raan)
        self.argp = float(argp)
        self.nu = float(nu)
        self.mu = float(mu)

    @property
    def period(self):
        return 2.0 * np.pi * np.sqrt(self.a ** 3 / self.mu)

    @property
    def mean_motion(self):
        return np.sqrt(self.mu / self.a ** 3)

    @property
    def mean_anomaly(self):
        E = 2.0 * np.arctan2(
            np.sin(self.nu) * np.sqrt(1.0 - self.e),
            np.cos(self.nu) + self.e,
        )
        return E - self.e * np.sin(E)

    def to_state(self):
        """Position/velocity in ECI [m], [m/s]."""
        e, nu = self.e, self.nu
        p = self.a * (1.0 - e * e)
        r_peri = p / (1.0 + e * np.cos(nu))
        r_pf = r_peri * np.array([np.cos(nu), np.sin(nu), 0.0])
        v_pf = np.sqrt(self.mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

        # Perifocal -> ECI rotation: Rz(raan) Rx(i) Rz(argp)
        co, so = np.cos(self.raan), np.sin(self.raan)
        cv, sv = np.cos(self.i), np.sin(self.i)
        cw, sw = np.cos(self.argp), np.sin(self.argp)
        R = np.array([
            [co * cw - so * sw * cv, -co * sw - so * cw * cv, so * sv],
            [so * cw + co * sw * cv, -so * sw + co * cw * cv, -co * sv],
            [sw * sv, cw * sv, cv],
        ])
        return R @ r_pf, R @ v_pf


def state_to_elements(r, v, mu=MU_EARTH):
    """Convert an ECI state vector to :class:`ClassicalElements`."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)

    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h)

    energy = 0.5 * vn * vn - mu / rn
    a = -mu / (2.0 * energy)

    e_vec = np.cross(v, h) / mu - r / rn
    e = np.linalg.norm(e_vec)

    i = np.arccos(np.clip(h[2] / hn, -1.0, 1.0))
    raan = 0.0 if np.linalg.norm(n_vec) < 1e-12 else np.arctan2(n_vec[1], n_vec[0])

    if e > 1e-10:
        argp = np.arctan2(
            np.dot(np.cross(n_vec, e_vec), h) / hn,
            np.dot(n_vec, e_vec),
        )
        nu = np.arctan2(
            np.dot(np.cross(e_vec, r), h) / hn,
            np.dot(e_vec, r),
        )
    else:
        argp = 0.0
        nu = np.arctan2(r[1], r[0])

    return ClassicalElements(a, e, i, raan, argp, nu, mu=mu)