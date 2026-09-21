"""Reference-frame helpers and Euler angle conversions.

Frames used across the project:
* ECI  - Earth-Centered Inertial (J2000-like), the inertial reference.
* BODY - rigid-body frame fixed to the spacecraft, defined by the attitude quaternion.
* LVLH - Local Vertical Local Horizontal frame for nadir-pointing missions.
"""

import numpy as np

from cubesat.attitude.quaternion import from_rotation_matrix, to_rotation_matrix


def body_to_eci(q, v_body):
    """Express a vector in ECI given its body-frame components."""
    return to_rotation_matrix(q) @ v_body


def eci_to_body(q, v_eci):
    """Express a vector in the body frame given its ECI components."""
    return to_rotation_matrix(q).T @ v_eci


def euler321_to_dcm(phi, theta, psi):
    """DCM from 3-2-1 (yaw-pitch-roll) Euler angles [rad].

    ``C = Rz(psi) @ Ry(theta) @ Rx(phi)`` maps body vectors to the reference frame.
    """
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cs, ss = np.cos(psi), np.sin(psi)
    return np.array([
        [ct * cs, sp * st * cs - cp * ss, cp * st * cs + sp * ss],
        [ct * ss, sp * st * ss + cp * cs, cp * st * ss - sp * cs],
        [-st, sp * ct, cp * ct],
    ])


def dcm_to_euler321(C):
    """3-2-1 Euler angles [rad] from a DCM."""
    theta = np.arcsin(np.clip(-C[2, 0], -1.0, 1.0))
    phi = np.arctan2(C[2, 1], C[2, 2])
    psi = np.arctan2(C[1, 0], C[0, 0])
    return phi, theta, psi


def lvlh_quaternion(r_eci, v_eci):
    """Quaternion mapping *body* -> *inertial* (ECI) for a frame aligned with LVLH.

    Axes (LVLH): x along cross-product normal, z along nadir (toward Earth),
    y completing the right-handed triad.
    """
    r = np.asarray(r_eci, dtype=float)
    v = np.asarray(v_eci, dtype=float)
    rhat = r / np.linalg.norm(r)
    z = -rhat
    y = np.cross(r, v)
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    C = np.column_stack([x, y, z])  # maps LVLH vectors to ECI
    return from_rotation_matrix(C)


def nadir_eci(r_eci):
    """Nadir (Earth-pointing) unit vector in ECI."""
    return -r_eci / np.linalg.norm(r_eci)


def sun_eci(r_eci, sun_vector_eci):
    """Unit vector from the spacecraft toward the Sun, in ECI."""
    to_sun = np.asarray(sun_vector_eci, dtype=float) - np.asarray(r_eci, dtype=float)
    return to_sun / np.linalg.norm(to_sun)