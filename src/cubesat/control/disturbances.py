"""External disturbance torque models."""

import numpy as np

from cubesat.attitude.rotations import eci_to_body


def gravity_gradient_torque(q, r_eci, inertia, mu):
    """Gravity-gradient torque in the body frame.

    ``tau_gg = (3 * mu / r^3) * (rhat_b x (J @ rhat_b))``
    with ``rhat_b`` the radial (Earth->spacecraft) unit vector in body frame.
    """
    r = np.asarray(r_eci, dtype=float)
    rn = np.linalg.norm(r)
    J = np.asarray(inertia, dtype=float)

    rad_eci = r / rn
    rad_b = eci_to_body(q, rad_eci)
    return (3.0 * mu / rn ** 3) * np.cross(rad_b, J @ rad_b)


def environment_torque(q, r_eci, inertia, mu, j2_enabled=True):
    """Sum of modeled environmental disturbances (gravity gradient only for now)."""
    return gravity_gradient_torque(q, r_eci, inertia, mu)