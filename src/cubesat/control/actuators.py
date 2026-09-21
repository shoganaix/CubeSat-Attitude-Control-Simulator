"""Actuator models: ideal torque source with saturation."""

import numpy as np


class IdealActuator:
    """Perfect torque execution clamped to a per-axis magnitude limit.

    Placeholder for reaction wheels / magnetorquers; wheel momentum dynamics,
    friction and magnetorquer dipole limits arrive in a later milestone.
    """

    def __init__(self, max_torque):
        self.max_torque = float(max_torque)

    def apply(self, torque_cmd):
        tau = np.asarray(torque_cmd, dtype=float)
        return np.clip(tau, -self.max_torque, self.max_torque)

    def is_saturated(self, torque_cmd):
        return np.any(np.abs(np.asarray(torque_cmd, dtype=float)) > self.max_torque + 1e-12)