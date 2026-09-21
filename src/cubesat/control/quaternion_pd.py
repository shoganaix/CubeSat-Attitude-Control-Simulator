"""Quaternion-error PD attitude controller (space-grade PD).

Control law (Wie's quaternion feedback):
    tau = -kp * sign(dq0) * dq_vec - kd * (omega - omega_ref)

The control error quaternion is ``dq = q_des^{-1} (x) q_cur``; its rotation
matrix ``C(dq) = C(q_des)^T C(q_cur)`` maps coordinates from the current body
frame into the desired body frame, so the vector part of ``dq`` is the
shortest-path rotation axis expressed in the current body frame. ``sign(dq0)``
handles the quaternion double cover.
"""

import numpy as np

from cubesat.attitude.quaternion import conjugate, identity, multiply


def tune_gains(J_diag, omega_n, zeta=1.0):
    """Scalar PD gains from a diagonal inertia and a design 2nd-order response.

    Uses the largest moment of inertia to be conservative across axes.
    """
    j_max = np.max(np.asarray(J_diag, dtype=float))
    kp = omega_n ** 2 * j_max
    kd = 2.0 * zeta * omega_n * j_max
    return float(kp), float(kd)


class QuaternionPD:
    """PD regulator (with optional reference rate) on quaternion attitude error."""

    def __init__(self, kp, kd):
        self.kp = float(kp)
        self.kd = float(kd)

    def compute_torque(self, q, omega, q_des, omega_des=None):
        q = np.asarray(q, dtype=float)
        omega = np.asarray(omega, dtype=float)
        q_des = np.asarray(q_des, dtype=float)
        omega_des = np.zeros(3) if omega_des is None else np.asarray(omega_des, dtype=float)

        dq = multiply(conjugate(q_des), q)         # rotate current body -> desired frame
        sign = 1.0 if dq[0] >= 0.0 else -1.0
        e_vec = sign * dq[1:4]
        omega_err = omega - omega_des

        return -self.kp * e_vec - self.kd * omega_err