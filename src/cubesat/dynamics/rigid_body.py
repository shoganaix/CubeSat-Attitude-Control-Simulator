"""Rigid-body rotational dynamics (Euler's equations + quaternion kinematics).

State vector: ``x = [q0, q1, q2, q3, wx, wy, wz]``
* ``q``   : attitude quaternion (body -> inertial)
* ``w``   : angular velocity in the *body* frame [rad/s]

Integration is a fixed-step classical RK4 with state normalization after each step.
"""

import numpy as np

from cubesat.attitude import quaternion as qm


def attitude_derivative(q, omega, inertia, torque):
    """Return ``(q_dot, omega_dot)`` for the true rigid-body dynamics."""
    omega = np.asarray(omega, dtype=float)
    J = np.asarray(inertia, dtype=float)
    tau = np.asarray(torque, dtype=float)

    # Euler's equations: J w_dot + w x (J w) = tau
    omega_dot = np.linalg.solve(J, tau - np.cross(omega, J @ omega))

    # Kinematics: q_dot = 0.5 * q (x) [0; w]
    wq = np.concatenate(([0.0], omega))
    q_dot = 0.5 * qm.multiply(q, wq)

    return q_dot, omega_dot


def state_derivative(x, inertia, torque_fn):
    """Wrapper returning a 7-vector derivative for integration.

    ``torque_fn(state)`` returns the total external torque in the body frame.
    """
    q = x[:4]
    omega = x[4:]
    tau = np.asarray(torque_fn(q, omega), dtype=float)
    q_dot, omega_dot = attitude_derivative(q, omega, inertia, tau)
    return np.concatenate((q_dot, omega_dot))


def rk4_step(deriv, x, dt, inertia, torque_fn):
    """Advance ``x`` by ``dt`` using one classical RK4 step."""
    k1 = deriv(x, inertia, torque_fn)
    k2 = deriv(x + 0.5 * dt * k1, inertia, torque_fn)
    k3 = deriv(x + 0.5 * dt * k2, inertia, torque_fn)
    k4 = deriv(x + dt * k3, inertia, torque_fn)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next[:4] = qm.normalize(x_next[:4])
    return x_next


def integrate_state(x0, dt, n_steps, inertia, torque_fn):
    """Integrate the rigid-body state for ``n_steps`` fixed steps; returns full (n+1, 7) array."""
    x = np.asarray(x0, dtype=float).copy()
    traj = np.empty((n_steps + 1, 7))
    traj[0] = x
    for i in range(n_steps):
        x = rk4_step(state_derivative, x, dt, inertia, torque_fn)
        traj[i + 1] = x
    return traj