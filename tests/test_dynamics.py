"""Unit tests for rigid-body dynamics and integrator invariants."""

import numpy as np

from cubesat.attitude.quaternion import (
    chordal_distance,
    from_axis_angle,
    identity,
    multiply,
    to_rotation_matrix,
)
from cubesat.dynamics.rigid_body import integrate_state


def _zero_torque(q, omega):
    return np.zeros(3)


def _inertial_angular_momentum(q, omega, J):
    # L is conserved in the *inertial* frame; in the body frame it precesses.
    return to_rotation_matrix(q) @ (J @ omega)


def test_torque_free_conserves_angular_momentum_and_kinematics():
    J = np.diag([0.020, 0.024, 0.008])
    omega0 = np.array([0.05, -0.03, 0.12])
    q0 = from_axis_angle([1.0, 2.0, -0.5], 1.2)
    x0 = np.concatenate((q0, omega0))

    dt = 0.01
    traj = integrate_state(x0, dt, 2000, J, _zero_torque)
    final_q = traj[-1, :4]
    final_w = traj[-1, 4:]

    # Kinematics / normalization
    assert np.isclose(np.linalg.norm(final_q), 1.0, atol=1e-12)
    assert chordal_distance(q0, final_q) > 1e-3  # the object actually rotated

    # Angular momentum L = J w must be conserved in the inertial frame
    L0 = _inertial_angular_momentum(q0, omega0, J)
    Lf = _inertial_angular_momentum(final_q, final_w, J)
    assert np.allclose(L0, Lf, atol=1e-9 * (1 + np.abs(L0)).max())

    # Rotational kinetic energy 0.5 * w^T J w
    E0 = 0.5 * omega0 @ J @ omega0
    Ef = 0.5 * final_w @ J @ final_w
    assert np.isclose(E0, Ef, rtol=1e-10)


def test_constant_principal_axis_spin_matches_closed_form():
    # Pure spin about a principal axis keeps omega constant in the body frame;
    # the quaternion must follow q(t) = q0 (x) rot(z, w*t).
    J = np.diag([0.020, 0.024, 0.008])
    w = 0.3  # rad/s
    omega0 = np.array([0.0, 0.0, w])
    q0 = from_axis_angle([1.0, 0.0, 0.0], 0.4)
    x0 = np.concatenate((q0, omega0))

    dt = 0.01
    t_end = 5.0
    n = int(t_end / dt)
    traj = integrate_state(x0, dt, n, J, _zero_torque)

    expected = multiply(q0, from_axis_angle([0.0, 0.0, 1.0], w * t_end))
    assert chordal_distance(traj[-1, :4], expected) < 1e-6
    assert np.allclose(traj[-1, 4:], omega0, atol=1e-9)


def test_state_derivative_reuses_exposed_api():
    # Sanity: the convenience derivative function type-checks and returns 7 values.
    from cubesat.dynamics.rigid_body import state_derivative

    J = np.diag([0.02, 0.02, 0.01])
    x = np.concatenate((identity(), np.zeros(3)))
    dx = state_derivative(x, J, _zero_torque)
    assert dx.shape == (7,)
    assert np.allclose(dx, 0.0)