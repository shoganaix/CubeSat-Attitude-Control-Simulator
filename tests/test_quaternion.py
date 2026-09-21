"""Unit tests for quaternion algebra (attitude/quaternion.py)."""

import numpy as np
import pytest

from cubesat.attitude.quaternion import (
    chordal_distance,
    conjugate,
    error,
    from_axis_angle,
    from_rotation_matrix,
    from_rotation_vector,
    identity,
    multiply,
    normalize,
    rotate,
    to_rotation_matrix,
)


def test_normalize_unit_and_inverse():
    q = np.array([0.3, 0.4, 0.5, 0.7])
    n = normalize(q)
    assert np.isclose(np.linalg.norm(n), 1.0)
    assert np.allclose(multiply(n, conjugate(n)), identity(), atol=1e-12)


def test_rotate_z_90_maps_x_to_y():
    q = from_axis_angle([0.0, 0.0, 1.0], np.pi / 2.0)
    assert np.allclose(rotate(q, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_rotate_z_180_maps_x_to_minus_x():
    q = from_axis_angle([0.0, 0.0, 1.0], np.pi)
    assert np.allclose(rotate(q, [1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-12)


def test_dcm_roundtrip():
    q = from_axis_angle([1.0, 2.0, -0.5], 1.1)
    q_back = from_rotation_matrix(to_rotation_matrix(q))
    assert chordal_distance(q, q_back) < 1e-12


def test_error_quaternion_identity_when_equal():
    q = from_axis_angle([0.4, 0.3, 0.2], 2.0)
    dq = error(q, q)
    assert np.isclose(abs(dq[0]), 1.0)
    assert np.allclose(dq[1:], 0.0, atol=1e-12)


def test_compose_error_then_recordering():
    a = from_axis_angle([1.0, 0.0, 0.0], 0.5)
    d = from_axis_angle([0.0, 1.0, 0.0], 0.8)
    dq = error(a, d)
    # applying dq to 'a' must land on 'd' (as a rotation)
    assert chordal_distance(multiply(dq, a), d) < 1e-12


def test_chordal_distance_matches_rotation_angle():
    theta = np.pi / 3.0
    q = from_axis_angle([0.0, 0.0, 1.0], theta)
    assert np.isclose(chordal_distance(identity(), q), theta)


def test_chordal_distance_shortest_path():
    # 270 deg rotation is measured as 90 deg (shortest path / double cover).
    q = from_axis_angle([0.0, 0.0, 1.0], 3.0 * np.pi / 2.0)
    assert np.isclose(chordal_distance(identity(), q), np.pi / 2.0)


def test_from_rotation_vector():
    rv = np.array([0.2, 0.3, 0.4]) * 1.2
    q = from_axis_angle(rv, np.linalg.norm(rv))
    assert np.isclose(chordal_distance(q, from_rotation_vector(rv)), 0.0)