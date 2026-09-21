"""Quaternion utilities used across the ADCS stack.

Convention
----------
* Hamilton convention, scalar-first layout: ``q = [q0, q1, q2, q3] = [w, x, y, z]``.
* Multiplication ``multiply(p, q)`` is the Hamilton product ``p (x) q``.
* A quaternion ``q`` rotates vectors from *body* to *inertial*:
  ``v_inertial = R(q) @ v_body`` (see ``to_rotation_matrix``).
* For unit quaternions the inverse equals the conjugate.
"""

import numpy as np


def identity():
    """Return the identity quaternion (no rotation)."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def normalize(q):
    """Normalize a quaternion to unit length in place of a NaN check."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-15:
        raise ValueError("zero-norm quaternion cannot be normalized")
    return q / n


def conjugate(q):
    """Conjugate (inverse for unit quaternions)."""
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def multiply(p, q):
    """Hamilton product ``p (x) q`` (apply the rotation of ``q`` first, then ``p``)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p0, p1, p2, p3 = p
    q0, q1, q2, q3 = q
    return np.array([
        p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
        p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
        p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
        p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0,
    ])


def from_axis_angle(axis, angle):
    """Build a unit quaternion that rotates ``angle`` [rad] about ``axis``."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return identity()
    axis = axis / n
    half = 0.5 * angle
    s = np.sin(half)
    return np.array([np.cos(half), s * axis[0], s * axis[1], s * axis[2]])


def from_rotation_vector(rv):
    """Build a unit quaternion from a rotation vector ``rv`` (angle ~= |rv|, axis ~= rv/|rv|)."""
    rv = np.asarray(rv, dtype=float)
    return from_axis_angle(rv, np.linalg.norm(rv))


def to_rotation_vector(q):
    """Rotation vector (angle [rad] x unit axis) from a unit quaternion (shortest rotation)."""
    q = np.asarray(q, dtype=float)
    if q[0] < 0.0:
        q = -q  # shortest-path representation
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    s = np.sqrt(1.0 - q[0] * q[0])
    if s < 1e-12:
        return np.zeros(3)
    return angle * q[1:] / s


def rotate(q, v):
    """Rotate a 3-vector ``v`` by quaternion ``q``: ``v' = q (x) v (x) q*``."""
    q = np.asarray(q, dtype=float)
    v = np.asarray(v, dtype=float)
    qv = np.array([0.0, v[0], v[1], v[2]])
    return multiply(multiply(q, qv), conjugate(q))[1:]


def to_rotation_matrix(q):
    """Rotation matrix ``R`` such that ``v_inertial = R @ v_body`` (body-to-inertial)."""
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q
    return np.array([
        [1.0 - 2.0 * (q2 * q2 + q3 * q3), 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
        [2.0 * (q1 * q2 + q0 * q3), 1.0 - 2.0 * (q1 * q1 + q3 * q3), 2.0 * (q2 * q3 - q0 * q1)],
        [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), 1.0 - 2.0 * (q1 * q1 + q2 * q2)],
    ])


def from_rotation_matrix(C):
    """Unit quaternion from a valid rotation matrix (Shoemake extraction)."""
    C = np.asarray(C, dtype=float)
    tr = np.trace(C)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q0 = 0.25 * s
        q1 = (C[2, 1] - C[1, 2]) / s
        q2 = (C[0, 2] - C[2, 0]) / s
        q3 = (C[1, 0] - C[0, 1]) / s
    elif C[0, 0] > C[1, 1] and C[0, 0] > C[2, 2]:
        s = np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2.0
        q0 = (C[2, 1] - C[1, 2]) / s
        q1 = 0.25 * s
        q2 = (C[0, 1] + C[1, 0]) / s
        q3 = (C[0, 2] + C[2, 0]) / s
    elif C[1, 1] > C[2, 2]:
        s = np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2]) * 2.0
        q0 = (C[0, 2] - C[2, 0]) / s
        q1 = (C[0, 1] + C[1, 0]) / s
        q2 = 0.25 * s
        q3 = (C[1, 2] + C[2, 1]) / s
    else:
        s = np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1]) * 2.0
        q0 = (C[1, 0] - C[0, 1]) / s
        q1 = (C[0, 2] + C[2, 0]) / s
        q2 = (C[1, 2] + C[2, 1]) / s
        q3 = 0.25 * s
    return normalize(np.array([q0, q1, q2, q3]))


def error(a, d):
    """Attitude error quaternion ``dQ = d (x) a*`` rotating from frame ``a`` to frame ``d``.

    The vector part of ``dQ`` is the rotation axis expressed in the current (body) frame.
    """
    return multiply(np.asarray(d, dtype=float), conjugate(np.asarray(a, dtype=float)))


def chordal_distance(a, d):
    """Shortest-path angular distance [rad] between two attitudes."""
    a = normalize(a)
    d = normalize(d)
    dq = error(a, d)
    dq = dq if dq[0] >= 0.0 else -dq
    angle = 2.0 * np.arccos(np.clip(dq[0], -1.0, 1.0))
    return angle