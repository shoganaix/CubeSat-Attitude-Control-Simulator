"""Noise and bias models for simulated sensors."""

import numpy as np


class GaussianNoise:
    """Additive white Gaussian noise with optional per-axis bias."""

    def __init__(self, std, bias=None, seed=None):
        self.std = np.asarray(std, dtype=float)
        self.bias = np.zeros(self.std.shape) if bias is None else np.asarray(bias, dtype=float)
        self.rng = np.random.default_rng(seed)

    def sample(self, shape):
        return self.rng.normal(loc=self.bias, scale=self.std, size=shape)


def unit_noise(unit_vector, std_rad, seed=None):
    """Perturb a unit vector by small angular noise and re-normalize.

    ``std_rad`` is the 1-sigma cone error of the measurement.
    """
    u = np.asarray(unit_vector, dtype=float)
    rng = np.random.default_rng(seed)
    v = u + rng.normal(scale=std_rad, size=3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return u
    return v / n