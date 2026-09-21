"""Rate gyroscope model: bias + white noise, per axis, in the body frame."""

import numpy as np

from cubesat.sensors.noise import GaussianNoise


class Gyro:
    """Models a MEMS rate gyro (e.g. MPU-6050 class sensor).

    ``measure(omega_true) -> omega_meas = omega_true + bias + noise``
    """

    def __init__(self, std=2e-3, bias=None, seed=None):
        # Defaults: 2e-3 rad/s (0.11 deg/s) 1-sigma noise, no bias.
        self.noise = GaussianNoise(std=np.full(3, float(std)), bias=bias, seed=seed)

    def measure(self, omega_true):
        omega_true = np.asarray(omega_true, dtype=float)
        return omega_true + self.noise.sample(omega_true.shape)