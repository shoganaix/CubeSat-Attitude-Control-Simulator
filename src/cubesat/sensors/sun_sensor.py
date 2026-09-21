"""Sun sensor model: noisy unit vector to the Sun in the body frame."""

import numpy as np

from cubesat.sensors.noise import unit_noise


class SunSensor:
    """Two-axis coarse sun sensor (fine sun sensor class).

    ``measure(sun_body_true)`` returns a noisy unit vector with a 1-sigma
    cone error of ``std_rad`` radians.
    """

    def __init__(self, std_rad=0.02, seed=None):
        self.std_rad = float(std_rad)
        self.seed = seed

    def measure(self, sun_body_true):
        return unit_noise(sun_body_true, self.std_rad, seed=self.seed)