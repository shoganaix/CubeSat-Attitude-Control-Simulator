"""Physical constants and mission parameters (SI units)."""

import numpy as np

MU_EARTH = 3.986004418e14  # Earth gravitational parameter [m^3/s^2]
R_EARTH = 6378.137e3       # Earth equatorial radius [m]
J2 = 1.08262668e-3         # Earth oblateness (zonal J2)

# Reference unit vector from Sun to Earth/spacecraft in ECI (fixed, configurable).
# Default points along +X; replace with a proper solar ephemeris for real missions.
SUN_VECTOR_ECI = np.array([1.0, 0.0, 0.0])

# Example 3U CubeSat: 10x10x34 cm, ~4 kg.
MASS = 4.0  # kg
INERTIA_DIAG = np.array([0.020, 0.024, 0.008])  # kg*m^2, principal-axis inertia

# Reference low-Earth orbit: ~550 km circular, 51.6 deg inclination (ISS-like).
REFERENCE_ALTITUDE = 550.0e3  # m
REFERENCE_INCLINATION_DEG = 51.6