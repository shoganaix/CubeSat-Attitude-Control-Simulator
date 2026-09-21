# CubeSat Attitude Control Simulator

An **embedded systems** project aimed at the space sector: simulation and attitude determination and control (ADCS) of a reference 3U CubeSat in low Earth orbit. Designed to iterate fast in Python and then be ported to C/C++ firmware on an ESP32 and mirrored in MATLAB/Simulink.

The goal is to model the rotational behaviour of a small satellite in orbit and stabilize it with space-grade feedback control (quaternions, attitude-error PD), using simulated sensors and a framework ready for state estimation (M1).

> Versión en español: [README.es.md](README.es.md)

---

## Project status (Milestone M0)

- [x] Quaternion algebra (Hamilton convention, scalar-first) + DCM.
- [x] 3-DOF rigid-body dynamics (Euler + quaternion kinematics) with RK4.
- [x] Orbit: Keplerian elements, two-body propagation + J2 secular rate, LVLH/nadir frames.
- [x] Quaternion-error PD controller (shortest-path, tuned gains).
- [x] Gravity-gradient disturbance + saturated ideal actuator.
- [x] Simulated sensors: gyroscope (bias + noise) and sun sensor (cone noise).
- [x] Closed-loop simulation orchestrator + CSV telemetry + plots.
- [x] Test suite (`pytest`).

---

## Architecture

```text
Desired Attitude (nadir / LVLH or fixed inertial)
        |
        v
+----------------+    torque cmd    +----------------+
|    Controller   | --------------> |   Actuator     |   (saturation)
| Quaternion PD   |                 +----------------+
+----------------+                        |  applied torque (cmd + disturbances)
        ^                                v
        |                    +----------------+
        | estimated state     |   Dynamics     |   Euler + quat kinematics (RK4)
        |    (M1: TRIAD/      |   3-DOF         |   + gravity gradient
        |     MEKF; today:    +----------------+
        |     true attitude)         |
        |                             v
        +------------------+  +----------------+
                           |  |    Sensors     |  gyro (bias+noise), sun sensor
                           |  +----------------+
                           +-------------------+
                        Orbit: Keplerian elements -> r(t), v(t), LVLH reference
```

Attitude is represented by a quaternion `q` that rotates vectors from *body* to *inertial*: `v_inertial = R(q) @ v_body`. The integrated state is `x = [q, ω_body]`.

---

## Getting started

```bash
python -m venv .venv && .\.venv\Scripts\activate
pip install -e .          # or: pip install -r requirements.txt
python examples\orbit_sim.py               # orbit propagation + 3D plot
python examples\attitude_control_demo.py   # nadir-pointing maneuver (500 s)
pytest -q                                  # run the test suite
```

Outputs land in `results/`: `telemetry.csv`, `attitude.png`, `orbit_attitude.png`, `orbit.png`.

---

## Repository layout

```
src/cubesat/
  attitude/      quaternion.py, rotations.py        (algebra, DCM, LVLH, euler321)
  dynamics/      rigid_body.py                      (Euler, kinematics, RK4)
  orbit/         orbital_elements.py, propagation.py (+ J2 secular)
  sensors/       gyro.py, sun_sensor.py, noise.py
  control/       quaternion_pd.py, actuators.py, disturbances.py
  sim/           simulation.py, telemetry.py, visualization.py
  utils/         constants.py
tests/           test_quaternion.py test_dynamics.py test_orbit.py test_controller.py
examples/        orbit_sim.py, attitude_control_demo.py
```

---

## Conventions (important)

- **Quaternions**: Hamilton, scalar-first `[q0, q1, q2, q3]`, conjugate = inverse.
- **Vector rotation**: `v_inertial = R(q) v_body`; `multiply(p, q)` composes by applying `q` first.
- **Control attitude error**: `dq = q_des^{-1} ⊗ q_cur`; the error axis is expressed in the current body frame and the proportional term is `-kp·sign(dq0)·dq_vec`. `sign(dq0)` selects the shortest path (double cover).
- **Frames**: ECI (inertial, J2000-like), BODY, LVLH (nadir). Sun vector defaults to a fixed ECI direction (a real ephemeris is in the roadmap).
- **Control law**: `τ = -kp·sign(dq0)·dq_vec − kd·(ω − ω_ref)`, with `ω_ref` the reference-frame angular velocity (follows the LVLH orbital rotation).

The closed loop uses *true* attitude in M0; sensor-based estimation (TRIAD + MEKF) arrives in M1.

---

## Roadmap

- **M0 (current)**: working dynamics + orbit + PD control simulation.
- **M1**: detumbling (B-dot) and a TRIAD + MEKF estimator using the sensors already modeled.
- **M2**: MATLAB/Simulink twin and cross-verification against Python.
- **M3 (embedded)**: C firmware on ESP32 — port the quaternion-PD controller, IMU/sensor drivers, UART telemetry and *hardware-in-the-loop* (the PC feeds the attitude state over serial).
- **M4**: realistic actuators (reaction wheels with saturation/friction, magnetorquers), full disturbance set (solar, aerodynamic, magnetic).
- **M5**: mission scenario (sun-pointing + nadir-pointing) and a final portfolio with campaigns documented in the git history.

---

## M0 results (nadir-pointing, 150° initial error)

| Metric | Value |
| --- | --- |
| Final error | 0.24 ° |
| Settling (<1°) | ~390 s |
| Peak torque | 5.4e-5 N·m |
| Gravity-gradient disturbance | ~1e-8 N·m |