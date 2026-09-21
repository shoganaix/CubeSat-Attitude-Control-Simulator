"""High-level simulation orchestrator: orbit + dynamics + control + sensors.

Closed loop per step (t -> t+dt):
    r(t)          = orbit.state_at(t)
    q_ref(t)      = reference attitude (nadir/LVLH or fixed inertial)
    tau_control   = QuaternionPD(q, omega, q_ref)
    tau_actuated  = clamp(tau_control)
    tau_env       = gravity-gradient torque
    integrate      rigid-body state (RK4) with total torque = tau_actuated + tau_env
    sensors        sampled for telemetry (estimator comes in a later milestone)
"""

import math
from dataclasses import dataclass, field

import numpy as np

from cubesat.attitude.quaternion import (
    from_axis_angle,
    multiply,
    conjugate,
    to_rotation_vector,
    chordal_distance,
)
from cubesat.attitude.rotations import eci_to_body, lvlh_quaternion
from cubesat.attitude.rotations import sun_eci as sun_eci_vector
from cubesat.control.actuators import IdealActuator
from cubesat.control.disturbances import gravity_gradient_torque
from cubesat.control.quaternion_pd import QuaternionPD, tune_gains
from cubesat.dynamics.rigid_body import rk4_step, state_derivative
from cubesat.orbit.orbital_elements import ClassicalElements
from cubesat.orbit.propagation import Orbit
from cubesat.sensors.gyro import Gyro
from cubesat.sensors.sun_sensor import SunSensor
from cubesat.sim.telemetry import write_csv
from cubesat.utils.constants import (
    INERTIA_DIAG,
    MU_EARTH,
    R_EARTH,
    REFERENCE_ALTITUDE,
    REFERENCE_INCLINATION_DEG,
    SUN_VECTOR_ECI,
)


@dataclass
class SimulationConfig:
    dt: float = 1.0
    duration: float = 500.0
    seed: int = 42

    # Orbit (two-body + optional J2 secular).
    mu: float = MU_EARTH
    j2_on: bool = True
    altitude: float = REFERENCE_ALTITUDE
    eccentricity: float = 0.0
    inclination_deg: float = REFERENCE_INCLINATION_DEG
    raan_deg: float = 0.0
    argp_deg: float = 0.0
    true_anomaly_deg: float = 0.0
    sun_vector_eci: np.ndarray = field(default_factory=lambda: SUN_VECTOR_ECI.copy())

    # Initial attitude: reference frame at t=0, then an extra body-fixed offset.
    initial_offset_deg: float = 150.0
    initial_offset_axis: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.8, 0.1]))
    initial_omega_body: np.ndarray = field(
        default_factory=lambda: np.array([2.0e-3, -1.0e-3, 3.0e-3])
    )

    # Rigid body.
    inertia: np.ndarray = field(default_factory=lambda: np.diag(INERTIA_DIAG))

    # Control: "nadir" tracks the LVLH frame, "inertial" holds a fixed orientation.
    reference_mode: str = "nadir"
    omega_n: float = 0.05
    zeta: float = 1.0
    max_torque: float = 1.0e-3
    enable_gravity_gradient: bool = True

    # Sensors (logged for now; used by the estimator in the next milestone).
    gyro_noise_std: float = 2.0e-3
    gyro_bias: np.ndarray = field(default_factory=lambda: np.array([1e-4, -5e-5, 2e-4]))
    sun_noise_std_rad: float = 0.02

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(self, k, np.asarray(v, dtype=float).copy())


@dataclass
class SimulationResult:
    t: np.ndarray
    quaternions: np.ndarray      # (N, 4) true attitude
    omegas: np.ndarray           # (N, 3) body rate
    q_refs: np.ndarray           # (N, 4) reference attitude
    errors_deg: np.ndarray       # (N,)  chordal error to reference [deg]
    tau_commanded: np.ndarray    # (N, 3) desired torque
    tau_actuated: np.ndarray     # (N, 3) applied actuator torque
    tau_disturbance: np.ndarray  # (N, 3) environmental torque
    r_ecis: np.ndarray           # (N, 3) inertial position
    gyro_meas: np.ndarray        # (N, 3) gyro samples
    sun_body: np.ndarray         # (N, 3) true Sun vector in body frame
    sun_meas: np.ndarray         # (N, 3) noisy Sun sensor samples
    config: SimulationConfig

    @property
    def n_steps(self):
        return len(self.t)

    def settling_time(self, threshold_deg=1.0):
        below = np.where(self.errors_deg <= threshold_deg)[0]
        if below.size == 0:
            return None
        first = int(below[0])
        if np.all(self.errors_deg[first:] <= threshold_deg):
            return float(self.t[first])
        return None

    def to_csv(self, path):
        write_csv(
            path,
            {
                "t_s": self.t,
                "q0": self.quaternions[:, 0],
                "q1": self.quaternions[:, 1],
                "q2": self.quaternions[:, 2],
                "q3": self.quaternions[:, 3],
                "wx_rad_s": self.omegas[:, 0],
                "wy_rad_s": self.omegas[:, 1],
                "wz_rad_s": self.omegas[:, 2],
                "error_deg": self.errors_deg,
                "tau_cmd_x": self.tau_commanded[:, 0],
                "tau_cmd_y": self.tau_commanded[:, 1],
                "tau_cmd_z": self.tau_commanded[:, 2],
                "tau_dist_x": self.tau_disturbance[:, 0],
                "tau_dist_y": self.tau_disturbance[:, 1],
                "tau_dist_z": self.tau_disturbance[:, 2],
            },
        )


class Simulation:
    def __init__(self, config=None):
        self.config = config or SimulationConfig()
        cfg = self.config
        self.inertia = np.asarray(cfg.inertia, dtype=float)

        elements = ClassicalElements(
            a=R_EARTH + cfg.altitude,
            e=cfg.eccentricity,
            i=np.deg2rad(cfg.inclination_deg),
            raan=np.deg2rad(cfg.raan_deg),
            argp=np.deg2rad(cfg.argp_deg),
            nu=np.deg2rad(cfg.true_anomaly_deg),
            mu=cfg.mu,
        )
        self.orbit = Orbit(elements, mu=cfg.mu, j2_on=cfg.j2_on)

        kp, kd = tune_gains(np.diag(cfg.inertia), cfg.omega_n, cfg.zeta)
        self.controller = QuaternionPD(kp, kd)
        self.actuator = IdealActuator(cfg.max_torque)

        self.gyro = Gyro(std=cfg.gyro_noise_std, bias=cfg.gyro_bias, seed=cfg.seed)
        self.sun_sensor = SunSensor(std_rad=cfg.sun_noise_std_rad, seed=cfg.seed)

    def reference_attitude(self, r_eci, v_eci, t):
        if self.config.reference_mode == "nadir":
            return lvlh_quaternion(r_eci, v_eci)
        if self.config.reference_mode == "inertial":
            return np.array([1.0, 0.0, 0.0, 0.0])
        raise ValueError(
            f"unknown reference_mode '{self.config.reference_mode}'; use 'nadir' or 'inertial'"
        )

    def initial_state(self, q_ref0):
        offset_deg = self.config.initial_offset_deg
        if offset_deg == 0.0:
            offset_q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            offset_q = from_axis_angle(
                self.config.initial_offset_axis, math.radians(offset_deg)
            )
        q0 = multiply(offset_q, q_ref0)
        return np.concatenate((q0, np.asarray(self.config.initial_omega_body, dtype=float)))

    def run(self):
        cfg = self.config
        n_steps = int(round(cfg.duration / cfg.dt))
        t = cfg.dt * np.arange(n_steps + 1)

        # Precompute the orbit trajectory and reference attitude at every time step.
        rv_all = np.array([self.orbit.state_at(ti) for ti in t])
        r_ecis = rv_all[:, 0, :]
        q_ref_all = np.array(
            [self.reference_attitude(rvec, vvec, ti) for (rvec, vvec), ti in zip(rv_all, t)]
        )

        N = n_steps + 1
        quats = np.empty((N, 4))
        omegas = np.empty((N, 3))
        errors = np.empty(N)
        tau_cmd = np.zeros((N, 3))
        tau_act = np.zeros((N, 3))
        tau_dist = np.zeros((N, 3))
        gyro_meas = np.zeros((N, 3))
        sun_body = np.zeros((N, 3))
        sun_meas = np.zeros((N, 3))

        x = self.initial_state(q_ref_all[0])
        quats[0] = x[:4]
        omegas[0] = x[4:]

        for i in range(n_steps):
            r = r_ecis[i]
            q_ref = q_ref_all[i]
            # Angular velocity of the reference frame, expressed in the (nearly
            # aligned) body frame; lets the derivative term track nadir motion.
            # dq_ref = conj(q_ref_i) (x) q_ref_{i+1}; its DCM is C(q_ref_i)^T C(q_ref_{i+1}).
            dq_ref = multiply(conjugate(q_ref), q_ref_all[i + 1])
            omega_ref = to_rotation_vector(dq_ref) / cfg.dt

            q = quats[i]
            omega = omegas[i]
            errors[i] = math.degrees(chordal_distance(q, q_ref))

            cmd = self.controller.compute_torque(q, omega, q_ref, omega_ref)
            tau_cmd[i] = cmd
            tau_act[i] = self.actuator.apply(cmd)

            def total_torque(q_stage, omega_stage):
                ctl = self.actuator.apply(
                    self.controller.compute_torque(q_stage, omega_stage, q_ref, omega_ref)
                )
                env = (
                    gravity_gradient_torque(q_stage, r, self.inertia, cfg.mu)
                    if cfg.enable_gravity_gradient
                    else np.zeros(3)
                )
                return ctl + env

            x = rk4_step(state_derivative, x, cfg.dt, self.inertia, total_torque)
            quats[i + 1] = x[:4]
            omegas[i + 1] = x[4:]

            tau_dist[i] = (
                gravity_gradient_torque(x[:4], r, self.inertia, cfg.mu)
                if cfg.enable_gravity_gradient
                else np.zeros(3)
            )

            gyro_meas[i] = self.gyro.measure(x[4:])
            sun_eci = sun_eci_vector(r, cfg.sun_vector_eci)
            sb = eci_to_body(x[:4], sun_eci)
            sun_body[i] = sb
            sun_meas[i] = self.sun_sensor.measure(sb)

        errors[n_steps] = math.degrees(chordal_distance(quats[n_steps], q_ref_all[n_steps]))

        return SimulationResult(
            t=t,
            quaternions=quats,
            omegas=omegas,
            q_refs=q_ref_all,
            errors_deg=errors,
            tau_commanded=tau_cmd,
            tau_actuated=tau_act,
            tau_disturbance=tau_dist,
            r_ecis=r_ecis,
            gyro_meas=gyro_meas,
            sun_body=sun_body,
            sun_meas=sun_meas,
            config=cfg,
        )