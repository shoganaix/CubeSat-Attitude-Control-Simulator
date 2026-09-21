# CubeSat Attitude Control Simulator

Proyecto de **sistemas embebidos** aplicado al sector espacial: simulación y control de actitud (ADCS) de un CubeSat de referencia de 3U en órbita baja. Planteado para iterar rápido en Python y traducirse luego a firmware C/C++ sobre ESP32 y a un gemelo MATLAB/Simulink.

El objetivo es modelar el comportamiento rotacional de un pequeño satélite en órbita y estabilizarlo con control por realimentación *space-grade* (quaterniones, PD con error de actitud), con sensores simulados y un entramado preparado para estimación de estado (M1).

---

## Estado del proyecto (Hito M0)

- [x] Álgebra de quaterniones (convención Hamilton, scalar-first) + DCM.
- [x] Dinámica de cuerpo rígido 3-DOF (Euler + cinemática de quaternion) con RK4.
- [x] Órbita: elementos keplerianos, propagación 2-cuerpos + tasa secular J2, marcos LVLH/nadir.
- [x] Controlador PD con error de quaternion (giro estereotipado, camino corto, ganancias sintonizadas).
- [x] Perturbación por gradiente de gravedad + actuador ideal saturado.
- [x] Sensores simulados: giroscopio (bias+ruido) y sensor solar (ruido cónico).
- [x] Orquestador de simulación en lazo cerrado + telemetría CSV + plots.
- [x] Suite de tests (`pytest`).

---

## Arquitectura

```text
Desired Attitude (nadir / LVLH o inercial fija)
        |
        v
+----------------+    torque cmd    +----------------+
|    Controlador  | --------------> |   Actuador     |   (saturación)
|  Quaternion PD  |                 +----------------+
+----------------+                        |  torque real (cmd + perturbaciones)
        ^                                v
        |                    +----------------+
        | estado estimado     |  Dinámica      |   Euler + cinemática quat (RK4)
        |      (M1: TRIAD/    |  3-DOF         |   + gradiente de gravedad
        |       MEKF, hoy:    +----------------+
        |       actitud real)         |
        |                             v
        +------------------+  +----------------+
                           |  |   Sensores     |  giro (bias+ruido), sensor solar
                           |  +----------------+
                           +-------------------+
                        Órbita: elementos keplerianos -> r(t), v(t), referencia LVLH
```

La actitud se representa con un quaternion `q` que rota vectores de *cuerpo* a *inercial*: `v_inertial = R(q) @ v_body`. El estado integrado es `x = [q, ω_body]`.

---

## Cómo usar

```bash
python -m venv .venv && .\.venv\Scripts\activate
pip install -e .          # o: pip install -r requirements.txt
python examples\orbit_sim.py               # propagación orbital + plot 3D
python examples\attitude_control_demo.py   # maniobra de apuntado nadir (500 s)
pytest -q                                  # suite de tests
```

Salidas en `results/`: `telemetry.csv`, `attitude.png`, `orbit_attitude.png`, `orbit.png`.

---

## Estructura

```
src/cubesat/
  attitude/      quaternion.py, rotations.py        (algebra, DCM, LVLH, euler321)
  dynamics/      rigid_body.py                      (Euler, cinemática, RK4)
  orbit/         orbital_elements.py, propagation.py (+ J2 secular)
  sensors/       gyro.py, sun_sensor.py, noise.py
  control/       quaternion_pd.py, actuators.py, disturbances.py
  sim/           simulation.py, telemetry.py, visualization.py
  utils/         constants.py
tests/           test_quaternion.py test_dynamics.py test_orbit.py test_controller.py
examples/        orbit_sim.py, attitude_control_demo.py
```

---

## Convenciones (importante)

- **Quaterniones**: Hamilton, scalar-first `[q0, q1, q2, q3]`, conjugado = inverso.
- **Rotación de vectores**: `v_inertial = R(q) v_body`; `multiply(p, q)` compone aplicando `q` primero.
- **Error de actitud de control**: `dq = q_des^{-1} ⊗ q_cur`; el eje del error es el del cuerpo actual y el término proporcional es `-kp·sign(dq0)·dq_vec`. El `sign(dq0)` elige el camino corto (doble cubierta).
- **Frames**: ECI (inercial, J2000-like), BODY, LVLH (nadir). Vector solar solar por defecto fijo en ECI (ephemeris real en roadmap).
- **Ley de control**: `τ = -kp·sign(dq0)·dq_vec − kd·(ω − ω_ref)`, con `ω_ref` la velocidad angular del marco de referencia (sigue el giro orbital del LVLH).

El control de lazo cerrado usa actitud "verdadera" en M0; la estimación por sensores (TRIAD + MEKF) llega en M1.

---

## Roadmap

- **M0 (ahora)**: simulación funcional de dinámica + órbita + control PD.
- **M1**: detumbling (B-dot) y estimador TRIAD + mehk (MEKF/EKF) con los sensores ya modelados.
- **M2**: gemelo MATLAB/Simulink y co-verificación contra Python.
- **M3 (embebido)**: firmware C sobre ESP32 — port del controlador quaternion-PD, drivers IMU/sensor, telemetría UART y *hardware-in-the-loop* (el PC alimenta el estado de actitud por serie).
- **M4**: actuadores realistas (ruedas con saturación/fricción, magnetorquers), perturbaciones completas (solar, aerodinámica, magnética).
- **M5**: escenario de misión (sun-pointing + nadir-pointing) y portafolio final con campañas documentadas en el histórico de git.

---

## Resultados M0 (nadir-pointing, 150° de error inicial)

| Métrica | Valor |
| --- | --- |
| Error final | 0.24 ° |
| Settling (<1°) | ~390 s |
| Torque pico | 5.4e-5 N·m |
| Perturbación (grad. gravedad) | ~1e-8 N·m |