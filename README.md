# CubeSat Attitude Control Simulator

A simulation project focused on the **attitude dynamics and control of a CubeSat**, developed to explore spacecraft orientation, closed-loop control systems, and sensor-based state estimation.

The project aims to model how a small satellite behaves in orbit and how its orientation can be stabilized and controlled using feedback control techniques.

---

## Project Goals

- Model the rotational dynamics of a CubeSat
- Simulate spacecraft attitude and orientation
- Implement a closed-loop attitude control system
- Process simulated sensor data
- Analyze system response and stability
- Explore control strategies applicable to small satellites

---

## Main Concepts

The project combines concepts from:

- Spacecraft attitude dynamics
- Control systems
- Feedback control
- Sensor processing
- Dynamic system simulation
- State estimation
- Satellite orientation and stabilization

---

## System Overview

The simulation is structured around three main components:

### 1. CubeSat Dynamics

The spacecraft rotational behavior is modeled using its physical properties and rotational equations of motion.

The simulation considers the evolution of the CubeSat orientation over time and its response to control inputs.

### 2. Sensor Processing

Simulated sensor measurements are used to estimate the CubeSat state and provide feedback to the controller.

Possible sensor inputs include:

- Gyroscope measurements
- Accelerometer data
- Magnetometer measurements
- Orientation estimates

### 3. Attitude Control

A closed-loop controller uses the estimated spacecraft state to reduce the error between the current and desired attitude.

The objective is to stabilize the CubeSat and drive its orientation toward a target reference.

---

## Control Architecture

```text
Desired Attitude
       |
       v
+----------------+
|   Controller   |
+----------------+
       |
       v
+----------------+
| CubeSat Model  |
|   Dynamics     |
+----------------+
       |
       v
+----------------+
|    Sensors     |
+----------------+
       |
       +--------------------+
                            |
                            v
                     State Feedback
