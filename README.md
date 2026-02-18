# Long-Range-Sim: A High-Fidelity Ballistics Engine (Game-Oriented)

A deterministic, real-time–capable external ballistics simulation designed for **high-fidelity game development**.  
The goal is to achieve **realistic long-range bullet behavior** (600–1500+ yards and beyond) using a **computationally efficient pseudo–6-DOF model**, with a clean upgrade path to full 6-DOF if ever required.

Initial development to be done using **Python 3**.

This project prioritizes:
- Physical correctness where it matters perceptually
- Determinism and performance
- Data-driven tuning and validation against real-world DOPE

## Simulator Design Philosophy

### Why Not Full 6-DOF (Initially)?
A true 6-DOF rigid-body simulation requires:
- Detailed aerodynamic moment coefficients
- Very small timesteps
- High CPU cost
- Complex debugging

For this use-case, **90–95% of perceived realism** comes from:
- Correct drag modeling (Mach-dependent)
- Wind (layered, 3D)
- Atmosphere
- Spin drift and Magnus effects
- Coriolis (long range)

This engine models **translation only (3-DOF)** while injecting **spin- and wind-related forces** that approximate angular effects. This approach is widely used in professional solvers and simulators.

---

## Scope

### Included
- External ballistics only (after the bullet has exited the barrel)
- Supersonic → transonic → subsonic flight
- Wind drift (including layered winds)
- Spin drift (RH/LH twist)
- Magnus lift
- Atmospheric density & Mach effects
- Coriolis effect
- RK4 integration

### Excluded
- Bullet tumbling / instability
- Full angular state (quaternions, inertia tensors)
- Interior ballistics (chamber pressure, burn rate)

---

## Coordinate System

- Right-handed Cartesian coordinates
- `+X`: downrange
- `+Y`: right
- `+Z`: down
- Gravity acts in `+Z`

---

## Data-Driven Models

### Simulation Object (bullet) Definition / Inputs
- Mass
- Diameter / cross-sectional area
- G7 ballistic coefficient
- Rifling twist rate
- Muzzle velocity


---

### Drag Tables
- Mach → Cd
- Linear interpolation
- Transonic scaling

---

### Atmosphere
- ICAO standard atmosphere
- Density as function of altitude
- Speed of sound from temperature

---

### Wind
- Arbitrary number of altitude layers
- 3D vectors
- Smooth interpolation

---

## Validation

The solver will be validated against **DOPE**  
(**Data On Previous Engagements**):

- Drop vs range
- Wind drift vs range
- Time of flight

If simulated DOPE matches known tables within a few percent, the model is considered correct for gameplay.

---

## State Estimation

A critical function of any good ballistics simulator is to maintain an accurate representation of the gun's position and orientation in space over time.

### High-Level Idea
A high-precision 6-DOF tracking system for a standalone VR rifle scope attachment capable of resolving micro-adjustments (~0.1°) in orientation and producing reliable position and orientation estimates for long-range shooting simulation.

### System Overview: Visual-Inertial State Estimator
- IMUs provide high-rate inertial sensing
	- Optional IMU array for noise/vibration suppression
- Inclinometers provide tilt/roll absolute orientation
- 1-2 cameras provide absolute drift correction
- Error-State Extended Kalman Filter (ES-EKF)
- Balance latency, accuracy, and compute cost by running each level at an appropriate rate

## Sensors

### IMUs
- BNO085 - Current candidate IMU
- Gyroscope -> $\omega$ (rad/s) rotation rate in x, y, and z (body axes)
	- Drives orientation propagation
- Accelerometers -> a (m/s<sup>2</sup>) specific force (acceleration - gravity) in x, y, and z (body axes)
	- Drives velocity and gravity alignment
- Rate: 500-2000 Hz
- Optional 4x4 array -> Reduces noise via averaging, rejects vibrations

### Inclinometers
- SCL3300-D01-PCB - Current candidate inclinometer ($0.0055^\circ$ resolution)
- Provides absolute orientation about an axis normal to gravity, like a level
- Roll and Pitch can be measured directly, caging the gyroscope orientation integration
- Represents a possible candidate upgrade for a "tactical" grade system
- Rate: 10-70 Hz

### Cameras
- Absolute pose constraints via feature tracking / optical flow
- Relative pose between frames
- Rate: 30-60 Hz
- Do not propagate state, they only correct drift

## Kalman Filter Schema

The **Extended Kalman Filter (EKF)** is a recursive Bayesian estimator that fuses noisy sensor measurements to produce optimal state estimates. It operates in two phases:

1. **Predict (Time Update)**: Propagate the state forward using a nonlinear motion model. The covariance grows due to process uncertainty.
2. **Update (Measurement Update)**: Ingest a sensor measurement, compute the innovation (residual), and blend it with the prediction using the optimal Kalman gain. State and covariance are corrected.

The **16-DOF quaternion-based state** tracks:
- **Position** (3D): rifle pose in NED coordinates
- **Velocity** (3D): linear rates
- **Orientation** (4D): unit quaternion for robust 3D rotations
- **Angular Rate** (3D): quaternion derivatives (rad/s)
- **Accel Bias** (3D): IMU accelerometer bias, estimated and corrected

Sensor measurements (GNSS, IMU, inclinometer) are nonlinearly combined via their measurement models and Jacobians. The EKF recursively minimizes estimation error subject to noise covariances, producing smooth, drift-free pose estimates suitable for ballistic computation.


---

## References
- Modern Exterior Ballistics (R. L. McCoy)
- Ingalls / US Army Ballistic Tables
- Applied Ballistics For Long Range Shooting (Bryan Litz)
- US Army FM 3-22.9
