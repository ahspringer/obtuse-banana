# Long-Range-Sim: A High-Fidelity Ballistics Engine (Game-Oriented)

A deterministic, real-time–capable external ballistics simulation designed for **high-fidelity game development**.  
The goal is to achieve **realistic long-range bullet behavior** (600–1500+ yards and beyond) using a **computationally efficient pseudo–6-DOF model**, with a clean upgrade path to full 6-DOF if ever required.

This project prioritizes:
- Physical correctness where it matters perceptually
- Determinism and performance
- Data-driven tuning and validation against real-world DOPE

---

## Design Philosophy

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
- Coriolis (optional)
- Deterministic RK4 integration

### Possible Expansions (in near-term)
- Pseudo-random perturbations/scattering

### Explicitly Excluded (for now)
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

## Mathematical Model

The following notation is used:

| Notation           | Meaning               |
| ------------------ | --------------------- |
| $\mathbf{v}$       | Vector quantity       |
| $\{\|\mathbf{v}\|}$     | Vector magnitude     |
| $\hat{\mathbf{v}}$ | Unit vector |


### State Vector (3-DOF)

At time `t`:

```math
\mathbf{x}(t) =
\begin{bmatrix}
\mathbf{p}(t) \\
\mathbf{v}(t)
\end{bmatrix}
```

Where:
- **p = (x, y, z)** — position (meters)
- **v = (vx, vy, vz)** — velocity (m/s)

---

### Relative Airflow

The bullet only interacts with **airflow relative to its motion**:

```math
\mathbf{v}_{rel} = \mathbf{v}_{bullet} - \mathbf{v}_{wind}
```

---

### Airflow Decomposition

Let velocity magnitude $\hat{\mathbf{v}}$:

```math
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}
```

**Axial (head/tail wind):**

```math
\mathbf{v}_{ax} = (\mathbf{v}_{rel} \cdot \hat{\mathbf{v}})\,\hat{\mathbf{v}}
```

**Normal (cross-flow):**

```math
\mathbf{v}_{n} = \mathbf{v}_{rel} - \mathbf{v}_{ax}
```

---

## Forces

### Gravity

```math
\mathbf{F}_g = m\,\mathbf{g}
```

---

### Drag (Axial Only)

```math
\mathbf{F}_d =
-\frac{1}{2}
\rho
C_d(M)
A
\|\mathbf{v}_{ax}\|^2
\hat{\mathbf{v}}
```

Where:
- `ρ` — air density
- `A` — bullet cross-sectional area
- `Cd(M)` — Mach-dependent drag coefficient (using G7 tables*)

*Note: G7 tables provide more accurate long-range trajectory predictions for modern boat-tail bullets because their reference model more closely mimics the aerodynamic drag characteristics of sleek, high-velocity projectiles.*

---

### Cross-Flow Side Force (Yaw-of-Repose Proxy)

Approximates aerodynamic side force due to crosswind:

```math
\mathbf{F}_{side} =
\frac{1}{2}
\rho
C_L
A
\|\mathbf{v}_n\|^2
\hat{\mathbf{v}}_n
```

Typical values:
- `C_L ≈ 0.01–0.03`

---

### Spin-Induced Magnus Lift

Spin axis is approximated as the velocity direction:

```math
\hat{\mathbf{s}} \approx \hat{\mathbf{v}}
```

Magnus direction:

```math
\hat{\mathbf{m}} =
\frac{\hat{\mathbf{s}} \times \mathbf{v}_{rel}}
{\|\hat{\mathbf{s}} \times \mathbf{v}_{rel}\|}
```

Force:

```math
\mathbf{F}_m =
C_m
\rho
A
\|\mathbf{v}_{rel}\|
\omega
\hat{\mathbf{m}}
```

Where:
- `ω` — spin rate (rad/s)
- `C_m ≈ 1e-4 – 5e-4`

This produces realistic **spin drift** without modeling full orientation dynamics.

*Note: Magnus force arises from circulation of airflow around a spinning body. It is caused by surface tangential velocity from spin interacting with bulk flow velocity. The Magnus effect is the same physics that makes a baseball curve, a soccer ball “bend,” or a tennis ball spin off course.*

---

### Coriolis Effect

```math
\mathbf{a}_c = 2(\mathbf{v} \times \boldsymbol{\Omega}_{earth})
```

*Note: The Coriolis effect is an apparent sideways deflection of a moving object caused by the Earth’s rotation, included in long-range ballistics because it slightly shifts a bullet’s impact point over hundreds or thousands of yards.*

---

### Total Acceleration

```math
\mathbf{a} =
\frac{
\mathbf{F}_g +
\mathbf{F}_d +
\mathbf{F}_{side} +
\mathbf{F}_m
}{m}
+ \mathbf{a}_c
```

---

## Numerical Integration

- **Integrator:** 4th-order Runge–Kutta (RK4)
- **Timestep:** 0.5–2.0 ms (configurable)

---

## Data-Driven Models

### Simulation Object (bullet) Definition / Inputs
- Mass
- Diameter / cross-sectional area
- G7 ballistic coefficient
- Rifling twist rate
- Muzzle velocity

Spin rate:

```math
\omega = \frac{2\pi\,v_{muzzle}}{\text{twist}}
```

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

## Project Structure (Proposed)

*TBD*

---

## References
- Modern Exterior Ballistics (R. L. McCoy)
- Ingalls / US Army Ballistic Tables
- Applied Ballistics For Long Range Shooting (Bryan Litz)
- US Army FM 3-22.9
