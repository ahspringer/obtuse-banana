import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional, List

# allow ``python src/filters/sensor_fusion.py`` to work by placing the
# package root on sys.path; this mirrors the behaviour of ``python -m``.
if __name__ == "__main__" and __package__ is None:
    import os, sys
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    __package__ = "src.filters"

# bring in concrete sensor implementations from the canonical sensors package
# use relative imports so module can be executed as part of the `src` package
from ..sensors.IMU import Accelerometer3Axis, Gyroscope3Axis
from ..sensors.GNSS import GNSSReceiver
from ..sensors.inclinometers import Inclinometer1Axis, Inclinometer2Axis
from ..util import quaternion_multiply, quaternion_normalize, quaternion_to_rotation_matrix, euler_from_dcm

"""
-------------------------------------------------------------------------------
OBJECT ORIENTED EXTENDED KALMAN FILTER (EKF) FRAMEWORK
--- QUATERNION-BASED (16-DOF STATE)
-------------------------------------------------------------------------------
Architecture:
1. State:           Holds the 'Truth' (x) and Uncertainty (P).
2. SystemModel:     Defines how the system moves (Physics/Kinetics).
3. Sensor (Base):   Defines how a sensor sees the world (Measurement Models).
   * concrete implementations now reside in ``src/sensors``; this module only
     contains lightweight adapters that expose a uniform API to the EKF engine.
4. EKF (Engine):    Performs the recursive estimation.

Conventions:
- State Vector (x): n x 1 numpy array
- Covariance (P):   n x n numpy array
- Measurement (z):  m x 1 numpy array
-------------------------------------------------------------------------------
"""



# =============================================================================
# STATE CLASS (16-DOF Quaternion-based)
# =============================================================================

class State:
    """
    The System State (Quaternion-based, 16 DOF).
    Encapsulates the state vector 'x' and the covariance matrix 'P'.
    
    State layout (16-DOF):
    0-2:   Position (px, py, pz) [m]
    3-5:   Velocity (vx, vy, vz) [m/s]
    6-9:   Quaternion (q0, q1, q2, q3) [scalar-first convention]
    10-12: Quaternion Rate (dq0, dq1, dq2, dq3) [rad/s as quaternion]
    13-15: Accelerometer Bias (bx, by, bz) [m/s²]
    """
    def __init__(self, dim: int = 16):
        self.dim = dim
        self.x = np.zeros((dim, 1))
        # Initialize quaternion to identity [1, 0, 0, 0]
        self.x[6, 0] = 1.0
        self.P = np.eye(dim) * 1.0  # Initialize with high uncertainty
        self.timestamp = 0.0

    def get_position(self) -> np.ndarray:
        return self.x[0:3]

    def get_velocity(self) -> np.ndarray:
        return self.x[3:6]
    
    def get_quaternion(self) -> np.ndarray:
        return self.x[6:10]
    
    def get_quaternion_rate(self) -> np.ndarray:
        return self.x[10:14]
    
    def get_euler_angles(self) -> np.ndarray:
        q = self.get_quaternion().flatten()
        return quaternion_to_euler(q)
    
    def get_accel_bias(self) -> np.ndarray:
        return self.x[13:16]
    
    def set_quaternion(self, q: np.ndarray):
        self.x[6:10] = quaternion_normalize(q).reshape((4, 1))

# The sensor abstraction used by the EKF is now a thin wrapper around the
# classes defined in the `src/sensors` package.  Those classes handle all of
# the simulation/hardware error modeling, buffering, frame transforms, etc.
#
# SensorFusion remains responsible only for providing a common interface to
# the Kalman filter (measurement_model, jacobian and noise covariance).

class EKFSensorAdapter:
    """Adapter that exposes a light-weight EKF-friendly API for any
    ``src.sensors`` object.

    The adapter holds a reference to the underlying sensor instance so that
    the simulation world can still step it forward independently, but the EKF
    only sees the parts it needs:

        * ``m_dim`` – measurement dimension
        * ``R``     – noise covariance used during the update
        * ``measurement_model(state)`` – computes expected z from a State
        * ``jacobian(state)``          – linearisation of ``h`` around ``state``

    Additional helper constructors (``adapt_gnss`` / ``adapt_accel`` etc.)
    are provided below to build common mappings to the 16‑DOF quaternion state.
    """

    def __init__(
        self,
        base_sensor,
        measurement_model: Callable[["State"], np.ndarray],
        jacobian: Callable[["State"], np.ndarray],
        R: Optional[np.ndarray] = None,
    ):
        self.base_sensor = base_sensor
        self.name = getattr(base_sensor, "sensor_id", str(base_sensor))
        self.m_dim = base_sensor.spec.dimension
        self.R = R if R is not None else np.eye(self.m_dim)
        self.measurement_model = measurement_model
        self.jacobian = jacobian

    def set_noise_covariance(self, R: np.ndarray):
        """Override the measurement noise covariance used by the EKF."""
        self.R = R


# helper for numeric jacobians reused by several adapters

def _numeric_jacobian(
    h_func: Callable[[State], np.ndarray],
    state: "State",
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute a simple finite–difference Jacobian of ``h_func`` at ``state``.

    ``h_func`` is expected to return an ``(m,1)`` column vector.  The returned
    matrix has shape ``(m,n)`` where ``n`` is ``state.dim``.
    """
    m = h_func(state).shape[0]
    n = state.dim
    H = np.zeros((m, n))
    base = h_func(state)
    for j in range(n):
        dx = np.zeros((n, 1))
        dx[j, 0] = eps
        tmp = State(dim=n)
        tmp.x = state.x.copy() + dx
        tmp.P = state.P.copy()
        H[:, j] = ((h_func(tmp) - base).reshape(m) / eps)
    return H

class ExtendedKalmanFilter:
    """
    The Math Engine. 
    A pure implementation of the Discrete Extended Kalman Filter.
    It knows Math, not Physics.
    """
    def __init__(self, state: State):
        self.state = state

    def predict(self, 
                dt: float, 
                f_transition: Callable[[np.ndarray, float], np.ndarray], 
                F_jacobian: Callable[[np.ndarray, float], np.ndarray],
                Q: np.ndarray):
        """
        Time Update Step (A Priori).
        
        Args:
            dt: Time delta.
            f_transition: Function f(x, dt) returning predicted x.
            F_jacobian: Function F(x, dt) returning Jacobian matrix F.
            Q: Process Noise Covariance Matrix (System uncertainty).
        """
        # 1. Project the State ahead
        # x_k|k-1 = f(x_k-1|k-1)
        self.state.x = f_transition(self.state.x, dt)

        # 2. Project the Error Covariance ahead
        # P_k|k-1 = F_k * P_k-1|k-1 * F_k^T + Q_k
        F = F_jacobian(self.state.x, dt)
        self.state.P = F @ self.state.P @ F.T + Q
        
        self.state.timestamp += dt

    def update(self, sensor, z_measurement: np.ndarray):
        """
        Measurement Update Step (A Posteriori).
        Fuses a sensor reading into the state.
        
        Args:
            sensor: An object with ``measurement_model``, ``jacobian`` and
                    ``R`` attributes (typically an ``EKFSensorAdapter``).
            z_measurement: The raw data from hardware (m x 1).
        """
        # 1. Calculate Innovation (Residual)
        # y = z - h(x_predicted)
        z_predicted = sensor.measurement_model(self.state)
        y = z_measurement - z_predicted

        # 2. Get Measurement Jacobian and Noise
        H = sensor.jacobian(self.state)
        R = sensor.R

        # 3. Calculate Innovation Covariance
        # S = H * P * H^T + R
        S = H @ self.state.P @ H.T + R

        # 4. Calculate Optimal Kalman Gain
        # K = P * H^T * S^-1
        try:
            K = self.state.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print(f"[{sensor.name}] Singular Matrix S. Skipping update.")
            return

        # 5. Update State Estimate
        # x = x + K * y
        self.state.x = self.state.x + (K @ y)

        # 6. Update Covariance Estimate
        # P = (I - K * H) * P
        I = np.eye(self.state.dim)
        self.state.P = (I - K @ H) @ self.state.P

    def __repr__(self):
        return f"EKF(State={self.state.x.flatten()}, P={self.state.P})"

# -----------------------------------------------------------------------------
# CONCRETE IMPLEMENTATION EXAMPLES (The "Plug and Play" Part)
# -----------------------------------------------------------------------------

class NavigationPhysics:
    """
    Defines the nonlinear transition models f(x) and Jacobian F.
    Uses quaternion kinematics for robust 3D rotation representation.
    
    Standard strapdown inertial navigation equations:
    - Position: p_k+1 = p_k + v_k * dt
    - Velocity: v_k+1 = v_k + (a_measured - bias - g_rotated) * dt
    - Quaternion: q_k+1 = q_k + 0.5 * q_k * [0; omega] * dt
    - Biases: constant (random walk with small process noise)
    """
    @staticmethod
    def transition_function(x: np.ndarray, dt: float, measured_accel: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Nonlinear state transition for 16-DOF quaternion navigation.
        
        State: [pos(3), vel(3), quat(4), quat_rate(4), accel_bias(3)]
        """
        new_x = x.copy()
        
        # Position update: p = p + v*dt
        new_x[0:3] += x[3:6] * dt
        
        # Quaternion update: q = q + 0.5 * q * [0; omega] * dt
        q = x[6:10].flatten()
        omega = x[10:13].flatten()  # Angular velocity from quat rate (simplified)
        
        # Quaternion kinematics: dq = 0.5 * q * [0; omega]
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        q_dot_quat = 0.5 * quaternion_multiply(q, omega_quat)
        new_q = q + q_dot_quat * dt
        new_x[6:10] = quaternion_normalize(new_q).reshape((4, 1))
        
        # Quaternion rate (angular velocity) - assume constant
        # Could be updated with gyro measurements or process noise
        # new_x[10:14] stays the same (no change in angular velocity)
        
        # Velocity update (simplified: no acceleration control input)
        # In a real system: v = v + (a_meas - bias - g_rotated)*dt
        # For now, just gravity compensation:
        R = quaternion_to_rotation_matrix(q)
        g_ned = np.array([0.0, 0.0, 9.81])
        g_body = R.T @ g_ned
        
        if measured_accel is not None:
            accel_bias = x[13:16].flatten()
            corrected_accel = measured_accel - accel_bias - g_body
            new_x[3:6] += corrected_accel.reshape((3, 1)) * dt
        
        # Biases stay constant (random walk with low process noise)
        
        return new_x

    @staticmethod
    def transition_jacobian(x: np.ndarray, dt: float) -> np.ndarray:
        """
        Jacobian of transition function (F matrix).
        Numerical approximation for nonlinear quaternion kinematics.
        """
        eps = 1e-7
        dim = x.shape[0]
        F = np.zeros((dim, dim))
        
        f0 = NavigationPhysics.transition_function(x, dt)
        
        for j in range(dim):
            dx = np.zeros((dim, 1))
            dx[j] = eps
            f_pert = NavigationPhysics.transition_function(x + dx, dt)
            F[:, j] = ((f_pert - f0) / eps).flatten()
        
        return F

# the concrete sensor definitions have been moved into `src.sensors`.
# instead of re-defining them here we provide a few helper constructors that
# create ``EKFSensorAdapter`` instances for the most common devices.


def adapt_gnss_sensor(gnss: GNSSReceiver) -> EKFSensorAdapter:
    """Return an EKF adapter that maps a GNSSReceiver to the 16‑DOF state.

    The underlying GNSSReceiver produces a 6‑vector [x,y,z,vx,vy,vz].  The
    measurement model simply pulls the same quantities out of the Kalman state
    (
        position = state[0:3], velocity = state[3:6]
    ).
    """
    def h(state: State) -> np.ndarray:
        return np.vstack((state.get_position(), state.get_velocity()))

    def H(state: State) -> np.ndarray:
        Hmat = np.zeros((6, state.dim))
        Hmat[0:3, 0:3] = np.eye(3)
        Hmat[3:6, 3:6] = np.eye(3)
        return Hmat

    adapter = EKFSensorAdapter(gnss, h, H)
    if hasattr(gnss, "white_noise_std"):
        adapter.R = np.diag(gnss.white_noise_std ** 2)
    return adapter


def adapt_accel_sensor(
    accel: Accelerometer3Axis,
    bias_index: int = 13,
) -> EKFSensorAdapter:
    """Wrap a 3‑axis accelerometer for use with the EKF state.

    ``bias_index`` indicates where in ``State.x`` the 3‑vector of accel bias
    lives (default matches the 16‑DOF layout defined above).
    """
    mounting = (
        accel.dcm_body_to_sensor
        if hasattr(accel, "dcm_body_to_sensor")
        else np.eye(3)
    )

    def h(state: State) -> np.ndarray:
        q = state.get_quaternion().flatten()
        Rb2n = quaternion_to_rotation_matrix(q)
        g_ned = np.array([0.0, 0.0, 9.81])
        g_body = Rb2n.T @ g_ned
        meas = -g_body.reshape((3, 1))
        bias = state.x[bias_index : bias_index + 3].reshape((3, 1))
        return mounting @ meas + bias

    def H(state: State) -> np.ndarray:
        return _numeric_jacobian(h, state)

    adapter = EKFSensorAdapter(accel, h, H)
    if hasattr(accel, "white_noise_std"):
        adapter.R = np.diag(accel.white_noise_std ** 2)
    return adapter


def adapt_gyro_sensor(
    gyro: Gyroscope3Axis,
    bias_index: Optional[int] = None,
) -> EKFSensorAdapter:
    """Wrap a 3‑axis gyro for EKF.

    ``bias_index`` may be ``None`` if there is no gyro‑bias state.
    """
    mounting = (
        gyro.dcm_body_to_sensor
        if hasattr(gyro, "dcm_body_to_sensor")
        else np.eye(3)
    )

    def h(state: State) -> np.ndarray:
        omega = state.x[10:13].reshape((3, 1))
        meas = mounting @ omega
        if bias_index is not None:
            bias = state.x[bias_index : bias_index + 3].reshape((3, 1))
            meas = meas + bias
        return meas

    def H(state: State) -> np.ndarray:
        return _numeric_jacobian(h, state)

    adapter = EKFSensorAdapter(gyro, h, H)
    if hasattr(gyro, "white_noise_std"):
        adapter.R = np.diag(gyro.white_noise_std ** 2)
    return adapter


def adapt_inclinometer_sensor(
    inclinometer: Inclinometer1Axis,
) -> EKFSensorAdapter:
    """Wrap a 1-axis or 2-axis inclinometer for EKF.

    Inclinometers measure tilt angles (pitch, roll) directly from the DCM derived
    from the quaternion state. This provides absolute orientation constraints.
    """
    def h(state: State) -> np.ndarray:
        q = state.get_quaternion().flatten()
        dcm = quaternion_to_rotation_matrix(q)
        # Extract pitch and roll from DCM
        pitch = np.arctan2(-dcm[2, 0], dcm[2, 2])
        roll = np.arctan2(dcm[2, 1], dcm[2, 2])
        if inclinometer.measurement_axis == "pitch":
            return np.array([[pitch]])
        elif inclinometer.measurement_axis == "roll":
            return np.array([[roll]])
        else:
            return np.array([[pitch, roll]]).T

    def H(state: State) -> np.ndarray:
        return _numeric_jacobian(h, state)

    adapter = EKFSensorAdapter(inclinometer, h, H)
    if hasattr(inclinometer, "white_noise_std"):
        adapter.R = np.diag(inclinometer.white_noise_std ** 2)
    return adapter

# composite adapter just stitches a list of adapters together

def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Uses DCM as intermediate representation.
    Returns angles in radians.
    """
    dcm = quaternion_to_rotation_matrix(q)
    return euler_from_dcm(dcm)

def composite_adapter(adapters: List[EKFSensorAdapter]) -> EKFSensorAdapter:
    total_dim = sum(a.m_dim for a in adapters)
    name = "+".join(a.name for a in adapters)

    def h(state: State) -> np.ndarray:
        parts = [a.measurement_model(state) for a in adapters]
        return np.vstack(parts)

    def H(state: State) -> np.ndarray:
        parts = [a.jacobian(state) for a in adapters]
        return np.vstack(parts)

    comp = EKFSensorAdapter(adapters[0].base_sensor, h, H)
    # assemble block diag R
    R = np.zeros((total_dim, total_dim))
    idx = 0
    for a in adapters:
        R[idx : idx + a.m_dim, idx : idx + a.m_dim] = a.R
        idx += a.m_dim
    comp.R = R
    comp.name = name
    return comp


if __name__ == "__main__":
    # demo showing EKF fusion with accelerometer, gyroscope, and inclinometer sensors
    nav_state = State(dim=16)  # 16‑DOF quaternion state
    ekf = ExtendedKalmanFilter(nav_state)

    # IMU sensors (accelerometer + gyroscope)
    accel = Accelerometer3Axis(sensor_id="accel_1")
    gyro = Gyroscope3Axis(sensor_id="gyro_1")

    # Inclinometer sensors for absolute orientation correction
    inclin_pitch = Inclinometer1Axis(sensor_id="inclin_pitch", measurement_axis="pitch")
    inclin_roll = Inclinometer1Axis(sensor_id="inclin_roll", measurement_axis="roll")

    # Create adapters
    acc_adapter = adapt_accel_sensor(accel)
    gyro_adapter = adapt_gyro_sensor(gyro)
    pitch_adapter = adapt_inclinometer_sensor(inclin_pitch)
    roll_adapter = adapt_inclinometer_sensor(inclin_roll)

    # Set small initial accel bias
    nav_state.x[13:16] = np.array([[0.02], [-0.01], [0.005]])
    Q_matrix = np.eye(nav_state.dim) * 0.01  # Process noise covariance

    # Set up true vectors for simulation validation
    print("--- Starting EKF Fusion demo (IMU + inclinometers) ---")
    print("Rolling from 0 to 5 degrees over 1 second...")
    print("\nTime      | True Roll | True Pitch | True Yaw | Est Roll | Est Pitch | Est Yaw")
    print("-" * 80)
    for t in range(10):
        dt = 0.1
        ekf.predict(
            dt=dt,
            f_transition=NavigationPhysics.transition_function,
            F_jacobian=NavigationPhysics.transition_jacobian,
            Q=Q_matrix,
        )

        # Step the sensors (they update internally with their error models)
        z_acc = accel.step(np.array([0.0, 0.0, -9.81]), dt, t)
        z_gyro = gyro.step(np.zeros(3), dt, t)
        
        # Create a test DCM with gradually increasing roll (0 to 5 degrees)
        # Roll angle increases linearly over the 10 timesteps
        roll_angle = (t / 9.0) * np.radians(5.0)  # 0 to 5 degrees
        
        # Create rotation matrix for pure roll about X-axis
        # In coordinate frame: +X downrange, +Y right, +Z down
        # Roll is rotation about X-axis
        cr = np.cos(roll_angle)
        sr = np.sin(roll_angle)
        dcm_body_to_nav = np.array([
            [1.0,   0.0,   0.0],
            [0.0,    cr,    sr],
            [0.0,   -sr,    cr]
        ])
        
        # Compute true Euler angles from the true DCM
        true_roll, true_pitch, true_yaw = euler_from_dcm(dcm_body_to_nav)
        
        z_pitch = inclin_pitch.step_from_dcm(dcm_body_to_nav, dt, t)
        z_roll = inclin_roll.step_from_dcm(dcm_body_to_nav, dt, t)

        # Fuse into state
        if z_acc is not None:
            ekf.update(acc_adapter, np.atleast_2d(z_acc).reshape(-1, 1))
        if z_gyro is not None:
            ekf.update(gyro_adapter, np.atleast_2d(z_gyro).reshape(-1, 1))
        if z_pitch is not None:
            ekf.update(pitch_adapter, np.atleast_2d(z_pitch).reshape(-1, 1))
        if z_roll is not None:
            ekf.update(roll_adapter, np.atleast_2d(z_roll).reshape(-1, 1))

        # Extract and convert quaternion to Euler angles
        roll_est, pitch_est, yaw_est = nav_state.get_euler_angles()
        
        # Format output with true and estimated values side by side
        print(f"t={t*dt:.2f}s | {np.degrees(true_roll):8.4f}° | {np.degrees(true_pitch):9.4f}° | {np.degrees(true_yaw):7.4f}° | {np.degrees(roll_est):7.4f}° | {np.degrees(pitch_est):8.4f}° | {np.degrees(yaw_est):7.4f}°")
    print("--- Demo finished ---")