import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional, List

"""
-------------------------------------------------------------------------------
OBJECT ORIENTED EXTENDED KALMAN FILTER (EKF) FRAMEWORK
--- QUATERNION-BASED (16-DOF STATE)
-------------------------------------------------------------------------------
Architecture:
1. State:           Holds the 'Truth' (x) and Uncertainty (P).
2. SystemModel:     Defines how the system moves (Physics/Kinetics).
3. Sensor (Base):   Defines how a sensor sees the world (Measurement Models).
4. EKF (Engine):    Performs the recursive estimation.

Conventions:
- State Vector (x): n x 1 numpy array
- Covariance (P):   n x n numpy array
- Measurement (z):  m x 1 numpy array
-------------------------------------------------------------------------------
"""

# =============================================================================
# QUATERNION UTILITIES
# =============================================================================

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion multiplication q1 * q2 (q = [q0, q1, q2, q3] scalar-first)."""
    q0_1, q1_1, q2_1, q3_1 = q1[0], q1[1], q1[2], q1[3]
    q0_2, q1_2, q2_2, q3_2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        q0_1*q0_2 - q1_1*q1_2 - q2_1*q2_2 - q3_1*q3_2,
        q0_1*q1_2 + q1_1*q0_2 + q2_1*q3_2 - q3_1*q2_2,
        q0_1*q2_2 - q1_1*q3_2 + q2_1*q0_2 + q3_1*q1_2,
        q0_1*q3_2 + q1_1*q2_2 - q2_1*q1_2 + q3_1*q0_2
    ])

def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate: [q0, -q1, -q2, -q3]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_norm(q: np.ndarray) -> float:
    """Euclidean norm of quaternion."""
    return np.linalg.norm(q)

def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit norm."""
    return q / quaternion_norm(q)

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (q0, q1, q2, q3) to 3x3 rotation matrix (body->NED)."""
    q = quaternion_normalize(q)
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [    2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2),     2*(q2*q3 - q0*q1)],
        [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])

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
    
    def get_accel_bias(self) -> np.ndarray:
        return self.x[13:16]
    
    def set_quaternion(self, q: np.ndarray):
        self.x[6:10] = quaternion_normalize(q).reshape((4, 1))

class Sensor(ABC):
    """
    Abstract Base Class for all Sensors (Accelerometer, Gyro, GPS, etc.).
    
    To add a new sensor (e.g., a Visual Odometry unit), subclass this,
    define the noise characteristics (R), and the measurement model (h).
    """
    def __init__(self, name: str, measurement_dim: int):
        self.name = name
        self.m_dim = measurement_dim
        # R: Measurement Noise Covariance Matrix
        self.R = np.eye(measurement_dim)

    def set_noise_covariance(self, noise_std_devs: List[float]):
        """Helper to set R based on standard deviations."""
        if len(noise_std_devs) != self.m_dim:
            raise ValueError(f"Expected {self.m_dim} noise values.")
        np.fill_diagonal(self.R, [x**2 for x in noise_std_devs])

    @abstractmethod
    def measurement_model(self, state: State) -> np.ndarray:
        """
        h(x): Nonlinear mapping from State Space to Measurement Space.
        Returns expected measurement z_hat (m x 1).
        """
        pass

    @abstractmethod
    def jacobian(self, state: State) -> np.ndarray:
        """
        H: Jacobian of the measurement model h(x) with respect to x.
        Returns H matrix (m x n).
        """
        pass

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

    def update(self, sensor: Sensor, z_measurement: np.ndarray):
        """
        Measurement Update Step (A Posteriori).
        Fuses a sensor reading into the state.
        
        Args:
            sensor: The Sensor object providing the model.
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

class GPSSensor(Sensor):
    """
    Example of a concrete sensor implementation.
    """
    def __init__(self):
        super().__init__("GPS", measurement_dim=3)
        # GPS usually has ~2.5m accuracy
        self.set_noise_covariance([2.5, 2.5, 5.0]) # x, y, z errors

    def measurement_model(self, state: State) -> np.ndarray:
        """
        GPS measures Position directly.
        h(x) = [px, py, pz].T
        """
        # Extract position from state (indices 0, 1, 2)
        return state.x[0:3]

    def jacobian(self, state: State) -> np.ndarray:
        """
        H Matrix. 
        Since measurement is linear (1-to-1 with position states),
        this is simple.
        """
        H = np.zeros((self.m_dim, state.dim))
        H[0, 0] = 1 # Measure px
        H[1, 1] = 1 # Measure py
        H[2, 2] = 1 # Measure pz
        return H


class AccelerometerSensor(Sensor):
    """
    Accelerometer sensor model.

    The sensor expects a `measurement_func` callable that returns the
    expected specific force (3x1) in the body frame *without* bias.
    The sensor will add the accelerometer bias (read from `state`) and
    an optional mounting rotation to produce the final expected measurement.
    """
    def __init__(self,
                 name: str = "ACC",
                 measurement_func: Optional[Callable[[State], np.ndarray]] = None,
                 bias_start_index: int = 9,
                 mounting_R: Optional[np.ndarray] = None):
        super().__init__(name, measurement_dim=3)
        self.measurement_func = measurement_func
        self.bias_start = bias_start_index
        self.mounting_R = np.eye(3) if mounting_R is None else mounting_R

    def _get_bias(self, state: State) -> np.ndarray:
        return state.x[self.bias_start:self.bias_start+3].reshape((3, 1))

    def measurement_model(self, state: State) -> np.ndarray:
        if self.measurement_func is None:
            raise RuntimeError("Accelerometer measurement_func not provided")
        true_specific_force = self.measurement_func(state).reshape((3, 1))
        # apply mounting rotation and bias
        meas = self.mounting_R @ true_specific_force + self._get_bias(state)
        return meas

    def jacobian(self, state: State) -> np.ndarray:
        # Numerical Jacobian w.r.t full state
        eps = 1e-6
        m = self.m_dim
        n = state.dim
        H = np.zeros((m, n))
        base = self.measurement_model(state)
        for i in range(n):
            dx = np.zeros((n, 1))
            dx[i, 0] = eps
            tmp_state = State(dim=n)
            tmp_state.x = state.x.copy() + dx
            tmp_state.P = state.P.copy()
            tmp = self.measurement_model(tmp_state)
            H[:, i] = ((tmp - base).reshape(m) / eps)
        return H


class GyroscopeSensor(Sensor):
    """
    Gyroscope sensor model.

    The sensor expects a `measurement_func` callable that returns the
    expected angular rate (3x1) in the body frame *without* bias.
    The sensor will add the gyro bias (read from `state`) and an optional
    mounting rotation to produce the final expected measurement.
    """
    def __init__(self,
                 name: str = "GYRO",
                 measurement_func: Optional[Callable[[State], np.ndarray]] = None,
                 bias_start_index: int = 12,
                 mounting_R: Optional[np.ndarray] = None):
        super().__init__(name, measurement_dim=3)
        self.measurement_func = measurement_func
        self.bias_start = bias_start_index
        self.mounting_R = np.eye(3) if mounting_R is None else mounting_R

    def _get_bias(self, state: State) -> np.ndarray:
        return state.x[self.bias_start:self.bias_start+3].reshape((3, 1))

    def measurement_model(self, state: State) -> np.ndarray:
        if self.measurement_func is None:
            raise RuntimeError("Gyro measurement_func not provided")
        true_omega = self.measurement_func(state).reshape((3, 1))
        meas = self.mounting_R @ true_omega + self._get_bias(state)
        return meas

    def jacobian(self, state: State) -> np.ndarray:
        # Numerical Jacobian w.r.t full state
        eps = 1e-6
        m = self.m_dim
        n = state.dim
        H = np.zeros((m, n))
        base = self.measurement_model(state)
        for i in range(n):
            dx = np.zeros((n, 1))
            dx[i, 0] = eps
            tmp_state = State(dim=n)
            tmp_state.x = state.x.copy() + dx
            tmp_state.P = state.P.copy()
            tmp = self.measurement_model(tmp_state)
            H[:, i] = ((tmp - base).reshape(m) / eps)
        return H


class CompositeSensor(Sensor):
    """
    Composite sensor that stacks multiple `Sensor` objects into a single
    measurement model and Jacobian. Useful when you have many accelerometers
    and gyroscopes to fuse at once.
    """
    def __init__(self, sensors: List[Sensor], name: str = "COMPOSITE"):
        total_dim = sum(s.m_dim for s in sensors)
        super().__init__(name, measurement_dim=total_dim)
        self.sensors = sensors
        # Build block-diagonal R
        R_blocks = [s.R for s in sensors]
        self.R = np.zeros((total_dim, total_dim))
        idx = 0
        for Rb in R_blocks:
            m = Rb.shape[0]
            self.R[idx:idx+m, idx:idx+m] = Rb
            idx += m

    def measurement_model(self, state: State) -> np.ndarray:
        parts = [s.measurement_model(state).reshape((s.m_dim, 1)) for s in self.sensors]
        return np.vstack(parts)

    def jacobian(self, state: State) -> np.ndarray:
        H_parts = [s.jacobian(state) for s in self.sensors]
        return np.vstack(H_parts)

# -----------------------------------------------------------------------------
# USAGE DEMO: multiple accelerometers + gyros fused via CompositeSensor
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialize a full 15-state navigation vector (pos, vel, att_err, acc_bias, gyro_bias)
    nav_state = State(dim=15)
    ekf = ExtendedKalmanFilter(nav_state)
    gps = GPSSensor()

    # Simple measurement functions for demo
    def accel_measurement_func(state: State) -> np.ndarray:
        # Very simple: specific force equals -gravity in body frame
        return np.array([0.0, 0.0, -9.81])

    def gyro_measurement_func(state: State) -> np.ndarray:
        # No rotation for this demo
        return np.zeros(3)

    # Create multiple accelerometers and gyros
    acc1 = AccelerometerSensor(name="ACC1", measurement_func=accel_measurement_func, bias_start_index=9)
    acc1.set_noise_covariance([0.05, 0.05, 0.05])

    acc2 = AccelerometerSensor(name="ACC2", measurement_func=accel_measurement_func, bias_start_index=9,
                               mounting_R=np.eye(3))
    acc2.set_noise_covariance([0.1, 0.1, 0.1])

    gyro1 = GyroscopeSensor(name="GYRO1", measurement_func=gyro_measurement_func, bias_start_index=12)
    gyro1.set_noise_covariance([0.001, 0.001, 0.001])

    gyro2 = GyroscopeSensor(name="GYRO2", measurement_func=gyro_measurement_func, bias_start_index=12)
    gyro2.set_noise_covariance([0.002, 0.002, 0.002])

    # Composite sensor stacking all IMU channels
    imu_composite = CompositeSensor([acc1, acc2, gyro1, gyro2])

    # Set small initial biases in the state (for demo visibility)
    nav_state.x[9:12] = np.array([[0.05], [-0.02], [0.01]])
    nav_state.x[12:15] = np.array([[0.005], [-0.003], [0.002]])

    # Process noise (Q) for full 15-state system
    Q_matrix = np.eye(15) * 0.01

    # Run a short simulation: predict with simple kinematics and update with IMU composite
    print("--- Starting EKF Fusion (IMU composite + GPS demo) ---")
    for t in range(10):
        dt = 0.1
        ekf.predict(
            dt=dt,
            f_transition=NavigationPhysics.transition_function,
            F_jacobian=NavigationPhysics.transition_jacobian,
            Q=Q_matrix,
        )

        # Simulate composite IMU measurement (true + noise)
        z_true = imu_composite.measurement_model(nav_state)
        noise = np.random.multivariate_normal(np.zeros(imu_composite.m_dim), imu_composite.R).reshape((imu_composite.m_dim, 1))
        z_noisy = z_true + noise

        ekf.update(imu_composite, z_noisy)

        # Occasionally fuse GPS as well
        if t % 5 == 0:
            fake_gps = np.vstack((nav_state.x[0:3] + np.random.randn(3, 1) * 1.0))
            ekf.update(gps, fake_gps)

        print(f"t={t*dt:.2f}s pos_x={nav_state.x[0,0]:.3f} acc_bias_x={nav_state.x[9,0]:.4f} gyro_bias_x={nav_state.x[12,0]:.5f}")

    print("--- Demo finished ---")