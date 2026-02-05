import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional, List

"""
-------------------------------------------------------------------------------
OBJECT ORIENTED EXTENDED KALMAN FILTER (EKF) FRAMEWORK
-------------------------------------------------------------------------------
Author: Jarvis (Model)
Context: Modular Sensor Fusion for Navigation

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

class State:
    """
    The System State.
    Encapsulates the state vector 'x' and the covariance matrix 'P'.
    
    For a typical IMU-driven Navigation system, a 15-DOF state is standard:
    0-2: Position (px, py, pz)
    3-5: Velocity (vx, vy, vz)
    6-8: Orientation Error (roll, pitch, yaw) or Quaternion 
    9-11: Accelerometer Bias
    12-14: Gyroscope Bias
    """
    def __init__(self, dim: int = 15):
        self.dim = dim
        self.x = np.zeros((dim, 1))
        self.P = np.eye(dim) * 1.0  # Initialize with high uncertainty
        self.timestamp = 0.0

    def get_position(self) -> np.ndarray:
        return self.x[0:3]

    def get_velocity(self) -> np.ndarray:
        return self.x[3:6]
    
    # Add getters/setters for specific state slices as needed for readability

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
    This is where the strapdown inertial navigation equations live.
    """
    @staticmethod
    def transition_function(x: np.ndarray, dt: float) -> np.ndarray:
        """
        Simple Kinematic Model for demonstration.
        x = [pos_x, vel_x, ...].T
        """
        # Create a copy to avoid modifying previous state in place
        new_x = x.copy()
        
        # Position += Velocity * dt (x = x + v*t)
        # Assumes state indices: 0-2 pos, 3-5 vel
        new_x[0:3] += x[3:6] * dt
        
        # Velocity logic would go here (requires control inputs or gravity handling)
        
        return new_x

    @staticmethod
    def transition_jacobian(x: np.ndarray, dt: float) -> np.ndarray:
        """
        Returns F matrix.
        """
        dim = x.shape[0]
        F = np.eye(dim)
        
        # Partial derivative of Position w.r.t Velocity is dt
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        
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

# -----------------------------------------------------------------------------
# USAGE DEMO
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Initialize
    nav_state = State(dim=6) # Using 6DOF for simple demo (Pos+Vel)
    ekf = ExtendedKalmanFilter(nav_state)
    gps = GPSSensor()
    
    # Process Noise (Q)
    Q_matrix = np.eye(6) * 0.1
    
    # 2. Simulation Loop
    print("--- Starting EKF Fusion ---")
    
    # Fake time steps
    for t in range(20):
        dt = 0.1
        
        # PREDICT (IMU Integration would happen here)
        ekf.predict(
            dt=dt, 
            f_transition=NavigationPhysics.transition_function,
            F_jacobian=NavigationPhysics.transition_jacobian,
            Q=Q_matrix
        )
        print(f"Time {t*dt:.1f}s | Pred X: {nav_state.x[0,0]:.2f}")
        
        # UPDATE (GPS Reading comes in)
        # Simulating a GPS reading at x=10.0 (with some conceptual noise)
        fake_gps_data = np.array([[1.0 + t*0.1], [0.0], [5.0]]) 
        ekf.update(gps, fake_gps_data)
        print(f"Time {t*dt:.1f}s | Update X: {fake_gps_data[0,0]:.2f}")
        
        print(f"Time {t*dt:.1f}s | Corr X: {nav_state.x[0,0]:.2f}\n")