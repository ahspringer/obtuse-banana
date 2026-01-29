# -*- coding: utf-8 -*-
"""
Filename: motion_gen.py
Description: Kinematic generator for rigid body motion. 
             Provides various motion profiles (Steady, Ramp, Harmonic, Step) 
             to drive an IMU simulation for filter development.
"""

import numpy as np
import IMU 

class MotionProfile:
    """Base class for rotational motion profiles about a fixed pivot."""
    def __init__(self, axis: np.ndarray = np.array([0, 1, 0])):
        self.axis = np.asarray(axis, dtype=float)
        self.axis = self.axis / np.linalg.norm(self.axis)

    def get_kinematics(self, t: float) -> tuple:
        """Override this to return (angle, rate, alpha) at time t."""
        raise NotImplementedError

    def get_state(self, t: float):
        """
        Calculates the full 3D state for the IMU.
        Returns: R_bn, omega_b, alpha_b, accel_origin_n
        """
        angle, rate, accel = self.get_kinematics(t)

        omega_body = rate * self.axis
        alpha_body = accel * self.axis
        
        # Rodrigues' Rotation Formula for DCM
        K = np.array([
            [0, -self.axis[2], self.axis[1]],
            [self.axis[2], 0, -self.axis[0]],
            [-self.axis[1], self.axis[0], 0]
        ])
        I = np.eye(3)
        dcm_body_to_nav = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        accel_origin_nav = np.zeros(3) # Pivot is fixed
        return dcm_body_to_nav, omega_body, alpha_body, accel_origin_nav


class HarmonicProfile(MotionProfile):
    """Sinusoidal oscillation (Pendulum/Metronome)."""
    def __init__(self, axis, amplitude_rad, freq_hz, phase=0.0):
        super().__init__(axis)
        self.A = amplitude_rad
        self.w = 2 * np.pi * freq_hz
        self.phi = phase

    def get_kinematics(self, t: float):
        angle = self.A * np.sin(self.w * t + self.phi)
        rate  = self.A * self.w * np.cos(self.w * t + self.phi)
        accel = -self.A * (self.w**2) * np.sin(self.w * t + self.phi)
        return angle, rate, accel


class SteadyRateProfile(MotionProfile):
    """Constant angular velocity (Centrifuge/Spinning)."""
    def __init__(self, axis, rate_rads, start_angle=0.0):
        super().__init__(axis)
        self.rate = rate_rads
        self.theta0 = start_angle

    def get_kinematics(self, t: float):
        angle = self.theta0 + self.rate * t
        rate  = self.rate
        accel = 0.0
        return angle, rate, accel


class AngularRampProfile(MotionProfile):
    """Constant angular acceleration (Spin-up)."""
    def __init__(self, axis, accel_rads2, start_rate=0.0):
        super().__init__(axis)
        self.alpha = accel_rads2
        self.w0 = start_rate

    def get_kinematics(self, t: float):
        angle = 0.5 * self.alpha * t**2 + self.w0 * t
        rate  = self.alpha * t + self.w0
        accel = self.alpha
        return angle, rate, accel


class StepCommandProfile(MotionProfile):
    """
    Simulates 'sawtooth' or 'square' velocity steps.
    Transitions rate from one value to another over a small window.
    """
    def __init__(self, axis, rates: list, durations: list):
        super().__init__(axis)
        self.rates = rates
        self.durations = np.cumsum(durations)
        
    def get_kinematics(self, t: float):
        # Find which segment we are in
        idx = np.searchsorted(self.durations, t)
        idx = min(idx, len(self.rates) - 1)
        
        # In this simple version, we return steady rate for the segment.
        # For a true sawtooth, you'd integrate the rates to get angle.
        rate = self.rates[idx]
        angle = rate * t # Simplified integration
        accel = 0.0 
        if t > 0 and idx > 0 and abs(t - self.durations[idx-1]) < 0.01:
            accel = (self.rates[idx] - self.rates[idx-1]) / 0.01 # Impulse approx
            
        return angle, rate, accel

# --- Simulation Runner ---

def run_simulation():
    dt = 0.005  # 200 Hz for better resolution on ramps
    duration = 10.0
    time = np.arange(0, duration, dt)
    gravity_nav = np.array([0, 0, 9.81]) 

    # --- PICK YOUR PROFILE HERE ---
    # Option 1: Steady Spin (Check Centripetal)
    # profile = SteadyRateProfile(axis=[0, 0, 1], rate_rads=np.radians(360)) 
    
    # Option 2: Angular Ramp (Check Alpha x R)
    profile = AngularRampProfile(axis=[0, 1, 0], accel_rads2=0.5)
    
    # Option 3: Harmonic
    # profile = HarmonicProfile(axis=[0, 1, 0], amplitude_rad=np.radians(30), freq_hz=0.5)

    imu = IMU.IMU(
        accel_config={'pos_sensor_body': [1.0, 0.0, 0.0], 'noise_std': 0.01},
        gyro_config={'noise_std': 0.001}
    )
    
    print(f"Running profile: {type(profile).__name__}")
    
    for t in time:
        R_bn, omega_b, alpha_b, accel_pivot_n = profile.get_state(t)
        
        meas_accel, meas_gyro = imu.step(
            accel_cg_nav=accel_pivot_n,
            omega_body=omega_b,
            alpha_body=alpha_b,
            dcm_body_to_nav=R_bn,
            gravity_nav=gravity_nav,
            dt=dt
        )
        
        # For a real test, you would log these to a CSV or Plotly
        if int(t/dt) % 100 == 0:
            print(f"T={t:.2f} | Gyro_y: {meas_gyro[1]:.3f} | Accel_z: {meas_accel[2]:.3f}")

if __name__ == "__main__":
    run_simulation()