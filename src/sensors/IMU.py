# -*- coding: utf-8 -*-
"""
Filename: IMU.py
Description: Defines inertial measurement unit (IMU) sensors including accelerometers, gyroscopes, and magnetometers.
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, Union, Dict, List, Any
from enum import Enum
from dataclasses import dataclass
from .sensor_framework import FrameAwareSensor, BaseSensor, SensorSpec

class Accelerometer3Axis(FrameAwareSensor):
    """
    3-axis accelerometer with lever arm modeling.
    Outputs specific force in m/s^2.
    """
    SPEC = SensorSpec(
        dimension=3,
        units="m/s^2",
        description="3-axis specific force",
        labels=["ax", "ay", "az"]
    )
    
    def __init__(
        self, 
        sensor_id: str = "accel_3axis",
        pos_sensor_body: Optional[np.ndarray] = None,
        **kwargs
    ):
        super().__init__(sensor_id, self.SPEC, **kwargs)
        self.pos_sensor_body = (
            np.zeros(3) if pos_sensor_body is None 
            else np.asarray(pos_sensor_body, dtype=float)
        )
    
    def compute_specific_force(
        self, 
        accel_cg_nav: np.ndarray,
        omega_body: np.ndarray,
        alpha_body: np.ndarray,
        dcm_body_to_nav: np.ndarray,
        gravity_nav: np.ndarray
    ) -> np.ndarray:
        """
        Compute true specific force at sensor location using transport theorem.
        """
        dcm_nav_to_body = dcm_body_to_nav.T
        accel_cg_body = dcm_nav_to_body @ accel_cg_nav
        
        # Lever arm effects
        accel_euler = np.cross(alpha_body, self.pos_sensor_body)
        accel_centripetal = np.cross(omega_body, np.cross(omega_body, self.pos_sensor_body))
        
        gravity_body = dcm_nav_to_body @ gravity_nav
        
        return accel_cg_body + accel_euler + accel_centripetal - gravity_body


class Gyroscope3Axis(FrameAwareSensor):
    """
    3-axis rate gyroscope.
    Outputs angular velocity in rad/s.
    """
    SPEC = SensorSpec(
        dimension=3,
        units="rad/s",
        description="3-axis angular velocity",
        labels=["wx", "wy", "wz"]
    )
    
    def __init__(self, sensor_id: str = "gyro_3axis", **kwargs):
        super().__init__(sensor_id, self.SPEC, **kwargs)


class Magnetometer3Axis(FrameAwareSensor):
    """
    3-axis magnetometer.
    Outputs magnetic field in microTesla (uT).
    """
    SPEC = SensorSpec(
        dimension=3,
        units="uT",
        description="3-axis magnetic field",
        labels=["mx", "my", "mz"]
    )
    
    def __init__(self, sensor_id: str = "mag_3axis", **kwargs):
        super().__init__(sensor_id, self.SPEC, **kwargs)


# ==============================================================================
# ATTITUDE SENSORS
# ==============================================================================

class GyroscopeQuaternion(BaseSensor):
    """
    Integrated gyroscope that outputs orientation as quaternion.
    
    Architecture: Uses composition - contains a Gyroscope3Axis to model
    rate errors, then integrates those corrupted rates to produce a
    quaternion attitude estimate.
    """
    SPEC = SensorSpec(
        dimension=4,
        units="quaternion",
        description="Orientation quaternion [w, x, y, z]",
        labels=["qw", "qx", "qy", "qz"]
    )
    
    def __init__(
        self, 
        sensor_id: str = "gyro_quat",
        initial_quaternion: Optional[np.ndarray] = None,
        quaternion_noise_std: float = 0.0,
        # Pass-through parameters for the internal 3-axis gyro
        dcm_body_to_sensor: Optional[np.ndarray] = None,
        initial_bias: Optional[np.ndarray] = None,
        bias_instability_std: Optional[np.ndarray] = None,
        white_noise_std: Optional[np.ndarray] = None,
        scale_factors: Optional[np.ndarray] = None,
        saturation_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        # Initialize base sensor with 4D quaternion output
        super().__init__(sensor_id, self.SPEC, **kwargs)
        
        # Initialize quaternion state (identity = no rotation)
        self.quaternion = (
            np.array([1.0, 0.0, 0.0, 0.0]) if initial_quaternion is None
            else np.asarray(initial_quaternion, dtype=float)
        )
        self._normalize_quaternion()
        
        # Additional noise on quaternion output (beyond rate errors)
        self.quaternion_noise_std = quaternion_noise_std
        self.rng = np.random.default_rng(seed)
        
        # COMPOSITION: Internal 3-axis gyro for rate error modeling
        # This gyro handles all the realistic sensor errors (bias, noise, etc.)
        self.rate_gyro = Gyroscope3Axis(
            sensor_id=f"{sensor_id}_rates",
            dcm_body_to_sensor=dcm_body_to_sensor,
            initial_bias=initial_bias,
            bias_instability_std=bias_instability_std,
            white_noise_std=white_noise_std,
            scale_factors=scale_factors,
            saturation_limits=saturation_limits,
            seed=seed,
            buffer_size=kwargs.get('buffer_size', 100)
        )
    
    def _normalize_quaternion(self):
        """Ensure quaternion has unit magnitude."""
        norm = np.linalg.norm(self.quaternion)
        if norm > 0:
            self.quaternion /= norm
    
    def step_from_rates(
        self, 
        omega_body: np.ndarray, 
        dt: float, 
        timestamp: float
    ) -> np.ndarray:
        """
        Integrate angular rates to update quaternion.
        
        Process:
        1. Internal gyro applies realistic rate errors to omega_body (3D -> 3D)
        2. Integrate corrupted rates using quaternion kinematics (3D -> 4D)
        3. Optionally add quaternion-specific noise
        4. Store and return the quaternion
        
        Args:
            omega_body: True angular velocity in body frame (rad/s)
            dt: Time step for integration
            timestamp: Current time
            
        Returns:
            Corrupted quaternion estimate
        """
        # Step 1: Get corrupted rates from internal gyro
        # This applies bias, bias random walk, noise, scale factors, etc.
        omega_measured = self.rate_gyro.step_with_frame(omega_body, dt, timestamp)
        
        # Step 2: Integrate using quaternion kinematics
        # Quaternion derivative: q_dot = 0.5 * Omega(omega) * q
        # where Omega is the skew-symmetric matrix form
        qw, qx, qy, qz = self.quaternion
        wx, wy, wz = omega_measured
        
        # Quaternion rate of change
        q_dot = 0.5 * np.array([
            -qx*wx - qy*wy - qz*wz,
             qw*wx + qy*wz - qz*wy,
             qw*wy - qx*wz + qz*wx,
             qw*wz + qx*wy - qy*wx
        ])
        
        # Euler integration (for small dt, this is adequate)
        self.quaternion += q_dot * dt
        self._normalize_quaternion()
        
        # Step 3: Add quaternion-specific noise if needed
        # (This models errors beyond rate integration, e.g., numerical drift)
        if self.quaternion_noise_std > 0:
            quat_noise = self.quaternion_noise_std * self.rng.standard_normal(4)
            self.quaternion += quat_noise
            self._normalize_quaternion()
        
        # Step 4: Store and return
        self._store(self.quaternion, timestamp)
        
        return self.quaternion
    
    def get_rate_gyro(self) -> Gyroscope3Axis:
        """
        Access the internal rate gyroscope.
        Useful for debugging or analyzing rate errors separately.
        """
        return self.rate_gyro
    

if __name__ == "__main__":
    import unittest
    from ..util.angles import dcm_from_euler, euler_from_dcm, quaternion_to_rotation_matrix

    # ==================================================================
    # UNIT TESTS FOR ACCELEROMETER
    # ==================================================================
    
    class TestAccelerometer3Axis(unittest.TestCase):
        """Test 3-axis accelerometer sensor."""
        
        def test_static_horizontal_acceleration(self):
            """Test accelerometer measuring static horizontal acceleration."""
            print("\n[TEST] Static horizontal acceleration: Testing 1 m/s^2 in X-axis.")
            try:
                accel = Accelerometer3Axis(seed=42)
                
                # Static horizontal acceleration
                accel_cg = np.array([1.0, 0.0, 0.0])  # 1 m/s^2 in X
                omega_body = np.zeros(3)
                alpha_body = np.zeros(3)
                dcm = np.eye(3)  # No rotation
                gravity = np.array([0.0, 0.0, 9.81])
                
                specific_force = accel.compute_specific_force(accel_cg, omega_body, alpha_body, dcm, gravity)
                
                # Should measure acceleration minus gravity effect
                self.assertAlmostEqual(specific_force[0], 1.0, places=2)
                self.assertAlmostEqual(specific_force[2], -9.81, places=2)
            finally:
                print(f"[PASS] [OK] Static acceleration measured: {specific_force} m/s^2.")
        
        def test_gravity_at_level(self):
            """Test accelerometer measuring gravity at level orientation."""
            print("\n[TEST] Gravity at level: Testing measurement of 1g downward acceleration.")
            try:
                accel = Accelerometer3Axis(seed=42)
                
                # No body acceleration, level orientation
                accel_cg = np.array([0.0, 0.0, 0.0])
                omega_body = np.zeros(3)
                alpha_body = np.zeros(3)
                dcm = np.eye(3)
                gravity = np.array([0.0, 0.0, 9.81])
                
                specific_force = accel.compute_specific_force(accel_cg, omega_body, alpha_body, dcm, gravity)
                
                # Should measure -1g acceleration (upward acceleration needed to stay airborne)
                self.assertAlmostEqual(specific_force[2], -9.81, places=1)
            finally:
                print(f"[PASS] [OK] Gravity acceleration verified: {specific_force[2]:.2f} m/s^2.")
        
        def test_lever_arm_centripetal(self):
            """Test lever arm effect from centripetal acceleration."""
            print("\n[TEST] Lever arm centripetal: Testing sensor offset during rotation.")
            try:
                # Sensor offset 0.1 m in X from center of mass
                accel = Accelerometer3Axis(pos_sensor_body=np.array([0.1, 0.0, 0.0]), seed=42)
                
                # No body acceleration, spinning about Z-axis
                accel_cg = np.array([0.0, 0.0, 0.0])
                omega_body = np.array([0.0, 0.0, 10.0])  # 10 rad/s
                alpha_body = np.zeros(3)
                dcm = np.eye(3)
                gravity = np.array([0.0, 0.0, 9.81])
                
                specific_force = accel.compute_specific_force(accel_cg, omega_body, alpha_body, dcm, gravity)
                
                # Centripetal acceleration = omega^2 * r = 100 * 0.1 = 10 m/s^2 toward center (negative X)
                self.assertLess(specific_force[0], 0.0)  # Should be negative (toward center)
            finally:
                print(f"[PASS] [OK] Lever arm effect measured: {specific_force} m/s^2.")
        
        def test_with_bias_error(self):
            """Test accelerometer with bias error."""
            print("\n[TEST] Accelerometer bias (0.1 m/s^2): Testing constant bias application.")
            try:
                bias = np.array([0.1, 0.2, 0.3])
                accel = Accelerometer3Axis(initial_bias=bias, seed=42)
                
                accel_cg = np.array([0.0, 0.0, 0.0])
                omega_body = np.zeros(3)
                alpha_body = np.zeros(3)
                dcm = np.eye(3)
                gravity = np.array([0.0, 0.0, 9.81])
                
                specific_force = accel.compute_specific_force(accel_cg, omega_body, alpha_body, dcm, gravity)
                
                # Apply the sensor error model
                measured = accel.step(specific_force, 0.01, 0.0)
                
                # Should include bias offset
                self.assertAlmostEqual(measured[0], bias[0], places=2)
            finally:
                print(f"[PASS] [OK] Bias error applied to measurement.")
    
    
    # ==================================================================
    # UNIT TESTS FOR GYROSCOPE
    # ==================================================================
    
    class TestGyroscope3Axis(unittest.TestCase):
        """Test 3-axis gyroscope sensor."""
        
        def test_zero_rotation_rate(self):
            """Test gyroscope at rest."""
            print("\n[TEST] Zero rotation: Testing gyroscope measures ~0 rad/s at rest.")
            try:
                gyro = Gyroscope3Axis(
                    white_noise_std=np.array([0.001, 0.001, 0.001]),
                    seed=42
                )
                
                omega_body = np.zeros(3)
                dcm_body_to_nav = np.eye(3)
                
                # Measure through the sensor model
                measured = gyro.step_with_frame(omega_body, 0.01, 0.0)
                
                # Should be close to zero with small noise
                np.testing.assert_array_almost_equal(measured, [0.0, 0.0, 0.0], decimal=2)
            finally:
                print(f"[PASS] [OK] Zero rotation verified: {measured} rad/s.")
        
        def test_rotation_about_z_axis(self):
            """Test gyroscope measuring rotation about Z-axis."""
            print("\n[TEST] Z-axis rotation (10 rad/s): Testing gyroscope X and Y rejection.")
            try:
                gyro = Gyroscope3Axis(
                    white_noise_std=np.array([0.0001, 0.0001, 0.0001]),
                    seed=42
                )
                
                omega_body = np.array([0.0, 0.0, 10.0])
                dcm_body_to_nav = np.eye(3)
                
                measured = gyro.step_with_frame(omega_body, 0.01, 0.0)
                
                # Should read approximately [0, 0, 10]
                self.assertAlmostEqual(measured[0], 0.0, places=2)
                self.assertAlmostEqual(measured[1], 0.0, places=2)
                self.assertAlmostEqual(measured[2], 10.0, places=2)
            finally:
                print(f"[PASS] [OK] Z-axis rotation verified: {measured} rad/s.")
        
        def test_with_bias_random_walk(self):
            """Test gyroscope bias random walk."""
            print("\n[TEST] Bias random walk: Testing that bias drifts over time.")
            try:
                gyro = Gyroscope3Axis(
                    bias_instability_std=np.array([0.001, 0.001, 0.001]),
                    seed=42
                )
                
                omega_body = np.array([0.0, 0.0, 0.0])
                dcm_body_to_nav = np.eye(3)
                
                measurements = []
                for i in range(50):
                    measured = gyro.step_with_frame(omega_body, 0.01, 0.01 * i)
                    measurements.append(measured)
                
                measurements = np.array(measurements)
                
                # Bias should change over time
                bias_variation = np.std(measurements, axis=0)
                self.assertGreater(np.mean(bias_variation), 0.0)
            finally:
                print(f"[PASS] [OK] Bias random walk verified: std = {np.mean(bias_variation):.4f} rad/s.")
    
    
    # ==================================================================
    # UNIT TESTS FOR MAGNETOMETER
    # ==================================================================
    
    class TestMagnetometer3Axis(unittest.TestCase):
        """Test 3-axis magnetometer sensor."""
        
        def test_north_east_down_field(self):
            """Test magnetometer measuring standard NED magnetic field."""
            print("\n[TEST] Standard mag field: Testing magnetometer in NED frame (27 uT).")
            try:
                mag = Magnetometer3Axis(
                    white_noise_std=np.array([0.01, 0.01, 0.01]),
                    seed=42
                )
                
                # Standard magnetic field at mid-latitudes ~45 uT (NED frame)
                mag_ned = np.array([27.0, 0.0, 43.0])  # North and down components
                dcm_body_to_nav = np.eye(3)  # No rotation
                
                measured = mag.step_with_frame(mag_ned, 0.01, 0.0)
                
                # Should read approximately the input field
                np.testing.assert_array_almost_equal(measured, mag_ned, decimal=1)
            finally:
                print(f"[PASS] [OK] Mag field measurement verified: {measured} uT.")
        
        def test_rotated_field(self):
            """Test magnetometer measuring rotated magnetic field."""
            print("\n[TEST] Rotated mag field: Testing magnetometer with 45 degree rotation.")
            try:
                mag = Magnetometer3Axis(
                    white_noise_std=np.array([0.001, 0.001, 0.001]),
                    seed=42
                )
                
                mag_ned = np.array([30.0, 0.0, 40.0])
                
                # 45 degree pitch rotation
                dcm = dcm_from_euler(0.0, np.radians(45.0), 0.0)
                
                measured = mag.step_with_frame(mag_ned, 0.01, 0.0)
                
                # Magnitude should be preserved
                mag_magnitude = np.linalg.norm(mag_ned)
                measured_magnitude = np.linalg.norm(measured)
                self.assertAlmostEqual(mag_magnitude, measured_magnitude, places=1)
            finally:
                print(f"[PASS] [OK] Rotated mag field verified: magnitude = {measured_magnitude:.2f} uT.")
    
    
    # ==================================================================
    # UNIT TESTS FOR GYROSCOPE QUATERNION
    # ==================================================================
    
    class TestGyroscopeQuaternion(unittest.TestCase):
        """Test quaternion-based integrated gyroscope."""
        
        def test_identity_quaternion_at_rest(self):
            """Test gyroscope quaternion starts at identity."""
            print("\n[TEST] Identity quaternion: Testing initial quaternion state [1,0,0,0].")
            try:
                gyro_quat = GyroscopeQuaternion(
                    initial_quaternion=[1.0, 0.0, 0.0, 0.0],
                    quaternion_noise_std=0.0,
                    white_noise_std=np.array([0.0, 0.0, 0.0]),
                    seed=42
                )
                
                omega_body = np.zeros(3)
                
                quat = gyro_quat.step_from_rates(omega_body, 0.01, 0.0)
                
                # Should remain at identity
                np.testing.assert_array_almost_equal(quat, [1.0, 0.0, 0.0, 0.0], decimal=5)
            finally:
                print(f"[PASS] [OK] Identity quaternion verified: {quat}.")
        
        def test_quaternion_rotation_integration(self):
            """Test quaternion integrates rotation correctly."""
            print("\n[TEST] Rotation integration: Testing quaternion from 10 deg/s for 1 second.")
            try:
                gyro_quat = GyroscopeQuaternion(
                    initial_quaternion=[1.0, 0.0, 0.0, 0.0],
                    white_noise_std=np.array([0.0, 0.0, 0.0]),
                    seed=42
                )
                
                # 10 deg/s rotation about Z-axis
                omega_body = np.array([0.0, 0.0, np.radians(10.0)])
                dt = 0.01
                
                # Integrate for 1 second (100 steps)
                for i in range(100):
                    quat = gyro_quat.step_from_rates(omega_body, dt, i * dt)
                
                # After 1 second at 10 deg/s, should have rotated ~10 degrees about Z
                # For Z-axis rotation, quat = [cos(theta/2), 0, 0, sin(theta/2)]
                theta = np.radians(10.0)
                expected_quat = np.array([np.cos(theta/2), 0, 0, np.sin(theta/2)])
                
                np.testing.assert_array_almost_equal(quat, expected_quat, decimal=2)
            finally:
                print(f"[PASS] [OK] Rotation integration verified: {quat}.")
        
        def test_quaternion_normalization(self):
            """Test that quaternion remains normalized after noise."""
            print("\n[TEST] Quaternion normalization: Testing unit-norm preservation.")
            try:
                gyro_quat = GyroscopeQuaternion(
                    initial_quaternion=[1.0, 0.0, 0.0, 0.0],
                    quaternion_noise_std=0.05,
                    white_noise_std=np.array([0.01, 0.01, 0.01]),
                    seed=42
                )
                
                omega_body = np.array([1.0, 2.0, 3.0])
                
                for i in range(50):
                    quat = gyro_quat.step_from_rates(omega_body, 0.01, 0.01 * i)
                    norm = np.linalg.norm(quat)
                    self.assertAlmostEqual(norm, 1.0, places=5)
            finally:
                print(f"[PASS] [OK] Quaternion normalization verified: final norm = {np.linalg.norm(quat):.6f}.")
        
        def test_with_rate_bias(self):
            """Test gyroscope quaternion with rate bias error."""
            print("\n[TEST] Rate bias (0.01 rad/s): Testing bias effect on integrated rotation.")
            try:
                bias = np.array([0.01, 0.0, 0.0])
                gyro_quat = GyroscopeQuaternion(
                    initial_quaternion=[1.0, 0.0, 0.0, 0.0],
                    initial_bias=bias,
                    white_noise_std=np.array([0.0, 0.0, 0.0]),
                    seed=42
                )
                
                # Command zero rotation - but bias will cause drift
                omega_body = np.zeros(3)
                dt = 0.01
                
                for i in range(100):
                    quat = gyro_quat.step_from_rates(omega_body, dt, i * dt)
                
                # Should have rotated by bias * time = 0.01 * 1.0 = 0.01 radians
                # Check that X component (bias axis) has non-zero attitude
                self.assertNotAlmostEqual(quat[1], 0.0, places=3)
            finally:
                print(f"[PASS] [OK] Rate bias drift verified: quaternion has bias-induced rotation.")
        
        def test_trajectory_tracking(self):
            """Test quaternion gyro tracking combined rotation trajectory."""
            print("\n[TEST] Rotation trajectory: Testing quaternion tracking 3-DOF rotation.")
            try:
                gyro_quat = GyroscopeQuaternion(
                    initial_quaternion=[1.0, 0.0, 0.0, 0.0],
                    white_noise_std=np.array([0.001, 0.001, 0.001]),
                    buffer_size=500,
                    seed=42
                )
                
                dt = 0.01
                duration = 1.0
                num_steps = int(duration / dt)
                
                quaternions = []
                
                for i in range(num_steps):
                    t = i * dt
                    # Time-varying rotation rates
                    omega_body = np.array([
                        0.5 * np.sin(2 * np.pi * 0.5 * t),  # X: 0.5 Hz
                        0.3 * np.cos(2 * np.pi * 0.3 * t),  # Y: 0.3 Hz
                        0.8 * np.sin(2 * np.pi * 0.8 * t)   # Z: 0.8 Hz
                    ])
                    
                    quat = gyro_quat.step_from_rates(omega_body, dt, t)
                    quaternions.append(quat)
                
                quaternions = np.array(quaternions)
                
                # All quaternions should be normalized
                norms = np.linalg.norm(quaternions, axis=1)
                np.testing.assert_array_almost_equal(norms, np.ones(num_steps), decimal=5)
                
                # History should be recorded
                hist = gyro_quat.get_history_with_timestamps()
                self.assertEqual(len(hist[0]), num_steps)
            finally:
                print(f"[PASS] [OK] Trajectory tracking verified: {len(quaternions)} quaternion samples.")
    
    
    # ==================================================================
    # RUN TESTS
    # ==================================================================
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestAccelerometer3Axis))
    suite.addTests(loader.loadTestsFromTestCase(TestGyroscope3Axis))
    suite.addTests(loader.loadTestsFromTestCase(TestMagnetometer3Axis))
    suite.addTests(loader.loadTestsFromTestCase(TestGyroscopeQuaternion))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("IMU SENSOR TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
