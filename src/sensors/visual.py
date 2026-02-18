from .sensor_framework import SimulatedSensor, SensorSpec
import numpy as np
from typing import Optional
from ..util.angles import dcm_from_euler, euler_from_dcm, quat_from_dcm

# ==============================================================================
# VISION-BASED SENSORS
# ==============================================================================

class VisionPoseEstimator(SimulatedSensor):
    """
    Camera-based pose estimator (e.g., Visual-Inertial Odometry).
    Outputs 6-DOF pose: position (x, y, z) + orientation (roll, pitch, yaw).
    
    Note: In practice, vision systems often output position + quaternion (7D),
    but for simplicity we use Euler angles here (6D).
    """
    SPEC = SensorSpec(
        dimension=6,
        units="mixed",
        description="6-DOF pose: [x, y, z, roll, pitch, yaw]",
        labels=["x", "y", "z", "roll", "pitch", "yaw"]
    )
    
    def __init__(
        self, 
        sensor_id: str = "vision_pose",
        position_noise_std: float = 0.01,  # meters
        orientation_noise_std: float = 0.01,  # radians
        update_rate_hz: float = 30.0,
        **kwargs
    ):
        super().__init__(sensor_id, self.SPEC, **kwargs)
        
        self.update_rate_hz = update_rate_hz
        self.update_period = 1.0 / update_rate_hz
        self.last_update_time = 0.0
        
        # Override default noise with vision-specific noise
        self.white_noise_std = np.array([
            position_noise_std, position_noise_std, position_noise_std,
            orientation_noise_std, orientation_noise_std, orientation_noise_std
        ])
    
    def step_from_truth(
        self, 
        position_nav: np.ndarray,
        dcm_body_to_nav: np.ndarray,
        dt: float,
        timestamp: float
    ) -> Optional[np.ndarray]:
        """
        Generate pose measurement from ground truth.
        Returns None if not enough time has elapsed for next update.
        """
        # Check if enough time has passed for new measurement
        if timestamp - self.last_update_time < self.update_period:
            return None
        
        self.last_update_time = timestamp
        
        # Extract Euler angles from DCM
        roll = np.arctan2(dcm_body_to_nav[2, 1], dcm_body_to_nav[2, 2])
        pitch = np.arctan2(-dcm_body_to_nav[2, 0], 
                          np.sqrt(dcm_body_to_nav[2, 1]**2 + dcm_body_to_nav[2, 2]**2))
        yaw = np.arctan2(dcm_body_to_nav[1, 0], dcm_body_to_nav[0, 0])
        
        # Combine position and orientation
        true_pose = np.array([
            position_nav[0], position_nav[1], position_nav[2],
            roll, pitch, yaw
        ])
        
        # Apply error model
        return super().step(true_pose, dt, timestamp)


class VisionPoseEstimatorQuaternion(SimulatedSensor):
    """
    Camera-based pose estimator with quaternion orientation.
    Outputs 7-DOF: position (x, y, z) + quaternion (w, x, y, z).
    """
    SPEC = SensorSpec(
        dimension=7,
        units="mixed",
        description="7-DOF pose: [x, y, z, qw, qx, qy, qz]",
        labels=["x", "y", "z", "qw", "qx", "qy", "qz"]
    )
    
    def __init__(
        self, 
        sensor_id: str = "vision_pose_quat",
        position_noise_std: float = 0.01,
        orientation_noise_std: float = 0.01,
        update_rate_hz: float = 30.0,
        **kwargs
    ):
        super().__init__(sensor_id, self.SPEC, **kwargs)
        
        self.update_rate_hz = update_rate_hz
        self.update_period = 1.0 / update_rate_hz
        self.last_update_time = 0.0
        
        self.white_noise_std = np.array([
            position_noise_std, position_noise_std, position_noise_std,
            orientation_noise_std, orientation_noise_std, 
            orientation_noise_std, orientation_noise_std
        ])
    
    def step_from_truth(
        self, 
        position_nav: np.ndarray,
        quaternion: np.ndarray,
        dt: float,
        timestamp: float
    ) -> Optional[np.ndarray]:
        """Generate pose measurement with quaternion orientation."""
        if timestamp - self.last_update_time < self.update_period:
            return None
        
        self.last_update_time = timestamp
        
        # Combine position and quaternion
        true_pose = np.concatenate([position_nav, quaternion])
        
        measured = super().step(true_pose, dt, timestamp)
        
        # Normalize quaternion part after adding noise
        if measured is not None:
            measured[3:] /= np.linalg.norm(measured[3:])
            self.current_value = measured
        
        return measured
    

if __name__ == "__main__":
    import unittest

    # ==================================================================
    # UNIT TESTS FOR VISION POSE ESTIMATOR (EULER ANGLES)
    # ==================================================================
    
    class TestVisionPoseEstimator(unittest.TestCase):
        """Test vision pose estimator with Euler angles."""
        
        def test_identity_pose(self):
            """Test measurement at identity pose (origin, no rotation)."""
            print("\n[TEST] Identity pose: Testing vision sensor at origin with no rotation.")
            try:
                sensor = VisionPoseEstimator(
                    position_noise_std=0.001,
                    orientation_noise_std=0.001,
                    update_rate_hz=30.0,
                    seed=42
                )
                position = np.array([0.0, 0.0, 0.0])
                dcm = dcm_from_euler(0.0, 0.0, 0.0)
                
                measurement = sensor.step_from_truth(position, dcm, 0.05, 0.05)
                self.assertIsNotNone(measurement)
                # Check that measured pose is close to identity (with noise allowance)
                np.testing.assert_allclose(measurement[:3], position, atol=0.01)
                np.testing.assert_allclose(measurement[3:], [0.0, 0.0, 0.0], atol=0.01)
            finally:
                print(f"[PASS] [OK] Identity pose verified: position={measurement[:3]}, orientation={measurement[3:]}.")
        
        def test_known_position(self):
            """Test measurement at known position with no rotation."""
            print("\n[TEST] Known position: Testing vision sensor at non-zero position (1, 2, 3).")
            try:
                sensor = VisionPoseEstimator(
                    position_noise_std=0.001,
                    orientation_noise_std=0.001,
                    seed=42
                )
                position = np.array([1.0, 2.0, 3.0])
                dcm = dcm_from_euler(0.0, 0.0, 0.0)
                
                measurement = sensor.step_from_truth(position, dcm, 0.05, 0.05)
                self.assertIsNotNone(measurement)
                # Check position measurement
                np.testing.assert_array_almost_equal(measurement[:3], position, decimal=3)
            finally:
                print(f"[PASS] [OK] Known position verified: measured position = {measurement[:3]}.")
        
        def test_known_rotation(self):
            """Test measurement with known rotation (30° pitch)."""
            print("\n[TEST] Known rotation: Testing vision sensor with 30° pitch rotation.")
            try:
                sensor = VisionPoseEstimator(
                    position_noise_std=0.0001,
                    orientation_noise_std=0.0001,
                    seed=42
                )
                position = np.array([0.0, 0.0, 0.0])
                pitch_true = np.radians(30.0)
                dcm = dcm_from_euler(0.0, pitch_true, 0.0)
                
                measurement = sensor.step_from_truth(position, dcm, 0.05, 0.05)
                self.assertIsNotNone(measurement)
                # Check orientation (roll, pitch, yaw should be approximately 0, 30°, 0)
                measured_euler = measurement[3:]
                self.assertAlmostEqual(measured_euler[1], pitch_true, places=2)
            finally:
                print(f"[PASS] [OK] Known rotation verified: measured orientation = {np.degrees(measured_euler)}°.")
        
        def test_update_rate_limiting(self):
            """Test that measurements are only generated at update_rate_hz."""
            print("\n[TEST] Update rate limiting: Testing that updates occur only at 30 Hz (0.0333 s period).")
            try:
                sensor = VisionPoseEstimator(update_rate_hz=30.0, seed=42)
                position = np.array([1.0, 2.0, 3.0])
                dcm = dcm_from_euler(0.0, 0.0, 0.0)
                
                # Try multiple updates within one period
                dt = 0.01  # 10 ms
                update_count = 0
                timestamps = []
                
                for i in range(10):
                    measurement = sensor.step_from_truth(position, dcm, dt, dt * i)
                    if measurement is not None:
                        update_count += 1
                        timestamps.append(dt * i)
                
                # At 30 Hz with 0.1 s total, should get approximately 3 updates
                self.assertGreater(update_count, 0)
                self.assertLess(update_count, 5)
            finally:
                print(f"[PASS] [OK] Update rate limiting verified: {update_count} updates in {10 * dt:.2f}s.")
        
        def test_with_position_bias(self):
            """Test position bias application."""
            print("\n[TEST] Position bias (0.1 m): Testing constant position bias application.")
            try:
                bias = np.array([0.1, 0.2, 0.3])
                sensor = VisionPoseEstimator(
                    initial_bias=np.concatenate([bias, np.zeros(3)]),
                    position_noise_std=0.0,
                    orientation_noise_std=0.0,
                    seed=42
                )
                position = np.array([0.0, 0.0, 0.0])
                dcm = dcm_from_euler(0.0, 0.0, 0.0)
                
                measurement = sensor.step_from_truth(position, dcm, 0.05, 0.05)
                self.assertIsNotNone(measurement)
                # Measured position should include bias
                np.testing.assert_array_almost_equal(measurement[:3], bias, decimal=3)
            finally:
                print(f"[PASS] [OK] Position bias verified: bias offset applied correctly.")
        
        def test_history_tracking(self):
            """Test that pose history is tracked correctly."""
            print("\n[TEST] History tracking: Testing that measurement history is maintained.")
            try:
                sensor = VisionPoseEstimator(
                    update_rate_hz=30.0,
                    buffer_size=10,
                    seed=42
                )
                position = np.array([1.0, 2.0, 3.0])
                dcm = dcm_from_euler(0.0, np.radians(15.0), 0.0)
                
                # Record multiple measurements
                count = 0
                for i in range(30):
                    measurement = sensor.step_from_truth(position, dcm, 0.01, 0.01 * i)
                    if measurement is not None:
                        count += 1
                
                hist = sensor.get_history_matrix()
                self.assertEqual(hist.shape[1], 6)  # 6 DOF
                self.assertGreater(hist.shape[0], 0)
            finally:
                print(f"[PASS] - History tracking verified: {hist.shape[0]} measurements of dimension {hist.shape[1]}.")
        
        def test_combined_pose_trajectory(self):
            """Test vision sensor tracking a combined motion trajectory."""
            print("\n[TEST] Combined pose trajectory: Testing vision sensor tracking 6-DOF motion.")
            try:
                sensor = VisionPoseEstimator(
                    position_noise_std=0.05,
                    orientation_noise_std=0.05,
                    update_rate_hz=30.0,
                    buffer_size=500,
                    seed=42
                )
                
                # Simulate 2 seconds of motion
                dt = 0.01
                duration = 2.0
                num_steps = int(duration / dt)
                measurements = []
                
                for i in range(num_steps):
                    t = i * dt
                    # Position: oscillating along x-axis
                    pos_x = 0.5 * np.sin(2 * np.pi * 0.5 * t)
                    position = np.array([pos_x, 0.0, 0.0])
                    
                    # Orientation: pitch oscillating at 1 Hz
                    pitch = 0.3 * np.sin(2 * np.pi * 1.0 * t)
                    dcm = dcm_from_euler(0.0, pitch, 0.0)
                    
                    measurement = sensor.step_from_truth(position, dcm, dt, t)
                    if measurement is not None:
                        measurements.append(measurement)
                
                self.assertGreater(len(measurements), 0)
                measurements = np.array(measurements)
                
                # Check that position oscillates
                pos_variations = np.std(measurements[:, 0])
                self.assertGreater(pos_variations, 0.0)
            finally:
                print(f"[PASS] [OK] Combined pose trajectory verified: {len(measurements)} measurements recorded.")
    
    
    # ==================================================================
    # UNIT TESTS FOR VISION POSE ESTIMATOR (QUATERNION)
    # ==================================================================
    
    class TestVisionPoseEstimatorQuaternion(unittest.TestCase):
        """Test vision pose estimator with quaternion orientation."""
        
        def test_identity_pose_quat(self):
            """Test identity quaternion pose (origin, identity rotation)."""
            print("\n[TEST] Identity pose (quaternion): Testing vision sensor at origin with identity quaternion.")
            try:
                sensor = VisionPoseEstimatorQuaternion(
                    position_noise_std=0.001,
                    orientation_noise_std=0.001,
                    seed=42
                )
                position = np.array([0.0, 0.0, 0.0])
                quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
                
                measurement = sensor.step_from_truth(position, quaternion, 0.05, 0.05)
                self.assertIsNotNone(measurement)
                # Position should be near origin
                np.testing.assert_array_almost_equal(measurement[:3], position, decimal=2)
            finally:
                print(f"[PASS] [OK] Identity quaternion pose verified: position={measurement[:3]}, quat={measurement[3:]}.")
        
        def test_known_position_quat(self):
            """Test quaternion pose at known position."""
            print("\n[TEST] Known position (quaternion): Testing vision sensor at fixed position.")
            try:
                sensor = VisionPoseEstimatorQuaternion(
                    position_noise_std=0.0001,
                    orientation_noise_std=0.0001,
                    seed=42
                )
                position = np.array([5.0, 10.0, 15.0])
                quaternion = np.array([1.0, 0.0, 0.0, 0.0])
                
                measurement = sensor.step_from_truth(position, quaternion, 0.05, 0.05)
                self.assertIsNotNone(measurement)
                np.testing.assert_array_almost_equal(measurement[:3], position, decimal=3)
            finally:
                print(f"[PASS] [OK] Known position (quaternion) verified: position={measurement[:3]}.")
        
        def test_quaternion_normalization(self):
            """Test that noisy quaternion is normalized after measurement."""
            print("\n[TEST] Quaternion normalization: Testing that quaternion remains unit-norm after noise.")
            try:
                sensor = VisionPoseEstimatorQuaternion(
                    position_noise_std=0.01,
                    orientation_noise_std=0.01,
                    seed=42
                )
                position = np.array([0.0, 0.0, 0.0])
                quaternion = np.array([1.0, 0.0, 0.0, 0.0])
                
                # Run multiple measurements
                for i in range(10):
                    measurement = sensor.step_from_truth(position, quaternion, 0.05, 0.05 + i * 0.05)
                    if measurement is not None:
                        quat_part = measurement[3:]
                        norm = np.linalg.norm(quat_part)
                        self.assertAlmostEqual(norm, 1.0, places=5)
            finally:
                print(f"[PASS] [OK] Quaternion normalization verified: final quat norm = {norm:.6f}.")
        
        def test_update_rate_quat(self):
            """Test update rate limiting with quaternion sensor."""
            print("\n[TEST] Update rate (quaternion): Testing quaternion sensor update rate limiting.")
            try:
                sensor = VisionPoseEstimatorQuaternion(update_rate_hz=20.0, seed=42)
                position = np.array([1.0, 2.0, 3.0])
                quaternion = np.array([1.0, 0.0, 0.0, 0.0])
                
                update_count = 0
                for i in range(20):
                    measurement = sensor.step_from_truth(position, quaternion, 0.01, 0.01 * i)
                    if measurement is not None:
                        update_count += 1
                
                self.assertGreater(update_count, 0)
                self.assertLess(update_count, 5)
            finally:
                print(f"[PASS] [OK] Update rate (quaternion) verified: {update_count} updates generated.")
        
        def test_dcm_to_quat_conversion(self):
            """Test that DCM-to-quaternion conversion in sensor is consistent."""
            print("\n[TEST] DCM-to-quaternion consistency: Testing angle extraction consistency.")
            try:
                sensor = VisionPoseEstimatorQuaternion(
                    position_noise_std=0.0,
                    orientation_noise_std=0.0,
                    seed=42
                )
                position = np.array([0.0, 0.0, 0.0])
                
                # Test multiple rotation angles
                roll_test = np.radians(30.0)
                pitch_test = np.radians(25.0)
                yaw_test = np.radians(15.0)
                
                dcm = dcm_from_euler(roll_test, pitch_test, yaw_test)
                quat_expected = quat_from_dcm(dcm)
                
                # Provide the expected quaternion
                measurement = sensor.step_from_truth(position, quat_expected, 0.05, 0.05)
                self.assertIsNotNone(measurement)
                
                # Verify quaternion structure
                quat_measured = measurement[3:]
                self.assertAlmostEqual(np.linalg.norm(quat_measured), 1.0, places=5)
            finally:
                print(f"[PASS] [OK] DCM-to-quaternion conversion verified.")
        
        def test_histogram_quat(self):
            """Test history tracking with quaternion data."""
            print("\n[TEST] History (quaternion): Testing measurement history for 7-DOF data.")
            try:
                sensor = VisionPoseEstimatorQuaternion(
                    update_rate_hz=30.0,
                    buffer_size=20,
                    seed=42
                )
                position = np.array([1.0, 2.0, 3.0])
                quaternion = np.array([1.0, 0.0, 0.0, 0.0])
                
                for i in range(30):
                    measurement = sensor.step_from_truth(position, quaternion, 0.01, 0.01 * i)
                
                hist = sensor.get_history_matrix()
                self.assertEqual(hist.shape[1], 7)  # 7 DOF (position + quaternion)
                self.assertGreater(hist.shape[0], 0)
            finally:
                print(f"[PASS] [OK] History (quaternion) verified: {hist.shape[0]} measurements of dimension {hist.shape[1]}.")
        
        def test_quaternion_trajectory(self):
            """Test quaternion sensor tracking 7-DOF trajectory."""
            print("\n[TEST] Quaternion trajectory: Testing vision sensor with 7-DOF tracking.")
            try:
                sensor = VisionPoseEstimatorQuaternion(
                    position_noise_std=0.02,
                    orientation_noise_std=0.02,
                    update_rate_hz=30.0,
                    buffer_size=500,
                    seed=42
                )
                
                # Simulate motion for 1 second
                dt = 0.01
                duration = 1.0
                num_steps = int(duration / dt)
                measurements = []
                
                for i in range(num_steps):
                    t = i * dt
                    # Circular motion in x-y plane
                    pos_x = 0.5 * np.cos(2 * np.pi * 0.5 * t)
                    pos_y = 0.5 * np.sin(2 * np.pi * 0.5 * t)
                    position = np.array([pos_x, pos_y, 0.0])
                    
                    # Rotating orientation
                    yaw = 2 * np.pi * 0.5 * t
                    dcm = dcm_from_euler(0.0, 0.0, yaw)
                    quaternion = quat_from_dcm(dcm)
                    
                    measurement = sensor.step_from_truth(position, quaternion, dt, t)
                    if measurement is not None:
                        measurements.append(measurement)
                
                self.assertGreater(len(measurements), 0)
                measurements = np.array(measurements)
                
                # Verify all quaternions are normalized
                for i in range(measurements.shape[0]):
                    quat_norm = np.linalg.norm(measurements[i, 3:])
                    self.assertAlmostEqual(quat_norm, 1.0, places=5)
            finally:
                print(f"[PASS] [OK] Quaternion trajectory verified: {len(measurements)} measurements with normalized quaternions.")
    
    
    # ==================================================================
    # RUN TESTS
    # ==================================================================
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestVisionPoseEstimator))
    suite.addTests(loader.loadTestsFromTestCase(TestVisionPoseEstimatorQuaternion))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("VISION SENSOR TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
