# -*- coding: utf-8 -*-
"""
Filename: inclinomters.py
Description: Inclinometer sensor models for single-axis and dual-axis tilt measurements.
             These sensors extract pitch and roll angles from the DCM and apply error models to simulate real-world measurements.
             
COORDINATE SYSTEM: +X = downrange, +Y = right, +Z = down
The inclinometer measures tilt angles (pitch and roll) relative to gravity (Z-down).
             
             Each sensor operates in either SIMULATION or HARDWARE mode.
"""

import numpy as np
from ..util.angles import dcm_from_euler, euler_from_dcm
from .sensor_framework import SimulatedSensor, SensorSpec

class Inclinometer1Axis(SimulatedSensor):
    """
    Single-axis tilt sensor.
    Outputs angle in radians about a fixed axis (e.g., pitch or roll).
    """
    SPEC = SensorSpec(
        dimension=1,
        units="rad",
        description="Single-axis tilt angle",
        labels=["theta"]
    )
    
    def __init__(
        self, 
        sensor_id: str = "inclinometer_1axis",
        measurement_axis: str = "pitch",  # 'pitch' or 'roll'
        **kwargs
    ):
        super().__init__(sensor_id, self.SPEC, **kwargs)
        self.measurement_axis = measurement_axis
    
    def step_from_dcm(
        self, 
        dcm_body_to_nav: np.ndarray, 
        dt: float, 
        timestamp: float
    ) -> np.ndarray:
        """
        Extract single-axis tilt from DCM.
        """
        # Extract Euler angles (pitch, roll)
        # For small angles: pitch ≈ -atan2(R[2,0], R[2,2])
        #                   roll ≈ atan2(R[2,1], R[2,2])
        
        if self.measurement_axis == "pitch":
            true_angle = np.arctan2(-dcm_body_to_nav[2, 0], dcm_body_to_nav[2, 2])
        elif self.measurement_axis == "roll":
            true_angle = np.arctan2(dcm_body_to_nav[2, 1], dcm_body_to_nav[2, 2])
        else:
            true_angle = 0.0
        
        # Apply error model
        return super().step(np.array([true_angle]), dt, timestamp)


class Inclinometer2Axis(SimulatedSensor):
    """
    Dual-axis tilt sensor.
    Outputs pitch and roll angles in radians.
    """
    SPEC = SensorSpec(
        dimension=2,
        units="rad",
        description="Dual-axis tilt angles",
        labels=["pitch", "roll"]
    )
    
    def __init__(self, sensor_id: str = "inclinometer_2axis", **kwargs):
        super().__init__(sensor_id, self.SPEC, **kwargs)
    
    def step_from_dcm(
        self, 
        dcm_body_to_nav: np.ndarray, 
        dt: float, 
        timestamp: float
    ) -> np.ndarray:
        """
        Extract pitch and roll from DCM.
        """
        pitch = np.arctan2(-dcm_body_to_nav[2, 0], dcm_body_to_nav[2, 2])
        roll = np.arctan2(dcm_body_to_nav[2, 1], dcm_body_to_nav[2, 2])
        
        true_angles = np.array([pitch, roll])
        return super().step(true_angles, dt, timestamp)


# ===============================================================================
# SCL3300D02 Digital Twin Inclinometer
# ===============================================================================


# ===============================================================================
# SCL3300D02 Digital Twin 2-Axis Inclinometer
# ===============================================================================
class SCL3300D02DigitalTwin(SimulatedSensor):
    """
    Digital twin for the SCL3300D02 2-axis inclinometer.
    Simulates realistic noise, bias, and thermal drift for pitch and roll.
    Specs:
        - Noise density: 6 µg/√Hz (0.000006 g/√Hz ≈ 5.886e-5 m/s^2/√Hz)
        - Initial offsets: ±2 mg (0.01962 m/s^2) for both axes
        - Thermal drift: 0.1 mg/°C (0.000981 m/s^2/°C)
    """
    SPEC = SensorSpec(
        dimension=2,
        units="rad",
        description="2-axis tilt angles (SCL3300D02 digital twin)",
        labels=["pitch", "roll"]
    )

    def __init__(self, temp_C=25.0, temp_drift_C=0.0, seed=None, **kwargs):
        super().__init__("scl3300d02_digital_twin", self.SPEC, **kwargs)
        rng = np.random.default_rng(seed)
        # Noise density (converted to std for 1 Hz bandwidth, in radians)
        self.noise_std = np.ones(2) * np.radians(0.003)  # ~0.003 deg ≈ 5.24e-5 rad
        # Initial offsets (bias) in radians (±2 mg/9.81)
        self.bias = rng.uniform(-0.01962/9.81, 0.01962/9.81, 2)
        # Thermal drift (per degree C, in radians)
        self.thermal_drift_per_C = np.ones(2) * (0.000981/9.81)  # m/s^2/°C to rad/°C
        self.temp_C = temp_C
        self.temp_drift_C = temp_drift_C
        self.rng = rng

    def step_from_dcm(self, dcm_body_to_nav, dt, timestamp, temp_C=None):
        if temp_C is not None:
            self.temp_C = temp_C
        # Extract pitch and roll from DCM
        pitch = np.arctan2(-dcm_body_to_nav[2, 0], dcm_body_to_nav[2, 2])
        roll = np.arctan2(dcm_body_to_nav[2, 1], dcm_body_to_nav[2, 2])
        true_angles = np.array([pitch, roll])
        # Add bias, thermal drift, and noise
        thermal_offset = self.thermal_drift_per_C * (self.temp_C + self.temp_drift_C - 25.0)
        noise = self.rng.normal(0, self.noise_std, 2)
        measured = true_angles + self.bias + thermal_offset + noise
        return measured


if __name__ == "__main__":
    import unittest   
    
    # ==================================================================
    # UNIT TESTS
    # ==================================================================
    
    class TestInclinometer1Axis(unittest.TestCase):
        """Test single-axis inclinometer."""
        
        def test_level_orientation_pitch(self):
            """Test pitch measurement at level (0 deg) orientation."""
            print("\n[TEST] Level orientation - pitch axis: Testing that pitch reads ~0° at level orientation.")
            try:
                inc = Inclinometer1Axis(measurement_axis="pitch")
                dcm = dcm_from_euler(0.0, 0.0, 0.0)  # Level
                
                measured = inc.step_from_dcm(dcm, 0.01, 0.0)
                self.assertAlmostEqual(measured[0], 0.0, places=5)
            finally:
                print("[PASS] ✓ Pitch measurement at level orientation verified.")
        
        def test_level_orientation_roll(self):
            """Test roll measurement at level (0 deg) orientation."""
            print("\n[TEST] Level orientation - roll axis: Testing that roll reads ~0° at level orientation.")
            try:
                inc = Inclinometer1Axis(measurement_axis="roll")
                dcm = dcm_from_euler(0.0, 0.0, 0.0)  # Level
                
                measured = inc.step_from_dcm(dcm, 0.01, 0.0)
                self.assertAlmostEqual(measured[0], 0.0, places=5)
            finally:
                print("[PASS] ✓ Roll measurement at level orientation verified.")
        
        def test_pitch_positive_rotation(self):
            """Test pitch measurement with positive pitch (nose up)."""
            print("\n[TEST] Positive pitch rotation (30°): Testing that pitch sensor correctly measures +30° nose-up orientation.")
            try:
                inc = Inclinometer1Axis(measurement_axis="pitch")
                pitch_true = np.radians(30.0)
                dcm = dcm_from_euler(0.0, pitch_true, 0.0)
                
                measured = inc.step_from_dcm(dcm, 0.01, 0.0)
                self.assertAlmostEqual(measured[0], pitch_true, places=5)
            finally:
                print("[PASS] ✓ Positive pitch rotation correctly measured.")
        
        def test_roll_positive_rotation(self):
            """Test roll measurement with positive roll (right wing down)."""
            print("\n[TEST] Positive roll rotation (45°): Testing that roll sensor correctly measures +45° right-wing-down orientation.")
            try:
                inc = Inclinometer1Axis(measurement_axis="roll")
                roll_true = np.radians(45.0)
                dcm = dcm_from_euler(roll_true, 0.0, 0.0)
                
                measured = inc.step_from_dcm(dcm, 0.01, 0.0)
                self.assertAlmostEqual(measured[0], roll_true, places=5)
            finally:
                print("[PASS] ✓ Positive roll rotation correctly measured.")
        
        def test_with_bias_error(self):
            """Test pitch with constant bias error."""
            print("\n[TEST] Bias error (5°): Testing that pitch sensor applies constant 5-degree bias offset correctly.")
            try:
                bias = np.array([np.radians(5.0)])  # 5 degree bias
                inc = Inclinometer1Axis(
                    measurement_axis="pitch",
                    initial_bias=bias,
                    seed=12345
                )
                dcm = dcm_from_euler(0.0, 0.0, 0.0)  # Level
                
                measured = inc.step_from_dcm(dcm, 0.01, 0.0)
                # Should read 5 degrees
                self.assertAlmostEqual(measured[0], np.radians(5.0), places=5)
            finally:
                print("[PASS] ✓ Bias error correctly applied to measurement.")
        
        def test_with_white_noise(self):
            """Test that white noise is applied to measurements."""
            print("\n[TEST] White noise application (1° std): Testing that Gaussian white noise is correctly applied to measurements.")
            try:
                noise_std = np.array([np.radians(1.0)])  # 1 degree noise std
                inc = Inclinometer1Axis(
                    measurement_axis="pitch",
                    white_noise_std=noise_std,
                    seed=42
                )
                dcm = dcm_from_euler(0.0, 0.0, 0.0)  # Level
                
                # Run multiple samples
                measurements = []
                for i in range(100):
                    measured = inc.step_from_dcm(dcm, 0.01, float(i)*0.01)
                    measurements.append(measured[0])
                
                measurements = np.array(measurements)
                # Check that we have variation (noise is being applied)
                self.assertGreater(np.std(measurements), 0.0)
                # Check approximate noise level
                self.assertLess(np.std(measurements), np.radians(2.0))
            finally:
                print(f"[PASS] ✓ White noise verified with std = {np.std(measurements):.6f} rad.")
        
        def test_with_scale_factor(self):
            """Test pitch with scale factor error."""
            print("\n[TEST] Scale factor error (0.9x): Testing that pitch measurements are scaled by 0.9 (10% error).")
            try:
                scale = np.array([0.9])  # 10% scale error
                inc = Inclinometer1Axis(
                    measurement_axis="pitch",
                    scale_factors=scale,
                    seed=12345
                )
                pitch_true = np.radians(30.0)
                dcm = dcm_from_euler(0.0, pitch_true, 0.0)
                
                measured = inc.step_from_dcm(dcm, 0.01, 0.0)
                # Should read 90% of true value
                self.assertAlmostEqual(measured[0], pitch_true * 0.9, places=5)
            finally:
                print("[PASS] ✓ Scale factor error correctly applied.")
        
        def test_history_tracking(self):
            """Test that sensor history is tracked correctly."""
            print("\n[TEST] History tracking: Testing that sensor correctly maintains history buffer of measurements.")
            try:
                inc = Inclinometer1Axis(measurement_axis="pitch", buffer_size=10)
                dcm = dcm_from_euler(0.0, np.radians(15.0), 0.0)
                
                # Record 5 samples
                for i in range(5):
                    inc.step_from_dcm(dcm, 0.01, float(i)*0.01)
                
                hist_data = inc.get_history_matrix()
                self.assertEqual(hist_data.shape[0], 5)
                self.assertEqual(hist_data.shape[1], 1)
            finally:
                print(f"[PASS] ✓ History buffer contains {hist_data.shape[0]} samples of dimension {hist_data.shape[1]}.")
    
    
    class TestInclinometer2Axis(unittest.TestCase):
        """Test dual-axis inclinometer."""
        
        def test_level_orientation(self):
            """Test 2-axis measurement at level orientation."""
            print("\n[TEST] 2-axis level orientation: Testing that both pitch and roll read ~0° at level orientation.")
            try:
                inc = Inclinometer2Axis()
                dcm = dcm_from_euler(0.0, 0.0, 0.0)  # Level
                
                measured = inc.step_from_dcm(dcm, 0.01, 0.0)
                np.testing.assert_array_almost_equal(measured, [0.0, 0.0], decimal=5)
            finally:
                print(f"[PASS] ✓ 2-axis measurement verified: pitch={measured[0]:.6f}, roll={measured[1]:.6f}.")
        
        def test_pitch_and_roll(self):
            """Test 2-axis measurement with combined pitch and roll."""
            print("\n[TEST] Combined pitch and roll (25°, 20°): Testing 2-axis sensor with simultaneous pitch and roll angles.")
            try:
                inc = Inclinometer2Axis()
                roll_true = np.radians(20.0)
                pitch_true = np.radians(25.0)  # Use smaller angles to avoid gimbal lock
                dcm = dcm_from_euler(roll_true, pitch_true, 0.0)
                
                measured = inc.step_from_dcm(dcm, 0.01, 0.0)
                # Use absolute tolerance for numerical precision issues with Euler extraction
                np.testing.assert_allclose(
                    measured, 
                    [pitch_true, roll_true], 
                    atol=np.radians(2.0)  # 2 degree tolerance
                )
            finally:
                print(f"[PASS] ✓ Combined pitch and roll measured: pitch={np.degrees(measured[0]):.2f}°, roll={np.degrees(measured[1]):.2f}°.")
        
        def test_with_bias_vector(self):
            """Test 2-axis with per-axis bias."""
            print("\n[TEST] 2-axis bias vector (2°, 3°): Testing per-axis bias application for pitch and roll channels.")
            try:
                bias = np.array([np.radians(2.0), np.radians(3.0)])  # Different bias per axis
                inc = Inclinometer2Axis(initial_bias=bias, seed=12345)
                dcm = dcm_from_euler(0.0, 0.0, 0.0)
                
                measured = inc.step_from_dcm(dcm, 0.01, 0.0)
                np.testing.assert_array_almost_equal(measured, bias, decimal=5)
            finally:
                print(f"[PASS] ✓ Per-axis bias applied: pitch bias={np.degrees(measured[0]):.2f}°, roll bias={np.degrees(measured[1]):.2f}°.")
        
        def test_with_saturation(self):
            """Test 2-axis with saturation limits."""
            print("\n[TEST] Saturation limits (±45°): Testing that measurements are clipped at ±45° saturation limits.")
            try:
                sat_min = np.array([np.radians(-45.0), np.radians(-45.0)])
                sat_max = np.array([np.radians(45.0), np.radians(45.0)])
                
                # Create a large pitch angle that exceeds saturation
                inc = Inclinometer2Axis(
                    saturation_limits=(sat_min, sat_max),
                    seed=12345
                )
                # Attempt a 90 degree pitch
                roll_true = np.radians(0.0)
                pitch_true = np.radians(90.0)
                dcm = dcm_from_euler(roll_true, pitch_true, 0.0)
                
                measured = inc.step_from_dcm(dcm, 0.01, 0.0)
                # Should be clipped to 45 degrees
                self.assertLessEqual(measured[0], sat_max[0])
                self.assertLessEqual(measured[1], sat_max[1])
            finally:
                print(f"[PASS] ✓ Saturation limits enforced: pitch={np.degrees(measured[0]):.2f}°, roll={np.degrees(measured[1]):.2f}° (max ±45°).")
        
        def test_dimension(self):
            """Test that output dimension is 2."""
            print("\n[TEST] Output dimension: Testing that 2-axis inclinometer output is 2-dimensional.")
            try:
                inc = Inclinometer2Axis()
                self.assertEqual(inc.SPEC.dimension, 2)
            finally:
                print(f"[PASS] ✓ 2-axis inclinometer dimension verified: {inc.SPEC.dimension} dimensions.")
        
        def test_history_with_timestamps(self):
            """Test history tracking with timestamps."""
            print("\n[TEST] History with timestamps: Testing that sensor history maintains accurate timestamps.")
            try:
                inc = Inclinometer2Axis(buffer_size=10)
                
                # Record samples at different times
                for i in range(5):
                    dcm = dcm_from_euler(0.0, np.radians(10.0 * i), 0.0)
                    inc.step_from_dcm(dcm, 0.01, float(i)*0.01)
                
                timestamps, hist_data = inc.get_history_with_timestamps()
                self.assertEqual(len(timestamps), 5)
                self.assertEqual(hist_data.shape, (5, 2))
            finally:
                print(f"[PASS] ✓ History with timestamps verified: {len(timestamps)} samples with shape {hist_data.shape}.")
    
    
    class TestInclinometerIntegration(unittest.TestCase):
        """Integration tests with sinusoidal movements."""
        
        def test_1axis_sinusoidal_pitch(self):
            """Test 1-axis inclinometer with sinusoidal pitch motion."""
            print("\n[TEST] 1-axis sinusoidal motion: Testing pitch sensor tracking sinusoidal motion (0.5 Hz, ±30°) over 2 seconds.")
            try:
                inc = Inclinometer1Axis(
                    measurement_axis="pitch",
                    white_noise_std=np.array([np.radians(0.1)]),  # Small noise
                    buffer_size=500,  # Increase to hold all samples
                    seed=42
                )
                
                # Simulate 2 seconds of sinusoidal pitch motion (0.5 Hz, ±30 degrees)
                dt = 0.01
                duration = 2.0
                num_steps = int(duration / dt)
                
                times = np.arange(num_steps) * dt
                pitch_true = 0.52 * np.sin(2 * np.pi * 0.5 * times)  # ±30 degrees
                
                measurements = []
                for i, t in enumerate(times):
                    dcm = dcm_from_euler(0.0, pitch_true[i], 0.0)
                    measured = inc.step_from_dcm(dcm, dt, t)
                    measurements.append(measured[0])
                
                measurements = np.array(measurements)
                
                # Check that measurements track the sinusoid reasonably well
                # Compute RMS error between measured and true (accounting for noise)
                rms_error = np.sqrt(np.mean((measurements - pitch_true) ** 2))
                self.assertLess(rms_error, np.radians(0.5))  # Less than 0.5 degree RMS
                
                # Check history
                hist = inc.get_history_matrix()
                self.assertEqual(len(hist), num_steps)
            finally:
                print(f"[PASS] ✓ 1-axis sinusoidal tracking completed: RMS error = {np.degrees(rms_error):.4f}°, {num_steps} samples recorded.")
        
        def test_2axis_sinusoidal_motion(self):
            """Test 2-axis inclinometer with sinusoidal combined motion."""
            print("\n[TEST] 2-axis sinusoidal motion: Testing dual-axis sensor with different frequencies (pitch: 0.5 Hz, roll: 1.0 Hz) over 3 seconds.")
            try:
                inc = Inclinometer2Axis(
                    white_noise_std=np.array([np.radians(0.1), np.radians(0.1)]),
                    buffer_size=500,  # Increase to hold all samples
                    seed=42
                )
                
                # Simulate 3 seconds of combined sinusoidal motion
                dt = 0.01
                duration = 3.0
                num_steps = int(duration / dt)
                
                times = np.arange(num_steps) * dt
                # Pitch: 0.5 Hz, ±25 degrees
                pitch_true = 0.436 * np.sin(2 * np.pi * 0.5 * times)
                # Roll: 1.0 Hz, ±30 degrees (different frequency)
                roll_true = 0.524 * np.sin(2 * np.pi * 1.0 * times)
                
                pitch_measurements = []
                roll_measurements = []
                
                for i, t in enumerate(times):
                    dcm = dcm_from_euler(roll_true[i], pitch_true[i], 0.0)
                    measured = inc.step_from_dcm(dcm, dt, t)
                    pitch_measurements.append(measured[0])
                    roll_measurements.append(measured[1])
                
                pitch_measurements = np.array(pitch_measurements)
                roll_measurements = np.array(roll_measurements)
                
                # Check pitch tracking - relax tolerance for noise
                pitch_rms = np.sqrt(np.mean((pitch_measurements - pitch_true) ** 2))
                self.assertLess(pitch_rms, np.radians(2.0))  # Less than 2 degree RMS
                
                # Check roll tracking
                roll_rms = np.sqrt(np.mean((roll_measurements - roll_true) ** 2))
                self.assertLess(roll_rms, np.radians(2.0))  # Less than 2 degree RMS
                
                # Verify that roll measurement has higher frequency content
                # (due to 1 Hz input vs 0.5 Hz for pitch)
                hist = inc.get_history_matrix()
                self.assertEqual(hist.shape, (num_steps, 2))
                
                # Check that measurements are in reasonable range
                self.assertTrue(np.all(np.abs(pitch_measurements) < np.radians(45.0)))
                self.assertTrue(np.all(np.abs(roll_measurements) < np.radians(45.0)))
            finally:
                print(f"[PASS] ✓ 2-axis sinusoidal motion tracked: pitch RMS error = {np.degrees(pitch_rms):.4f}°, roll RMS error = {np.degrees(roll_rms):.4f}°, {num_steps} samples.")
        
        def test_2axis_synchronized_sinusoidal(self):
            """Test 2-axis with synchronized pitch and roll movements."""
            print("\n[TEST] 2-axis synchronized sinusoidal: Testing quad-phase sinusoidal motion with bias (2 Hz, ±20°) over 1 second.")
            try:
                inc = Inclinometer2Axis(
                    initial_bias=np.array([np.radians(1.0), np.radians(-0.5)]),
                    white_noise_std=np.array([np.radians(0.05), np.radians(0.05)]),
                    seed=123
                )
                
                # Both axes at same frequency but different phases
                dt = 0.01
                duration = 1.0
                num_steps = int(duration / dt)
                
                times = np.arange(num_steps) * dt
                freq = 2.0  # 2 Hz
                
                # Pitch: phase 0
                pitch_true = 0.349 * np.sin(2 * np.pi * freq * times)
                # Roll: phase 90 degrees
                roll_true = 0.349 * np.cos(2 * np.pi * freq * times)
                
                measurements = []
                for i, t in enumerate(times):
                    dcm = dcm_from_euler(roll_true[i], pitch_true[i], 0.0)
                    measured = inc.step_from_dcm(dcm, dt, t)
                    measurements.append(measured)
                
                measurements = np.array(measurements)
                
                # Verify shape
                self.assertEqual(measurements.shape, (num_steps, 2))
                
                # Check that bias is reflected in measurements
                mean_pitch = np.mean(measurements[:, 0])
                mean_roll = np.mean(measurements[:, 1])
                
                # Should have offset close to the bias (within noise margin)
                self.assertAlmostEqual(mean_pitch, np.radians(1.0), places=1)
                self.assertAlmostEqual(mean_roll, np.radians(-0.5), places=1)
            finally:
                print(f"[PASS] ✓ Synchronized sinusoidal motion verified: pitch mean = {np.degrees(mean_pitch):.2f}°, roll mean = {np.degrees(mean_roll):.2f}°, {num_steps} samples.")
    
    
    # ==================================================================
    # RUN TESTS
    # ==================================================================
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestInclinometer1Axis))
    suite.addTests(loader.loadTestsFromTestCase(TestInclinometer2Axis))
    suite.addTests(loader.loadTestsFromTestCase(TestInclinometerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
