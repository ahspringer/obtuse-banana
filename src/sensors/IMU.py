# -*- coding: utf-8 -*-
"""
Filename: imu_sim.py
Description: Object-Oriented simulation of Inertial Measurement Unit (IMU) sensors.
             Includes detailed error models for Accelerometers and Gyroscopes,
             including Saturation and Quantization (ADC) effects.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict

class InertialSensorModel:
    """
    Base class for inertial sensors (Accelerometers and Gyroscopes).
    Handles the common error sources: Bias, Random Walk, White Noise, Scale Factors,
    Saturation, and Quantization.
    """
    
    def __init__(
        self,
        dcm_body_to_sensor: np.ndarray = np.eye(3),
        initial_bias: np.ndarray = np.zeros(3),
        bias_instability_std: float = 0.0,
        white_noise_std: float = 0.0,
        scale_factors: np.ndarray = np.ones(3),
        saturation_limit: Optional[float] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the sensor error model.

        Args:
            dcm_body_to_sensor (np.ndarray): 3x3 Direction Cosine Matrix (Body -> Sensor).
            initial_bias (np.ndarray): Initial static bias (3-vector).
            bias_instability_std (float): Standard deviation of bias random walk (per sqrt(sec)).
            white_noise_std (float): Standard deviation of white noise.
            scale_factors (np.ndarray): Scale factor per axis (usually close to 1.0).
            saturation_limit (float, optional): Magnitude at which sensor saturates (clips). None = Infinite.
            seed (int, optional): Random seed for reproducibility.
        """
        # Coordinate transformation
        self.dcm_body_to_sensor = np.asarray(dcm_body_to_sensor, dtype=float)

        # Error Parameters
        self.bias = np.asarray(initial_bias, dtype=float)
        self.bias_instability_std = bias_instability_std
        self.white_noise_std = white_noise_std
        self.scale_factors = np.asarray(scale_factors, dtype=float)
        
        # Saturation (clipping)
        self.saturation_limit = abs(saturation_limit) if saturation_limit is not None else None

        # Quantization (ADC) settings - defaults to off
        self.quantization_enabled = False
        self.quant_bit_depth = 32
        self.quant_max_range = 100.0
        self.quant_step_size = 0.0

        # Random Number Generator
        self.rng = np.random.default_rng(seed)

    def enable_quantization(self, bit_depth: int = 32, max_range: Optional[float] = None):
        """
        Enable ADC quantization simulation.

        Args:
            bit_depth (int): Resolution of the ADC (e.g., 16, 24).
            max_range (float, optional): The full-scale range (+/- value) of the ADC. 
                                         If None, defaults to saturation_limit.
                                         If both are None, raises ValueError.
        """
        if max_range is None:
            if self.saturation_limit is not None:
                max_range = self.saturation_limit
            else:
                raise ValueError("Must provide max_range if saturation_limit is not set.")
        
        self.quantization_enabled = True
        self.quant_bit_depth = int(bit_depth)
        self.quant_max_range = abs(float(max_range))
        
        # Calculate LSB (Least Significant Bit) size
        # Assuming signed symmetric range: 2 * range / 2^bits
        self.quant_step_size = (2.0 * self.quant_max_range) / (2 ** self.quant_bit_depth)

    def apply_errors(self, true_signal_body: np.ndarray, dt: float) -> np.ndarray:
        """
        Applies misalignment, scale factors, bias drift, noise, saturation, and quantization.
        
        Args:
            true_signal_body (np.ndarray): The true physical quantity in the Body frame.
            dt (float): Timestep in seconds.

        Returns:
            np.ndarray: The corrupted measurement in the Sensor frame.
        """
        # 1. Rotate True Signal to Sensor Frame (Misalignment)
        # We rotate first so that errors (which exist on the sensor axis) are applied correctly.
        signal_sensor_frame = self.dcm_body_to_sensor @ true_signal_body

        # 2. Update Bias (Random Walk)
        if self.bias_instability_std > 0.0:
            walk_step = self.bias_instability_std * np.sqrt(dt) * self.rng.standard_normal(3)
            self.bias += walk_step

        # 3. Generate White Noise
        noise = np.zeros(3)
        if self.white_noise_std > 0.0:
            noise = self.white_noise_std * self.rng.standard_normal(3)

        # 4. Apply Scale Factors, Bias, and Noise (Sensor Frame)
        signal_corrupted = (self.scale_factors * signal_sensor_frame) + self.bias + noise

        # 5. Saturation (Clipping)
        if self.saturation_limit is not None:
            signal_corrupted = np.clip(
                signal_corrupted, 
                -self.saturation_limit, 
                self.saturation_limit
            )

        # 6. Quantization (ADC Simulation)
        if self.quantization_enabled and self.quant_step_size > 0:
            # Round to nearest step
            signal_corrupted = (
                np.round(signal_corrupted / self.quant_step_size) * self.quant_step_size
            )
            
            # Re-clip to ensure we didn't quantize outside the range 
            # (though physically the ADC would just saturate at the max code)
            signal_corrupted = np.clip(
                signal_corrupted,
                -self.quant_max_range,
                self.quant_max_range
            )

        return signal_corrupted


class Accelerometer(InertialSensorModel):
    """
    Simulates a 3-axis Accelerometer.
    """

    def __init__(
        self,
        pos_sensor_body: np.ndarray,
        **kwargs
    ):
        """
        Args:
            pos_sensor_body (np.ndarray): [x, y, z] position of sensor relative to CG in Body Frame (meters).
            **kwargs: Passed to InertialSensorModel.
        """
        super().__init__(**kwargs)
        self.pos_sensor_body = np.asarray(pos_sensor_body, dtype=float)

    def step(
        self,
        accel_cg_nav: np.ndarray,
        omega_body: np.ndarray,
        alpha_body: np.ndarray,
        dcm_body_to_nav: np.ndarray,
        gravity_nav: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Calculate accelerometer measurement based on kinematic state.
        """
        
        # Transpose DCM to get Nav -> Body
        dcm_nav_to_body = dcm_body_to_nav.T

        # 1. Rotate linear acceleration (CG) into Body frame
        accel_cg_body = dcm_nav_to_body @ accel_cg_nav

        # 2. Lever-Arm Effects
        accel_euler = np.cross(alpha_body, self.pos_sensor_body)
        accel_centripetal = np.cross(omega_body, np.cross(omega_body, self.pos_sensor_body))

        # 3. Gravity in Body Frame
        gravity_body = dcm_nav_to_body @ gravity_nav

        # 4. True Specific Force (Body Frame)
        f_true_body = accel_cg_body + accel_euler + accel_centripetal - gravity_body

        # 5. Apply Sensor Errors
        return self.apply_errors(f_true_body, dt)


class Gyroscope(InertialSensorModel):
    """
    Simulates a 3-axis Gyroscope.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, omega_body: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculate gyroscope measurement.
        """
        return self.apply_errors(omega_body, dt)


class IMU:
    """
    Container class for a complete 6-DOF IMU (Accelerometer Triad + Gyroscope Triad).
    """
    
    def __init__(
        self,
        accel_config: Dict = {},
        gyro_config: Dict = {}
    ):
        """
        Initialize the IMU with configuration dictionaries for the sensors.
        
        Args:
            accel_config (Dict): kwargs passed to Accelerometer constructor.
            gyro_config (Dict): kwargs passed to Gyroscope constructor.
        """
        # Ensure position is in accel config, default to zero if missing
        if 'pos_sensor_body' not in accel_config:
            accel_config['pos_sensor_body'] = np.zeros(3)
            
        self.accel = Accelerometer(**accel_config)
        self.gyro = Gyroscope(**gyro_config)

    def step(
        self,
        accel_cg_nav: np.ndarray,
        omega_body: np.ndarray,
        alpha_body: np.ndarray,
        dcm_body_to_nav: np.ndarray,
        gravity_nav: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance the simulation for both Accel and Gyro.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (accel_meas, gyro_meas)
        """
        
        a_meas = self.accel.step(
            accel_cg_nav=accel_cg_nav,
            omega_body=omega_body,
            alpha_body=alpha_body,
            dcm_body_to_nav=dcm_body_to_nav,
            gravity_nav=gravity_nav,
            dt=dt
        )
        
        w_meas = self.gyro.step(
            omega_body=omega_body, 
            dt=dt
        )
        
        return a_meas, w_meas

    def enable_quantization(self, accel_bits: int = 32, gyro_bits: int = 32, 
                           accel_range: Optional[float] = None, gyro_range: Optional[float] = None):
        """
        Helper to enable quantization on both sensors simultaneously.
        """
        self.accel.enable_quantization(bit_depth=accel_bits, max_range=accel_range)
        self.gyro.enable_quantization(bit_depth=gyro_bits, max_range=gyro_range)