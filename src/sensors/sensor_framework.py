# -*- coding: utf-8 -*-
"""
Filename: sensors.py
Description: Extended Object-Oriented framework for multi-modal sensor fusion.
             Supports variable-dimension sensors:
             - IMU sensors (3-axis accelerometers, gyros, magnetometers)
             - Quaternion-output gyroscopes
             - Inclinometers (1-axis or 2-axis)
             - Vision-based pose estimators
             - GPS/GNSS receivers
             
             Each sensor operates in either SIMULATION or HARDWARE mode.
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, Union, Dict, List, Any
from enum import Enum
from dataclasses import dataclass

class SensorMode(Enum):
    SIMULATION = "SIMULATION"
    HARDWARE = "HARDWARE"

@dataclass
class SensorSpec:
    """Specification for sensor output format and interpretation."""
    dimension: int
    units: str
    description: str
    labels: List[str]  # e.g., ['x', 'y', 'z'] or ['qw', 'qx', 'qy', 'qz']

# ==============================================================================
# ABSTRACT BASE CLASSES
# ==============================================================================

class BaseSensor:
    """
    Dimension-agnostic foundation for any sensor.
    Handles data management, timestamping, and history buffering for outputs of any size.
    
    Attributes:
        sensor_id (str): Unique identifier for logging or multi-sensor setups.
        spec (SensorSpec): Output format specification.
        buffer_size (int): Max number of samples to keep in the ring buffer.
        history (deque): A sliding window of (timestamp, value) pairs.
        current_value (np.ndarray): The most recent measurement.
        update_rate_hz (float, optional): If set, limits sensor update rate to this frequency.
    """
    def __init__(
        self, 
        sensor_id: str, 
        spec: SensorSpec,
        buffer_size: int = 100,
        update_rate_hz: Optional[float] = None
    ):
        self.sensor_id = sensor_id
        self.spec = spec
        self.buffer_size = buffer_size
        
        # Ring buffer for history
        self.history = deque(maxlen=buffer_size)
        
        # Initialize state with appropriate dimension
        self.current_value: np.ndarray = np.zeros(spec.dimension)
        self.current_timestamp: float = 0.0
        
        # Update rate limiting (Hz). If None, the sensor updates every step.
        self.update_rate_hz = update_rate_hz
        if update_rate_hz is not None:
            self.update_period = 1.0 / update_rate_hz
            self.last_update_time = 0.0
        else:
            self.update_period = 0.0
            self.last_update_time = 0.0

    def _store(self, data: np.ndarray, timestamp: float):
        """
        Internal utility to update current state and append to the sliding window.
        Args:
            data (np.ndarray): Sensor measurement of shape specified by spec.dimension.
            timestamp (float): System or simulation time.
        """
        if data.shape[0] != self.spec.dimension:
            raise ValueError(
                f"Data dimension {data.shape[0]} doesn't match spec dimension {self.spec.dimension}"
            )
        self.current_value = data.copy()
        self.current_timestamp = timestamp
        self.history.append((timestamp, data.copy()))

    def get_latest(self) -> np.ndarray:
        """Access the most recent measurement vector."""
        return self.current_value
    
    def should_update(self, timestamp: float) -> bool:
        """
        Check if enough time has passed since the last update to generate a new measurement.
        Used for limiting sensor update rates (e.g., 10 Hz GPS, 100 Hz IMU, 400 Hz accelerometer).
        
        Args:
            timestamp: Current simulation or system time (seconds)
            
        Returns:
            True if update should occur, False if not enough time has passed.
            Always returns True if no update_rate_hz was specified.
        """
        if self.update_period == 0.0:
            # No rate limiting
            return True
        
        if timestamp - self.last_update_time >= self.update_period:
            self.last_update_time = timestamp
            return True
        return False

    def get_history_matrix(self) -> np.ndarray:
        """
        Converts the history buffer into a 2D NumPy array (N x D).
        Where D is the sensor dimension.
        """
        if not self.history:
            return np.empty((0, self.spec.dimension))
        return np.array([item[1] for item in self.history])
    
    def get_history_with_timestamps(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns separate arrays for timestamps and data.
        Returns:
            Tuple of (timestamps (N,), data (N x D))
        """
        if not self.history:
            return np.empty(0), np.empty((0, self.spec.dimension))
        timestamps = np.array([item[0] for item in self.history])
        data = np.array([item[1] for item in self.history])
        return timestamps, data


class SimulatedSensor(BaseSensor):
    """
    Generic simulated sensor with configurable error models and optional update rate limiting.
    
    Error chain: Truth -> Scale -> Bias -> Noise -> Saturation -> Quantization -> Output
    
    Update Rate:
        All sensors can optionally have a limited update rate (e.g., 10 Hz for GPS, 100 Hz for IMU).
        If update_rate_hz is not None, the sensor will return None until the update period has elapsed.
    """
    def __init__(
        self, 
        sensor_id: str,
        spec: SensorSpec,
        initial_bias: Optional[np.ndarray] = None,
        bias_instability_std: Optional[np.ndarray] = None,
        white_noise_std: Optional[np.ndarray] = None,
        scale_factors: Optional[np.ndarray] = None,
        saturation_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        quantization_bits: Optional[int] = None,
        quantization_range: Optional[float] = None,
        update_rate_hz: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        # Pass update_rate_hz to parent BaseSensor
        super().__init__(sensor_id, spec, update_rate_hz=update_rate_hz, **kwargs)
        
        # Initialize error parameters with proper dimensions
        dim = spec.dimension
        
        self.bias = np.zeros(dim) if initial_bias is None else np.asarray(initial_bias, dtype=float)
        self.bias_instability_std = (
            np.zeros(dim) if bias_instability_std is None 
            else np.asarray(bias_instability_std, dtype=float)
        )
        self.white_noise_std = (
            np.zeros(dim) if white_noise_std is None 
            else np.asarray(white_noise_std, dtype=float)
        )
        self.scale_factors = (
            np.ones(dim) if scale_factors is None 
            else np.asarray(scale_factors, dtype=float)
        )
        
        # Saturation: (min_values, max_values) per dimension
        self.saturation_limits = saturation_limits
        
        # Quantization
        self.quant_step = 0.0
        if quantization_bits and quantization_range:
            self.quant_step = (2.0 * quantization_range) / (2 ** quantization_bits)
            self.quant_max = abs(quantization_range)
        else:
            self.quant_max = None
        
        self.rng = np.random.default_rng(seed)
    
    def step(self, true_signal: np.ndarray, dt: float, timestamp: float) -> np.ndarray:
        """
        Apply error model to truth signal.
        
        Args:
            true_signal: The perfect physical measurement
            dt: Time step for bias random walk integration
            timestamp: Current time
        """
        # 1. Bias Random Walk
        if np.any(self.bias_instability_std > 0):
            self.bias += self.bias_instability_std * np.sqrt(dt) * self.rng.standard_normal(self.spec.dimension)
        
        # 2. White Noise
        noise = np.zeros(self.spec.dimension)
        if np.any(self.white_noise_std > 0):
            noise = self.white_noise_std * self.rng.standard_normal(self.spec.dimension)
        
        # 3. Scale + Bias + Noise
        measured = (self.scale_factors * true_signal) + self.bias + noise
        
        # 4. Saturation
        if self.saturation_limits:
            min_vals, max_vals = self.saturation_limits
            measured = np.clip(measured, min_vals, max_vals)
        
        # 5. Quantization
        if self.quant_step > 0:
            measured = np.round(measured / self.quant_step) * self.quant_step
            if self.quant_max:
                measured = np.clip(measured, -self.quant_max, self.quant_max)
        
        self._store(measured, timestamp)
        return measured


class HardwareSensor(BaseSensor):
    """
    Generic hardware sensor with calibration.
    Applies inverse error model: Raw -> Calibration -> Output
    """
    def __init__(
        self, 
        sensor_id: str,
        spec: SensorSpec,
        calibration_bias: Optional[np.ndarray] = None,
        calibration_scale: Optional[np.ndarray] = None,
        **kwargs
    ):
        super().__init__(sensor_id, spec, **kwargs)
        
        dim = spec.dimension
        self.calib_bias = (
            np.zeros(dim) if calibration_bias is None 
            else np.asarray(calibration_bias, dtype=float)
        )
        self.calib_scale = (
            np.ones(dim) if calibration_scale is None 
            else np.asarray(calibration_scale, dtype=float)
        )
    
    def ingest(self, raw_data: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Process raw hardware data through calibration.
        """
        raw_vec = np.asarray(raw_data, dtype=float)
        
        # Apply calibration
        calibrated = (raw_vec - self.calib_bias) / np.where(
            self.calib_scale != 0, self.calib_scale, 1.0
        )
        
        self._store(calibrated, timestamp)
        return calibrated
    
    def recalibrate(self, new_bias: np.ndarray, new_scale: np.ndarray):
        """Update calibration parameters."""
        self.calib_bias = np.asarray(new_bias, dtype=float)
        self.calib_scale = np.asarray(new_scale, dtype=float)


# ==============================================================================
# COORDINATE FRAME AWARE SENSORS
# ==============================================================================

class FrameAwareSensor(SimulatedSensor):
    """
    Base for sensors that need coordinate frame transformations.
    Adds DCM (Direction Cosine Matrix) support for mounting alignment.
    """
    def __init__(
        self,
        sensor_id: str,
        spec: SensorSpec,
        dcm_body_to_sensor: Optional[np.ndarray] = None,
        **kwargs
    ):
        super().__init__(sensor_id, spec, **kwargs)
        self.dcm_body_to_sensor = (
            np.eye(3) if dcm_body_to_sensor is None 
            else np.asarray(dcm_body_to_sensor, dtype=float)
        )
    
    def step_with_frame(
        self, 
        true_signal_body: np.ndarray, 
        dt: float, 
        timestamp: float
    ) -> np.ndarray:
        """Apply frame transformation before error model."""
        # Rotate to sensor frame
        signal_sensor = self.dcm_body_to_sensor @ true_signal_body
        # Apply error model
        return super().step(signal_sensor, dt, timestamp)


# ==============================================================================
# SENSOR SUITE CONTAINER
# ==============================================================================

class SensorSuite:
    """
    High-level container for multiple heterogeneous sensors.
    Manages a collection of sensors operating in the same mode (SIM or HARDWARE).
    """
    def __init__(
        self,
        suite_id: str = "sensor_suite",
        mode: Union[SensorMode, str] = SensorMode.SIMULATION,
        time: float = 0.0
    ):
        if isinstance(mode, str):
            mode = SensorMode[mode.upper()]
        
        self.suite_id = suite_id
        self.mode = mode
        self.time = time
        
        # Dictionary of sensors by type and id
        self.sensors: Dict[str, BaseSensor] = {}
    
    def add_sensor(self, sensor_name: str, sensor: BaseSensor):
        """Add a sensor to the suite."""
        self.sensors[sensor_name] = sensor
    
    def get_sensor(self, sensor_name: str) -> BaseSensor:
        """Retrieve a specific sensor."""
        return self.sensors.get(sensor_name)
    
    def get_all_readings(self) -> Dict[str, Dict[str, Any]]:
        """
        Get latest readings from all sensors.
        Returns:
            Dict mapping sensor_name -> {timestamp, data, spec}
        """
        readings = {}
        for name, sensor in self.sensors.items():
            readings[name] = {
                'timestamp': sensor.current_timestamp,
                'data': sensor.get_latest(),
                'spec': sensor.spec
            }
        return readings
    
    def get_all_histories(self) -> Dict[str, np.ndarray]:
        """Get history matrices for all sensors."""
        return {
            name: sensor.get_history_matrix() 
            for name, sensor in self.sensors.items()
        }


if __name__ == "__main__":
    print('This is the sensor framework module. Please run sim.py for a demonstration of the sensor suite in action.')