# -*- coding: utf-8 -*-
"""
Filename: imu_core.py
Description: A unified Object-Oriented framework for Inertial Measurement Units.
             Handles both:
             1. SIMULATION: Generating realistic data from physics (Truth -> Errors -> Data)
             2. HARDWARE: Ingesting real data from drivers (Raw -> Calibration -> Data)
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, Union, Dict, List
from enum import Enum

class SensorMode(Enum):
    SIMULATION = "SIMULATION"
    HARDWARE = "HARDWARE"

class BaseSensor:
    """
    The abstract foundation for any 3-axis inertial sensor.
    Handles the boilerplate of data management, timestamping, and history buffering.
    
    Attributes:
        sensor_id (str): Unique identifier for logging or multi-sensor setups.
        buffer_size (int): Max number of samples to keep in the ring buffer.
        history (deque): A sliding window of (timestamp, value) pairs.
        current_value (np.ndarray): The most recent [x, y, z] measurement.
    """
    def __init__(self, sensor_id: str, buffer_size: int = 100):
        self.sensor_id = sensor_id
        self.buffer_size = buffer_size
        
        # Ring buffer for history (useful for filtering/debugging)
        self.history = deque(maxlen=buffer_size)
        
        # Initialize state
        self.current_value: np.ndarray = np.zeros(3)
        self.current_timestamp: float = 0.0

    def _store(self, data: np.ndarray, timestamp: float):
        """
        Internal utility to update current state and append to the sliding window.
        Args:
            data (np.ndarray): 3-vector of sensor data.
            timestamp (float): System or simulation time.
        """
        self.current_value = data
        self.current_timestamp = timestamp
        self.history.append((timestamp, data))

    def get_latest(self) -> np.ndarray:
        """Access the most recent measurement vector."""
        return self.current_value

    def get_history_matrix(self) -> np.ndarray:
        """
        Converts the history buffer into a 2D NumPy array (N x 3).
        Useful for plotting or feeding into batch filters.
        """
        if not self.history:
            return np.empty((0, 3))
        # Extract just the data vectors, ignore timestamps for matrix
        return np.array([item[1] for item in self.history])


class SimulatedSensor(BaseSensor):
    """
    A 'Digital Twin' sensor that models physical imperfections.
    It takes 'Truth' as an input and applies a sequence of error transformations.
    
    Error Order of Operations:
    1. Misalignment (DCM) -> 2. Scale Factors -> 3. Bias Drift -> 4. Noise -> 5. Saturation -> 6. Quantization
    """
    def __init__(
        self, 
        sensor_id: str = "sim_sensor",
        dcm_body_to_sensor: np.ndarray = np.eye(3),
        initial_bias: np.ndarray = np.zeros(3),
        bias_instability_std: float = 0.0,
        white_noise_std: float = 0.0,
        scale_factors: np.ndarray = np.ones(3),
        saturation_limit: Optional[float] = None,
        quantization_bits: Optional[int] = None,
        quantization_range: float = 100.0,
        seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(sensor_id, **kwargs)
        
        # -- Physical Configuration --
        # Frame Transformation: Represents how the sensor chip is physically skewed 
        # relative to the vehicle's primary axes.
        self.dcm_body_to_sensor = np.asarray(dcm_body_to_sensor, dtype=float)

        # -- Error Model Parameters --
        # Deterministic Errors: Scaling and static initial bias
        self.bias = np.asarray(initial_bias, dtype=float)
        self.bias_instability_std = bias_instability_std

        # Stochastic Errors: Random walk (drift) and White Noise
        self.white_noise_std = white_noise_std
        self.scale_factors = np.asarray(scale_factors, dtype=float)

        # Nonlinearities: clipping and resolution limits
        self.saturation_limit = abs(saturation_limit) if saturation_limit else None
        
        # -- Quantization --
        self.quant_step = 0.0
        self.quant_max = abs(quantization_range)
        if quantization_bits:
            # LSB Size = Total Range / (2 ^ bit_depth)
            self.quant_step = (2.0 * self.quant_max) / (2 ** quantization_bits)

        # Set random number generator for reproducibility
        self.rng = np.random.default_rng(seed)

    def step(self, true_signal_body: np.ndarray, dt: float, timestamp: float) -> np.ndarray:
        """
        Corrupts physical truth to generate a realistic sensor reading.
        Args:
            true_signal_body (np.ndarray): The perfect physical quantity (e.g., acceleration in body frame).
            dt (float): Change in time (integration step for bias drift).
            timestamp (float): Current simulation time.
        """
        # 1. Coordinate Misalignment: Rotate truth into the sensor's skewed frame
        signal_s = self.dcm_body_to_sensor @ true_signal_body

        # 2. Bias Instability (Random Walk): 
        # b_k = b_{k-1} + N(0, sigma^2 * dt)
        if self.bias_instability_std > 0.0:
            self.bias += self.bias_instability_std * np.sqrt(dt) * self.rng.standard_normal(3)

        # 3. White Noise: High-frequency jitter
        noise = np.zeros(3)
        if self.white_noise_std > 0.0:
            noise = self.white_noise_std * self.rng.standard_normal(3)

        # 4. Scale + Bias + Noise
        measured = (self.scale_factors * signal_s) + self.bias + noise

        # 5. Saturation: The physical limit of the MEMS structure or internal amplifier
        if self.saturation_limit:
            measured = np.clip(measured, -self.saturation_limit, self.saturation_limit)

        # 6. Quantization: The Analog-to-Digital Converter
        if self.quant_step > 0:
            measured = np.round(measured / self.quant_step) * self.quant_step
            measured = np.clip(measured, -self.quant_max, self.quant_max)

        # Store and Return
        self._store(measured, timestamp)
        return measured


class HardwareSensor(BaseSensor):
    """
    A proxy for a physical sensor. 
    Instead of adding errors, it attempts to remove them using calibration parameters.
    """
    def __init__(
        self, 
        sensor_id: str = "real_sensor",
        calibration_bias: np.ndarray = np.zeros(3),
        calibration_scale: np.ndarray = np.ones(3),
        dcm_sensor_to_body: np.ndarray = np.eye(3),
        **kwargs
    ):
        super().__init__(sensor_id, **kwargs)

        # Calibration: Used to correct raw measurements based on lab characterization
        self.calib_bias = np.asarray(calibration_bias, dtype=float)
        self.calib_scale = np.asarray(calibration_scale, dtype=float)

        # Alignment: Rotates sensor-local data back into body frame
        self.dcm_sensor_to_body = np.asarray(dcm_sensor_to_body, dtype=float)

    def ingest(self, raw_data: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Processes a raw packet from hardware.
        1. Strips estimated bias. 2. Rescales. 3. Rotates to Body Frame.
        """
        raw_vec = np.asarray(raw_data, dtype=float)
        
        # 1. Apply Calibration (inverse of simulation model)
        # Estimate = (Raw - Bias) / Scale
        # Note: Avoid divide by zero
        calibrated_s = (raw_vec - self.calib_bias) / np.where(self.calib_scale != 0, self.calib_scale, 1.0)
        
        # 2. Align to Body Frame
        # Real sensors are often mounted weirdly. We rotate them into the vehicle frame here.
        body_frame_data = self.dcm_sensor_to_body @ calibrated_s
        
        self._store(body_frame_data, timestamp)
        return body_frame_data

    def recalibrate(self, new_bias: np.ndarray, new_scale: np.ndarray):
        """Update calibration parameters on-the-fly."""
        self.calib_bias = np.asarray(new_bias, dtype=float)
        self.calib_scale = np.asarray(new_scale, dtype=float)


# --- Concrete Implementations ---

class AccelerometerSim(SimulatedSensor):
    """
    Simulated Accelerometer that accounts for dynamics and mounting position.
    """
    def __init__(self, pos_sensor_body: np.ndarray, **kwargs):
        super().__init__(sensor_id="accel_sim", **kwargs)
        # Offset from Pivot / Center of Gravity (m)
        self.pos_sensor_body = np.asarray(pos_sensor_body, dtype=float)

    def compute_specific_force(self, accel_cg_n, omega_b, alpha_b, R_bn, gravity_n):
        """
        Calculates the True Specific Force at the sensor location.
        Uses the Transport Theorem to account for accelerations experienced 
        due to being offset from the pivot/CG.
        """
        R_nb = R_bn.T
        accel_cg_b = R_nb @ accel_cg_n
        
        # Tangential Acceleration (Euler): alpha x r
        accel_euler = np.cross(alpha_b, self.pos_sensor_body)

        # Centripetal Acceleration: omega x (omega x r)
        accel_centripetal = np.cross(omega_b, np.cross(omega_b, self.pos_sensor_body))
        
        # Gravity in Body Frame
        gravity_b = R_nb @ gravity_n
        
        # True Specific Force = Kinematic Accel - Gravity Accel
        return accel_cg_b + accel_euler + accel_centripetal - gravity_b


class GyroscopeSim(SimulatedSensor):
    """
    Simulated Gyroscope.
    Acts as a distinct class to allow for future G-sensitivity or 
    vibration rectification implementations.
    """
    def __init__(self, **kwargs):
        super().__init__(sensor_id="gyro_sim", **kwargs)


class MagnetometerSim(SimulatedSensor):
    """
    Simulated Magnetometer.
    Acts as a distinct class to allow for future Soft-Iron / Hard-Iron 
    distortion modeling.
    """
    def __init__(self, **kwargs):
        super().__init__(sensor_id="mag_sim", **kwargs)


class IMU:
    """
    High-level IMU controller. 
    Encapsulates a triad of Accelerometers, Gyroscopes, and Magnetometers.
    Can be initialized in SIMULATION or HARDWARE mode.
    """
    def __init__(
        self,
        id: str = "imu_1",
        mode: Union[SensorMode, str] = SensorMode.SIMULATION,
        accel_config: Dict = {},
        gyro_config: Dict = {},
        mag_config: Dict = {},
        time: float = 0.0
    ):
        """
        Args:
            mode: 'SIMULATION' or 'HARDWARE'.
            accel_config: Params for Accel (e.g., pos_sensor_body, noise_std).
            gyro_config: Params for Gyro (e.g., bias_instability_std).
            mag_config: Params for Magnetometer.
        """
        if isinstance(mode, str):
            mode = SensorMode[mode.upper()]
        self.mode = mode
        self.id = id
        
        # Type hints for the digital twin architecture: sensors switch between 
        # physics-based simulators or calibrated hardware interfaces based on mode.
        self.accel: Union[AccelerometerSim, HardwareSensor]
        self.gyro: Union[GyroscopeSim, HardwareSensor]
        self.mag: Union[MagnetometerSim, HardwareSensor]

        if self.mode == SensorMode.SIMULATION:
            # Initialize Physics-based Simulators
            # Extract position for lever arm, default to 0 (pivot/CG)
            pos = accel_config.pop('pos_sensor_body', np.zeros(3))
            
            self.accel = AccelerometerSim(pos_sensor_body=pos, **accel_config)
            self.gyro = GyroscopeSim(sensor_id="gyro_sim", **gyro_config)
            self.mag = MagnetometerSim(sensor_id="mag_sim", **mag_config)
            
        else:
            # Initialize Hardware Containers
            self.accel = HardwareSensor(sensor_id=f"accel_real_{self.id}", **accel_config)
            self.gyro = HardwareSensor(sensor_id=f"gyro_real_{self.id}", **gyro_config)
            self.mag = HardwareSensor(sensor_id=f"mag_real_{self.id}", **mag_config)
            
        self.time = time

    # --- SIMULATION INTERFACE ---
    def update_sim(
        self,
        dt: float,
        accel_cg_nav: np.ndarray,
        omega_body: np.ndarray,
        alpha_body: np.ndarray,
        dcm_body_to_nav: np.ndarray,
        gravity_nav: np.ndarray = np.array([0, 0, 9.81]),
        mag_field_nav: np.ndarray = np.array([20.0, 0.0, 40.0]) # uT example
    ):
        """
        Drive the simulation forward by processing kinematic ground truth into sensor readings.

        Note: This function is only usable in SIMULATION mode. It calculates the specific force 
        at the offset sensor location (lever-arm) and applies the stochastic error models.

        Args:
            dt (float): Integration timestep (seconds).
            accel_cg_nav (np.ndarray): Linear acceleration of the vehicle CG in the Navigation frame (m/s^2).
            omega_body (np.ndarray): Angular velocity of the body relative to Navigation, in Body frame (rad/s).
            alpha_body (np.ndarray): Angular acceleration of the body in Body frame (rad/s^2).
            dcm_body_to_nav (np.ndarray): 3x3 Rotation matrix representing vehicle attitude.
            gravity_nav (np.ndarray): Local gravity vector in Navigation frame (m/s^2).
            mag_field_nav (np.ndarray): Local magnetic field vector in Navigation frame (e.g., microTeslas).

        Returns:
            Dict[str, np.ndarray]: The latest corrupted sensor measurements.
        """
        if self.mode != SensorMode.SIMULATION:
            raise RuntimeError("Cannot call update_sim() when IMU is in HARDWARE mode.")
        
        self.time += dt
        
        # 1. Accelerometer Physics
        f_true = self.accel.compute_specific_force(
            accel_cg_nav, omega_body, alpha_body, dcm_body_to_nav, gravity_nav
        )
        self.accel.step(f_true, dt, self.time)
        
        # 2. Gyro Physics (Rate)
        self.gyro.step(omega_body, dt, self.time)
        
        # 3. Magnetometer Physics
        # Rotate Nav magnetic field into Body frame
        mag_true_body = dcm_body_to_nav.T @ mag_field_nav
        self.mag.step(mag_true_body, dt, self.time)
        
        return self.get_readings()

    # --- HARDWARE INTERFACE ---
    def ingest_data(
        self,
        accel_vec: np.ndarray,
        gyro_vec: np.ndarray,
        mag_vec: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ):
        """
        Ingest and process raw data from physical hardware.

        Note: This function is only usable in HARDWARE mode. It applies inverse-model calibration 
        (bias/scale correction) and rotates data into the vehicle's body frame.

        Args:
            accel_vec (np.ndarray): Raw acceleration vector from hardware driver [x, y, z].
            gyro_vec (np.ndarray): Raw angular rate vector from hardware driver [x, y, z].
            mag_vec (Optional[np.ndarray]): Raw magnetic field vector [x, y, z].
            timestamp (Optional[float]): The hardware-provided timestamp. If None, 
                auto-increments based on a default 100Hz assumption.

        Returns:
            Dict[str, np.ndarray]: The latest calibrated and aligned sensor measurements.
        """
        if self.mode != SensorMode.HARDWARE:
            raise RuntimeError("Cannot call ingest_data() when IMU is in SIMULATION mode.")
            
        if timestamp is None:
            self.time += 0.01 # Assume 100Hz if not provided
            timestamp = self.time
        else:
            self.time = timestamp

        self.accel.ingest(accel_vec, timestamp)
        self.gyro.ingest(gyro_vec, timestamp)
        
        if mag_vec is not None:
            self.mag.ingest(mag_vec, timestamp)
            
        return self.get_readings()

    # --- COMMON INTERFACE ---
    def get_readings(self) -> Dict[str, np.ndarray]:
        """
        Retrieve the most recent measurement set from the sensors.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing 'timestamp', 'accel', 'gyro', and 'mag'.
        """
        return {
            'timestamp': self.time,
            'accel': self.accel.get_latest(),
            'gyro': self.gyro.get_latest(),
            'mag': self.mag.get_latest()
        }

    def get_history(self) -> Dict[str, np.ndarray]:
        """
        Retrieve the full history for all sensors stored in the internal ring buffers.

        Returns:
            Dict[str, np.ndarray]: A dictionary where each key contains an (N x 3) matrix 
                of historical data.
        """
        return {
            'accel': self.accel.get_history_matrix(),
            'gyro': self.gyro.get_history_matrix(),
            'mag': self.mag.get_history_matrix()
        }