# -*- coding: utf-8 -*-
"""
Filename: GNSS.py
Description: Defines a GNSS/GPS receiver sensor that outputs 3D position and velocity.
"""

from sensor_framework import SimulatedSensor, SensorSpec
import numpy as np
from typing import Optional

# ==============================================================================
# GNSS/GPS SENSORS
# ==============================================================================

class GNSSReceiver(SimulatedSensor):
    """
    GNSS/GPS receiver outputting 3D position + velocity.
    6-DOF: [latitude, longitude, altitude, vel_north, vel_east, vel_down]
    
    Note: For simplicity, position is in meters (local NED frame).
    Real systems output lat/lon/alt which requires geodetic conversions.
    """
    SPEC = SensorSpec(
        dimension=6,
        units="mixed",
        description="GNSS position + velocity: [x, y, z, vx, vy, vz]",
        labels=["x", "y", "z", "vx", "vy", "vz"]
    )
    
    def __init__(
        self, 
        sensor_id: str = "gnss",
        position_noise_std: float = 2.0,  # meters (horizontal)
        velocity_noise_std: float = 0.1,  # m/s
        update_rate_hz: float = 10.0,
        **kwargs
    ):
        super().__init__(sensor_id, self.SPEC, **kwargs)
        
        self.update_rate_hz = update_rate_hz
        self.update_period = 1.0 / update_rate_hz
        self.last_update_time = 0.0
        
        # GNSS noise characteristics (horizontal better than vertical)
        self.white_noise_std = np.array([
            position_noise_std, position_noise_std, position_noise_std * 2,
            velocity_noise_std, velocity_noise_std, velocity_noise_std
        ])
    
    def step_from_truth(
        self, 
        position_nav: np.ndarray,
        velocity_nav: np.ndarray,
        dt: float,
        timestamp: float
    ) -> Optional[np.ndarray]:
        """Generate GNSS measurement."""
        if timestamp - self.last_update_time < self.update_period:
            return None
        
        self.last_update_time = timestamp
        
        true_state = np.concatenate([position_nav, velocity_nav])
        return super().step(true_state, dt, timestamp)