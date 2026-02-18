# -*- coding: utf-8 -*-
"""
Filename: GNSS.py
Description: Defines a GNSS/GPS receiver sensor that outputs 3D position and velocity.
"""

from .sensor_framework import SimulatedSensor, SensorSpec
import numpy as np
from typing import Optional

# ==============================================================================
# GNSS/GPS SENSORS
# ==============================================================================

class GNSSReceiver(SimulatedSensor):
    """
    GNSS/GPS receiver outputting 3D position + velocity.
    
    COORDINATE SYSTEM: +X = downrange, +Y = right, +Z = down (NED convention)
    Output: [x, y, z, vx, vy, vz] in local NED frame (meters, m/s)
    
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
        # GNSS noise characteristics (horizontal better than vertical)
        white_noise_std = np.array([
            position_noise_std, position_noise_std, position_noise_std * 2,
            velocity_noise_std, velocity_noise_std, velocity_noise_std
        ])
        
        super().__init__(
            sensor_id, 
            self.SPEC, 
            white_noise_std=white_noise_std,
            update_rate_hz=update_rate_hz,
            **kwargs
        )
    
    def step_from_truth(
        self, 
        position_nav: np.ndarray,
        velocity_nav: np.ndarray,
        dt: float,
        timestamp: float
    ) -> Optional[np.ndarray]:
        """Generate GNSS measurement if update period has elapsed, otherwise return None."""
        if not self.should_update(timestamp):
            return None
        
        true_state = np.concatenate([position_nav, velocity_nav])
        return super().step(true_state, dt, timestamp)