"""Sensor Fusion / Kalman Filter package.

Most client code should import conveniently from here rather than diving into
individual submodules. The core EKF engine and state definitions live in
``sensor_fusion.py``.

Example::

    from src.filters import (
        State,
        ExtendedKalmanFilter,
        NavigationPhysics,
        adapt_gnss_sensor,
        adapt_accel_sensor,
        adapt_inclinometer_sensor,
    )

"""

# re-export commonly used EKF classes and functions from submodules
from .sensor_fusion import (
    State,
    ExtendedKalmanFilter,
    NavigationPhysics,
    EKFSensorAdapter,
    adapt_gnss_sensor,
    adapt_accel_sensor,
    adapt_gyro_sensor,
    adapt_inclinometer_sensor,
    quaternion_to_euler,
    composite_adapter,
)

__all__ = [
    # State and EKF engine
    "State",
    "ExtendedKalmanFilter",
    "NavigationPhysics",
    # Sensor adapters
    "EKFSensorAdapter",
    "adapt_gnss_sensor",
    "adapt_accel_sensor",
    "adapt_gyro_sensor",
    "adapt_inclinometer_sensor",
    "composite_adapter",
    # Utilities
    "quaternion_to_euler",
]
