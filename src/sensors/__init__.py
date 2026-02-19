"""Sensors package.

Most client code should import conveniently from here rather than diving into
individual submodules. The IMU class is in ``IMU.py``, the GNSS receiver is in ``GNSS.py``,
and the inclinometer models are in ``inclinometers.py``.

Example::

    from src.sensors import Accelerometer3Axis, GyroscopeQuaternion, Inclinometer2Axis, GNSSReceiver

"""

# re-export commonly used sensor classes from submodules

from .IMU import (
    Accelerometer3Axis,
    Gyroscope3Axis,
    GyroscopeQuaternion,
    Magnetometer3Axis,
    BNO085IMU
)

from .inclinometers import (
    Inclinometer1Axis,
    Inclinometer2Axis
)

from .GNSS import GNSSReceiver

__all__ = [
    # IMU sensors
    "Accelerometer3Axis",
    "Gyroscope3Axis",
    "GyroscopeQuaternion",
    "Magnetometer3Axis",
    "BNO085IMU",
    # inclinometers
    "Inclinometer1Axis",
    "Inclinometer2Axis",
    # GNSS receiver
    "GNSSReceiver"
]