"""
Test script demonstrating the update rate functionality now available to all SimulatedSensor objects.

Previously, only GNSS had update rate limiting. Now any sensor can be created with an optional
update_rate_hz parameter to simulate sensors that don't update on every simulation step.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.sensors import (
    Accelerometer3Axis, 
    GyroscopeQuaternion, 
    Inclinometer2Axis, 
    GNSSReceiver
)


def test_update_rates():
    """Demonstrate update rate functionality for various sensors."""
    
    print("=" * 70)
    print("Testing Update Rate Functionality for All Sensors")
    print("=" * 70)
    
    # Example 1: GNSS at 10 Hz (default)
    gnss_10hz = GNSSReceiver(
        sensor_id="gnss_10hz",
        update_rate_hz=10.0
    )
    print(f"\n✓ GNSS (10 Hz):")
    print(f"  - Update period: {gnss_10hz.update_period:.3f} seconds")
    print(f"  - should_update(t=0.00): {gnss_10hz.should_update(0.00)}")
    print(f"  - should_update(t=0.05): {gnss_10hz.should_update(0.05)}")
    print(f"  - should_update(t=0.10): {gnss_10hz.should_update(0.10)}")
    
    # Example 2: GNSS at 50 Hz (faster)
    gnss_50hz = GNSSReceiver(
        sensor_id="gnss_50hz",
        update_rate_hz=50.0
    )
    print(f"\n✓ GNSS (50 Hz):")
    print(f"  - Update period: {gnss_50hz.update_period:.3f} seconds")
    print(f"  - should_update(t=0.00): {gnss_50hz.should_update(0.00)}")
    print(f"  - should_update(t=0.02): {gnss_50hz.should_update(0.02)}")
    print(f"  - should_update(t=0.04): {gnss_50hz.should_update(0.04)}")
    
    # Example 3: Accelerometer with no rate limit (updates every step)
    accel_unlimited = Accelerometer3Axis(
        sensor_id="accel_unlimited",
        update_rate_hz=None  # No rate limiting
    )
    print(f"\n✓ Accelerometer (Unlimited):")
    print(f"  - Update rate: {accel_unlimited.update_rate_hz} (unlimited)")
    print(f"  - should_update(t=0.0001): {accel_unlimited.should_update(0.0001)}")
    print(f"  - should_update(t=0.0002): {accel_unlimited.should_update(0.0002)}")
    
    # Example 4: Accelerometer with optional 200 Hz limit
    accel_200hz = Accelerometer3Axis(
        sensor_id="accel_200hz",
        update_rate_hz=200.0
    )
    print(f"\n✓ Accelerometer (200 Hz):")
    print(f"  - Update period: {accel_200hz.update_period:.4f} seconds")
    print(f"  - should_update(t=0.0000): {accel_200hz.should_update(0.0000)}")
    print(f"  - should_update(t=0.0025): {accel_200hz.should_update(0.0025)}")
    print(f"  - should_update(t=0.0050): {accel_200hz.should_update(0.0050)}")
    
    # Example 5: Inclinometer at 50 Hz
    inclin_50hz = Inclinometer2Axis(
        sensor_id="inclin_50hz",
        update_rate_hz=50.0
    )
    print(f"\n✓ Inclinometer (50 Hz):")
    print(f"  - Update period: {inclin_50hz.update_period:.3f} seconds")
    print(f"  - should_update(t=0.00): {inclin_50hz.should_update(0.00)}")
    print(f"  - should_update(t=0.01): {inclin_50hz.should_update(0.01)}")
    print(f"  - should_update(t=0.02): {inclin_50hz.should_update(0.02)}")
    
    # Example 6: Gyroscope with high rate
    gyro_400hz = GyroscopeQuaternion(
        sensor_id="gyro_400hz",
        update_rate_hz=400.0
    )
    print(f"\n✓ Gyroscope Quaternion (400 Hz):")
    print(f"  - Update period: {gyro_400hz.update_period:.4f} seconds")
    
    print("\n" + "=" * 70)
    print("All sensors now support optional update_rate_hz parameter!")
    print("=" * 70)
    print("\nUsage Examples:")
    print("  - GNSS at 10 Hz:        GNSSReceiver(update_rate_hz=10.0)")
    print("  - Accelerometer 100 Hz: Accelerometer3Axis(update_rate_hz=100.0)")
    print("  - Inclinometer 50 Hz:   Inclinometer2Axis(update_rate_hz=50.0)")
    print("  - Gyroscope 400 Hz:     GyroscopeQuaternion(update_rate_hz=400.0)")
    print("  - Unlimited rate:       Accelerometer3Axis()  # or update_rate_hz=None")
    print("\n")


if __name__ == "__main__":
    test_update_rates()
