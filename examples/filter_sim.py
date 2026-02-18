"""
Extended Kalman Filter Functionality Test
==========================================

This script reads simulated sensor data from a CSV file and processes it through
an EKF to generate state estimates. It visualizes the results by overlaying the
filtered estimates on the measured sensor data.

Flow:
1. Read sensor_sim_results.csv
2. Initialize 16-DOF quaternion-based EKF
3. Set up sensor adapters (accelerometer, inclinometer)
4. Step through data: predict → update
5. Plot results with filtered state in bold red
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # Add project root to path

from src.filters import (
    State,
    ExtendedKalmanFilter,
    NavigationPhysics,
    adapt_accel_sensor,
    adapt_inclinometer_sensor,
    quaternion_to_euler,
)
from src.sensors import Accelerometer3Axis, Inclinometer2Axis

SAVE_DIR = Path(r".\examples\images")
SAVE_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(r".\examples\data")
DATA_DIR.mkdir(exist_ok=True)

def load_sensor_data(csv_path):
    """Load sensor simulation results from CSV file."""
    print(f"[1/5] Loading sensor data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"     [OK] Loaded {len(df)} data points")
    return df


def initialize_ekf():
    """Initialize the 16-DOF quaternion-based EKF."""
    print("[2/5] Initializing 16-DOF quaternion-based EKF")
    
    # Create 16-DOF state (pos, vel, quat, quat_rate, accel_bias)
    nav_state = State(dim=16)
    
    # Set initial uncertainty (high until filter converges)
    P_init = np.eye(16)
    P_init[0:3, 0:3] *= 100.0      # Position uncertainty: 100 m²
    P_init[3:6, 3:6] *= 10.0       # Velocity uncertainty: 10 (m/s)²
    P_init[6:10, 6:10] *= 0.1      # Quaternion uncertainty: small
    P_init[10:13, 10:13] *= 0.01   # Quaternion rate uncertainty: small
    P_init[13:16, 13:16] *= 0.01   # Accel bias uncertainty: small
    nav_state.P = P_init
    
    ekf = ExtendedKalmanFilter(nav_state)
    print("     [OK] EKF initialized with identity quaternion [1, 0, 0, 0]")
    
    return ekf


def setup_sensor_adapters():
    """Create sensor adapters for the EKF."""
    print("[3/5] Setting up sensor adapters")
    
    # Accelerometer (3-axis)
    accel = Accelerometer3Axis()
    accel_adapter = adapt_accel_sensor(accel)
    print(f"     [OK] Accelerometer adapter created (dim={accel_adapter.m_dim})")
    
    # Inclinometer (2-axis: pitch + roll)
    inclin = Inclinometer2Axis()
    inclin_adapter = adapt_inclinometer_sensor(inclin)
    print(f"     [OK] Inclinometer adapter created (dim={inclin_adapter.m_dim})")
    
    return {
        "accel": (accel_adapter, "accel_meas"),
        "inclin": (inclin_adapter, "inclin_meas"),
    }


def process_data_through_ekf(ekf, df, sensor_adapters):
    """Feed data through the EKF, collecting estimates."""
    print("[4/5] Processing sensor data through EKF")
    
    # Process noise covariance (tuned for the system)
    Q = np.eye(16) * 0.001
    Q[0:3, 0:3] *= 0.01      # Position process noise
    Q[3:6, 3:6] *= 0.01      # Velocity process noise
    Q[6:10, 6:10] *= 0.001   # Quaternion process noise
    Q[10:13, 10:13] *= 0.01  # Angular velocity process noise
    Q[13:16, 13:16] *= 0.001 # Accel bias process noise
    
    # Storage for results
    results = {
        "time": [],
        "est_roll": [],
        "est_pitch": [],
        "est_yaw": [],
        "meas_pitch": [],
        "meas_roll": [],
        "true_roll": [],
        "true_pitch": [],
        "true_yaw": [],
    }
    
    num_rows = len(df)
    prev_time = 0.0
    
    for idx, row in df.iterrows():
        current_time = row["time(s)"]
        dt = current_time - prev_time if idx > 0 else 0.0
        
        # Print progress every 500 rows
        if idx % 500 == 0:
            print(f"     Processing row {idx}/{num_rows} (t={current_time:.2f}s)")
        
        # ===== PREDICT STEP =====
        if dt > 0.0:
            ekf.predict(
                dt=dt,
                f_transition=NavigationPhysics.transition_function,
                F_jacobian=NavigationPhysics.transition_jacobian,
                Q=Q,
            )
        
        # ===== MEASUREMENT UPDATE STEPS =====
        
        # Extract measurements from CSV columns
        z_accel = np.array([
            row["accel_meas_x(m/s^2)"],
            row["accel_meas_y(m/s^2)"],
            row["accel_meas_z(m/s^2)"],
        ]).reshape(-1, 1)
        
        z_inclin = np.array([
            row["inclin_meas_pitch(rad)"],
            row["inclin_meas_roll(rad)"],
        ]).reshape(-1, 1)
        
        # Update with accelerometer
        if not np.any(np.isnan(z_accel)):
            accel_adapter = sensor_adapters["accel"][0]
            try:
                ekf.update(accel_adapter, z_accel)
            except:
                pass
        
        # Update with inclinometer
        if not np.any(np.isnan(z_inclin)):
            inclin_adapter = sensor_adapters["inclin"][0]
            try:
                ekf.update(inclin_adapter, z_inclin)
            except:
                pass
        
        # ===== COLLECT RESULTS =====
        state = ekf.state
        roll_est, pitch_est, yaw_est = quaternion_to_euler(state.get_quaternion().flatten())
        
        # True Euler angles from CSV
        true_roll = row["euler_true_roll(rad)"]
        true_pitch = row["euler_true_pitch(rad)"]
        true_yaw = row["euler_true_yaw(rad)"]
        
        results["time"].append(current_time)
        results["est_roll"].append(roll_est)
        results["est_pitch"].append(pitch_est)
        results["est_yaw"].append(yaw_est)
        results["meas_pitch"].append(row["inclin_meas_pitch(rad)"])
        results["meas_roll"].append(row["inclin_meas_roll(rad)"])
        results["true_roll"].append(true_roll)
        results["true_pitch"].append(true_pitch)
        results["true_yaw"].append(true_yaw)
        
        prev_time = current_time
    
    print(f"     [OK] Processed all {num_rows} rows")
    return pd.DataFrame(results)


def plot_results(results_df):
    """Create comprehensive plots of sensor data with filtered estimates overlaid."""
    print("[5/5] Generating visualization")
    
    time = results_df["time"].values
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("EKF Functionality Test: Orientation Estimation", fontsize=16, fontweight="bold")
    
    # Euler angles (Roll, Pitch, Yaw)
    roll_deg = np.degrees(results_df["est_roll"].values)
    roll_meas_deg = np.degrees(results_df["meas_roll"].values)
    roll_true_deg = np.degrees(results_df["true_roll"].values)
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(time, roll_meas_deg, "b.-", label="Inclinometer Measurement", alpha=0.7, linewidth=0.8)
    ax1.plot(time, roll_true_deg, "g--", label="True Roll", alpha=0.7, linewidth=1.2)
    ax1.plot(time, roll_deg, "r-", label="EKF Estimate", linewidth=2.5)
    ax1.set_ylabel("Roll (degrees)", fontsize=10)
    ax1.set_xlabel("Time (s)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    pitch_deg = np.degrees(results_df["est_pitch"].values)
    pitch_meas_deg = np.degrees(results_df["meas_pitch"].values)
    pitch_true_deg = np.degrees(results_df["true_pitch"].values)
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(time, pitch_meas_deg, "b.-", label="Inclinometer Measurement", alpha=0.7, linewidth=0.8)
    ax2.plot(time, pitch_true_deg, "g--", label="True Pitch", alpha=0.7, linewidth=1.2)
    ax2.plot(time, pitch_deg, "r-", label="EKF Estimate", linewidth=2.5)
    ax2.set_ylabel("Pitch (degrees)", fontsize=10)
    ax2.set_xlabel("Time (s)", fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    yaw_deg = np.degrees(results_df["est_yaw"].values)
    yaw_true_deg = np.degrees(results_df["true_yaw"].values)
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(time, yaw_true_deg, "g--", label="True Yaw", alpha=0.7, linewidth=1.2)
    ax3.plot(time, yaw_deg, "r-", label="EKF Estimate", linewidth=2.5)
    ax3.set_ylabel("Yaw (degrees)", fontsize=10)
    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Error plots
    roll_error_deg = roll_deg - roll_true_deg
    pitch_error_deg = pitch_deg - pitch_true_deg
    yaw_error_deg = yaw_deg - yaw_true_deg
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(time, roll_error_deg, "b-", label="Roll Error", linewidth=1.5, alpha=0.8)
    ax4.plot(time, pitch_error_deg, "g-", label="Pitch Error", linewidth=1.5, alpha=0.8)
    ax4.plot(time, yaw_error_deg, "r-", label="Yaw Error", linewidth=1.5, alpha=0.8)
    ax4.set_ylabel("Error (degrees)", fontsize=10)
    ax4.set_xlabel("Time (s)", fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add legend explaining line styles
    legend_elements = [
        Patch(facecolor="blue", alpha=0.7, label="Sensor Measurements"),
        Patch(facecolor="red", alpha=1.0, label="EKF Estimates (bold)"),
        Patch(facecolor="green", alpha=0.7, label="True State"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    # Save figure
    output_path = os.path.join(SAVE_DIR, "filter_sim_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"     [OK] Plot saved to: {output_path}")
    
    plt.show()


def main():
    """Main execution flow."""
    print("=" * 80)
    print("EKF FUNCTIONALITY TEST - SENSOR FUSION ON SIMULATED DATA")
    print("=" * 80)
    print()
    
    # Construct path to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "data", "sensor_sim_results.csv")
    
    try:
        # 1. Load data
        df = load_sensor_data(csv_path)
        print()
        
        # 2. Initialize EKF
        ekf = initialize_ekf()
        print()
        
        # 3. Setup sensor adapters
        adapters = setup_sensor_adapters()
        print()
        
        # 4. Process through EKF
        results_df = process_data_through_ekf(ekf, df, adapters)
        print()
        
        # 5. Generate plots
        plot_results(results_df)
        print()
        print("=" * 80)
        print("[SUCCESS] EKF FUNCTIONALITY TEST COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[FAILED] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
