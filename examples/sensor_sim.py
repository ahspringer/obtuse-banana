import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))  # Add project root to path

from src.sensors.IMU import BNO085IMU
from src.sensors import SCL3300D02DigitalTwin
from src.util import dcm_from_euler, quat_from_dcm, euler_from_dcm

"""
SENSOR SIMULATION COORDINATE SYSTEM:
    +X = Downrange
    +Y = Right (starboard)
    +Z = Down

All positions, velocities, and accelerations follow this convention.
Gravity points in the +Z direction (downward).
"""

SAVE_DIR = Path(r".\examples\images")
SAVE_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(r".\examples\data")
DATA_DIR.mkdir(exist_ok=True)


def sensor_data_sim():

    # Initialize BNO085 IMU digital twin
    bno085_imu = BNO085IMU(seed=42)

    # Use SCL3300D02 digital twin inclinometer
    inclin = SCL3300D02DigitalTwin(seed=42)

    # Simulation parameters
    dt = 1/400  # 400 Hz
    total_duration = 20.0  # 20 seconds of static motion
    n_steps = int(total_duration / dt)

    print(f"\nSimulating {total_duration} seconds at {1/dt:.0f} Hz ({n_steps} steps)...")

    # Storage arrays
    time_array = np.zeros(n_steps)
    accel_true = np.zeros((n_steps, 3))
    accel_measured = np.zeros((n_steps, 3))
    quat_true = np.zeros((n_steps, 4))
    quat_measured = np.zeros((n_steps, 4))
    inclin_true = np.zeros((n_steps, 2))  # [pitch, roll]
    inclin_measured = np.zeros((n_steps, 2))
    euler_true = np.zeros((n_steps, 3))  # [roll, pitch, yaw]


    # Simulation loop
    for i in range(n_steps):
        t = i * dt
        time_array[i] = t

        if t%2 == 0:
            print(f"Simulating time: {t:.2f} seconds", end='\r')
        
        # TRUE MOTION: No movement, flat orientation (roll=0, pitch=0, yaw=0)
        # COORDINATE SYSTEM: +X is downrange, +Y is right, +Z is down
        position_true = np.array([0.0, 0.0, -1.0])  # x, y, z (meters); z=-1 means 1m above reference (negative Z is up)
        velocity_true = np.array([0.0, 0.0, 0.0])  # vx, vy, vz (m/s); no movement
        roll = 0.0
        pitch = 0.0
        yaw = 0.0
        omega = np.array([0.0, 0.0, 0.0])

        if t > 10.0:
            # After 10 seconds, introduce a small pitch up to 10 degrees
            pitch = np.radians(10.0) * (t - 10.0) / 2.0  # Ramp up over 2 seconds
            if pitch > np.radians(10.0):
                pitch = np.radians(10.0)
        
        # Build DCM from Euler angles using util function
        dcm = dcm_from_euler(roll, pitch, yaw)
        
        # Store true Euler angles
        euler_true[i] = np.array([roll, pitch, yaw])
        
        # Convert true Euler to quaternion
        quat_true[i] = quat_from_dcm(dcm)
        
        # True acceleration (assume static, gravity is only acceleration)
        acceleration_body = np.array([0.0, 0.0, 0.0])
        alpha = np.zeros(3)  # No angular acceleration
        # Gravity vector in the global frame (+Z is down, so gravity points in +Z direction)
        gravity_ned = np.array([0.0, 0.0, 9.81])
        

        # Use BNO085 IMU digital twin for accel/gyro
        accel_meas, quat_meas = bno085_imu.step(acceleration_body, omega, alpha, dcm, gravity_ned, dt, t)
        # For true values, use the same as before (static, gravity only)
        accel_truth = bno085_imu.accel.compute_specific_force(acceleration_body, omega, alpha, dcm, gravity_ned)
        accel_true[i] = accel_truth
        accel_measured[i] = accel_meas
        quat_measured[i] = quat_meas

        # Inclinometer measures pitch and roll from DCM (sensor returns [pitch, roll])
        inclin_measured[i] = inclin.step_from_dcm(dcm, dt, t)

        # True inclinometer reading: extract Euler angles from DCM and
        # store as [pitch, roll] to match the sensor's output ordering.
        roll_true, pitch_true, yaw_true = euler_from_dcm(dcm)
        inclin_true[i] = np.array([pitch_true, roll_true])

    print(f"Simulation complete: {n_steps} steps")

    # ===========================================================================
    # PLOTTING
    # ===========================================================================

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Sensor Simulation (X-Downrange, Y-Right, Z-Down)", fontsize=16)

    # =========================================================================
    # ACCELEROMETER SUBPLOTS
    # =========================================================================
    for axis_idx in range(3):
        ax = plt.subplot(4, 4, axis_idx + 1)
        axis_name = ['X', 'Y', 'Z'][axis_idx]
        
        ax.plot(time_array, accel_true[:, axis_idx], 'b-', linewidth=2, label='True', alpha=0.7)
        ax.plot(time_array, accel_measured[:, axis_idx], 'r.', markersize=3, label='Measured', alpha=0.7)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Accel {axis_name} (m/s^2)')
        ax.set_title(f'Accelerometer {axis_name}-axis')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # =========================================================================
    # QUATERNION SUBPLOTS (W, X, Y, Z)
    # =========================================================================
    for quat_idx in range(4):
        ax = plt.subplot(4, 4, quat_idx + 5)
        quat_name = ['W', 'X', 'Y', 'Z'][quat_idx]
        
        ax.plot(time_array, quat_true[:, quat_idx], 'b-', linewidth=2, label='True', alpha=0.7)
        ax.plot(time_array, quat_measured[:, quat_idx], 'r.', markersize=3, label='Measured', alpha=0.7)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Quat {quat_name}')
        ax.set_title(f'Quaternion {quat_name}')
        ax.grid(True, alpha=0.3)
        if quat_idx == 3:
            ax.legend()

    # =========================================================================
    # INCLINOMETER SUBPLOTS (Pitch, Roll)
    # =========================================================================
    for incl_idx in range(2):
        ax = plt.subplot(4, 4, incl_idx + 9)
        incl_name = ['Pitch', 'Roll'][incl_idx]

        # Convert to degrees for readability
        ax.plot(time_array, np.degrees(inclin_true[:, incl_idx]), 'b-', linewidth=2, label='True', alpha=0.7)
        ax.plot(time_array, np.degrees(inclin_measured[:, incl_idx]), 'r.', markersize=3, label='Measured', alpha=0.7)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{incl_name} (degrees)')
        ax.set_title(f'Inclinometer {incl_name}')
        ax.grid(True, alpha=0.3)
        if incl_idx == 1:
            ax.legend()



    # Add a summary text box
    ax_summary = plt.subplot(4, 4, 11)
    ax_summary.axis('off')
    summary_text = (
        f'Simulation Parameters:\n'
        f'Duration: {total_duration} seconds\n'
        f'Sample Rate: {1/dt:.0f} Hz\n'
        f'Total Steps: {n_steps}\n\n'
        f'Motion:\n'
        f'0-20s: Static level flight\n\n'
        f'Sensors:\n'
        f'- Accelerometer 3-axis\n'
        f'- Gyroscope Quaternion\n'
        f'- Inclinometer 2-axis'
    )
    ax_summary.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig_filename = 'sensor_sim_results.png'
    plt.savefig(SAVE_DIR / fig_filename, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as '{fig_filename}' in the '{SAVE_DIR}' directory.")
    plt.close()
    
    # Save sensor data to CSV
    save_sensor_data_to_csv(time_array, accel_true, accel_measured, quat_true, quat_measured,
                            inclin_true, inclin_measured, euler_true, fig_filename)

    print("\n" + "=" * 70)
    print("Simulation visualization complete!")
    print("=" * 70)


def save_sensor_data_to_csv(time_array, accel_true, accel_measured, quat_true, quat_measured,
                             inclin_true, inclin_measured, euler_true, fig_filename):
    """
    Save all sensor data to a timestamped CSV file.
    Missing GPS data points are left blank in the CSV.
    """
    # Generate timestamp from figure filename
    csv_filename = f"sensor_sim_results.csv"
    csv_path = DATA_DIR / csv_filename
    
    # Create a DataFrame with all data
    n_steps = len(time_array)
    
    data_dict = {
        'time(s)': time_array,
        # Accelerometer
        'accel_true_x(m/s^2)': accel_true[:, 0],
        'accel_true_y(m/s^2)': accel_true[:, 1],
        'accel_true_z(m/s^2)': accel_true[:, 2],
        'accel_meas_x(m/s^2)': accel_measured[:, 0],
        'accel_meas_y(m/s^2)': accel_measured[:, 1],
        'accel_meas_z(m/s^2)': accel_measured[:, 2],
        # Quaternion
        'quat_true_w': quat_true[:, 0],
        'quat_true_x': quat_true[:, 1],
        'quat_true_y': quat_true[:, 2],
        'quat_true_z': quat_true[:, 3],
        'quat_meas_w': quat_measured[:, 0],
        'quat_meas_x': quat_measured[:, 1],
        'quat_meas_y': quat_measured[:, 2],
        'quat_meas_z': quat_measured[:, 3],
        # Inclinometer
        'inclin_true_pitch(rad)': inclin_true[:, 0],
        'inclin_true_roll(rad)': inclin_true[:, 1],
        'inclin_meas_pitch(rad)': inclin_measured[:, 0],
        'inclin_meas_roll(rad)': inclin_measured[:, 1],
        # Euler angles
        'euler_true_roll(rad)': euler_true[:, 0],
        'euler_true_pitch(rad)': euler_true[:, 1],
        'euler_true_yaw(rad)': euler_true[:, 2],
    }
    
    df = pd.DataFrame(data_dict)
    df.to_csv(csv_path, index=False)
    print(f"Sensor data saved to '{csv_filename}' in the '{DATA_DIR}' directory.")


if __name__ == "__main__":
    sensor_data_sim()