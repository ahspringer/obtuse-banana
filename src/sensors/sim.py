import matplotlib.pyplot as plt
import numpy as np
from IMU import Accelerometer3Axis, GyroscopeQuaternion
from inclinometers import Inclinometer2Axis
from GNSS import GNSSReceiver
import util

def sim():
    # Storage arrays
    time_array = np.zeros(n_steps)
    accel_true = np.zeros((n_steps, 3))
    accel_measured = np.zeros((n_steps, 3))
    quat_true = np.zeros((n_steps, 4))
    quat_measured = np.zeros((n_steps, 4))
    inclin_true = np.zeros((n_steps, 2))  # [pitch, roll]
    inclin_measured = np.zeros((n_steps, 2))
    euler_true = np.zeros((n_steps, 3))  # [roll, pitch, yaw]
    gps_pos_true = np.zeros((n_steps, 3))  # [x, y, z] in meters
    gps_pos_measured = np.zeros((n_steps, 3))
    gps_vel_true = np.zeros((n_steps, 3))  # [vx, vy, vz]
    gps_vel_measured = np.zeros((n_steps, 3))
    gps_measurement_count = [0]  # Track number of valid GPS updates

    # Simulation loop
    for i in range(n_steps):
        t = i * dt
        time_array[i] = t

        if t%2 == 0:
            print(f"Simulating time: {t:.2f} seconds", end='\r')
        
        # TRUE MOTION: No movement, flat orientation (roll=0, pitch=0, yaw=0)
        position_true = np.array([0.0, 0.0, 1.0])  # x, y, z (meters); z = altitude
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
        dcm = util.dcm_from_euler(roll, pitch, yaw)
        
        # Store true Euler angles
        euler_true[i] = np.array([roll, pitch, yaw])
        
        # Convert true Euler to quaternion
        quat_true[i] = util.quat_from_dcm(dcm)
        
        # True acceleration (assume static, gravity is only acceleration)
        acceleration_body = np.array([0.0, 0.0, 0.0])
        alpha = np.zeros(3)  # No angular acceleration
        gravity_ned = np.array([0.0, 0.0, 9.81])
        
        # Compute specific force (what accelerometer measures before noise)
        accel_truth = accel.compute_specific_force(acceleration_body, omega, alpha, dcm, gravity_ned)
        accel_true[i] = accel_truth
        
        # Measure with IMU sensors
        accel_measured[i] = accel.step_with_frame(accel_truth, dt, t)
        quat_measured[i] = gyro_quat.step_from_rates(omega, dt, t)
        
        # Inclinometer measures pitch and roll from DCM (sensor returns [pitch, roll])
        inclin_measured[i] = inclin.step_from_dcm(dcm, dt, t)

        # True inclinometer reading: extract Euler angles from DCM and
        # store as [pitch, roll] to match the sensor's output ordering.
        roll_true, pitch_true, yaw_true = util.euler_from_dcm(dcm)
        inclin_true[i] = np.array([pitch_true, roll_true])
        
        # True position and velocity (stationary at origin)
        gps_pos_true[i] = position_true
        gps_vel_true[i] = velocity_true
        
        # Measure with GPS (lower update rate handled internally)
        gps_result = gnss.step_from_truth(position_true, velocity_true, dt, t)
        if gps_result is not None:
            gps_pos_measured[i] = gps_result[0:3]
            gps_vel_measured[i] = gps_result[3:6]
            gps_measurement_count[0] += 1
        else:
            # Propagate last valid measurement if available
            if i > 0:
                gps_pos_measured[i] = gps_pos_measured[i-1]
                gps_vel_measured[i] = gps_vel_measured[i-1]

    print(f"Simulation complete: {n_steps} steps")

    # ===========================================================================
    # PLOTTING
    # ===========================================================================

    fig = plt.figure(figsize=(20, 16))

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

    # =========================================================================
    # GPS POSITION SUBPLOTS (X, Y, Z)
    # =========================================================================
    for pos_idx in range(3):
        ax = plt.subplot(4, 4, pos_idx + 11)
        pos_name = ['X', 'Y', 'Z'][pos_idx]
        
        ax.plot(time_array, gps_pos_true[:, pos_idx], 'b-', linewidth=2, label='True', alpha=0.7)
        ax.plot(time_array, gps_pos_measured[:, pos_idx], 'r.', markersize=3, label='Measured', alpha=0.7)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'Position {pos_name} (m)')
        ax.set_title(f'GPS Position {pos_name}')
        ax.grid(True, alpha=0.3)
        if pos_idx == 2:
            ax.legend()

    # Add a summary text box
    ax_summary = plt.subplot(4, 4, 16)
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
        f'- Inclinometer 2-axis\n'
        f'- GPS Receiver'
    )
    ax_summary.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('sensor_simulation_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'sensor_simulation_results.png'")
    plt.close()

    print("\n" + "=" * 70)
    print("Simulation visualization complete!")
    print("=" * 70)

if __name__ == "__main__":
    # Initialize sensors
    accel = Accelerometer3Axis(
        sensor_id="accel_main",
        white_noise_std=np.array([0.02, 0.02, 0.02]),
        initial_bias=np.array([0.03, -0.02, 0.01]),
        seed=42
    )

    gyro_quat = GyroscopeQuaternion(
        sensor_id="gyro_quat",
        initial_quaternion=[1.0, 0.0, 0.0, 0.0],
        white_noise_std=np.array([0.001, 0.001, 0.001]),
        quaternion_noise_std=0.01,
        seed=42
    )

    inclin = Inclinometer2Axis(
        sensor_id="inclinometer",
        white_noise_std=np.array([0.02, 0.02]),
        seed=42
    )

    gnss = GNSSReceiver(
        sensor_id="gps",
        position_noise_std=2.0,
        velocity_noise_std=0.1,
        update_rate_hz=10.0,
        seed=42
    )

    # Simulation parameters
    dt = 1/400  # 400 Hz
    total_duration = 20.0  # 20 seconds of static motion
    n_steps = int(total_duration / dt)

    print(f"\nSimulating {total_duration} seconds at {1/dt:.0f} Hz ({n_steps} steps)...")
    
    sim()