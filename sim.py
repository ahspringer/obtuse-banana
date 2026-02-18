import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.sensors.IMU import IMU, SensorMode
from src.filters.sensor_fusion import State, ExtendedKalmanFilter, adapt_accel_sensor, adapt_gyro_sensor, adapt_gnss_sensor

def playback_csv_to_imu(file_path: str):
    """
    Reads IMU data from a CSV and feeds it into the IMU hardware proxy.
    """
    # 1. Load the data
    try:
        # file uses space-separated columns (no commas) — read with whitespace delimiter
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the CSV exists.")
        return
    
    # Remove potential surrounding whitespace from column headers
    df.columns = df.columns.str.strip()
    
    print(df)

    # 2. Initialize IMU in HARDWARE mode
    # We provide a calibration config (here just defaults, but could be loaded from file)
    imu = IMU(
        mode=SensorMode.HARDWARE,
        accel_config={'calibration_bias': np.zeros(3), 'buffer_size': 5000},
        gyro_config={'calibration_bias': np.zeros(3), 'buffer_size': 5000},
        mag_config={'calibration_bias': np.zeros(3), 'buffer_size': 5000}
    )

    print(f"Starting playback of {len(df)} samples...")

    # 3. Iterate and ingest (simulating a live feed)
    t = 0.0
    has_timestamp = 'timestamp' in df.columns
    if not has_timestamp:
        print("Note: 'timestamp' column missing in CSV.\n\t-> Using 0.1s sample interval.")

    i = 0
    for _, row in df.iterrows():
        if has_timestamp:
            t = row['timestamp']
        else:
            t += 0.1
        
        accel_raw = np.array([row['ax_ms2'], row['ay_ms2'], row['az_ms2']])
        gyro_raw = np.array([row['gx_rads'], row['gy_rads'], row['gz_rads']])
        mag_raw = np.array([row['mx_uT'], row['my_uT'], row['mz_uT']])

        # Ingest: This applies calibration and stores in internal ring buffers
        imu.ingest_data(
            accel_vec=accel_raw,
            gyro_vec=gyro_raw,
            mag_vec=mag_raw,
            timestamp=t
        )
        i+=1

    print(f"Ended playback.")
    # print(imu.get_history())

    # 4. Retrieve the processed history from the IMU object
    history = imu.get_history()
    
    # 5. Plotting the results: 9 subplots (3 sensors × 3 axes)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True)
    
    # Sensor info: (data_key, title_prefix, ylabel, axis_labels)
    sensors = [
        ('accel', 'Accelerometer', 'm/s²'),
        ('gyro', 'Gyroscope', 'rad/s'),
        ('mag', 'Magnetometer', 'µT')
    ]
    axis_labels = ['X', 'Y', 'Z']
    
    for row, (sensor_key, sensor_title, ylabel) in enumerate(sensors):
        data = history[sensor_key]  # shape: (num_samples, 3)
        for col in range(3):
            ax = axes[row, col]
            ax.scatter(range(len(data[:, col])), data[:, col], s=10, alpha=0.6)
            ax.set_title(f'{sensor_title} - {axis_labels[col]} Axis')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if row == 2:  # bottom row
                ax.set_xlabel('Sample Index (Buffer)')
    
    plt.tight_layout()
    plt.show()


def filter_csv_imu_data(file_path: str):
    """
    Reads IMU data from a CSV, runs it through an EKF sensor fusion, and plots the results.
    """
    # 1. Load the data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the CSV exists.")
        return
    
    # Remove potential surrounding whitespace from column headers
    df.columns = df.columns.str.strip()
    
    print(df)

    # 2. Initialize IMU in HARDWARE mode
    # We provide a calibration config (here just defaults, but could be loaded from file)
    print(f"Initializing IMU...")
    imu = IMU(
        mode=SensorMode.HARDWARE,
        accel_config={'calibration_bias': np.zeros(3), 'buffer_size': 5000},
        gyro_config={'calibration_bias': np.zeros(3), 'buffer_size': 5000},
        mag_config={'calibration_bias': np.zeros(3), 'buffer_size': 5000}
    )

    print(f"Initialized IMU: {imu.id}\n -> Accelerometer: {imu.accel.sensor_id}\n -> Gyroscope: {imu.gyro.sensor_id}\n -> Magnetometer: {imu.mag.sensor_id}")

    # 3. Initialize EKF sensor fusion
    print(f"Initializing EKF sensor fusion...")
    nav_state = State(dim=16) # 16-DOF: [pos(3), vel(3), quat(4), quat_rate(4), accel_bias(3)]
    ekf = ExtendedKalmanFilter(nav_state)

    from src.filters.sensor_fusion import quaternion_to_rotation_matrix
    
    def accel_measurement_func(state: State) -> np.ndarray:
        """
        Accelerometer measurement model using quaternion representation.
        
        Returns specific force in body frame:
            f_specific = a_imu_body - g_body
        
        Where g_body is gravity rotated into the body frame from NED.
        
        State layout (16-DOF):
        [0-2]: position, [3-5]: velocity, [6-9]: quaternion,
        [10-13]: quaternion_rate, [13-15]: accel_bias
        """
        # Extract quaternion from state (indices 6-9)
        q = state.x[6:10].flatten()
        
        # Compute rotation matrix (body to NED) from quaternion
        R_b2n = quaternion_to_rotation_matrix(q)
        
        # Gravity in NED frame (North, East, Down)
        g_ned = np.array([0.0, 0.0, 9.81])
        
        # Rotate gravity to body frame (R_n2b = R_b2n^T)
        g_body = R_b2n.T @ g_ned
        
        # Estimate true acceleration from velocity (simplified)
        # In production, integrate measured accel with bias correction
        a_body = np.zeros(3)  # Placeholder
        
        # Specific force = acceleration minus gravity
        specific_force = a_body - g_body
        
        return specific_force
    
    def gyro_measurement_func(state: State) -> np.ndarray:
        """
        Gyroscope measurement model using quaternion kinematics.
        
        Returns angular velocity directly from quaternion rate state:
            omega = state[10:13]
        
        This is the body angular velocity in rad/s.
        
        State layout (16-DOF):
        [0-2]: position, [3-5]: velocity, [6-9]: quaternion,
        [10-13]: quaternion_rate, [13-15]: accel_bias
        """
        # Extract angular velocity from quaternion rate (simplified: use first 3 components)
        # In practice, this comes from quaternion rate kinematics
        # omega = 2 * Q_left(q)^T @ dq, where dq is quaternion derivative
        omega = state.x[10:13].flatten()  # Simplified: assume linear mapping
        
        return omega

    # 4. Iterate and ingest (simulating a live feed)
    t = 0.0
    dt = 0.1  # default sample interval if no timestamps are provided
    has_timestamp = 'timestamp' in df.columns
    if not has_timestamp:
        print(f"Note: 'timestamp' column missing in CSV.\n\t-> Using {dt}s sample interval.")

    i = 0
    for _, row in df.iterrows():
        if has_timestamp:
            t = row['timestamp']
        else:
            t += dt
        
        # Read raw sensor data from the CSV row
        accel_raw = np.array([row['ax_ms2'], row['ay_ms2'], row['az_ms2']])
        gyro_raw = np.array([row['gx_rads'], row['gy_rads'], row['gz_rads']])
        mag_raw = np.array([row['mx_uT'], row['my_uT'], row['mz_uT']])

        # Ingest: This applies calibration and stores in internal ring buffers
        imu.ingest_data(
            accel_vec=accel_raw,
            gyro_vec=gyro_raw,
            mag_vec=mag_raw,
            timestamp=t
        )

        # For demonstration, we can also run the EKF update here (though in a real system, this might be on a separate thread or at a different rate)


        i+=1


if __name__ == "__main__":
    # Example usage: Reading IMU data from a csv file and plotting it
    # playback_csv_to_imu(r'C:\Users\sprin\Documents\Aretex Labs\obtuse-banana\src\sensors\data\data_01292026.csv')

    # Example usage: Running the EKF fusion demo with recorded csv data
    filter_csv_imu_data(r'C:\Users\sprin\Documents\Aretex Labs\obtuse-banana\src\sensors\data\data_01292026.csv')

    pass