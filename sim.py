import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.sensors.IMU import IMU, SensorMode

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

if __name__ == "__main__":
    # Example usage:
    playback_csv_to_imu(r'C:\Users\sprin\Documents\Aretex Labs\obtuse-banana\src\sensors\data\data_01292026.csv')
    pass