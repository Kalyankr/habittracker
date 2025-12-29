# Hair Touching Habit Tracker - ML Pipeline

This repository contains an industry-standard Machine Learning pipeline for detecting hair touching gestures using wearable sensor data (Accelerometer/Gyroscope).

## Project Structure

```
habit-tracker-ml/
├── configs/                # Hydra configuration files
├── data/                   # Data storage
│   ├── raw/                # Place your 'sensor_data.csv' here
│   └── processed/          # Generated features
├── src/                    # Source code
│   ├── data/               # Data loading and cleaning
│   ├── features/           # Signal processing (Filtering, Windowing, FFT)
│   └── models/             # Training (XGBoost) and MLflow logging
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data:**
    *   Export your sensor data to CSV.
    *   Ensure it has columns: `timestamp`, `acc_x`, `acc_y`, `acc_z`, `gyro_x`, `gyro_y`, `gyro_z`, `label`.
    *   Place the file at `data/raw/sensor_data.csv`.

3.  **Run Pipeline:**
    ```bash
    python main.py
    ```

## Features

*   **Hydra Configuration:** Easily swap model parameters and processing settings in `configs/`.
*   **MLflow Tracking:** Automatically logs metrics, parameters, and artifacts. Run `mlflow ui` to view results.
*   **Signal Processing:** Includes Butterworth filtering and sliding window segmentation.
*   **CoreML Export:** Ready for iOS/WatchOS deployment.
