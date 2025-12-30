//
//  README.md
//  datacollectorapp
//
//  Created by Kalyan reddy Katla on 12/30/25.
//

# Hair-Touching Habit Tracker Feature Engineering

This repository contains code to process Apple Watch sensor data and generate **ML-ready features** for detecting hair-touching habits.

---

## 📂 Data Used

1. **WatchAccelerometerUncalibrated.csv** – Raw accelerometer data (`accelerationX/Y/Z`)
2. **WristMotion.csv** – Motion data including gyroscope (`rotationRateX/Y/Z`) and quaternions (`quaternionW/X/Y/Z`)

---

## Features

The pipeline produces **windowed features** for each 5-second segment (50% overlap):

- **Accelerometer Features:** mean, std, min, max, energy for `accelerationX/Y/Z`
- **Gyroscope Features:** mean, std, min, max for `rotationRateX/Y/Z`
- **Orientation Features:** pitch, roll, yaw (computed from quaternions) – mean, std, range
- **Motion Magnitude:** mean and std of acceleration magnitude

---

## How It Works

1. Load accelerometer and motion CSVs.
2. Convert timestamps from **nanoseconds** to pandas datetime.
3. Merge accelerometer and motion data using `merge_asof` on nearest timestamps.
4. Convert quaternions to **Euler angles** (pitch, roll, yaw).
5. Window the data into **5-second segments** with **50% overlap**.
6. Extract features from accelerometer, gyroscope, and orientation.
7. Produce a **final ML-ready DataFrame** (`ml_df`) with features and label (if available).

---

## Usage

```python
import pandas as pd
# Run the feature engineering script
ml_df = feature_engineering_pipeline(accel_csv, motion_csv)

# ml_df is now ready for ML model training
print(ml_df.head())
