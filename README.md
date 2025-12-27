# Habit Tracking with Apple Watch & Machine Learning

## Overview

This project is a **personal habit tracking system** using **Apple Watch motion sensors** and **machine learning** to detect specific habits, like touching hair. The system uses **state-based labeling** (YES/NO) and applies **temporal windowing**, **feature extraction**, and **batch-aware model evaluation** for accurate predictions.

It is designed to work in **multiple sessions (batches)** and handle real-world variations in hand movements.

---

## Features

- **Apple Watch Data Collection**: Collects motion data including:
  - Acceleration (`accelerationX/Y/Z`)
  - Rotation rates (`rotationRateX/Y/Z`)
  - Gravity and acceleration magnitude
  - Timestamp and batch/session ID

- **State-based Labeling**:
  - Users label events in real-time: `YES` when performing the habit, `NO` when not.
  - Labels persist until changed, creating a **dense supervision signal**.

- **Batch Handling**:
  - Supports multiple sessions/batches
  - Aligns labels to sensor data per batch
  - Prevents temporal leakage in ML evaluation

- **Feature Extraction**:
  - Window-based sliding approach (configurable window size and overlap)
  - Extracts rich features per window:
    - Mean, standard deviation, maximum, and energy of each sensor axis

- **Machine Learning**:
  - Random Forest Classifier with class balancing
  - Batch-level train/test split for honest evaluation
  - Probability threshold tuning and temporal smoothing for precision improvement

- **Evaluation**:
  - Precision, Recall, F1-score
  - Confusion matrix visualization
  - Batch-level cross-validation support

---

## Collect Apple Watch Data

- Use the custom Watch app or any motion logger app to record sensor data and label habits.
- Place each session in a batch_* folder with:
- WristMotion.csv — sensor readings
- Annotation.csv — labels (YES/NO)