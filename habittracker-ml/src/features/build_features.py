import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


def apply_filters(
    data: np.ndarray, fs: float, low_cut: float, high_cut: float
) -> np.ndarray:
    """
    Apply Butterworth filters to the signal.
    1. High-pass filter to remove gravity/DC component.
    2. Low-pass filter to remove high-frequency noise.
    """
    nyquist = 0.5 * fs

    # High-pass filter
    b_high, a_high = signal.butter(4, low_cut / nyquist, btype="high")
    data_high = signal.filtfilt(b_high, a_high, data, axis=0)

    # Low-pass filter
    b_low, a_low = signal.butter(4, high_cut / nyquist, btype="low")
    data_filtered = signal.filtfilt(b_low, a_low, data_high, axis=0)

    return data_filtered


def create_windows(
    df: pd.DataFrame, window_size: int, step_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Segment data into sliding windows.
    Assumes 'label' column exists.
    """
    X = []
    y = []

    # Columns to use for features (exclude timestamp and label)
    feature_cols = [c for c in df.columns if c not in ["timestamp", "label"]]
    data = df[feature_cols].values
    labels = df["label"].values

    for i in range(0, len(data) - window_size, step_size):
        window = data[i : i + window_size]
        # Use the mode of the labels in the window as the window label
        label_mode = pd.Series(labels[i : i + window_size]).mode()[0]

        X.append(window)
        y.append(label_mode)

    return np.array(X), np.array(y)


def extract_features(window: np.ndarray) -> dict:
    """
    Extract time and frequency domain features from a single window.
    Window shape: (time_steps, num_channels)
    """
    features = {}
    num_channels = window.shape[1]

    for ch in range(num_channels):
        channel_data = window[:, ch]
        prefix = f"ch{ch}"

        # Time domain
        features[f"{prefix}_mean"] = np.mean(channel_data)
        features[f"{prefix}_std"] = np.std(channel_data)
        features[f"{prefix}_min"] = np.min(channel_data)
        features[f"{prefix}_max"] = np.max(channel_data)
        features[f"{prefix}_rms"] = np.sqrt(np.mean(channel_data**2))
        features[f"{prefix}_skew"] = skew(channel_data)
        features[f"{prefix}_kurtosis"] = kurtosis(channel_data)

        # Frequency domain (FFT)
        fft_vals = np.fft.rfft(channel_data)
        fft_power = np.abs(fft_vals) ** 2
        features[f"{prefix}_energy"] = np.sum(fft_power)

    return features


def build_features(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """
    Main function to process raw dataframe into features dataframe.
    """
    log.info("Building features...")

    fs = cfg.processing.sampling_rate
    window_sec = cfg.processing.window_size_sec
    overlap = cfg.processing.overlap_percent

    window_size = int(window_sec * fs)
    step_size = int(window_size * (1 - overlap))

    # 1. Filter data
    feature_cols = [c for c in df.columns if c not in ["timestamp", "label"]]
    # Assuming columns are like acc_x, acc_y, acc_z, gyro_x...

    # Apply filters to sensor columns
    filtered_data = apply_filters(
        df[feature_cols].values,
        fs,
        cfg.processing.cutoff_freq_low,
        cfg.processing.cutoff_freq_high,
    )

    # Create a temporary dataframe with filtered data to pass to windowing
    df_filtered = df.copy()
    df_filtered[feature_cols] = filtered_data

    # 2. Windowing
    X_windows, y_windows = create_windows(df_filtered, window_size, step_size)
    log.info(f"Created {len(X_windows)} windows.")

    # 3. Feature Extraction
    feature_list = []
    for i in range(len(X_windows)):
        feats = extract_features(X_windows[i])
        feats["label"] = y_windows[i]
        feature_list.append(feats)

    features_df = pd.DataFrame(feature_list)
    log.info(f"Extracted features shape: {features_df.shape}")

    return features_df
