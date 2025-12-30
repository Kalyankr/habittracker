import os
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import hydra
from omegaconf import DictConfig

from src.data.make_dataset import load_batches

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
    data = signal.filtfilt(b_low, a_low, data_high, axis=0)

    return data


def create_windows(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows per batch.
    Returns:
        X -> (num_windows, window_size, num_channels)
        y -> (num_windows,)
    """
    X, y = [], []

    for batch_id, batch_df in df.groupby("batch_id"):
        batch_df = batch_df.reset_index(drop=True)

        for start in range(0, len(batch_df) - window_size + 1, step_size):
            window_df = batch_df.iloc[start : start + window_size]

            X.append(window_df[sensor_cols].values)
            y.append(window_df["label"].mode()[0])

    log.info("Created %d windows", len(X))
    return np.array(X), np.array(y)


def extract_features(window: np.ndarray) -> dict:
    """
    Extract time-domain features from a single window.
    Shape: (window_size, num_channels)
    """
    features = {}
    num_channels = window.shape[1]

    for ch in range(num_channels):
        x = window[:, ch]
        prefix = f"ch{ch}"

        features[f"{prefix}_mean"] = np.mean(x)
        features[f"{prefix}_std"] = np.std(x)
        features[f"{prefix}_min"] = np.min(x)
        features[f"{prefix}_max"] = np.max(x)
        features[f"{prefix}_rms"] = np.sqrt(np.mean(x**2))

        s = skew(x)
        k = kurtosis(x)

        features[f"{prefix}_skew"] = 0.0 if np.isnan(s) else s
        features[f"{prefix}_kurtosis"] = 0.0 if np.isnan(k) else k

        # Frequency domain (FFT)
        # fft_vals = np.fft.rfft(channel_data)
        # fft_power = np.abs(fft_vals) ** 2
        # features[f"{prefix}_energy"] = np.sum(fft_power)

    return features


def build_features(cfg: DictConfig) -> pd.DataFrame:
    """
    Main function to process raw dataframe into features dataframe.
    """
    log.info("Building features...")

    fs = cfg.processing.sampling_rate
    window_sec = cfg.processing.window_size_sec
    overlap = cfg.processing.overlap_percent
    data_cols = cfg.processing.data_columns

    window_size = int(window_sec * fs)
    step_size = max(1, int(window_size * (1 - overlap)))

    # load data
    df, batch_names = load_batches(Path(cfg.data.raw_path))

    required_cols = set(data_cols + ["label", "batch_id"])
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    all_features = []

    for idx, batch in enumerate(batch_names):
        log.info("Processing batch %d / %d", idx + 1, len(batch_names))

        batch_df = df[df["batch_id"] == batch].reset_index(drop=True)

        # # Optional filtering
        # filtered = apply_filters(
        #     batch_df[data_cols].values,
        #     fs,
        #     cfg.processing.cutoff_freq_low,
        #     cfg.processing.cutoff_freq_high,
        # )
        # batch_df[data_cols] = filtered

        X, y = create_windows(
            batch_df,
            data_cols,
            window_size,
            step_size,
        )

        for i in range(len(X)):
            feats = extract_features(X[i])
            feats["label"] = y[i]
            feats["batch_id"] = batch
            all_features.append(feats)

    features_df = pd.DataFrame(all_features)

    os.makedirs(Path(cfg.data.processed_path).parent, exist_ok=True)
    features_df.to_csv(cfg.data.processed_path, index=False)

    log.info("Saved features: %s", cfg.data.processed_path)
    log.info("Final shape: %s", features_df.shape)

    return features_df


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Orchestrate the data loading and cleaning process."""
    build_features(cfg)


if __name__ == "__main__":
    main()
