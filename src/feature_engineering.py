import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from utils.logger import init_logger
import logging


def load_batches(data_root: Path):
    logging.info("Loading batches from %s", data_root)
    all_batches = []

    for batch_dir in sorted(data_root.glob("batch_*")):
        logging.debug("Processing %s", batch_dir.name)
        data = pd.read_csv(batch_dir / "WristMotion.csv").sort_values("time")
        labels = pd.read_csv(batch_dir / "Annotation.csv").sort_values("time")

        data["batch_id"] = batch_dir.name
        labels["label"] = labels["text"].map({"no": 0, "yes": 1})

        aligned = pd.merge_asof(
            data, labels[["time", "label"]], on="time", direction="backward"
        )

        aligned = aligned.dropna(subset=["label"])
        aligned["label"] = aligned["label"].astype(int)

        all_batches.append(aligned)
    logging.info("Loaded %d batches", len(all_batches))
    return pd.concat(all_batches, ignore_index=True)


def create_windows(df, features, window_size, step_size):
    logging.info("Creating windows: size=%d step=%d", window_size, step_size)

    X, y, batch_ids = [], [], []

    for batch_id, batch_df in df.groupby("batch_id"):
        logging.debug("Windowing batch %s", batch_id)
        batch_df = batch_df.reset_index(drop=True)

        for start in range(0, len(batch_df) - window_size, step_size):
            window = batch_df.iloc[start : start + window_size]

            label = int(window["label"].mean() >= 0.5)

            X.append(window[features].values)
            y.append(label)
            batch_ids.append(batch_id)
    logging.info("Created %d windows", len(X))
    return np.array(X), np.array(y), np.array(batch_ids)


def extract_features(X):
    logging.info("Extracting statistical features")
    feats = []
    for window in X:
        mean = window.mean(axis=0)
        std = window.std(axis=0)
        maxv = window.max(axis=0)
        energy = np.sum(window**2, axis=0)
        feats.append(np.concatenate([mean, std, maxv, energy]))
    logging.info("Feature extraction complete")
    return np.array(feats)


FEATURES = [
    "accelerationX",
    "accelerationY",
    "accelerationZ",
    "rotationRateX",
    "rotationRateY",
    "rotationRateZ",
    "accelerationMagnitude",
]


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument(
        "--output", type=str, default="./data/processed/output_features.npz"
    )
    args = parser.parse_args()

    # Initialize logger
    init_logger()

    logging.info("Starting feature engineering pipeline")

    df = load_batches(Path(args.data_root))
    X_raw, y, batch_ids = create_windows(df, FEATURES, args.window_size, args.step_size)
    X_feat = extract_features(X_raw)

    logging.info("Saving features to %s", args.output)
    np.savez(args.output, X=X_feat, y=y, batch_ids=batch_ids)

    logging.info("Feature engineering pipeline completed")


if __name__ == "__main__":
    main()
