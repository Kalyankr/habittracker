import logging
from pathlib import Path
import pandas as pd


log = logging.getLogger(__name__)


def load(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    log.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        log.error(f"File not found: {file_path}")
        raise


def load_batches(data_root: Path):
    log.info("Loading batches from %s", data_root)
    all_batches = []
    batch_name_lst = []

    for batch_dir in sorted(data_root.glob("batch_*")):
        log.debug("Processing %s", batch_dir.name)
        data = pd.read_csv(batch_dir / "WristMotion.csv").sort_values("time")
        labels = pd.read_csv(batch_dir / "Annotation.csv").sort_values("time")

        data["batch_id"] = batch_dir.name
        batch_name_lst.append(batch_dir.name)
        labels["label"] = labels["text"].map({"no": 0, "yes": 1})

        data = clean_data(data)
        aligned = pd.merge_asof(
            data, labels[["time", "label"]], on="time", direction="backward"
        )

        aligned = aligned.dropna(subset=["label"])
        aligned["label"] = aligned["label"].astype(int)

        all_batches.append(aligned)

    log.info("Loaded %d batches", len(all_batches))

    return pd.concat(all_batches, ignore_index=True), batch_name_lst


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning.
    - Drop duplicates
    - Handle missing values
    - Ensure correct data types
    """
    log.info("Cleaning data...")
    initial_shape = df.shape

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop rows with missing values
    df = df.dropna()

    # Ensure timestamp is sorted
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    log.info(
        f"Data shape after cleaning: {df.shape} (dropped {initial_shape[0] - df.shape[0]} rows)"
    )
    return df
