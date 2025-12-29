import pandas as pd
import numpy as np
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    log.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        log.error(f"File not found: {file_path}")
        raise


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

    # Ensure timestamp is sorted if present
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    log.info(
        f"Data shape after cleaning: {df.shape} (dropped {initial_shape[0] - df.shape[0]} rows)"
    )
    return df


def process_data(cfg: DictConfig) -> pd.DataFrame:
    """Orchestrate the data loading and cleaning process."""
    df = load_data(cfg.data.raw_path)
    df = clean_data(df)
    return df
