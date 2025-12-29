import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import pandas as pd
from src.data.make_dataset import process_data
from src.features.build_features import build_features
from src.models.train_model import train

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.get("train_only"):
        if os.path.exists(cfg.data.processed_path):
            log.info(f"Loading processed features from {cfg.data.processed_path}")
            df_features = pd.read_csv(cfg.data.processed_path)
            train(df_features, cfg)
            return
        else:
            log.warning(
                f"Processed file not found at {cfg.data.processed_path}. Running full pipeline."
            )

    # 1. Load and Clean Data
    # Check if raw data exists
    if not os.path.exists(cfg.data.raw_path):
        log.error(
            f"Raw data not found at {cfg.data.raw_path}. Please place your sensor data there."
        )
        return

    df_raw = process_data(cfg)

    # 2. Feature Engineering
    df_features = build_features(df_raw, cfg)

    # Save processed features
    os.makedirs(os.path.dirname(cfg.data.processed_path), exist_ok=True)
    df_features.to_csv(cfg.data.processed_path, index=False)
    log.info(f"Saved processed features to {cfg.data.processed_path}")

    # 3. Train Model
    train(df_features, cfg)


if __name__ == "__main__":
    main()
