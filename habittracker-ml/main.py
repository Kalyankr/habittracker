import os
import logging
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra

from src.data.build_features import build_features
from src.models.train_model import train
from src.models.test_model import (
    test_model,
)  # We will adapt test_model as a callable function

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Load processed data
    if cfg.train_only:
        if os.path.exists(cfg.data.processed_path):
            log.info(f"Loading processed features from {cfg.data.processed_path}")
            df_features = pd.read_csv(cfg.data.processed_path)
        else:
            log.warning(
                f"Processed file not found at {cfg.data.processed_path}. Running full pipeline."
            )
            df_features = build_features(cfg)
    else:
        df_features = build_features(cfg)

    # Train model
    _ = train(df_features, cfg)

    # Test model
    if cfg.get("test", False):
        if not cfg.data.get("test_path"):
            log.warning("test_path not set in config, skipping test step")
        else:
            log.info("Running test on test set...")
            test_model(cfg)


if __name__ == "__main__":
    main()
