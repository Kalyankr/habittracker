import pandas as pd
from pathlib import Path


def load_batches(data_root: Path):
    all_batches = []

    for batch_dir in sorted(data_root.glob("batch_*")):
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

    return pd.concat(all_batches, ignore_index=True)
