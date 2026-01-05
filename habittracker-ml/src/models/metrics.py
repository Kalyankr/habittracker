import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import mlflow

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from hydra.core.hydra_config import HydraConfig


# Utils
def _get_artifact_path(filename: str) -> str:
    return os.path.join(
        HydraConfig.get().runtime.output_dir,
        filename,
    )


def _plot_confusion_matrix(cm, title, path, normalize=False):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(6, 6))
    disp.plot(
        cmap=plt.cm.Blues,
        values_format=".2f" if normalize else "d",
    )
    plt.title(title)
    plt.savefig(path)
    plt.close()


# Fold Metrics
def log_fold_metrics(
    y_true,
    y_pred,
    fold: int,
    feature_names=None,
    feature_importances=None,
):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    mlflow.log_metric(f"accuracy_fold_{fold}", acc)
    mlflow.log_metric(f"f1_fold_{fold}", f1)

    # Per-class recall (VERY important for habits)
    recalls = recall_score(y_true, y_pred, average=None)
    for i, r in enumerate(recalls):
        mlflow.log_metric(f"recall_class_{i}_fold_{fold}", float(r))

    cm = confusion_matrix(y_true, y_pred)

    # Raw CM
    cm_path = _get_artifact_path(f"cm_fold_{fold}.png")
    _plot_confusion_matrix(cm, f"Fold {fold} Confusion Matrix", cm_path)
    mlflow.log_artifact(cm_path)

    # Normalized CM
    cm_norm_path = _get_artifact_path(f"cm_fold_{fold}_normalized.png")
    _plot_confusion_matrix(
        cm, f"Fold {fold} Normalized CM", cm_norm_path, normalize=True
    )
    mlflow.log_artifact(cm_norm_path)

    # Feature importance
    if feature_importances is not None and feature_names is not None:
        fi_path = _get_artifact_path(f"feature_importance_fold_{fold}.json")
        with open(fi_path, "w") as f:
            json.dump(
                {
                    name: float(val)
                    for name, val in zip(feature_names, feature_importances)
                },
                f,
                indent=2,
            )
        mlflow.log_artifact(fi_path)

    return acc, f1


# Aggregated CV
def log_aggregated_cv_metrics(y_true_all, y_pred_all):
    cm = confusion_matrix(y_true_all, y_pred_all)

    raw_path = _get_artifact_path("cm_cv_aggregated.png")
    norm_path = _get_artifact_path("cm_cv_aggregated_normalized.png")

    _plot_confusion_matrix(cm, "Aggregated CV Confusion Matrix", raw_path)
    _plot_confusion_matrix(cm, "Aggregated CV Normalized CM", norm_path, normalize=True)

    mlflow.log_artifact(raw_path)
    mlflow.log_artifact(norm_path)

    # Per-class recall (aggregated)
    recalls = recall_score(y_true_all, y_pred_all, average=None)
    for i, r in enumerate(recalls):
        mlflow.log_metric(f"recall_class_{i}_cv", float(r))


# Final Model
def log_final_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    path = _get_artifact_path("cm_final_train.png")
    _plot_confusion_matrix(cm, "Final Train Confusion Matrix", path)
    mlflow.log_artifact(path)


# Latency
def log_prediction_latency(model, X, n_runs=100):
    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(X)
    elapsed = (time.perf_counter() - start) / n_runs

    mlflow.log_metric("prediction_latency_ms", elapsed * 1000)


# Drift Check
def log_batch_drift(X, batch_ids):
    """
    Simple statistical drift check using mean feature shift.
    """
    global_mean = X.mean()

    for batch in batch_ids.unique():
        batch_mean = X[batch_ids == batch].mean()
        drift = np.linalg.norm(batch_mean - global_mean)
        mlflow.log_metric(f"feature_drift_batch_{batch}", float(drift))
