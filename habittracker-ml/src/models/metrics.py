import json
import os
import numpy as np
import matplotlib.pyplot as plt
import mlflow

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from hydra.core.hydra_config import HydraConfig


def _get_artifact_path(filename: str) -> str:
    """
    Returns an absolute path inside Hydra's output directory.
    """
    run_dir = HydraConfig.get().runtime.output_dir
    return os.path.join(run_dir, filename)


def log_fold_metrics(
    y_true,
    y_pred,
    fold: int,
    feature_names=None,
    feature_importances=None,
):
    """
    Logs metrics and artifacts for a single CV fold.
    """

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")

    mlflow.log_metric(f"accuracy_fold_{fold}", acc)
    mlflow.log_metric(f"f1_fold_{fold}", f1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title(f"Confusion Matrix - Fold {fold}")

    cm_path = _get_artifact_path(f"confusion_matrix_fold_{fold}.png")
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    # Feature Importance
    if feature_importances is not None and feature_names is not None:
        fi_dict = {
            name: float(val) for name, val in zip(feature_names, feature_importances)
        }

        fi_path = _get_artifact_path(f"feature_importance_fold_{fold}.json")
        with open(fi_path, "w") as f:
            json.dump(fi_dict, f, indent=2)

        mlflow.log_artifact(fi_path)

    return acc, f1


def log_final_confusion_matrix(
    y_true,
    y_pred,
    label: str = "final",
):
    """
    Logs a final confusion matrix after training on full data.
    """

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Final Confusion Matrix")

    cm_path = _get_artifact_path(f"confusion_matrix_{label}.png")
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)


def log_aggregated_cv_confusion_matrix(
    y_true_all,
    y_pred_all,
    label: str = "cv_aggregated",
):
    """
    Logs a confusion matrix built from all CV held-out predictions.
    This is the most reliable real-world evaluation.
    """

    cm = confusion_matrix(y_true_all, y_pred_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Aggregated CV Confusion Matrix")

    cm_path = _get_artifact_path(f"confusion_matrix_{label}.png")
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)
