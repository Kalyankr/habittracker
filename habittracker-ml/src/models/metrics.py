import logging
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import mlflow

log = logging.getLogger(__name__)


def log_classification_metrics(
    y_true,
    y_pred,
    feature_names=None,
    feature_importances=None,
    fold: int = None,
):
    """
    Logs classification metrics (accuracy, F1), confusion matrix, and feature importances to MLflow.
    Returns metrics, confusion matrix path, and feature importance path (if applicable).
    """

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    metrics = {
        f"accuracy{'_fold' + str(fold) if fold is not None else ''}": acc,
        f"f1_score{'_fold' + str(fold) if fold is not None else ''}": f1,
    }

    # Log metrics to MLflow
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title(f"Confusion Matrix{' Fold ' + str(fold) if fold is not None else ''}")
    cm_path = f"confusion_matrix{'_fold_' + str(fold) if fold is not None else ''}.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    fi_path = None
    if feature_importances is not None and feature_names is not None:
        avg_importance = np.array(feature_importances)
        if avg_importance.ndim > 1:
            avg_importance = np.mean(avg_importance, axis=0)

        # Convert to native float for JSON
        fi_dict = {name: float(val) for name, val in zip(feature_names, avg_importance)}

        fi_path = (
            f"feature_importance{'_fold_' + str(fold) if fold is not None else ''}.json"
        )
        with open(fi_path, "w") as f:
            json.dump(fi_dict, f, indent=2)
        mlflow.log_artifact(fi_path)

    return metrics, cm_path, fi_path
