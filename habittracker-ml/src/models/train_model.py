import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import coremltools as ct

from omegaconf import DictConfig
from src.models.metrics import (
    log_fold_metrics,
    log_aggregated_cv_metrics,
    log_final_confusion_matrix,
    log_prediction_latency,
    log_batch_drift,
)

log = logging.getLogger(__name__)


def train(df: pd.DataFrame, cfg: DictConfig):
    log.info("Starting training")

    X = df.drop(columns=["label", "batch_id"])
    y = df["label"]
    batch_ids = df["batch_id"]

    mlflow.xgboost.autolog(disable=True)

    all_acc, all_f1 = [], []
    cv_y_true, cv_y_pred = [], []

    with mlflow.start_run(run_name=cfg.experiment.name):
        mlflow.log_params(cfg.model.params)

        # Drift check
        log_batch_drift(X, batch_ids)

        # CV
        for fold, batch in enumerate(batch_ids.unique()):
            train_idx = batch_ids != batch
            test_idx = batch_ids == batch

            model = xgb.XGBClassifier(
                **cfg.model.params,
                tree_method="hist",
                random_state=cfg.data.random_state,
            )

            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])

            acc, f1 = log_fold_metrics(
                y[test_idx],
                y_pred,
                fold,
                X.columns,
                model.feature_importances_,
            )

            all_acc.append(acc)
            all_f1.append(f1)

            cv_y_true.extend(y[test_idx].tolist())
            cv_y_pred.extend(y_pred.tolist())

            log.info(
                "Fold %d → Accuracy: %.4f | F1: %.4f",
                fold,
                acc,
                f1,
            )

        # Aggregated CV
        log_aggregated_cv_metrics(cv_y_true, cv_y_pred)

        mlflow.log_metric("accuracy_mean", float(np.mean(all_acc)))
        mlflow.log_metric("accuracy_std", float(np.std(all_acc)))
        mlflow.log_metric("f1_mean", float(np.mean(all_f1)))
        mlflow.log_metric("f1_std", float(np.std(all_f1)))

        # Final model
        final_model = xgb.XGBClassifier(
            **cfg.model.params,
            tree_method="hist",
            random_state=cfg.data.random_state,
        )
        final_model.fit(X, y)

        log_final_confusion_matrix(y, final_model.predict(X))

        # Latency
        log_prediction_latency(final_model, X.iloc[:1])

        # Model logging
        mlflow.xgboost.log_model(final_model, name="model")

        # CoreML export
        if cfg.coreML_export:
            try:
                coreml_model = ct.converters.xgboost.convert(
                    final_model,
                    feature_names=list(X.columns),
                    mode="classifier",
                )

                metadata = {
                    "class_labels": list(map(int, np.unique(y))),
                    "decision_threshold": cfg.model.get("threshold", 0.5),
                }
                coreml_model.user_defined_metadata.update(
                    {k: str(v) for k, v in metadata.items()}
                )

                coreml_path = "habit_tracker.mlpackage"
                coreml_model.save(coreml_path)
                mlflow.log_artifact(coreml_path)

                log.info("CoreML model exported successfully")

            except Exception as e:
                log.warning("CoreML export failed: %s", e)

    return final_model
