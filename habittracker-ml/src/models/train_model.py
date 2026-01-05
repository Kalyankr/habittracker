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
    log_final_confusion_matrix,
    log_aggregated_cv_confusion_matrix,
)

log = logging.getLogger(__name__)


def train(df: pd.DataFrame, cfg: DictConfig):
    """
    Train XGBoost using Leave-One-Batch-Out cross-validation.

    Logs:
      - Per-fold accuracy & F1
      - Per-fold confusion matrices
      - Aggregated CV confusion matrix (REAL performance)
      - Mean / std CV metrics
      - Final confusion matrix (sanity check)
      - Trained XGBoost model
      - Optional CoreML model
    """

    log.info("Starting training...")

    # Validation
    required_cols = {"label", "batch_id"}
    if not required_cols.issubset(df.columns):
        raise ValueError("DataFrame must contain 'label' and 'batch_id' columns")

    # Data
    X = df.drop(columns=["label", "batch_id"])
    y = df["label"]
    batch_ids = df["batch_id"]

    unique_batches = batch_ids.unique()
    log.info("Found %d batches for CV", len(unique_batches))

    # MLflow
    mlflow.xgboost.autolog(disable=True)

    all_acc, all_f1 = [], []
    cv_y_true, cv_y_pred = [], []

    with mlflow.start_run(run_name=cfg.experiment.name):
        # Log hyperparameters once
        mlflow.log_params(cfg.model.params)

        # Cross Validation
        for fold, batch in enumerate(unique_batches):
            log.info("Fold %d | Holding out batch: %s", fold, batch)

            train_idx = batch_ids != batch
            test_idx = batch_ids == batch

            X_train, X_test = X.loc[train_idx], X.loc[test_idx]
            y_train, y_test = y.loc[train_idx], y.loc[test_idx]

            model = xgb.XGBClassifier(
                n_estimators=cfg.model.params.n_estimators,
                max_depth=cfg.model.params.max_depth,
                learning_rate=cfg.model.params.learning_rate,
                objective=cfg.model.params.objective,
                eval_metric=cfg.model.params.eval_metric,
                tree_method="hist",
                random_state=cfg.data.random_state,
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc, f1 = log_fold_metrics(
                y_true=y_test,
                y_pred=y_pred,
                fold=fold,
                feature_names=X.columns,
                feature_importances=model.feature_importances_,
            )

            all_acc.append(acc)
            all_f1.append(f1)

            # Collect for aggregated CV CM
            cv_y_true.extend(y_test.tolist())
            cv_y_pred.extend(y_pred.tolist())

            log.info(
                "Fold %d → Accuracy: %.4f | F1: %.4f",
                fold,
                acc,
                f1,
            )

        # Aggregated CV Confusion Matrix
        log_aggregated_cv_confusion_matrix(
            y_true_all=cv_y_true,
            y_pred_all=cv_y_pred,
        )

        # Aggregate CV Metrics
        mlflow.log_metric("accuracy_mean", float(np.mean(all_acc)))
        mlflow.log_metric("accuracy_std", float(np.std(all_acc)))
        mlflow.log_metric("f1_mean", float(np.mean(all_f1)))
        mlflow.log_metric("f1_std", float(np.std(all_f1)))

        log.info(
            "CV Summary → Accuracy: %.4f ± %.4f | F1: %.4f ± %.4f",
            np.mean(all_acc),
            np.std(all_acc),
            np.mean(all_f1),
            np.std(all_f1),
        )

        # Final Model
        final_model = xgb.XGBClassifier(
            n_estimators=cfg.model.params.n_estimators,
            max_depth=cfg.model.params.max_depth,
            learning_rate=cfg.model.params.learning_rate,
            objective=cfg.model.params.objective,
            eval_metric=cfg.model.params.eval_metric,
            tree_method="hist",
            random_state=cfg.data.random_state,
        )

        final_model.fit(X, y)

        # Final confusion matrix (sanity check only)
        y_final_pred = final_model.predict(X)
        log_final_confusion_matrix(
            y_true=y,
            y_pred=y_final_pred,
        )

        # Log trained model
        mlflow.xgboost.log_model(final_model, name="model")

        # CoreML Export
        if cfg.coreML_export:
            try:
                coreml_model = ct.converters.xgboost.convert(
                    final_model,
                    feature_names=list(X.columns),
                )
                coreml_path = "habit_tracker.mlpackage"
                coreml_model.save(coreml_path)
                mlflow.log_artifact(coreml_path)

                log.info("CoreML model exported successfully")

            except Exception as e:
                log.warning("CoreML export failed: %s", e)

    return final_model, all_acc, all_f1
