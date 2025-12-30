import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from omegaconf import DictConfig
import logging
import coremltools as ct
from src.models.metrics import log_classification_metrics

log = logging.getLogger(__name__)


def train(df: pd.DataFrame, cfg: DictConfig):
    """
    Train XGBoost using Leave-One-Batch-Out cross-validation with MLflow logging.
    Optionally exports a CoreML model.
    """
    log.info("Starting training...")

    if "batch_id" not in df.columns:
        raise ValueError("DataFrame must contain 'batch_id' column for CV")

    X = df.drop(columns=["label", "batch_id"])
    y = df["label"]
    batch_ids = df["batch_id"]
    unique_batches = batch_ids.unique()
    log.info("Found %d batches", len(unique_batches))

    mlflow.xgboost.autolog(disable=True)

    all_acc, all_f1 = [], []
    feature_importances = []

    with mlflow.start_run(run_name=cfg.experiment.name):
        # Log static model params once
        mlflow.log_params(cfg.model.params)

        for fold, batch in enumerate(unique_batches):
            log.info("Training fold: holding out batch %s", batch)

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

            # Log metrics, confusion matrix, feature importance
            fi = (
                model.feature_importances_
                if hasattr(model, "feature_importances_")
                else None
            )
            metrics, cm_path, fi_path = log_classification_metrics(
                y_test, y_pred, X.columns, fi, fold
            )

            all_acc.append(metrics[f"accuracy_fold{fold}"])
            all_f1.append(metrics[f"f1_score_fold{fold}"])
            if fi is not None:
                feature_importances.append(fi)

            log.info(
                "Validation on Fold %d -> Accuracy: %.4f | F1: %.4f",
                fold,
                metrics[f"accuracy_fold{fold}"],
                metrics[f"f1_score_fold{fold}"],
            )

        # Aggregate CV metrics
        mlflow.log_metric("accuracy_mean", np.mean(all_acc))
        mlflow.log_metric("accuracy_std", np.std(all_acc))
        mlflow.log_metric("f1_mean", np.mean(all_f1))
        mlflow.log_metric("f1_std", np.std(all_f1))
        log.info(
            "Cross Validation completed -> Accuracy: %.4f ± %.4f | F1: %.4f ± %.4f",
            np.mean(all_acc),
            np.std(all_acc),
            np.mean(all_f1),
            np.std(all_f1),
        )

        # Train final model on all data
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
        mlflow.xgboost.log_model(final_model, name="model")

        # CoreML export
        if cfg.coreML_export:
            try:
                coreml_model = ct.converters.xgboost.convert(
                    final_model,
                    feature_names=list(X.columns),
                )
                coreml_model.save("habit_tracker.mlpackage")
                mlflow.log_artifact("habit_tracker.mlpackage")
                log.info("CoreML model exported successfully")
            except Exception as e:
                log.warning("CoreML conversion failed: %s", e)

    return final_model, all_acc, all_f1, feature_importances
