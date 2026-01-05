import os
import logging
import pandas as pd
from omegaconf import DictConfig

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test_model(cfg: DictConfig):
    """Run test using cfg"""
    test_path = cfg.data.test_path
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found: {test_path}")

    import xgboost as xgb
    import mlflow
    from src.models.metrics import (
        log_fold_metrics,
        log_final_confusion_matrix,
        log_aggregated_cv_metrics,
        log_prediction_latency,
    )

    df_test = pd.read_csv(test_path)
    if "label" not in df_test.columns:
        raise ValueError("'label' column is required in test data")

    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    # Load model: either local or from MLflow
    if cfg.model_path:
        model = xgb.XGBClassifier()
        model.load_model(cfg.model_path)
    else:
        mlflow_client = mlflow.tracking.MlflowClient()
        exp = mlflow_client.get_experiment_by_name(cfg.experiment.name)
        runs = mlflow_client.search_runs(
            exp.experiment_id, order_by=["attributes.start_time DESC"]
        )
        latest_run = runs[0]
        model_uri = f"runs:/{latest_run.info.run_id}/model"
        model = mlflow.xgboost.load_model(model_uri)

    # Predict & log
    y_pred = model.predict(X_test)
    acc, f1 = log_fold_metrics(
        y_test,
        y_pred,
        fold=0,
        feature_names=X_test.columns,
        feature_importances=model.feature_importances_
        if hasattr(model, "feature_importances_")
        else None,
    )
    logging.info("Test Metrics → Accuracy: %.4f | F1: %.4f", acc, f1)
    log_final_confusion_matrix(y_test, y_pred)
    log_prediction_latency(model, X_test.iloc[:1])
    log_aggregated_cv_metrics(y_test.tolist(), y_pred.tolist())
