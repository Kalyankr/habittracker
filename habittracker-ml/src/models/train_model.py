import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.xgboost
from omegaconf import DictConfig
import logging
import coremltools as ct

log = logging.getLogger(__name__)


def train(df: pd.DataFrame, cfg: DictConfig):
    """
    Train the model and log to MLflow.
    """
    log.info("Starting training...")

    # Prepare data
    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )

    # Enable MLflow autologging
    mlflow.xgboost.autolog()

    with mlflow.start_run(run_name=cfg.experiment.name):
        # Log params
        mlflow.log_params(cfg.model.params)

        # Initialize model
        model = xgb.XGBClassifier(
            n_estimators=cfg.model.params.n_estimators,
            max_depth=cfg.model.params.max_depth,
            learning_rate=cfg.model.params.learning_rate,
            objective=cfg.model.params.objective,
            eval_metric=cfg.model.params.eval_metric,
        )

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        log.info(f"Accuracy: {acc}")
        log.info(f"F1 Score: {f1}")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Convert to CoreML (Optional, for deployment)
        # Note: CoreML conversion for XGBoost requires specific handling
        # This is a placeholder for the conversion logic
        try:
            coreml_model = ct.converters.xgboost.convert(
                model, feature_names=list(X.columns)
            )
            coreml_model.save("habit_tracker.mlpackage")
            mlflow.log_artifact("habit_tracker.mlpackage")
            log.info("Model converted to CoreML and saved.")
        except Exception as e:
            log.warning(f"CoreML conversion failed: {e}")

    return model
