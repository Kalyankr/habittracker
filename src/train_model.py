import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utils.smoothing import smooth_predictions
import joblib
from utils.logger import init_logger
import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", type=str, default="./data/processed/output_features.npz"
    )
    parser.add_argument("--test_batch", type=str, default="batch_02")
    parser.add_argument("--n_estimators", type=int, default=400)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--smooth_window", type=int, default=3)
    parser.add_argument("--save_model", type=str, default="models/model.pkl")
    args = parser.parse_args()

    # Initialize logger
    init_logger()

    logging.info("Loading features from %s", args.features)

    data = np.load(args.features)
    X, y, batch_ids = data["X"], data["y"], data["batch_ids"]

    logging.info("Training model with %d samples", len(X))
    logging.info(
        "Using test batch %s with %d samples",
        args.test_batch,
        len(X[batch_ids == args.test_batch]),
    )

    train_idx = batch_ids != args.test_batch
    test_idx = batch_ids == args.test_batch

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logging.info("Training model with %d samples", len(X_train))

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=42,
    )

    logging.info("Model: %s", model)

    model.fit(X_train, y_train)

    probs = model.predict(X_test)
    y_pred_raw = (probs >= args.threshold).astype(int)
    y_pred = smooth_predictions(y_pred_raw, args.smooth_window)

    logging.info("Classification Report:")
    logging.info(classification_report(y_test, y_pred))

    logging.info("Confusion Matrix:")
    logging.info(confusion_matrix(y_test, y_pred))

    logging.info("Reports saved to results")
    with open("./results/classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))

    joblib.dump(model, args.save_model)
    logging.info("Model saved to: %s", args.save_model)


if __name__ == "__main__":
    main()
