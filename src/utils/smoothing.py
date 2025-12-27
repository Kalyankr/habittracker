import numpy as np


def smooth_predictions(preds, window=3):
    smoothed = []
    for i in range(len(preds)):
        start = max(0, i - window + 1)
        smoothed.append(int(np.mean(preds[start : i + 1]) >= 0.5))
    return np.array(smoothed)
