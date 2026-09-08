"""Training metrics and scalar recorder."""

import numpy as np
import torch
from sklearn import metrics


def calculate_metrics_for_train(labels, logits):
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == labels).float().mean().item()
    truth = labels.detach().cpu().numpy()
    scores = probabilities.detach().cpu().numpy()
    average_precision = metrics.average_precision_score(truth, scores)

    if np.unique(truth).size < 2:
        return None, None, accuracy, average_precision
    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(truth, scores)
    auc = metrics.auc(false_positive_rate, true_positive_rate)
    false_negative_rate = 1 - true_positive_rate
    eer = false_positive_rate[np.nanargmin(np.abs(false_negative_rate - false_positive_rate))]
    return auc, eer, accuracy, average_precision


class Recorder:
    def __init__(self):
        self.clear()

    def update(self, value, count=1):
        if value is not None:
            self.sum += value * count
            self.count += count

    def average(self):
        return None if self.count == 0 else self.sum / self.count

    def clear(self):
        self.sum = 0.0
        self.count = 0
