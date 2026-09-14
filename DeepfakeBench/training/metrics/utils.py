"""Frame- and video-level evaluation metrics."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn import metrics


def _binary_metrics(labels, scores):
    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(labels, scores)
    auc = metrics.auc(false_positive_rate, true_positive_rate)
    false_negative_rate = 1 - true_positive_rate
    eer = false_positive_rate[np.nanargmin(np.abs(false_negative_rate - false_positive_rate))]
    average_precision = metrics.average_precision_score(labels, scores)
    accuracy = ((scores > 0.5).astype(int) == np.clip(labels, 0, 1)).mean()
    return {"acc": accuracy, "auc": auc, "eer": eer, "ap": average_precision}


def _video_metrics(image_paths, scores, labels):
    grouped = defaultdict(list)
    for path, score, label in zip(image_paths, scores, labels):
        # Manifests may contain Windows separators even when running on Linux.
        normalized_path = str(path).replace("\\", "/")
        grouped[str(Path(normalized_path).parent)].append((float(score), int(label)))
    video_scores = np.asarray([np.mean([item[0] for item in values]) for values in grouped.values()])
    video_labels = np.asarray([round(np.mean([item[1] for item in values])) for values in grouped.values()])
    values = _binary_metrics(video_labels, video_scores)
    return {
        "video_auc": values["auc"],
        "video_eer": values["eer"],
        "video_acc": values["acc"],
    }


def get_test_metrics(y_pred, y_true, img_names):
    scores = np.asarray(y_pred).squeeze()
    labels = np.asarray(y_true).squeeze()
    result = _binary_metrics(labels, scores)
    if img_names and not isinstance(img_names[0], list):
        result.update(_video_metrics(img_names, scores, labels))
    else:
        result.update(
            video_auc=result["auc"],
            video_eer=result["eer"],
            video_acc=result["acc"],
        )
    result.update(pred=scores, label=labels)
    return result


def parse_metric_for_print(metric_dict):
    if not metric_dict:
        return "no validation result"
    parts = []
    for dataset, values in metric_dict.items():
        scalar_values = [
            f"{name}={value:.4f}"
            for name, value in values.items()
            if isinstance(value, (int, float, np.floating))
        ]
        parts.append(f"{dataset}: " + ", ".join(scalar_values))
    return " | ".join(parts)
