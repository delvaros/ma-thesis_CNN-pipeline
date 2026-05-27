import torch

from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd


def cm_breakdown(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    correct_class_0 = cm[0, 0]
    correct_class_1 = cm[1, 1]

    total_class_0 = cm[0, 0] + cm[0, 1]
    total_class_1 = cm[1, 0] + cm[1, 1]

    percent_correct_class_0 = (
        (correct_class_0 / total_class_0) * 100 if total_class_0 > 0 else 0
    )
    percent_correct_class_1 = (
        (correct_class_1 / total_class_1) * 100 if total_class_1 > 0 else 0
    )

    return f"  class0: {percent_correct_class_0:.2f}% of {total_class_0:6d} ,   class1: {percent_correct_class_1:.2f}% of {total_class_1:6d}"


def cm_breakdown_multi(y_true, y_pred):
    y_true_indices = np.argmax(y_true, axis=1)
    y_pred_indices = np.argmax(y_pred, axis=1)

    num_classes = len(y_true[0])

    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=range(num_classes))

    for class_idx in range(num_classes):
        total_samples = cm[class_idx, :].sum()

        s = f"Class {class_idx} ({total_samples})> "

        for pred_class_idx in range(num_classes):
            pred_count = cm[class_idx, pred_class_idx]
            pred_percentage = (
                100.0 * pred_count / total_samples if total_samples > 0 else 0
            )
            s += f"  {pred_class_idx}: {pred_count} )"
        for pred_class_idx in range(num_classes):
            pred_count = cm[class_idx, pred_class_idx]
            pred_percentage = (
                100.0 * pred_count / total_samples if total_samples > 0 else 0
            )
            s += f"  {pred_class_idx}: {pred_percentage:.2f}%"
        print(s)


def dump_predictions(output_file, ys, y_preds, keys):
    if isinstance(y_preds[0], np.ndarray) or isinstance(y_preds[0], list):
        if len(y_preds[0]) == 1:
            y_preds = [k[0] for k in y_preds]

    df = pd.DataFrame({"true": ys, "pred": y_preds, "key": keys})

    df.to_csv(output_file, index=False)
