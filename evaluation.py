import math
import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    mean_squared_error,
    mean_absolute_error,
)


def calc_classification_metrics(pred_scores, pred_labels, labels):
    if len(np.unique(labels)) == 2:  # binary classification
        roc_auc_pred_score = roc_auc_score(labels, pred_scores)
        precisions, recalls, thresholds = precision_recall_curve(labels, pred_scores)
        fscore = (2 * precisions * recalls) / (precisions + recalls)
        fscore[np.isnan(fscore)] = 0
        ix = np.argmax(fscore)
        threshold = thresholds[ix].item()
        pr_auc = auc(recalls, precisions)
        tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
        result = {
            "roc_auc": roc_auc_pred_score,
            "threshold": threshold,
            "pr_auc": pr_auc,
            "recall": recalls[ix].item(),
            "precision": precisions[ix].item(),
            "f1": fscore[ix].item(),
            "tn": tn.item(),
            "fp": fp.item(),
            "fn": fn.item(),
            "tp": tp.item(),
        }
    else:
        acc = (pred_labels == labels).mean()
        f1_micro = f1_score(y_true=labels, y_pred=pred_labels, average="micro")
        f1_macro = f1_score(y_true=labels, y_pred=pred_labels, average="macro")
        f1_weighted = f1_score(y_true=labels, y_pred=pred_labels, average="weighted")

        result = {
            "acc": acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "mcc": matthews_corrcoef(labels, pred_labels),
        }

    return result


def calc_regression_metrics(preds, labels):
    mse = mean_squared_error(labels, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
    }

def calc_imputation_metrics(cat_logits, cat_labels, numerical_logits, numerical_labels):
    output = {}

    correct = 0
    total = 0
    for i, (cat_logit, cat_label) in enumerate(zip(cat_logits, cat_labels)):
        a = (np.argmax(cat_logit, axis=1) == cat_label).sum()
        b = len(cat_label)
        correct += a
        total += b
        output[f"acc{i}"] = a / b
    output["acc"] = correct / total

    numerical_logits_flat = []
    numerical_labels_flat = []
    # print("dbg", numerical_logits[0] - numerical_labels[0])
    for i, (numerical_logit, numerical_label) in enumerate(zip(numerical_logits, numerical_labels)):
        numerical_logits_flat.extend(numerical_logit)
        numerical_labels_flat.extend(numerical_label)
        output[f"rmse{i}"] = math.sqrt(mean_squared_error(np.array(numerical_logit), np.array(numerical_label)))
    output["rmse"] = math.sqrt(mean_squared_error(np.array(numerical_logits_flat), np.array(numerical_labels_flat)))
    
    return output