from typing import List, Dict, Any, Optional


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
) -> Dict[str, Any]:
    """Compute raw count and row-normalized confusion matrices for given classes."""
    n_classes = len(classes)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    counts = [[0 for _ in range(n_classes)] for _ in range(n_classes)]

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in class_to_idx and pred_label in class_to_idx:
            row = class_to_idx[true_label]
            col = class_to_idx[pred_label]
            counts[row][col] += 1

    # Row-normalized matrix (percentages of class true labels)
    normalized = [[0.0 for _ in range(n_classes)] for _ in range(n_classes)]
    for row in range(n_classes):
        row_sum = sum(counts[row])
        if row_sum > 0:
            for col in range(n_classes):
                normalized[row][col] = round(counts[row][col] / row_sum, 4)

    return {
        "classes": classes,
        "counts": counts,
        "normalized": normalized,
    }


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
    top_3_preds: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    """Compute overall accuracy, per-class precision/recall/F1/support,
    macro averages, and optional Top-3 accuracy.
    """
    total = len(y_true)
    if total == 0:
        return {
            "total_samples": 0,
            "correct_samples": 0,
            "incorrect_samples": 0,
            "overall_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "top_3_accuracy": None,
        }

    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred, classes)
    counts = cm["counts"]
    n_classes = len(classes)

    per_class: Dict[str, Dict[str, Any]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    correct_total = 0

    for i, cls_name in enumerate(classes):
        tp = counts[i][i]
        correct_total += tp
        fp = sum(counts[r][i] for r in range(n_classes) if r != i)
        fn = sum(counts[i][c] for c in range(n_classes) if c != i)
        support = tp + fn

        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        rec = round(tp / support, 4) if support > 0 else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        per_class[cls_name] = {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    overall_accuracy = round(correct_total / total, 4)
    macro_precision = round(sum(precisions) / n_classes, 4) if n_classes > 0 else 0.0
    macro_recall = round(sum(recalls) / n_classes, 4) if n_classes > 0 else 0.0
    macro_f1 = round(sum(f1s) / n_classes, 4) if n_classes > 0 else 0.0

    # Top-3 Accuracy calculation
    top_3_accuracy: Optional[float] = None
    if top_3_preds is not None and len(top_3_preds) == total:
        top_3_correct = sum(
            1 for true_lbl, top_3_list in zip(y_true, top_3_preds)
            if true_lbl in top_3_list
        )
        top_3_accuracy = round(top_3_correct / total, 4)

    return {
        "total_samples": total,
        "correct_samples": correct_total,
        "incorrect_samples": total - correct_total,
        "overall_accuracy": overall_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "top_3_accuracy": top_3_accuracy,
    }
