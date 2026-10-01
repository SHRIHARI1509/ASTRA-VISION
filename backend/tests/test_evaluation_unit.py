import pytest
from typing import List
from app.core.config import settings
from app.services.classification_service import classification_service
from app.evaluation.metrics import (
    compute_confusion_matrix,
    compute_classification_metrics,
)

SAMPLE_CLASSES = [
    "Fighter Aircraft",
    "Helicopter",
    "Tank",
    "Ship",
    "Military Vehicle",
    "Drone",
]


def test_metric_calculation_perfect_predictions():
    """1. Perfect predictions: 100% accuracy, precision, recall, and F1."""
    y_true = ["Tank", "Helicopter", "Ship", "Fighter Aircraft", "Drone", "Military Vehicle"]
    y_pred = ["Tank", "Helicopter", "Ship", "Fighter Aircraft", "Drone", "Military Vehicle"]

    metrics = compute_classification_metrics(y_true, y_pred, SAMPLE_CLASSES)

    assert metrics["total_samples"] == 6
    assert metrics["correct_samples"] == 6
    assert metrics["incorrect_samples"] == 0
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f1"] == 1.0

    for cls_name in SAMPLE_CLASSES:
        assert metrics["per_class"][cls_name]["precision"] == 1.0
        assert metrics["per_class"][cls_name]["recall"] == 1.0
        assert metrics["per_class"][cls_name]["f1_score"] == 1.0
        assert metrics["per_class"][cls_name]["support"] == 1


def test_metric_calculation_completely_incorrect():
    """2. Completely incorrect predictions: 0% accuracy, precision, recall, and F1."""
    y_true = ["Tank", "Helicopter", "Ship"]
    y_pred = ["Ship", "Tank", "Helicopter"]  # All rotated / incorrect

    metrics = compute_classification_metrics(y_true, y_pred, SAMPLE_CLASSES)

    assert metrics["total_samples"] == 3
    assert metrics["correct_samples"] == 0
    assert metrics["incorrect_samples"] == 3
    assert metrics["overall_accuracy"] == 0.0

    for cls_name in ["Tank", "Helicopter", "Ship"]:
        assert metrics["per_class"][cls_name]["precision"] == 0.0
        assert metrics["per_class"][cls_name]["recall"] == 0.0
        assert metrics["per_class"][cls_name]["f1_score"] == 0.0


def test_metric_calculation_mixed_predictions():
    """3. Mixed predictions with known expected precision/recall values."""
    # 2 Tanks (1 correct, 1 predicted as Vehicle)
    # 2 Helicopters (both correct)
    # 1 Ship (predicted as Tank)
    y_true = ["Tank", "Tank", "Helicopter", "Helicopter", "Ship"]
    y_pred = ["Tank", "Military Vehicle", "Helicopter", "Helicopter", "Tank"]

    metrics = compute_classification_metrics(y_true, y_pred, SAMPLE_CLASSES)

    assert metrics["total_samples"] == 5
    assert metrics["correct_samples"] == 3
    assert metrics["overall_accuracy"] == round(3 / 5, 4)

    # Tank: TP=1, FP=1 (Ship predicted as Tank), FN=1 (Tank predicted as Vehicle)
    # Precision = 1 / (1 + 1) = 0.5; Recall = 1 / 2 = 0.5
    tank_metrics = metrics["per_class"]["Tank"]
    assert tank_metrics["precision"] == 0.5
    assert tank_metrics["recall"] == 0.5
    assert tank_metrics["f1_score"] == 0.5
    assert tank_metrics["support"] == 2

    # Helicopter: TP=2, FP=0, FN=0 -> Precision=1.0, Recall=1.0
    helo_metrics = metrics["per_class"]["Helicopter"]
    assert helo_metrics["precision"] == 1.0
    assert helo_metrics["recall"] == 1.0
    assert helo_metrics["support"] == 2

    # Ship: TP=0, FP=0, FN=1 -> Precision=0.0, Recall=0.0
    ship_metrics = metrics["per_class"]["Ship"]
    assert ship_metrics["precision"] == 0.0
    assert ship_metrics["recall"] == 0.0
    assert ship_metrics["support"] == 1


def test_metric_calculation_single_class_edge_case():
    """4. Single-class taxonomy edge case handled safely."""
    classes = ["Tank"]
    y_true = ["Tank", "Tank", "Tank"]
    y_pred = ["Tank", "Tank", "Tank"]

    metrics = compute_classification_metrics(y_true, y_pred, classes)

    assert metrics["total_samples"] == 3
    assert metrics["overall_accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["per_class"]["Tank"]["support"] == 3


def test_metric_calculation_missing_class_in_predictions():
    """5. Missing class in predictions: model never predicts Drone."""
    y_true = ["Drone", "Tank", "Ship"]
    y_pred = ["Tank", "Tank", "Ship"]  # Drone predicted as Tank, Drone never predicted

    metrics = compute_classification_metrics(y_true, y_pred, SAMPLE_CLASSES)

    drone_metrics = metrics["per_class"]["Drone"]
    assert drone_metrics["support"] == 1
    assert drone_metrics["true_positives"] == 0
    assert drone_metrics["precision"] == 0.0
    assert drone_metrics["recall"] == 0.0
    assert drone_metrics["f1_score"] == 0.0


def test_metric_calculation_class_imbalance():
    """6. Class imbalance: support varies widely across classes."""
    # 5 Tanks, 1 Helicopter
    y_true = ["Tank"] * 5 + ["Helicopter"]
    y_pred = ["Tank"] * 5 + ["Tank"]  # Helicopter misclassified as Tank

    metrics = compute_classification_metrics(y_true, y_pred, SAMPLE_CLASSES)

    assert metrics["total_samples"] == 6
    assert metrics["correct_samples"] == 5
    # Overall accuracy is 5/6 = 0.8333
    assert metrics["overall_accuracy"] == 0.8333

    # Macro recall: Tank=1.0, Helicopter=0.0, others=0.0 -> mean across 6 classes
    assert metrics["per_class"]["Tank"]["support"] == 5
    assert metrics["per_class"]["Helicopter"]["support"] == 1
    assert metrics["macro_recall"] == round(1.0 / 6, 4)


def test_confusion_matrix_generation_and_normalization():
    """7. Confusion matrix generation produces correct dimensions and row-normalized probabilities."""
    y_true = ["Tank", "Tank", "Helicopter", "Drone"]
    y_pred = ["Tank", "Ship", "Helicopter", "Drone"]

    cm = compute_confusion_matrix(y_true, y_pred, SAMPLE_CLASSES)

    assert len(cm["classes"]) == 6
    assert len(cm["counts"]) == 6
    assert len(cm["normalized"]) == 6

    # Verify counts for Tank row (index 2 in SAMPLE_CLASSES)
    tank_idx = SAMPLE_CLASSES.index("Tank")
    ship_idx = SAMPLE_CLASSES.index("Ship")
    assert cm["counts"][tank_idx][tank_idx] == 1  # 1 predicted as Tank
    assert cm["counts"][tank_idx][ship_idx] == 1  # 1 predicted as Ship

    # Verify normalized row sums to 1.0 for classes with support
    assert sum(cm["normalized"][tank_idx]) == 1.0
    assert cm["normalized"][tank_idx][tank_idx] == 0.5
    assert cm["normalized"][tank_idx][ship_idx] == 0.5


def test_metric_calculation_empty_inputs():
    """8. Empty input arrays return zeroed metrics without exception."""
    metrics = compute_classification_metrics([], [], SAMPLE_CLASSES)

    assert metrics["total_samples"] == 0
    assert metrics["overall_accuracy"] == 0.0
    assert metrics["macro_precision"] == 0.0


def test_deterministic_repeated_evaluation():
    """9. Evaluation is strictly deterministic across repeated runs."""
    y_true = ["Tank", "Helicopter", "Ship", "Drone"]
    y_pred = ["Tank", "Helicopter", "Helicopter", "Drone"]

    run1 = compute_classification_metrics(y_true, y_pred, SAMPLE_CLASSES)
    run2 = compute_classification_metrics(y_true, y_pred, SAMPLE_CLASSES)

    assert run1 == run2


def test_invalid_or_missing_labels_handled_safely():
    """10. Unrecognized labels outside the taxonomy do not cause crash."""
    y_true = ["UnknownClass", "Tank"]
    y_pred = ["Tank", "AlienCraft"]

    metrics = compute_classification_metrics(y_true, y_pred, SAMPLE_CLASSES)
    assert metrics["total_samples"] == 2
    # Neither unknown class counted towards known class True Positives
    assert metrics["overall_accuracy"] == 0.0


def test_top1_accuracy_strictly_used_for_primary_metrics():
    """11. Primary accuracy strictly relies on Top-1 predictions."""
    y_true = ["Tank"]
    y_pred = ["Military Vehicle"]  # Top-1 is wrong
    top_3 = [["Military Vehicle", "Tank", "Drone"]]  # Top-3 contains Tank

    metrics = compute_classification_metrics(y_true, y_pred, SAMPLE_CLASSES, top_3_preds=top_3)

    # Primary overall accuracy must be 0.0
    assert metrics["overall_accuracy"] == 0.0


def test_optional_top3_accuracy_clearly_separated():
    """12. Top-3 accuracy is computed as a clearly labeled secondary metric."""
    y_true = ["Tank", "Helicopter"]
    y_pred = ["Military Vehicle", "Helicopter"]  # 1/2 correct Top-1
    top_3 = [
        ["Military Vehicle", "Tank", "Drone"],      # Tank is rank 2 -> Top-3 match!
        ["Helicopter", "Fighter Aircraft", "Ship"], # Helicopter is rank 1 -> Top-3 match!
    ]

    metrics = compute_classification_metrics(y_true, y_pred, SAMPLE_CLASSES, top_3_preds=top_3)

    assert metrics["overall_accuracy"] == 0.5      # Top-1 accuracy is 50%
    assert metrics["top_3_accuracy"] == 1.0        # Top-3 accuracy is 100%


def test_production_inference_behavior_remains_unmodified():
    """13. Evaluation utilities do not modify production service taxonomy or thresholds."""
    assert settings.CANDIDATE_CATEGORIES == [
        "Fighter Aircraft",
        "Helicopter",
        "Tank",
        "Ship",
        "Military Vehicle",
        "Drone",
    ]
    assert settings.UNCERTAINTY_SCORE_THRESHOLD == 0.0100
    assert settings.UNCERTAINTY_MARGIN_THRESHOLD == 0.0200
    assert classification_service is not None
