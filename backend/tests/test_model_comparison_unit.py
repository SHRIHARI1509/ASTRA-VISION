import pytest
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock

from app.core.config import settings
from app.services.classification_service import classification_service
from app.models.base_adapter import ModelAdapter, ModelInferenceResult, Prediction
from app.models.siglip2_adapter import SigLIP2Adapter
from app.evaluation.clip_adapter import CLIPAdapter
from app.evaluation.comparator import ModelComparator, COMPARISON_CLASSES


class MockAdapter(ModelAdapter):
    """Controlled mock adapter for testing comparison metrics and validation."""

    def __init__(self, model_id: str, return_predictions: list):
        self._model_id = model_id
        self._predictions = return_predictions
        self._is_loaded = True
        self._call_count = 0

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def initialization_time_s(self) -> float:
        return 0.1

    def load(self) -> None:
        pass

    def predict(self, image: Image.Image, candidate_labels=None) -> ModelInferenceResult:
        self._call_count += 1
        preds = self._predictions.pop(0) if self._predictions else [
            Prediction(label=COMPARISON_CLASSES[0], score=0.99)
        ]
        return ModelInferenceResult(
            model=self._model_id,
            device="cpu",
            predictions=preds,
            inference_time_ms=10.0,
            status="success",
        )


@pytest.fixture
def synthetic_manifest():
    return [
        {
            "id": f"sample_{i:04d}",
            "filename": f"sample_{i}.jpg",
            "path": f"held_out_dataset/{COMPARISON_CLASSES[i % 6]}/sample_{i}.jpg",
            "ground_truth": COMPARISON_CLASSES[i % 6],
            "split": "held_out_test",
        }
        for i in range(30)
    ]


def test_1_identical_test_set_membership(synthetic_manifest):
    """1. Verify both models evaluate the exact same test-set membership without divergence."""
    preds_a = [[Prediction(label=s["ground_truth"], score=0.9)] for s in synthetic_manifest]
    preds_b = [[Prediction(label=s["ground_truth"], score=0.8)] for s in synthetic_manifest]

    adapter_a = MockAdapter("model_a", preds_a)
    adapter_b = MockAdapter("model_b", preds_b)

    comparator = ModelComparator(
        model_a_adapter=adapter_a,
        model_b_adapter=adapter_b,
    )

    # Fake images in evaluate_model using a mock Image.open
    with pytest.MonkeyPatch.context() as m:
        m.setattr(Image, "open", lambda p: Image.new("RGB", (50, 50)))
        res_a = comparator.evaluate_model(adapter_a, "Model A", synthetic_manifest)
        res_b = comparator.evaluate_model(adapter_b, "Model B", synthetic_manifest)

    # Identical sample count and order
    assert len(res_a["per_sample"]) == len(synthetic_manifest)
    assert len(res_b["per_sample"]) == len(synthetic_manifest)
    for sa, sb, orig in zip(res_a["per_sample"], res_b["per_sample"], synthetic_manifest):
        assert sa["id"] == sb["id"] == orig["id"]
        assert sa["ground_truth"] == sb["ground_truth"] == orig["ground_truth"]


def test_2_identical_class_taxonomy():
    """2. Verify both models are constrained to the identical 6-class taxonomy."""
    expected = [
        "Tank",
        "Drone",
        "Fighter Aircraft",
        "Military Vehicle",
        "Ship",
        "Helicopter",
    ]
    assert COMPARISON_CLASSES == expected
    # All classes exist in centralized settings
    for cls_name in COMPARISON_CLASSES:
        assert cls_name in settings.CANDIDATE_CATEGORIES


def test_3_deterministic_evaluation(synthetic_manifest):
    """3. Verify repeated evaluation on the same inputs yields identical metric values."""
    preds_1 = [[Prediction(label=s["ground_truth"], score=0.9)] for s in synthetic_manifest]
    preds_2 = [[Prediction(label=s["ground_truth"], score=0.9)] for s in synthetic_manifest]

    adapter_1 = MockAdapter("model_test", preds_1)
    adapter_2 = MockAdapter("model_test", preds_2)

    comparator = ModelComparator(model_a_adapter=adapter_1, model_b_adapter=adapter_2)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(Image, "open", lambda p: Image.new("RGB", (50, 50)))
        r1 = comparator.evaluate_model(adapter_1, "Model", synthetic_manifest)
        r2 = comparator.evaluate_model(adapter_2, "Model", synthetic_manifest)

    assert r1["top1_accuracy"] == r2["top1_accuracy"] == 1.0
    assert r1["macro_precision"] == r2["macro_precision"] == 1.0
    assert r1["macro_recall"] == r2["macro_recall"] == 1.0
    assert r1["macro_f1"] == r2["macro_f1"] == 1.0
    assert r1["confusion_matrix"]["counts"] == r2["confusion_matrix"]["counts"]


def test_4_correct_top1_calculation(synthetic_manifest):
    """4. Correct Top-1 calculation when exactly 15 of 30 samples are correct."""
    # First 15 correct, next 15 incorrect
    preds = []
    for i, s in enumerate(synthetic_manifest):
        if i < 15:
            preds.append([Prediction(label=s["ground_truth"], score=0.8)])
        else:
            wrong_label = COMPARISON_CLASSES[(COMPARISON_CLASSES.index(s["ground_truth"]) + 1) % 6]
            preds.append([Prediction(label=wrong_label, score=0.7)])

    adapter = MockAdapter("test_top1", preds)
    comparator = ModelComparator(model_a_adapter=adapter)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(Image, "open", lambda p: Image.new("RGB", (50, 50)))
        res = comparator.evaluate_model(adapter, "Model", synthetic_manifest)

    assert res["top1_accuracy"] == 0.5000


def test_5_to_8_macro_and_per_class_metrics():
    """5-8. Correct calculation of macro precision, recall, F1, and per-class metrics."""
    from app.evaluation.metrics import compute_classification_metrics

    # 2 Tank (both correct), 2 Drone (1 correct, 1 predicted as Tank)
    y_true = ["Tank", "Tank", "Drone", "Drone"]
    y_pred = ["Tank", "Tank", "Drone", "Tank"]

    metrics = compute_classification_metrics(y_true, y_pred, COMPARISON_CLASSES)

    # Tank: TP=2, FP=1, FN=0 -> Prec=2/3=0.6667, Rec=2/2=1.0, F1=0.8000
    tank = metrics["per_class"]["Tank"]
    assert tank["precision"] == 0.6667
    assert tank["recall"] == 1.0000
    assert tank["f1_score"] == 0.8000

    # Drone: TP=1, FP=0, FN=1 -> Prec=1/1=1.0, Rec=1/2=0.5, F1=0.6667
    drone = metrics["per_class"]["Drone"]
    assert drone["precision"] == 1.0000
    assert drone["recall"] == 0.5000
    assert drone["f1_score"] == 0.6667

    # Others: Prec=0, Rec=0, F1=0
    # Macro across 6 classes:
    expected_prec = round((0.6667 + 1.0000) / 6, 4)
    expected_rec = round((1.0000 + 0.5000) / 6, 4)
    assert metrics["macro_precision"] == expected_prec
    assert metrics["macro_recall"] == expected_rec


def test_9_confusion_matrix_dimensions_and_order():
    """9. Correct confusion matrix dimensions and identical class ordering."""
    from app.evaluation.metrics import compute_confusion_matrix

    y_true = ["Tank", "Drone", "Helicopter"]
    y_pred = ["Tank", "Drone", "Ship"]

    cm = compute_confusion_matrix(y_true, y_pred, COMPARISON_CLASSES)

    assert cm["classes"] == COMPARISON_CLASSES
    assert len(cm["counts"]) == 6
    assert all(len(row) == 6 for row in cm["counts"])
    # Helicopter row is index 5
    helo_idx = COMPARISON_CLASSES.index("Helicopter")
    ship_idx = COMPARISON_CLASSES.index("Ship")
    assert cm["counts"][helo_idx][ship_idx] == 1


def test_10_empty_input_handling():
    """10. Empty manifest handled gracefully without division by zero."""
    from app.evaluation.metrics import compute_classification_metrics

    metrics = compute_classification_metrics([], [], COMPARISON_CLASSES)
    assert metrics["total_samples"] == 0
    assert metrics["overall_accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0


def test_11_malformed_prediction_handling(synthetic_manifest):
    """11. Malformed predictions with non-taxonomy labels do not crash metrics."""
    preds = [[Prediction(label="UnrecognizedForeignObject", score=0.5)] for _ in synthetic_manifest]
    adapter = MockAdapter("malformed_test", preds)
    comparator = ModelComparator(model_a_adapter=adapter)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(Image, "open", lambda p: Image.new("RGB", (50, 50)))
        res = comparator.evaluate_model(adapter, "Model", synthetic_manifest)

    assert res["top1_accuracy"] == 0.0


def test_12_model_a_b_result_schema_validation(synthetic_manifest):
    """12. Validate that Model A and Model B results follow identical expected schema."""
    preds = [[Prediction(label=s["ground_truth"], score=0.8)] for s in synthetic_manifest]
    adapter = MockAdapter("schema_test", preds)
    comparator = ModelComparator(model_a_adapter=adapter)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(Image, "open", lambda p: Image.new("RGB", (50, 50)))
        res = comparator.evaluate_model(adapter, "Schema Model", synthetic_manifest)

    required_keys = [
        "name", "identifier", "device", "top1_accuracy", "top3_accuracy",
        "macro_precision", "macro_recall", "macro_f1", "per_class",
        "confusion_matrix", "latency", "per_sample"
    ]
    for key in required_keys:
        assert key in res


def test_13_production_inference_remains_untouched():
    """13. Production classification service continues to use SigLIP 2 with unaltered config."""
    assert settings.MODEL_ID == "google/siglip2-base-patch16-512"
    assert classification_service is not None
    assert settings.UNCERTAINTY_SCORE_THRESHOLD == 0.0100
    assert settings.UNCERTAINTY_MARGIN_THRESHOLD == 0.0200


def test_14_no_ensemble_behavior(synthetic_manifest):
    """14. Models are evaluated strictly in isolation; outputs are never averaged or combined."""
    preds_a = [[Prediction(label="Tank", score=0.9)] for _ in synthetic_manifest]
    preds_b = [[Prediction(label="Ship", score=0.9)] for _ in synthetic_manifest]

    adapter_a = MockAdapter("a", preds_a)
    adapter_b = MockAdapter("b", preds_b)

    comparator = ModelComparator(model_a_adapter=adapter_a, model_b_adapter=adapter_b)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(Image, "open", lambda p: Image.new("RGB", (50, 50)))
        res_a = comparator.evaluate_model(adapter_a, "A", synthetic_manifest)
        res_b = comparator.evaluate_model(adapter_b, "B", synthetic_manifest)

    # Model A predicts Tank, Model B predicts Ship; no ensemble voting takes place
    assert all(s["predicted"] == "Tank" for s in res_a["per_sample"])
    assert all(s["predicted"] == "Ship" for s in res_b["per_sample"])


def test_15_no_test_set_mutation(synthetic_manifest):
    """15. The held-out test manifest remains immutable during and after evaluation."""
    original_ids = [s["id"] for s in synthetic_manifest]
    original_labels = [s["ground_truth"] for s in synthetic_manifest]

    preds = [[Prediction(label="Tank", score=0.9)] for _ in synthetic_manifest]
    adapter = MockAdapter("test_mut", preds)
    comparator = ModelComparator(model_a_adapter=adapter)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(Image, "open", lambda p: Image.new("RGB", (50, 50)))
        comparator.evaluate_model(adapter, "Test", synthetic_manifest)

    assert [s["id"] for s in synthetic_manifest] == original_ids
    assert [s["ground_truth"] for s in synthetic_manifest] == original_labels


def test_16_no_second_inference_from_production_api():
    """16. Verify that production API service routes do not instantiate or query Model B."""
    from app.services.inference_service import inference_service
    # Production inference service model adapter is SigLIP2Adapter, not CLIPAdapter
    assert isinstance(inference_service.adapter, SigLIP2Adapter)
    assert not isinstance(inference_service.adapter, CLIPAdapter)
