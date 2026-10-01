"""Phase 7A Unit Tests — Top-3 Predictions SHOULD-HAVE Feature."""

import pytest
from unittest.mock import MagicMock
from app.services.classification_service import ClassificationService, ClassificationError
from app.services.inference_service import InferenceService
from app.core.config import settings


def _make_mock_inference_service(predictions):
    svc = MagicMock(spec=InferenceService)
    svc.run_inference.return_value = {
        "model": "google/siglip2-base-patch16-512",
        "device": "cpu",
        "predictions": predictions,
        "inference_time_ms": 150.0,
        "status": "success",
    }
    return svc


# 1. Six candidates produce exactly three top-ranked entries for top-3 output
def test_six_candidates_produce_exactly_three_top_entries():
    candidates = [
        {"label": "Drone", "score": 0.05},
        {"label": "Helicopter", "score": 0.42},
        {"label": "Tank", "score": 0.31},
        {"label": "Ship", "score": 0.12},
        {"label": "Fighter Aircraft", "score": 0.08},
        {"label": "Military Vehicle", "score": 0.02},
    ]
    mock_svc = _make_mock_inference_service(candidates)
    service = ClassificationService(inference_svc=mock_svc)

    result = service.classify_image(b"dummy_bytes", filename="test.jpg")

    assert "top_3" in result
    assert len(result["top_3"]) == 3
    assert result["top_3"][0]["label"] == "Helicopter"
    assert result["top_3"][1]["label"] == "Tank"
    assert result["top_3"][2]["label"] == "Ship"


# 2. Correct ranking (strict descending score order)
def test_top3_correct_ranking_order():
    candidates = [
        {"label": "Ship", "score": 0.15},
        {"label": "Tank", "score": 0.85},
        {"label": "Helicopter", "score": 0.55},
        {"label": "Drone", "score": 0.02},
    ]
    mock_svc = _make_mock_inference_service(candidates)
    service = ClassificationService(inference_svc=mock_svc)

    result = service.classify_image(b"dummy_bytes", filename="test.jpg")
    top_3 = result["top_3"]

    assert len(top_3) == 3
    assert top_3[0]["score"] >= top_3[1]["score"] >= top_3[2]["score"]
    assert top_3[0]["label"] == "Tank"
    assert top_3[1]["label"] == "Helicopter"
    assert top_3[2]["label"] == "Ship"


# 3. Deterministic ranking across multiple invocations
def test_top3_deterministic_ranking():
    candidates = [
        {"label": "Drone", "score": 0.20},
        {"label": "Military Vehicle", "score": 0.60},
        {"label": "Fighter Aircraft", "score": 0.40},
    ]
    mock_svc = _make_mock_inference_service(candidates)
    service = ClassificationService(inference_svc=mock_svc)

    result1 = service.classify_image(b"dummy_bytes", filename="test.jpg")
    result2 = service.classify_image(b"dummy_bytes", filename="test.jpg")

    assert result1["top_3"] == result2["top_3"]


# 4. Correct label-score pairing is preserved
def test_top3_correct_label_score_pairing():
    candidates = [
        {"label": "Fighter Aircraft", "score": 0.7712},
        {"label": "Tank", "score": 0.1234},
        {"label": "Drone", "score": 0.0555},
    ]
    mock_svc = _make_mock_inference_service(candidates)
    service = ClassificationService(inference_svc=mock_svc)

    result = service.classify_image(b"dummy_bytes", filename="test.jpg")
    top_3 = result["top_3"]

    assert top_3[0] == {"label": "Fighter Aircraft", "score": 0.7712}
    assert top_3[1] == {"label": "Tank", "score": 0.1234}
    assert top_3[2] == {"label": "Drone", "score": 0.0555}


# 5. Tie handling is deterministic
def test_top3_tie_handling_is_deterministic():
    # Two identical scores; tie-breaker must use centralized taxonomy index
    candidates = [
        {"label": "Helicopter", "score": 0.50},  # taxonomy index 1
        {"label": "Fighter Aircraft", "score": 0.50},  # taxonomy index 0
        {"label": "Tank", "score": 0.10},
    ]
    mock_svc = _make_mock_inference_service(candidates)
    service = ClassificationService(inference_svc=mock_svc)

    result = service.classify_image(b"dummy_bytes", filename="test.jpg")
    top_3 = result["top_3"]

    # Fighter Aircraft precedes Helicopter due to taxonomy order
    assert top_3[0]["label"] == "Fighter Aircraft"
    assert top_3[1]["label"] == "Helicopter"
    assert top_3[2]["label"] == "Tank"


# 6. Fewer-than-three candidate handling (safe behavior without fabricating entries)
def test_top3_fewer_than_three_candidates():
    # Only 2 candidates provided
    candidates = [
        {"label": "Tank", "score": 0.70},
        {"label": "Ship", "score": 0.30},
    ]
    mock_svc = _make_mock_inference_service(candidates)
    service = ClassificationService(inference_svc=mock_svc)

    result = service.classify_image(b"dummy_bytes", filename="test.jpg")
    top_3 = result["top_3"]

    assert len(top_3) == 2
    assert top_3[0]["label"] == "Tank"
    assert top_3[1]["label"] == "Ship"

    # Only 1 candidate provided
    mock_svc_single = _make_mock_inference_service([{"label": "Drone", "score": 0.99}])
    service_single = ClassificationService(inference_svc=mock_svc_single)
    result_single = service_single.classify_image(b"dummy_bytes", filename="test.jpg")

    assert len(result_single["top_3"]) == 1
    assert result_single["top_3"][0]["label"] == "Drone"


# 7. Empty candidate handling
def test_top3_empty_candidates_handled_safely():
    mock_svc = _make_mock_inference_service([])
    service = ClassificationService(inference_svc=mock_svc)

    with pytest.raises(ClassificationError) as exc_info:
        service.classify_image(b"dummy_bytes", filename="test.jpg")

    assert exc_info.value.code == "INVALID_INFERENCE_RESULT"


# 8. Malformed candidate handling
def test_top3_malformed_candidates_handled_safely():
    mock_svc = _make_mock_inference_service("not-a-list")
    service = ClassificationService(inference_svc=mock_svc)

    with pytest.raises(ClassificationError) as exc_info:
        service.classify_image(b"dummy_bytes", filename="test.jpg")

    assert exc_info.value.code == "INVALID_INFERENCE_RESULT"


# 9. No second model inference (inference service called exactly once)
def test_top3_no_second_model_inference():
    candidates = [
        {"label": "Tank", "score": 0.50},
        {"label": "Helicopter", "score": 0.30},
        {"label": "Ship", "score": 0.20},
    ]
    mock_svc = _make_mock_inference_service(candidates)
    service = ClassificationService(inference_svc=mock_svc)

    service.classify_image(b"dummy_bytes", filename="test.jpg")

    # Strictly 1 invocation of inference engine
    assert mock_svc.run_inference.call_count == 1


# 10. Existing primary prediction remains unchanged (rank 1 is identical)
def test_top3_primary_prediction_remains_unchanged():
    candidates = [
        {"label": "Drone", "score": 0.10},
        {"label": "Military Vehicle", "score": 0.75},
        {"label": "Tank", "score": 0.15},
    ]
    mock_svc = _make_mock_inference_service(candidates)
    service = ClassificationService(inference_svc=mock_svc)

    result = service.classify_image(b"dummy_bytes", filename="test.jpg")

    assert result["prediction"]["label"] == result["top_3"][0]["label"]
    assert result["prediction"]["score"] == result["top_3"][0]["score"]
    assert result["prediction"]["label"] == "Military Vehicle"
    assert result["prediction"]["score"] == 0.75
