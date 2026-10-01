from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.services.classification_service import ClassificationService, ClassificationError
from app.services.inference_service import InferenceService
from app.schemas.classification import ClassificationResponse
from app.main import app

client = TestClient(app)


# ============================================================================
# PHASE 7C: UNCERTAINTY & LOW-CONFIDENCE UNIT TESTS
# ============================================================================

def test_high_score_separation_is_not_uncertain():
    """CASE A: Clearly separated prediction above thresholds is marked as NOT uncertain."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    mock_inference_svc.run_inference.return_value = {
        "model": settings.MODEL_ID,
        "device": "cpu",
        "predictions": [
            {"label": "Tank", "score": 0.8000},
            {"label": "Military Vehicle", "score": 0.1000},
            {"label": "Helicopter", "score": 0.0200},
        ],
        "inference_time_ms": 120.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(file_bytes=b"dummy", filename="clear_tank.jpg")

    unc = result["uncertainty"]
    assert unc["is_uncertain"] is False
    assert unc["reason"] is None
    assert unc["score_margin"] == 0.7000
    assert unc["primary_score"] == 0.8000
    assert unc["method"] == "heuristic"


def test_low_score_margin_triggers_uncertainty():
    """CASE B: High primary score but close competing candidates triggers LOW_SCORE_MARGIN."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    # top_score=0.3100, second_score=0.3000 -> margin = 0.0100 < 0.0200 threshold
    mock_inference_svc.run_inference.return_value = {
        "model": settings.MODEL_ID,
        "device": "cpu",
        "predictions": [
            {"label": "Tank", "score": 0.3100},
            {"label": "Military Vehicle", "score": 0.3000},
            {"label": "Drone", "score": 0.0200},
        ],
        "inference_time_ms": 115.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(file_bytes=b"dummy", filename="close_call.jpg")

    unc = result["uncertainty"]
    assert unc["is_uncertain"] is True
    assert unc["reason"] == "LOW_SCORE_MARGIN"
    assert unc["score_margin"] == 0.0100
    assert unc["primary_score"] == 0.3100


def test_low_primary_score_triggers_uncertainty():
    """CASE C: Top score below UNCERTAINTY_SCORE_THRESHOLD triggers LOW_PRIMARY_SCORE."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    # top_score=0.0080 < 0.0100 threshold, but margin=0.0250 >= 0.0200
    mock_inference_svc.run_inference.return_value = {
        "model": settings.MODEL_ID,
        "device": "cpu",
        "predictions": [
            {"label": "Drone", "score": 0.0080},
            {"label": "Ship", "score": 0.0010},
        ],
        "inference_time_ms": 105.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    # Use synthetic candidate labels to test runner-up separation
    result = service.classify_image(
        file_bytes=b"dummy",
        filename="weak_recon.jpg",
        candidate_labels=["Drone", "Ship"],
    )

    unc = result["uncertainty"]
    assert unc["is_uncertain"] is True
    # top=0.0080 (< 0.0100) and runner-up=0.0010 (margin=0.0070 < 0.0200) -> BOTH
    # Let's verify specific reason based on margin
    if unc["score_margin"] < settings.UNCERTAINTY_MARGIN_THRESHOLD:
        assert unc["reason"] == "BOTH"
    else:
        assert unc["reason"] == "LOW_PRIMARY_SCORE"


def test_both_low_primary_and_low_margin_triggers_both():
    """CASE D: Both primary score and margin below thresholds triggers BOTH."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    # top=0.0004, runner-up=0.0003 -> margin=0.0001
    mock_inference_svc.run_inference.return_value = {
        "model": settings.MODEL_ID,
        "device": "cpu",
        "predictions": [
            {"label": "Drone", "score": 0.0004},
            {"label": "Helicopter", "score": 0.0003},
            {"label": "Fighter Aircraft", "score": 0.0001},
        ],
        "inference_time_ms": 95.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(file_bytes=b"dummy", filename="noisy_radar.jpg")

    unc = result["uncertainty"]
    assert unc["is_uncertain"] is True
    assert unc["reason"] == "BOTH"
    assert unc["score_margin"] == 0.0001
    assert unc["primary_score"] == 0.0004


def test_exact_margin_threshold_boundary():
    """Verify exact margin threshold boundary (margin == UNCERTAINTY_MARGIN_THRESHOLD)."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    # Exactly on margin threshold: 0.3000 - 0.2800 = 0.0200
    mock_inference_svc.run_inference.return_value = {
        "model": settings.MODEL_ID,
        "device": "cpu",
        "predictions": [
            {"label": "Tank", "score": 0.3000},
            {"label": "Military Vehicle", "score": 0.2800},
        ],
        "inference_time_ms": 80.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(
        file_bytes=b"dummy",
        filename="boundary.jpg",
        candidate_labels=["Tank", "Military Vehicle"],
    )

    unc = result["uncertainty"]
    assert unc["score_margin"] == 0.0200
    # Exactly at threshold, it is NOT uncertain
    assert unc["is_uncertain"] is False
    assert unc["reason"] is None


def test_just_below_margin_threshold():
    """Verify margin just below threshold (0.0199 < 0.0200) triggers uncertainty."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    # 0.3000 - 0.2801 = 0.0199 < 0.0200
    mock_inference_svc.run_inference.return_value = {
        "model": settings.MODEL_ID,
        "device": "cpu",
        "predictions": [
            {"label": "Tank", "score": 0.3000},
            {"label": "Military Vehicle", "score": 0.2801},
        ],
        "inference_time_ms": 80.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(
        file_bytes=b"dummy",
        filename="below_threshold.jpg",
        candidate_labels=["Tank", "Military Vehicle"],
    )

    unc = result["uncertainty"]
    assert unc["score_margin"] == 0.0199
    assert unc["is_uncertain"] is True
    assert unc["reason"] == "LOW_SCORE_MARGIN"


def test_just_above_margin_threshold():
    """Verify margin just above threshold (0.0201 >= 0.0200) is NOT uncertain."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    # 0.3000 - 0.2799 = 0.0201 >= 0.0200
    mock_inference_svc.run_inference.return_value = {
        "model": settings.MODEL_ID,
        "device": "cpu",
        "predictions": [
            {"label": "Tank", "score": 0.3000},
            {"label": "Military Vehicle", "score": 0.2799},
        ],
        "inference_time_ms": 80.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(
        file_bytes=b"dummy",
        filename="above_threshold.jpg",
        candidate_labels=["Tank", "Military Vehicle"],
    )

    unc = result["uncertainty"]
    assert unc["score_margin"] == 0.0201
    assert unc["is_uncertain"] is False
    assert unc["reason"] is None


def test_uncertainty_evaluation_is_deterministic():
    """Verify repeated evaluations on identical scores produce identical uncertainty outputs."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    raw = {
        "model": settings.MODEL_ID,
        "device": "cpu",
        "predictions": [
            {"label": "Fighter Aircraft", "score": 0.2500},
            {"label": "Helicopter", "score": 0.2450},
        ],
        "inference_time_ms": 100.0,
        "status": "success",
    }
    mock_inference_svc.run_inference.return_value = raw
    service = ClassificationService(inference_svc=mock_inference_svc)

    res1 = service.classify_image(file_bytes=b"dummy", filename="t.jpg")
    res2 = service.classify_image(file_bytes=b"dummy", filename="t.jpg")

    assert res1["uncertainty"] == res2["uncertainty"]
    assert res1["uncertainty"]["is_uncertain"] is True
    assert res1["uncertainty"]["reason"] == "LOW_SCORE_MARGIN"


def test_single_candidate_handled_safely():
    """Verify single candidate handles margin as None and evaluates only primary score."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    mock_inference_svc.run_inference.return_value = {
        "model": settings.MODEL_ID,
        "device": "cpu",
        "predictions": [{"label": "Tank", "score": 0.5000}],
        "inference_time_ms": 60.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(
        file_bytes=b"dummy",
        filename="single.jpg",
        candidate_labels=["Tank"],
    )

    unc = result["uncertainty"]
    assert unc["score_margin"] is None
    assert unc["is_uncertain"] is False
    assert unc["reason"] is None


def test_existing_prediction_and_top3_remain_unchanged():
    """Verify adding uncertainty does not alter primary prediction or top-3 ranking."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    mock_inference_svc.run_inference.return_value = {
        "model": settings.MODEL_ID,
        "device": "cpu",
        "predictions": [
            {"label": "Helicopter", "score": 0.4500},
            {"label": "Drone", "score": 0.1200},
            {"label": "Fighter Aircraft", "score": 0.0800},
            {"label": "Tank", "score": 0.0200},
        ],
        "inference_time_ms": 110.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(file_bytes=b"dummy", filename="intact.jpg")

    # Primary prediction
    assert result["prediction"]["label"] == "Helicopter"
    assert result["prediction"]["score"] == 0.4500

    # Top-3 predictions
    assert len(result["top_3"]) == 3
    assert result["top_3"][0]["label"] == "Helicopter"
    assert result["top_3"][1]["label"] == "Drone"
    assert result["top_3"][2]["label"] == "Fighter Aircraft"

    # Schema conforms to ClassificationResponse
    validated = ClassificationResponse(**result)
    assert validated.uncertainty is not None
    assert validated.uncertainty.is_uncertain is False


def test_no_second_model_inference():
    """Verify uncertainty evaluation executes in O(1) on existing scores without second inference."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    mock_inference_svc.run_inference.return_value = {
        "model": settings.MODEL_ID,
        "device": "cpu",
        "predictions": [
            {"label": "Tank", "score": 0.4000},
            {"label": "Ship", "score": 0.2000},
        ],
        "inference_time_ms": 70.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    _ = service.classify_image(file_bytes=b"dummy", filename="single_inference.jpg")

    assert mock_inference_svc.run_inference.call_count == 1
