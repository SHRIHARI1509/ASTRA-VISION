import re
from unittest.mock import MagicMock
import pytest

from app.core.config import settings
from app.services.classification_service import ClassificationService
from app.services.inference_service import InferenceService
from app.schemas.classification import ClassificationResponse

PROHIBITED_MARKETING_TERMS = [
    r"\bbest\b",
    r"\bhighest accuracy\b",
    r"\bguaranteed\b",
    r"\b100%\b",
    r"\bmilitary-grade\b",
    r"\bstate-of-the-art\b",
    r"\bcalibrated confidence\b",
    r"\bcertainty\b",
    r"\bsuperior\b",
    r"\bflawless\b",
    r"\bunbeatable\b",
]


def test_model_fit_configuration_present():
    """Verify centralized MODEL_FIT_JUSTIFICATION configuration exists and has required fields."""
    config = settings.MODEL_FIT_JUSTIFICATION
    assert isinstance(config, dict)
    assert config["model_name"] == "SigLIP 2 Base"
    assert config["model_id"] == "google/siglip2-base-patch16-512"
    assert "Zero-Shot" in config["task"]
    assert len(config["justification"]) > 50


def test_model_fit_word_count_conciseness():
    """Verify justification text length conforms strictly to 80-150 words target."""
    text = settings.MODEL_FIT_JUSTIFICATION["justification"]
    words = text.strip().split()
    word_count = len(words)
    assert 80 <= word_count <= 150, f"Justification word count ({word_count}) is outside 80-150 words range."


def test_model_fit_no_prohibited_accuracy_or_marketing_claims():
    """Content validation: Assert justification contains zero prohibited claims."""
    text = settings.MODEL_FIT_JUSTIFICATION["justification"].lower()
    for pattern in PROHIBITED_MARKETING_TERMS:
        match = re.search(pattern, text)
        assert match is None, f"Prohibited term/phrase '{pattern}' detected in justification: '{text}'"


def test_model_fit_does_not_claim_calibrated_confidence():
    """Verify text does not falsely claim probabilistic or calibrated confidence."""
    text = settings.MODEL_FIT_JUSTIFICATION["justification"].lower()
    assert "calibrated" not in text
    assert "% confidence" not in text
    assert "probability" not in text


def test_model_fit_included_in_classification_response():
    """Verify classification service outputs structured model_fit metadata matching schema."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    mock_inference_svc.run_inference.return_value = {
        "model": settings.MODEL_ID,
        "device": "cpu",
        "predictions": [
            {"label": "Tank", "score": 0.8500},
            {"label": "Military Vehicle", "score": 0.1000},
            {"label": "Helicopter", "score": 0.0500},
        ],
        "inference_time_ms": 110.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(file_bytes=b"dummy", filename="tank.jpg")

    assert "model_fit" in result
    model_fit = result["model_fit"]
    assert model_fit["model_name"] == "SigLIP 2 Base"
    assert model_fit["model_id"] == "google/siglip2-base-patch16-512"
    assert len(model_fit["justification"]) > 50

    # Schema validation
    validated = ClassificationResponse(**result)
    assert validated.model_fit is not None
    assert validated.model_fit.model_id == "google/siglip2-base-patch16-512"


def test_existing_pipeline_remains_unchanged():
    """Verify primary prediction, top-3, and uncertainty logic remain unaffected."""
    mock_inference_svc = MagicMock(spec=InferenceService)
    mock_inference_svc.run_inference.return_value = {
        "model": settings.MODEL_ID,
        "device": "cpu",
        "predictions": [
            {"label": "Helicopter", "score": 0.6000},
            {"label": "Drone", "score": 0.2000},
            {"label": "Fighter Aircraft", "score": 0.1000},
        ],
        "inference_time_ms": 95.0,
        "status": "success",
    }

    service = ClassificationService(inference_svc=mock_inference_svc)
    result = service.classify_image(file_bytes=b"dummy", filename="heli.jpg")

    assert result["prediction"]["label"] == "Helicopter"
    assert result["prediction"]["score"] == 0.6000
    assert len(result["top_3"]) == 3
    assert result["uncertainty"]["is_uncertain"] is False
    assert mock_inference_svc.run_inference.call_count == 1
