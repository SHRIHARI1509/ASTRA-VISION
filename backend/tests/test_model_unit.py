import io
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
import torch
from fastapi.testclient import TestClient

from app.core.config import settings
from app.models.base_adapter import ModelAdapter, ModelInferenceResult, Prediction
from app.models.siglip2_adapter import (
    SigLIP2Adapter,
    ModelInitializationError,
    ModelInferenceError,
)
from app.services.inference_service import InferenceService, ModelUnavailableError
from app.main import app

client = TestClient(app)


def create_test_image(format: str = "JPEG", size=(100, 100), color="blue") -> bytes:
    buf = io.BytesIO()
    mode = "RGB" if format.upper() in ["JPEG", "JPG"] else "RGBA"
    img = Image.new(mode, size, color=color)
    fmt = "JPEG" if format.upper() in ["JPEG", "JPG"] else format.upper()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ============================================================================
# 1. MODEL CONFIGURATION TESTS
# ============================================================================

def test_model_configuration_id():
    """Verify primary model identifier is google/siglip2-base-patch16-512."""
    assert settings.MODEL_ID == "google/siglip2-base-patch16-512"


def test_candidate_categories_centralized():
    """Verify all 6 required candidate categories are defined in settings."""
    expected_categories = [
        "Fighter Aircraft",
        "Helicopter",
        "Tank",
        "Ship",
        "Military Vehicle",
        "Drone",
    ]
    assert settings.CANDIDATE_CATEGORIES == expected_categories
    assert len(settings.CANDIDATE_CATEGORIES) == 6


def test_prompt_templates_coverage():
    """Verify prompt templates exist for all candidate categories."""
    for category in settings.CANDIDATE_CATEGORIES:
        assert category in settings.PROMPT_TEMPLATES
        prompt = settings.PROMPT_TEMPLATES[category]
        assert prompt.startswith("a photo of")
        assert len(prompt) > 10


# ============================================================================
# 2. DEVICE SELECTION TESTS
# ============================================================================

def test_device_selection_explicit_cpu():
    """Verify explicit CPU preference resolves to 'cpu'."""
    adapter = SigLIP2Adapter(device_preference="cpu")
    assert adapter.device == "cpu"


def test_device_selection_cuda_fallback_when_unavailable():
    """Verify requesting CUDA when unavailable cleanly falls back to CPU."""
    with patch("torch.cuda.is_available", return_value=False):
        adapter = SigLIP2Adapter(device_preference="cuda")
        assert adapter.device == "cpu"


def test_device_selection_cuda_used_when_available():
    """Verify requesting CUDA or auto when CUDA is available resolves to 'cuda'."""
    with patch("torch.cuda.is_available", return_value=True):
        adapter = SigLIP2Adapter(device_preference="cuda")
        assert adapter.device == "cuda"

        adapter_auto = SigLIP2Adapter(device_preference="auto")
        assert adapter_auto.device == "cuda"


# ============================================================================
# 3. MODEL INITIALIZATION & CACHING TESTS (LIGHTWEIGHT UNIT MOCKS)
# ============================================================================

def test_model_lazy_loading_and_caching():
    """Verify model and processor are initialized once and remain cached in memory."""
    mock_processor = MagicMock()
    mock_model = MagicMock()

    with patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor) as mock_proc_load, \
         patch("transformers.AutoModel.from_pretrained", return_value=mock_model) as mock_model_load:

        adapter = SigLIP2Adapter(device_preference="cpu")
        assert not adapter.is_loaded

        # First load
        adapter.load()
        assert adapter.is_loaded
        assert adapter.initialization_time_s is not None
        assert mock_proc_load.call_count == 1
        assert mock_model_load.call_count == 1

        # Second load (must reuse cached instance, no re-initialization)
        adapter.load()
        assert mock_proc_load.call_count == 1
        assert mock_model_load.call_count == 1


def test_model_initialization_failure_raises_custom_error():
    """Verify initialization failure raises ModelInitializationError without unhandled crash."""
    with patch("transformers.AutoProcessor.from_pretrained", side_effect=RuntimeError("Download failed")):
        adapter = SigLIP2Adapter(device_preference="cpu")
        with pytest.raises(ModelInitializationError) as exc_info:
            adapter.load()
        assert "Download failed" in str(exc_info.value)
        assert not adapter.is_loaded


def test_inference_service_graceful_startup_failure():
    """Verify InferenceService catches model startup failures without terminating backend."""
    mock_adapter = MagicMock(spec=ModelAdapter)
    mock_adapter.load.side_effect = RuntimeError("GPU out of memory")
    mock_adapter.model_id = "test-model"
    mock_adapter.is_loaded = False

    service = InferenceService(adapter=mock_adapter)
    loaded = service.initialize_model()
    assert loaded is False
    assert not service.is_ready


# ============================================================================
# 4. API INFERENCE ENDPOINT TESTS (UNIT WITH CONTROLLED RESPONSES)
# ============================================================================

def test_api_inference_missing_image():
    """Verify POST /api/inference/test without image returns 400 MISSING_FILE."""
    response = client.post("/api/inference/test")
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "MISSING_FILE"


def test_api_inference_empty_image():
    """Verify POST /api/inference/test with empty bytes returns 400 EMPTY_FILE."""
    response = client.post(
        "/api/inference/test",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "EMPTY_FILE"


def test_api_inference_corrupted_image():
    """Verify POST /api/inference/test with corrupt bytes returns 400 error."""
    corrupt_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01junk_payload"
    response = client.post(
        "/api/inference/test",
        files={"image": ("corrupted.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] in ["CORRUPTED_IMAGE", "UNIDENTIFIED_IMAGE"]


def test_api_inference_unsupported_format_text():
    """Verify POST /api/inference/test with plain text file returns 400."""
    response = client.post(
        "/api/inference/test",
        files={"image": ("document.txt", b"plain text report", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] in ["UNSUPPORTED_MIME_TYPE", "UNIDENTIFIED_IMAGE"]


def test_api_inference_dimensions_too_small():
    """Verify POST /api/inference/test with 5x5 image returns 400 DIMENSIONS_TOO_SMALL."""
    tiny_bytes = create_test_image("PNG", size=(5, 5))
    response = client.post(
        "/api/inference/test",
        files={"image": ("tiny.png", tiny_bytes, "image/png")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "DIMENSIONS_TOO_SMALL"


def test_api_inference_model_unavailable_returns_503():
    """Verify when model is unavailable, endpoint returns 503 MODEL_UNAVAILABLE."""
    img_bytes = create_test_image("JPEG", size=(100, 100))

    with patch("app.api.routes.inference_service.run_inference", side_effect=ModelUnavailableError("Model not loaded")):
        response = client.post(
            "/api/inference/test",
            files={"image": ("test.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "MODEL_UNAVAILABLE"


def test_api_inference_mocked_success_structure():
    """Verify POST /api/inference/test returns required structured schema."""
    img_bytes = create_test_image("JPEG", size=(200, 200))
    mock_result = {
        "model": "google/siglip2-base-patch16-512",
        "device": "cpu",
        "predictions": [
            {"label": "Fighter Aircraft", "score": 0.8312},
            {"label": "Helicopter", "score": 0.1205},
            {"label": "Tank", "score": 0.0341},
            {"label": "Ship", "score": 0.0102},
            {"label": "Military Vehicle", "score": 0.0035},
            {"label": "Drone", "score": 0.0005},
        ],
        "inference_time_ms": 350.25,
        "status": "success",
    }

    with patch("app.api.routes.inference_service.run_inference", return_value=mock_result):
        response = client.post(
            "/api/inference/test",
            files={"image": ("mock_jet.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "google/siglip2-base-patch16-512"
        assert data["device"] == "cpu"
        assert len(data["predictions"]) == 6
        assert data["predictions"][0]["label"] == "Fighter Aircraft"
        assert data["predictions"][0]["score"] == 0.8312
        assert data["inference_time_ms"] == 350.25
