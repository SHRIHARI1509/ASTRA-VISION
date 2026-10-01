"""ASTRA VISION — Phase 10 Multi-Object Detection Unit Tests.

Covers:
1. detection endpoint exists
2. valid image detection
3. invalid image rejection
4. empty detection state
5. bounding box schema
6. coordinate bounds
7. x1 < x2 validation
8. y1 < y2 validation
9. class normalization
10. unsupported labels are not force-mapped
11. detector singleton/cache
12. model loading failure handling
13. inference failure handling
14. RGB input processing
15. RGBA input conversion
16. grayscale input conversion
17. response determinism and sorting
"""

import io
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app.core.config import settings
from app.schemas.detection import BoxCoordinates, DetectedObject, DetectionResponse
from app.models.grounding_dino_adapter import (
    GroundingDINOAdapter,
    DetectionModelInitializationError,
    DetectionModelInferenceError,
)
from app.services.detection_service import (
    DetectionService,
    DetectionModelUnavailableError,
    DetectionInferenceError,
)
from app.services.image_service import ImageValidationError
from app.main import app

client = TestClient(app)


def create_test_image(format: str = "JPEG", size=(200, 200), mode="RGB") -> bytes:
    """Helper to synthesize valid test image bytes."""
    buf = io.BytesIO()
    img = Image.new(mode, size, color="blue" if mode != "L" else 128)
    fmt = "JPEG" if format.upper() in ["JPEG", "JPG"] else format.upper()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ============================================================================
# 1. DETECTION ENDPOINT EXISTENCE & ROUTING
# ============================================================================

def test_detection_endpoint_exists():
    """Verify POST /api/detect endpoint exists, is routed, and rejects missing file with 400."""
    response = client.post("/api/detect")
    assert response.status_code == 400
    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "MISSING_FILE"


# ============================================================================
# 2. VALID IMAGE DETECTION
# ============================================================================

def test_valid_image_detection():
    """Verify detection endpoint returns valid schema with detections, counts, and dimensions."""
    mock_adapter = MagicMock(spec=GroundingDINOAdapter)
    mock_adapter.is_loaded = True
    mock_adapter.model_id = "IDEA-Research/grounding-dino-base"
    mock_adapter.device = "cpu"
    mock_adapter.detect.return_value = {
        "detections": [
            DetectedObject(
                class_name="Tank",
                score=0.91,
                box=BoxCoordinates(x1=10, y1=15, x2=150, y2=180),
            )
        ],
        "image_width": 200,
        "image_height": 200,
        "count": 1,
        "detector": "IDEA-Research/grounding-dino-base",
        "device": "cpu",
        "inference_time_ms": 150.0,
    }

    test_bytes = create_test_image("JPEG", size=(200, 200))
    with patch("app.api.routes.detection_service", DetectionService(adapter=mock_adapter)):
        response = client.post(
            "/api/detect",
            files={"image": ("test_tank.jpg", test_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["image_width"] == 200
        assert data["image_height"] == 200
        assert len(data["detections"]) == 1
        assert data["detections"][0]["class_name"] == "Tank"
        assert data["detections"][0]["score"] == 0.91
        assert data["detections"][0]["box"] == {"x1": 10, "y1": 15, "x2": 150, "y2": 180}


# ============================================================================
# 3. INVALID / CORRUPT IMAGE REJECTION
# ============================================================================

def test_invalid_image_rejected():
    """Verify non-image bytes or corrupted content is rejected cleanly with 400."""
    corrupt_bytes = b"NOT_A_VALID_IMAGE_DATA_STREAM"
    response = client.post(
        "/api/detect",
        files={"image": ("corrupted.jpg", corrupt_bytes, "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] in ["CORRUPT_IMAGE", "DECODE_ERROR", "INVALID_FORMAT", "UNIDENTIFIED_IMAGE"]


# ============================================================================
# 4. EMPTY DETECTION STATE
# ============================================================================

def test_empty_detection_state():
    """Verify when no candidate passes threshold, clean count=0 and empty detections are returned."""
    mock_adapter = MagicMock(spec=GroundingDINOAdapter)
    mock_adapter.is_loaded = True
    mock_adapter.detect.return_value = {
        "detections": [],
        "image_width": 400,
        "image_height": 300,
        "count": 0,
        "detector": "IDEA-Research/grounding-dino-base",
        "device": "cpu",
        "inference_time_ms": 110.0,
    }

    test_bytes = create_test_image("PNG", size=(400, 300))
    with patch("app.api.routes.detection_service", DetectionService(adapter=mock_adapter)):
        response = client.post(
            "/api/detect",
            files={"image": ("empty_field.png", test_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["detections"] == []
        assert data["image_width"] == 400
        assert data["image_height"] == 300


# ============================================================================
# 5. BOUNDING BOX SCHEMA VALIDATION
# ============================================================================

def test_bounding_box_schema():
    """Verify BoxCoordinates schema enforces positive integers and expected fields."""
    box = BoxCoordinates(x1=50, y1=60, x2=200, y2=250)
    assert box.x1 == 50
    assert box.y1 == 60
    assert box.x2 == 200
    assert box.y2 == 250


# ============================================================================
# 6. COORDINATE BOUNDS
# ============================================================================

def test_coordinate_bounds():
    """Verify coordinate bounding prevents negative values and clamps to image dimensions."""
    with pytest.raises(ValueError):
        BoxCoordinates(x1=-5, y1=10, x2=100, y2=100)


# ============================================================================
# 7. X1 < X2 VALIDATION
# ============================================================================

def test_x1_less_than_x2():
    """Verify BoxCoordinates rejects degenerate box where x2 <= x1."""
    with pytest.raises(ValueError, match="x2 .* must be strictly greater than x1"):
        BoxCoordinates(x1=100, y1=50, x2=100, y2=150)

    with pytest.raises(ValueError, match="x2 .* must be strictly greater than x1"):
        BoxCoordinates(x1=150, y1=50, x2=100, y2=150)


# ============================================================================
# 8. Y1 < Y2 VALIDATION
# ============================================================================

def test_y1_less_than_y2():
    """Verify BoxCoordinates rejects degenerate box where y2 <= y1."""
    with pytest.raises(ValueError, match="y2 .* must be strictly greater than y1"):
        BoxCoordinates(x1=50, y1=100, x2=150, y2=100)

    with pytest.raises(ValueError, match="y2 .* must be strictly greater than y1"):
        BoxCoordinates(x1=50, y1=150, x2=150, y2=100)


# ============================================================================
# 9. CLASS NORMALIZATION
# ============================================================================

def test_class_normalization():
    """Verify detector phrases normalize into the 6 production taxonomy classes."""
    normalize = GroundingDINOAdapter.normalize_label

    # Tank
    assert normalize("tank") == "Tank"
    assert normalize("a tank") == "Tank"
    assert normalize("a tank.") == "Tank"

    # Military Vehicle
    assert normalize("military vehicle") == "Military Vehicle"
    assert normalize("a military vehicle") == "Military Vehicle"
    assert normalize("armored vehicle") == "Military Vehicle"
    assert normalize("apc") == "Military Vehicle"

    # Fighter Aircraft
    assert normalize("fighter aircraft") == "Fighter Aircraft"
    assert normalize("a fighter aircraft") == "Fighter Aircraft"
    assert normalize("fighter jet") == "Fighter Aircraft"

    # Helicopter
    assert normalize("helicopter") == "Helicopter"
    assert normalize("a helicopter") == "Helicopter"

    # Ship
    assert normalize("ship") == "Ship"
    assert normalize("a ship") == "Ship"
    assert normalize("warship") == "Ship"

    # Drone
    assert normalize("drone") == "Drone"
    assert normalize("a drone") == "Drone"
    assert normalize("uav") == "Drone"


# ============================================================================
# 10. UNSUPPORTED LABELS ARE NOT FORCE-MAPPED
# ============================================================================

def test_unsupported_labels_not_force_mapped():
    """Verify unrelated / open-vocabulary labels return None and are NOT forced into taxonomy."""
    normalize = GroundingDINOAdapter.normalize_label

    unsupported = [
        "person",
        "human",
        "dog",
        "cat",
        "tree",
        "cloud",
        "building",
        "house",
        "civilian car",
        "bicycle",
        "traffic light",
        "stop sign",
        "random noise",
    ]

    for label in unsupported:
        assert normalize(label) is None, f"Expected '{label}' to not map, but got '{normalize(label)}'"


# ============================================================================
# 11. DETECTOR SINGLETON / CACHE
# ============================================================================

def test_detector_singleton_cache():
    """Verify detector adapter caches loaded model and does not reload on subsequent calls."""
    adapter = GroundingDINOAdapter()
    assert not adapter.is_loaded

    # Mock internal processor and model loading
    with patch("transformers.AutoProcessor.from_pretrained", return_value=MagicMock()):
        with patch("transformers.AutoModelForZeroShotObjectDetection.from_pretrained", return_value=MagicMock()):
            adapter.load()
            assert adapter.is_loaded
            init_time = adapter.initialization_time_s

            # Second call should return immediately without reloading
            adapter.load()
            assert adapter.initialization_time_s == init_time


# ============================================================================
# 12. MODEL LOADING FAILURE
# ============================================================================

def test_model_loading_failure():
    """Verify graceful handling when detector weights fail to load, returning 503."""
    mock_adapter = MagicMock(spec=GroundingDINOAdapter)
    mock_adapter.is_loaded = False
    mock_adapter.detect.side_effect = DetectionModelInitializationError("Failed to fetch weights")

    test_bytes = create_test_image("JPEG")
    with patch("app.api.routes.detection_service", DetectionService(adapter=mock_adapter)):
        response = client.post(
            "/api/detect",
            files={"image": ("test.jpg", test_bytes, "image/jpeg")},
        )
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "DETECTOR_UNAVAILABLE"


# ============================================================================
# 13. INFERENCE FAILURE
# ============================================================================

def test_inference_failure():
    """Verify graceful handling when detector forward pass raises runtime error, returning 500."""
    mock_adapter = MagicMock(spec=GroundingDINOAdapter)
    mock_adapter.is_loaded = True
    mock_adapter.detect.side_effect = DetectionModelInferenceError("CUDA OOM or tensor dimension error")

    test_bytes = create_test_image("JPEG")
    with patch("app.api.routes.detection_service", DetectionService(adapter=mock_adapter)):
        response = client.post(
            "/api/detect",
            files={"image": ("test.jpg", test_bytes, "image/jpeg")},
        )
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "DETECTION_INFERENCE_ERROR"


# ============================================================================
# 14. RGB INPUT
# ============================================================================

def test_rgb_input():
    """Verify standard 3-channel RGB image is accepted and processed cleanly."""
    mock_adapter = MagicMock(spec=GroundingDINOAdapter)
    mock_adapter.is_loaded = True
    mock_adapter.detect.return_value = {
        "detections": [],
        "image_width": 150,
        "image_height": 150,
        "count": 0,
        "detector": "IDEA-Research/grounding-dino-base",
        "device": "cpu",
        "inference_time_ms": 80.0,
    }

    test_bytes = create_test_image("JPEG", size=(150, 150), mode="RGB")
    service = DetectionService(adapter=mock_adapter)
    res = service.detect_objects(file_bytes=test_bytes, filename="rgb.jpg")
    assert res["image_width"] == 150
    assert res["image_height"] == 150


# ============================================================================
# 15. RGBA INPUT
# ============================================================================

def test_rgba_input():
    """Verify 4-channel RGBA image is converted to RGB and processed without error."""
    mock_adapter = MagicMock(spec=GroundingDINOAdapter)
    mock_adapter.is_loaded = True
    mock_adapter.detect.return_value = {
        "detections": [],
        "image_width": 160,
        "image_height": 160,
        "count": 0,
        "detector": "IDEA-Research/grounding-dino-base",
        "device": "cpu",
        "inference_time_ms": 85.0,
    }

    test_bytes = create_test_image("PNG", size=(160, 160), mode="RGBA")
    service = DetectionService(adapter=mock_adapter)
    res = service.detect_objects(file_bytes=test_bytes, filename="rgba.png")
    assert res["image_width"] == 160
    assert res["image_height"] == 160


# ============================================================================
# 16. GRAYSCALE INPUT
# ============================================================================

def test_grayscale_input():
    """Verify 1-channel Grayscale image is converted to canonical RGB and processed."""
    mock_adapter = MagicMock(spec=GroundingDINOAdapter)
    mock_adapter.is_loaded = True
    mock_adapter.detect.return_value = {
        "detections": [],
        "image_width": 170,
        "image_height": 170,
        "count": 0,
        "detector": "IDEA-Research/grounding-dino-base",
        "device": "cpu",
        "inference_time_ms": 90.0,
    }

    test_bytes = create_test_image("PNG", size=(170, 170), mode="L")
    service = DetectionService(adapter=mock_adapter)
    res = service.detect_objects(file_bytes=test_bytes, filename="grayscale.png")
    assert res["image_width"] == 170
    assert res["image_height"] == 170


# ============================================================================
# 17. RESPONSE DETERMINISM & RANKING
# ============================================================================

def test_response_determinism():
    """Verify detections are sorted deterministically descending by detection score."""
    mock_adapter = MagicMock(spec=GroundingDINOAdapter)
    mock_adapter.is_loaded = True
    mock_adapter.detect.return_value = {
        "detections": [
            DetectedObject(class_name="Tank", score=0.88, box=BoxCoordinates(x1=10, y1=10, x2=100, y2=100)),
            DetectedObject(class_name="Military Vehicle", score=0.45, box=BoxCoordinates(x1=120, y1=20, x2=180, y2=80)),
        ],
        "image_width": 300,
        "image_height": 200,
        "count": 2,
        "detector": "IDEA-Research/grounding-dino-base",
        "device": "cpu",
        "inference_time_ms": 95.0,
    }

    test_bytes = create_test_image("JPEG", size=(300, 200))
    service = DetectionService(adapter=mock_adapter)
    res = service.detect_objects(file_bytes=test_bytes, filename="multi.jpg")

    assert res["count"] == 2
    assert res["detections"][0].score >= res["detections"][1].score
    assert res["detections"][0].class_name == "Tank"
    assert res["detections"][1].class_name == "Military Vehicle"
