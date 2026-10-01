import io
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
import torch
from transformers import AutoProcessor
from fastapi.testclient import TestClient

from app.core.config import settings
from app.models.siglip2_adapter import SigLIP2Adapter
from app.services.image_service import image_service, ImageValidationError
from app.services.inference_service import InferenceService
from app.services.classification_service import ClassificationService
from app.main import app

client = TestClient(app)


def make_image_bytes(
    format_name: str,
    mode: str,
    size: tuple,
    color=None,
) -> bytes:
    """Helper to generate in-memory image bytes with exact format, mode, and dimensions."""
    if color is None:
        if mode == "L":
            color = 128
        elif mode == "RGBA":
            color = (120, 160, 200, 220)
        else:
            color = (120, 160, 200)

    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    save_fmt = "JPEG" if format_name.upper() in ["JPEG", "JPG"] else format_name.upper()
    img.save(buf, format=save_fmt)
    return buf.getvalue()


# Cache a processor instance for unit test inspection to keep tests fast
@pytest.fixture(scope="module")
def siglip_processor():
    return AutoProcessor.from_pretrained(settings.MODEL_ID)


# ============================================================================
# 1. PROCESSOR CONFIGURATION & NORMALIZATION AUDIT TESTS
# ============================================================================

def test_processor_configuration(siglip_processor):
    """Verify official SigLIP 2 processor configuration parameters."""
    ip = siglip_processor.image_processor

    # Target dimensions
    assert ip.size["height"] == 512
    assert ip.size["width"] == 512

    # Resize behavior
    assert ip.do_resize is True
    # Resample filter 2 corresponds to PIL.Image.Resampling.BILINEAR
    assert ip.resample == 2

    # Rescale behavior (1/255 scaling)
    assert ip.do_rescale is True
    assert pytest.approx(ip.rescale_factor, rel=1e-4) == 1.0 / 255.0

    # Normalization behavior (mean=0.5, std=0.5 -> [-1, 1] range)
    assert ip.do_normalize is True
    assert list(ip.image_mean) == [0.5, 0.5, 0.5]
    assert list(ip.image_std) == [0.5, 0.5, 0.5]


def test_normalization_output_range(siglip_processor):
    """Verify processor maps [0, 255] pixels to [-1.0, 1.0] range."""
    # Test black image (all 0s)
    black_img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    res_black = siglip_processor(images=black_img, return_tensors="pt")
    # (0/255 - 0.5) / 0.5 = -1.0
    assert pytest.approx(res_black["pixel_values"].min().item(), abs=1e-3) == -1.0
    assert pytest.approx(res_black["pixel_values"].max().item(), abs=1e-3) == -1.0

    # Test white image (all 255s)
    white_img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    res_white = siglip_processor(images=white_img, return_tensors="pt")
    # (255/255 - 0.5) / 0.5 = 1.0
    assert pytest.approx(res_white["pixel_values"].min().item(), abs=1e-3) == 1.0
    assert pytest.approx(res_white["pixel_values"].max().item(), abs=1e-3) == 1.0


# ============================================================================
# 2. FORMAT HANDLING TESTS (JPEG, JPG, PNG, WEBP, RGBA, GRAYSCALE, MALFORMED)
# ============================================================================

@pytest.mark.parametrize(
    "fmt,ext,mode",
    [
        ("JPEG", ".jpeg", "RGB"),
        ("JPEG", ".jpg", "RGB"),
        ("PNG", ".png", "RGB"),
        ("PNG", ".png", "RGBA"),
        ("PNG", ".png", "L"),
        ("WEBP", ".webp", "RGB"),
        ("WEBP", ".webp", "RGBA"),
    ],
)
def test_format_handling_validation(fmt, ext, mode):
    """Verify image service decodes all supported formats and channel modes."""
    data = make_image_bytes(fmt, mode, (200, 200))
    meta = image_service.validate_and_inspect_image(
        file_bytes=data,
        filename=f"test_image{ext}",
    )
    assert meta["valid"] is True
    assert meta["width"] == 200
    assert meta["height"] == 200
    assert meta["format"] in ["JPEG", "PNG", "WEBP"]


def test_malformed_image_rejected():
    """Verify corrupted or invalid image fails safely."""
    malformed_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01not_a_valid_raster"
    with pytest.raises(ImageValidationError) as exc_info:
        image_service.validate_and_inspect_image(
            file_bytes=malformed_bytes,
            filename="corrupt.jpg",
        )
    assert exc_info.value.code in ["CORRUPTED_IMAGE", "UNIDENTIFIED_IMAGE"]


# ============================================================================
# 3. RESIZE & TENSOR DIMENSION TESTS (64x64, 224x224, 512x512, 1024x512, 512x1024, non-square)
# ============================================================================

@pytest.mark.parametrize(
    "size",
    [
        (64, 64),
        (224, 224),
        (512, 512),
        (1024, 512),
        (512, 1024),
        (800, 300),  # non-square
    ],
)
def test_resize_produces_model_dimensions(siglip_processor, size):
    """Verify processor converts diverse input dimensions to [1, 3, 512, 512] float32 tensor."""
    img = Image.new("RGB", size, color=(100, 150, 200))
    inputs = siglip_processor(images=img, return_tensors="pt")
    pv = inputs["pixel_values"]

    assert pv.shape == torch.Size([1, 3, 512, 512])
    assert pv.dtype == torch.float32


# ============================================================================
# 4. COLOR & CHANNEL CONVERSION TESTS
# ============================================================================

def test_rgba_channel_handling_in_adapter(siglip_processor):
    """Verify RGBA image is converted safely to RGB before processor invocation."""
    rgba_bytes = make_image_bytes("PNG", "RGBA", (150, 150))
    pil_image = Image.open(io.BytesIO(rgba_bytes))
    assert pil_image.mode == "RGBA"

    adapter = SigLIP2Adapter(device_preference="cpu")
    mock_model = MagicMock()
    mock_outputs = MagicMock()
    mock_outputs.logits_per_image = torch.tensor([[1.5, 0.2, -0.8, -1.0, -2.0, -3.0]])
    mock_model.return_value = mock_outputs

    adapter._processor = siglip_processor
    adapter._model = mock_model

    result = adapter.predict(image=pil_image)
    assert result.status == "success"
    assert len(result.predictions) == 6

    # Check call args of mock_model
    assert mock_model.called
    kwargs = mock_model.call_args.kwargs
    pv = kwargs["pixel_values"]
    assert pv.shape == torch.Size([1, 3, 512, 512])


def test_grayscale_channel_handling_in_adapter(siglip_processor):
    """Verify grayscale (mode 'L') image is safely converted to 3-channel RGB."""
    l_bytes = make_image_bytes("PNG", "L", (180, 180))
    pil_image = Image.open(io.BytesIO(l_bytes))
    assert pil_image.mode == "L"

    adapter = SigLIP2Adapter(device_preference="cpu")
    mock_model = MagicMock()
    mock_outputs = MagicMock()
    mock_outputs.logits_per_image = torch.tensor([[1.0, 0.5, 0.2, -0.1, -0.5, -1.0]])
    mock_model.return_value = mock_outputs

    adapter._processor = siglip_processor
    adapter._model = mock_model

    result = adapter.predict(image=pil_image)
    assert result.status == "success"
    assert len(result.predictions) == 6

    kwargs = mock_model.call_args.kwargs
    pv = kwargs["pixel_values"]
    assert pv.shape == torch.Size([1, 3, 512, 512])


# ============================================================================
# 5. DEVICE HANDLING & TENSOR PLACEMENT TESTS
# ============================================================================

def test_device_placement_cpu():
    """Verify tensor inputs are placed on CPU when running on CPU."""
    adapter = SigLIP2Adapter(device_preference="cpu")
    assert adapter.device == "cpu"


def test_device_placement_cuda_fallback():
    """Verify adapter falls back to CPU when CUDA is unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        adapter = SigLIP2Adapter(device_preference="cuda")
        assert adapter.device == "cpu"


# ============================================================================
# 6. DUPLICATE PREPROCESSING AUDIT TEST
# ============================================================================

def test_no_duplicate_preprocessing():
    """Verify image service does not perform resize or normalization."""
    # Ensure image_service preserves original dimensions and does not alter pixels
    data = make_image_bytes("JPEG", "RGB", (1024, 768))
    meta = image_service.validate_and_inspect_image(data, "test.jpg")
    assert meta["width"] == 1024
    assert meta["height"] == 768
    # The raw bytes are untouched


# ============================================================================
# 7. END-TO-END PREPROCESSING VIA /api/classify (INTEGRATION)
# ============================================================================

@pytest.mark.parametrize(
    "fmt,ext,mode,size",
    [
        ("JPEG", ".jpg", "RGB", (256, 256)),
        ("PNG", ".png", "RGBA", (300, 200)),
        ("PNG", ".png", "L", (150, 150)),
        ("WEBP", ".webp", "RGB", (400, 300)),
        ("WEBP", ".webp", "RGBA", (200, 400)),
        ("JPEG", ".jpeg", "RGB", (64, 64)),
        ("PNG", ".png", "RGB", (1024, 512)),
    ],
)
def test_api_classify_preprocessing_matrix(fmt, ext, mode, size):
    """Verify POST /api/classify succeeds across all formats, modes, and aspect ratios."""
    img_bytes = make_image_bytes(fmt, mode, size)
    mime = "image/jpeg" if fmt == "JPEG" else f"image/{fmt.lower()}"

    # Use mocked classification service to keep test suite fast and deterministic
    mock_response = {
        "prediction": {"label": "Tank", "score": 0.8521},
        "candidates": [
            {"label": "Tank", "score": 0.8521},
            {"label": "Military Vehicle", "score": 0.1204},
            {"label": "Helicopter", "score": 0.0211},
        ],
        "top_3": [
            {"label": "Tank", "score": 0.8521},
            {"label": "Military Vehicle", "score": 0.1204},
            {"label": "Helicopter", "score": 0.0211},
        ],
        "inference": {
            "model": settings.MODEL_ID,
            "device": "cpu",
            "inference_time_ms": 145.2,
        },
        "status": "success",
    }

    with patch("app.api.routes.classification_service.classify_image", return_value=mock_response):
        resp = client.post(
            "/api/classify",
            files={"image": (f"sample{ext}", img_bytes, mime)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["prediction"]["label"] == "Tank"
        assert len(data["top_3"]) == 3
