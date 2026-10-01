"""Phase 3 Integration & Smoke Tests.

NOTE: This test module requires the actual SigLIP 2 model weights
(google/siglip2-base-patch16-512) and executes live inference on hardware.
"""

import io
from pathlib import Path
import pytest
import httpx
from PIL import Image

from backend.app.main import app
from backend.app.models.siglip2_adapter import SigLIP2Adapter
from backend.app.services.inference_service import inference_service


FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.mark.anyio
async def test_actual_processor_and_model_initialization():
    """Verify actual SigLIP 2 processor and model load and remain cached in memory."""
    adapter = SigLIP2Adapter()
    assert not adapter.is_loaded

    # Load model
    adapter.load()
    assert adapter.is_loaded
    assert adapter.device in ["cuda", "cpu"]
    assert adapter.initialization_time_s is not None
    assert adapter.initialization_time_s > 0

    # Ensure cached in memory (subsequent call returns immediately)
    cached_init_time = adapter.initialization_time_s
    adapter.load()
    assert adapter.initialization_time_s == cached_init_time


@pytest.mark.anyio
async def test_actual_api_inference_endpoint():
    """Verify live POST /api/inference/test endpoint against actual model."""
    test_img_path = FIXTURES_DIR / "smoke_helicopter.jpg"
    assert test_img_path.exists(), f"Missing test fixture: {test_img_path}"

    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/inference/test",
            files={"image": ("test_helo.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        payload = response.json()

        # Contract checks
        assert payload["model"] == "google/siglip2-base-patch16-512"
        assert payload["device"] in ["cuda", "cpu"]
        assert payload["status"] == "success"
        assert "inference_time_ms" in payload
        assert payload["inference_time_ms"] > 0

        predictions = payload["predictions"]
        assert len(predictions) == 6
        for p in predictions:
            assert "label" in p
            assert "score" in p
            assert 0.0 <= p["score"] <= 1.0


@pytest.mark.anyio
async def test_actual_inference_smoke_all_categories():
    """Run model smoke test across representative images for all 6 target categories.

    Sanity check demonstrating model accepts validated images, processes candidate
    labels, and produces structured prediction scores and timing metadata.
    """
    smoke_targets = [
        ("aircraft", FIXTURES_DIR / "smoke_aircraft.jpg", "Fighter Aircraft"),
        ("helicopter", FIXTURES_DIR / "smoke_helicopter.jpg", "Helicopter"),
        ("tank", FIXTURES_DIR / "smoke_tank.jpg", "Tank"),
        ("ship", FIXTURES_DIR / "smoke_ship.jpg", "Ship"),
        ("military vehicle", FIXTURES_DIR / "smoke_vehicle.jpg", "Military Vehicle"),
        ("drone", FIXTURES_DIR / "smoke_drone.jpg", "Drone"),
    ]

    adapter = SigLIP2Adapter()
    adapter.load()

    smoke_records = []

    for name, path, expected_label in smoke_targets:
        assert path.exists(), f"Smoke image fixture missing: {path}"
        img = Image.open(path)
        res = adapter.predict(img)

        # Verification of structured output
        assert res.model == "google/siglip2-base-patch16-512"
        assert res.device in ["cuda", "cpu"]
        assert res.inference_time_ms > 0
        assert len(res.predictions) == 6

        top_pred = res.predictions[0]
        smoke_records.append({
            "category": name,
            "image": str(path.name),
            "top_candidate": top_pred.label,
            "top_score": top_pred.score,
            "device": res.device,
            "inference_time_ms": res.inference_time_ms,
            "scores": {p.label: p.score for p in res.predictions},
        })

    # Assert all 6 categories were processed
    assert len(smoke_records) == 6
