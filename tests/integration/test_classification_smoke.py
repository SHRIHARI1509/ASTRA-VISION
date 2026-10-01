"""Phase 4 Classification Integration & Smoke Tests.

Verifies end-to-end classification pipeline:
Image -> Validation -> SigLIP 2 -> Candidate scores -> Classification logic -> Structured result.
"""

from pathlib import Path
import pytest
import httpx

from backend.app.main import app

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.mark.anyio
async def test_api_classify_live_valid_image():
    """Verify POST /api/classify executes complete pipeline and returns structured result."""
    img_path = FIXTURES_DIR / "smoke_helicopter.jpg"
    assert img_path.exists(), f"Missing fixture: {img_path}"

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/classify",
            files={"image": ("helo_mission.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()

        # 1. Structure conformance
        assert "prediction" in data
        assert "candidates" in data
        assert "inference" in data
        assert data.get("status") == "success"

        # 2. Prediction integrity
        primary = data["prediction"]
        assert "label" in primary
        assert "score" in primary
        assert primary["label"] == "Helicopter"
        assert primary["score"] > 0.0

        # 3. Candidate ranking & tie handling
        candidates = data["candidates"]
        assert len(candidates) == 6
        scores = [c["score"] for c in candidates]
        assert scores == sorted(scores, reverse=True), "Candidates must be sorted descending by score"
        assert primary["label"] == candidates[0]["label"]
        assert primary["score"] == candidates[0]["score"]

        # 4. Inference metadata
        inf = data["inference"]
        assert inf["model"] == "google/siglip2-base-patch16-512"
        assert inf["device"] in ["cuda", "cpu"]
        assert inf["inference_time_ms"] > 0


@pytest.mark.anyio
async def test_api_classify_live_tank_image():
    """Verify classification on tank fixture produces primary prediction matching highest score."""
    img_path = FIXTURES_DIR / "smoke_tank.jpg"
    assert img_path.exists(), f"Missing fixture: {img_path}"

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/classify",
            files={"image": ("recon_tank.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()

        primary = data["prediction"]
        candidates = data["candidates"]

        assert primary["label"] == "Tank"
        assert primary["label"] == candidates[0]["label"]
        assert primary["score"] == candidates[0]["score"]


@pytest.mark.anyio
async def test_api_classify_missing_image():
    """Verify POST /api/classify with missing file returns 400 MISSING_FILE."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/classify")
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "MISSING_FILE"


@pytest.mark.anyio
async def test_api_classify_invalid_image():
    """Verify POST /api/classify with non-image payload fails safely via Phase 2 validation."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/classify",
            files={"image": ("bad.exe", b"executable bytes", "application/octet-stream")},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] in ["UNIDENTIFIED_IMAGE", "CORRUPTED_IMAGE", "UNSUPPORTED_FORMAT"]
