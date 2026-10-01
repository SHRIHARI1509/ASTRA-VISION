import io
import pytest
import httpx
from PIL import Image
from backend.app.main import app


@pytest.mark.anyio
async def test_api_health_contract():
    """Verify that backend application satisfies Phase 1 API health contract."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload == {"status": "ok", "service": "astra-vision"}


@pytest.mark.anyio
async def test_api_image_validate_contract():
    """Verify that backend application satisfies Phase 2 image validation contract."""
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color="green").save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/images/validate",
            files={"image": ("recon_test.jpg", img_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["filename"] == "recon_test.jpg"
        assert data["format"] == "JPEG"
        assert data["width"] == 640
        assert data["height"] == 480
        assert data["size_bytes"] == len(img_bytes)


@pytest.mark.anyio
async def test_api_image_validate_invalid_contract():
    """Verify that backend rejects non-image payload with structured error contract."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/images/validate",
            files={"image": ("malicious.exe", b"not a valid image", "application/octet-stream")},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["valid"] is False
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
