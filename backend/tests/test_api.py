from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify GET /api/health returns HTTP 200 with expected JSON payload."""
    response = client.get("/api/health")

    # 1. Verify HTTP 200
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # 2. Verify valid JSON response
    data = response.json()
    assert isinstance(data, dict), "Response must be a JSON dictionary"

    # 3. Verify expected health status and service identification
    assert data.get("status") == "ok", f"Expected status 'ok', got {data.get('status')}"
    assert data.get("service") == "astra-vision", f"Expected service 'astra-vision', got {data.get('service')}"


def test_not_found_error_envelope():
    """Verify 404 responses conform to standardized error envelope."""
    response = client.get("/api/nonexistent-endpoint")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == 404
