import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Verify health endpoint returns 200 and the specified response schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["message"] == "Personal Finance Backend is running"


def test_swagger_docs_accessible():
    """Verify OpenAPI documentation is served at /docs."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema_generation():
    """Verify OpenAPI JSON schema includes registered routes."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema.get("paths", {})

    expected_routes = [
        "/health",
        "/chat",
        "/upload-statement",
        "/transactions",
        "/analytics/summary",
        "/analytics/categories",
        "/recommendations/budget",
        "/recommendations/savings",
        "/detection/overspending",
        "/detection/anomalies",
    ]
    for route in expected_routes:
        assert route in paths, f"Route {route} should be registered in OpenAPI schema"


def test_validation_error_format():
    """Verify validation errors conform to standardized error envelope."""
    # Sending invalid body to /chat
    response = client.post("/chat", json={})
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert "Validation error" in data["message"]
    assert "details" in data

