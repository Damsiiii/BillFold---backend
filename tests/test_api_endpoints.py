"""
Integration tests for FastAPI endpoints in main.py, app/api/analytics.py, and app/api/recommendations.py.
Tests all 6 endpoints in Swagger documentation using FastAPI TestClient:
- GET /analytics/summary
- GET /analytics/categories
- GET /recommendations/budget
- GET /recommendations/savings
- GET /detection/overspending
- GET /detection/anomalies
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Fixture providing FastAPI test client."""
    return TestClient(app)


def test_root_endpoint(client: TestClient):
    """Verify root health check endpoint returns online status and endpoint list."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["docs_url"] == "/docs"
    assert "/analytics/summary" in data["endpoints"]
    assert "/detection/anomalies" in data["endpoints"]


def test_swagger_docs_accessible(client: TestClient):
    """Verify OpenAPI Swagger UI /docs endpoint is accessible."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "html" in response.text.lower()


# ==============================================================================
# /analytics Endpoints
# ==============================================================================

def test_analytics_summary_endpoint(client: TestClient):
    """Test GET /analytics/summary."""
    response = client.get("/analytics/summary")
    assert response.status_code == 200
    data = response.json()

    assert "total_income" in data
    assert "total_expenses" in data
    assert "net_savings" in data
    assert "savings_rate" in data
    assert data["total_income"] > 0
    assert data["total_expenses"] > 0
    assert data["net_savings"] == round(data["total_income"] - data["total_expenses"], 2)


def test_analytics_summary_with_filters(client: TestClient):
    """Test GET /analytics/summary with month filter."""
    response = client.get("/analytics/summary?month=2026-06")
    assert response.status_code == 200
    data = response.json()
    assert data["total_income"] == 275000.0


def test_analytics_categories_endpoint(client: TestClient):
    """Test GET /analytics/categories."""
    response = client.get("/analytics/categories")
    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) > 0
    for cat in data:
        assert "category" in cat
        assert "total_amount" in cat
        assert "transaction_count" in cat
        assert "average_amount" in cat
        assert "percentage_share" in cat


# ==============================================================================
# /recommendations Endpoints
# ==============================================================================

def test_recommendations_budget_endpoint(client: TestClient):
    """Test GET /recommendations/budget."""
    response = client.get("/recommendations/budget?month=2026-10")
    assert response.status_code == 200
    data = response.json()

    assert "baseline_months" in data
    assert "total_suggested_budget" in data
    assert "category_budgets" in data
    assert len(data["category_budgets"]) > 0


def test_recommendations_savings_endpoint(client: TestClient):
    """Test GET /recommendations/savings."""
    response = client.get("/recommendations/savings")
    assert response.status_code == 200
    data = response.json()

    assert "savings_summary" in data
    assert "discretionary_cuts" in data
    assert "recurring_subscriptions" in data
    assert "shortfall_covered_percentage" in data

    # Verify planted Netflix subscription is exposed in API
    subs = data["recurring_subscriptions"]
    assert any("Netflix" in s["merchant"] for s in subs)


# ==============================================================================
# /detection Endpoints
# ==============================================================================

def test_detection_overspending_endpoint(client: TestClient):
    """Test GET /detection/overspending."""
    response = client.get("/detection/overspending?month=2026-10")
    assert response.status_code == 200
    data = response.json()

    assert data["target_month"] == "2026-10"
    assert data["overall_status"] in ["ok", "warning", "over"]
    assert "summary" in data
    assert "categories" in data

    # October shopping was planted as overspent
    shopping_cat = next((c for c in data["categories"] if c["category"] == "Shopping"), None)
    assert shopping_cat is not None
    assert shopping_cat["status"] == "over"


def test_detection_anomalies_endpoint(client: TestClient):
    """Test GET /detection/anomalies."""
    response = client.get("/detection/anomalies")
    assert response.status_code == 200
    data = response.json()

    assert "anomalies" in data
    assert "total_anomalies" in data
    assert "summary_by_type" in data
    assert data["total_anomalies"] > 0

    # Verify planted anomalies are returned
    descriptions = [a["description"] for a in data["anomalies"]]
    assert any("SriLankan Airlines" in d for d in descriptions)
    assert any("Ministry of Crab" in d for d in descriptions)

