import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api.integrations import registry

client = TestClient(app)


def test_member1_upload_statement_unintegrated():
    """Verify upload statement returns 501 when Member 1 parser is unlinked."""
    files = {"file": ("statement.csv", b"Date,Amount,Description\n2026-01-01,50,Food", "text/csv")}
    response = client.post("/upload-statement?user_id=demo_user", files=files)
    assert response.status_code == 501
    assert "Member 1" in response.json()["message"]


def test_member2_transactions_unintegrated():
    """Verify transactions endpoint returns 501 when Member 2 DB handler is unlinked."""
    response = client.get("/transactions?user_id=demo_user")
    assert response.status_code == 501
    assert "Member 2" in response.json()["message"]


def test_member3_analytics_unintegrated():
    """Verify Member 3 analytics endpoints return 501 when unlinked."""
    endpoints = [
        "/analytics/summary",
        "/analytics/categories",
        "/recommendations/budget",
        "/recommendations/savings",
        "/detection/overspending",
        "/detection/anomalies",
    ]
    for ep in endpoints:
        resp = client.get(f"{ep}?user_id=demo_user")
        assert resp.status_code == 501
        assert "Member 3" in resp.json()["message"]


def test_member_integration_pluggable_handler():
    """Verify that when a member attaches their handler, the endpoint responds with 200."""
    async def sample_summary_handler(user_id: str):
        return {"total_income": 4000.0, "total_expenses": 1500.0, "net": 2500.0}

    registry.member3_summary_handler = sample_summary_handler
    try:
        response = client.get("/analytics/summary?user_id=alice")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data"]["net"] == 2500.0
    finally:
        registry.member3_summary_handler = None

