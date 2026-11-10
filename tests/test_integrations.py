from datetime import date

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.api import integrations
from app.api.integrations import registry
from app.database.database import Base
from app.models.transaction import Transaction
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

client = TestClient(app)

HANDLER_NAMES = (
    "member1_upload_handler",
    "member2_transactions_handler",
    "member3_summary_handler",
    "member3_categories_handler",
    "member3_budget_handler",
    "member3_savings_handler",
    "member3_overspending_handler",
    "member3_anomalies_handler",
)


@pytest.fixture(autouse=True)
def reset_registry():
    original_handlers = {name: getattr(registry, name) for name in HANDLER_NAMES}
    yield
    for name, handler in original_handlers.items():
        setattr(registry, name, handler)


@pytest.fixture
def sqlite_database(monkeypatch, tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'integrations.db'}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    test_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(integrations, "SessionLocal", test_session)

    db = test_session()
    db.add_all(
        [
            Transaction(
                user_id="alice",
                date=date(2026, 1, 1),
                description="Monthly salary",
                amount=4000.0,
                type="credit",
                category="Income",
                currency="LKR",
            ),
            Transaction(
                user_id="alice",
                date=date(2026, 1, 2),
                description="Grocery store",
                amount=500.0,
                type="debit",
                category="Food",
                currency="LKR",
            ),
            Transaction(
                user_id="bob",
                date=date(2026, 1, 3),
                description="Other user's purchase",
                amount=1000.0,
                type="debit",
                category="Shopping",
                currency="LKR",
            ),
        ]
    )
    db.commit()
    db.close()
    yield
    engine.dispose()


def test_member1_upload_statement_unintegrated():
    """Verify upload statement returns 501 when Member 1 parser is unlinked."""
    files = {"file": ("statement.csv", b"Date,Amount,Description\n2026-01-01,50,Food", "text/csv")}
    response = client.post("/upload-statement?user_id=demo_user", files=files)
    assert response.status_code == 501
    assert "Member 1" in response.json()["message"]


def test_member2_transactions_successfully_retrieves_database_rows(sqlite_database):
    response = client.get("/transactions?user_id=alice")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["count"] == 2
    assert {row["description"] for row in body["data"]["transactions"]} == {
        "Monthly salary",
        "Grocery store",
    }


def test_member3_integrated_endpoints_return_successful_responses(sqlite_database):
    endpoints = (
        "/analytics/summary",
        "/analytics/categories",
        "/recommendations/budget",
        "/recommendations/savings",
        "/detection/overspending",
        "/detection/anomalies",
    )
    for endpoint in endpoints:
        response = client.get(f"{endpoint}?user_id=alice")
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "success"
        assert response.json()["data"] is not None


def test_member3_analytics_summary_uses_user_database_transactions(sqlite_database):
    response = client.get("/analytics/summary?user_id=alice")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total_income"] == 4000.0
    assert data["total_expenses"] == 500.0
    assert data["net_savings"] == 3500.0
    assert data["income_transaction_count"] == 1
    assert data["expense_transaction_count"] == 1


def test_member_integration_pluggable_handler():
    """Verify that when a member attaches their handler, the endpoint responds with 200."""
    async def sample_summary_handler(user_id: str):
        return {"total_income": 4000.0, "total_expenses": 1500.0, "net": 2500.0}

    registry.member3_summary_handler = sample_summary_handler
    response = client.get("/analytics/summary?user_id=alice")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["net"] == 2500.0
