from contextlib import nullcontext
from datetime import date
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.chatbot import FinancialContext
from app.database.database import Base
from app.models.transaction import Transaction
from app.services import context_service
from app.services.context_service import FinancialContextProvider, set_context_provider
from app.services.llm_service import LLMService
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

client = TestClient(app)


@pytest.fixture(autouse=True)
def chatbot_database(monkeypatch, tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'chatbot.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    test_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(context_service, "SessionLocal", test_session)
    yield test_session
    engine.dispose()


def _add_transaction(
    db,
    user_id: str,
    *,
    date_value: date,
    description: str,
    amount: float,
    tx_type: str,
    category: str,
    currency: str | None = "LKR",
):
    db.add(
        Transaction(
            user_id=user_id,
            date=date_value,
            description=description,
            amount=amount,
            type=tx_type,
            category=category,
            currency=currency,
        )
    )
    db.commit()


def test_chat_valid_message_no_context():
    """Verify chatbot response when user financial data is not yet loaded."""
    response = client.post(
        "/chat",
        json={"message": "How much did I spend this month?", "user_id": "test_user"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "source" in data
    assert data["source"] == "system"
    assert "not yet been loaded" in data["response"]


def test_context_provider_loads_user_transactions(chatbot_database):
    db = chatbot_database()
    _add_transaction(
        db,
        "csv_test_20261110",
        date_value=date(2026, 10, 4),
        description="Market",
        amount=125.5,
        tx_type="debit",
        category="Groceries",
    )
    _add_transaction(
        db,
        "csv_test_20261110",
        date_value=date(2026, 10, 6),
        description="Salary",
        amount=1000,
        tx_type="credit",
        category="Income",
    )
    db.close()

    context = FinancialContextProvider().get_user_context("csv_test_20261110")

    assert context.user_id == "csv_test_20261110"
    assert context.has_data is True
    assert context.monthly_income == 1000
    assert context.monthly_expenses == 125.5
    assert context.category_spending == {"Groceries": 125.5}
    assert context.currency == "LKR"
    assert context.recent_transactions[0]["description"] == "Salary"


def test_context_provider_reports_no_data_for_user_without_transactions():
    context = FinancialContextProvider().get_user_context("no_transactions")
    assert context.user_id == "no_transactions"
    assert context.has_data is False
    response, source = LLMService().generate_response(
        "Show my recurring subscriptions.", context
    )
    assert source == "system"
    assert "not yet been loaded" in response


def test_context_provider_keeps_user_ids_isolated(chatbot_database):
    db = chatbot_database()
    _add_transaction(
        db,
        "user-a",
        date_value=date(2026, 10, 4),
        description="A purchase",
        amount=20,
        tx_type="debit",
        category="Food",
    )
    _add_transaction(
        db,
        "user-b",
        date_value=date(2026, 10, 4),
        description="B income",
        amount=300,
        tx_type="credit",
        category="Income",
        currency="USD",
    )
    db.close()

    provider = FinancialContextProvider()
    context_a = provider.get_user_context("user-a")
    context_b = provider.get_user_context("user-b")

    assert context_a.user_id == "user-a"
    assert context_a.monthly_expenses == 20
    assert context_a.monthly_income == 0
    assert context_a.recent_transactions[0]["description"] == "A purchase"
    assert context_b.user_id == "user-b"
    assert context_b.monthly_income == 300
    assert context_b.monthly_expenses == 0
    assert context_b.currency == "USD"
    assert context_b.recent_transactions[0]["description"] == "B income"


def test_context_contains_period_merchant_budget_savings_and_anomaly_data(
    chatbot_database,
):
    db = chatbot_database()
    transactions = [
        (date(2026, 7, 1), "Salary", 3000, "credit", "Income", 5000),
        (date(2026, 7, 5), "Video Subscription", 12, "debit", "Entertainment", 4988),
        (date(2026, 8, 1), "Salary", 3000, "credit", "Income", 7000),
        (date(2026, 8, 5), "Video Subscription", 12, "debit", "Entertainment", 6988),
        (date(2026, 9, 1), "Salary", 3000, "credit", "Income", 9988),
        (date(2026, 9, 5), "Video Subscription", 12, "debit", "Entertainment", 9976),
        (date(2026, 9, 8), "Grocery", 80, "debit", "Food", 9896),
        (date(2026, 9, 9), "Grocery", 2000, "debit", "Food", 7796),
    ]
    for tx_date, description, amount, tx_type, category, balance in transactions:
        db.add(
            Transaction(
                user_id="rich-context",
                date=tx_date,
                description=description,
                amount=amount,
                type=tx_type,
                category=category,
                balance=balance,
                currency="USD",
            )
        )
    db.commit()
    db.close()

    context = FinancialContextProvider().get_user_context("rich-context")
    currency_data = context.financial_data["currencies"]["USD"]

    assert len(currency_data["monthly_summary"]) == 3
    assert currency_data["weekly_summary"]
    assert currency_data["latest_balance"] == {
        "amount": 7796.0,
        "date": "2026-09-09",
    }
    assert currency_data["merchant_spending"][0]["name"] == "Grocery"
    assert currency_data["budgets"]["category_budgets"]
    assert currency_data["overspending"]["categories"]
    assert currency_data["savings"]["recurring_subscriptions"]
    assert currency_data["anomalies"]["anomalies"]


def test_unknown_currency_is_not_assumed_and_limitations_are_reported(
    chatbot_database,
):
    db = chatbot_database()
    _add_transaction(
        db,
        "unknown-currency-user",
        date_value=date(2026, 10, 4),
        description="Unlabeled purchase",
        amount=45,
        tx_type="debit",
        category="Other",
        currency=None,
    )
    db.close()

    context = FinancialContextProvider().get_user_context("unknown-currency-user")
    unknown = context.financial_data["currencies"]["Unknown"]
    response, source = LLMService().generate_response(
        "How much did I spend?", context
    )

    assert context.has_data is True
    assert context.currency is None
    assert unknown["summary"]["currency"] is None
    assert unknown["budget_recommendations"] is None
    assert "Currency is unknown" in unknown["limitations"][0]
    assert "currency is unknown" in response.lower()
    assert source == "financial_data"


def test_chat_answers_using_transactions_loaded_from_database(chatbot_database):
    db = chatbot_database()
    _add_transaction(
        db,
        "chat-user",
        date_value=date(2026, 10, 4),
        description="Market",
        amount=42.75,
        tx_type="debit",
        category="Groceries",
    )
    db.close()

    response = client.post(
        "/chat",
        json={
            "message": "How much did I spend this month?",
            "user_id": "chat-user",
        },
    )

    assert response.status_code == 200
    assert response.json()["source"] == "financial_data"
    assert "LKR 42.75" in response.json()["response"]


def test_configured_model_receives_arbitrary_question_and_retrieved_facts(
    monkeypatch,
):
    captured = {}

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors):
            captured["prompt"] = prompt
            return {"input_ids": SimpleNamespace(shape=(1, 2))}

        def decode(self, tokens, skip_special_tokens):
            return "Your verified data shows the spending trend."

    class FakeModel:
        def generate(self, **kwargs):
            return [[0, 0, 1]]

    service = LLMService()
    service.enabled = True
    service.init_model = lambda: True
    service._model = FakeModel()
    service._tokenizer = FakeTokenizer()
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(no_grad=lambda: nullcontext())
    )
    context = FinancialContext(
        user_id="dynamic-user",
        has_data=True,
        financial_data={
            "currencies": {
                "USD": {
                    "monthly_summary": [{"month": "2026-09", "expenses": 225.0}],
                    "recent_transactions": [
                        {"description": "Bookshop", "amount": 25.0}
                    ],
                }
            }
        },
    )

    response, source = service.generate_response(
        "Compare my spending in September and explain what drove it.", context
    )

    assert source == "financial_data"
    assert response == "Your verified data shows the spending trend."
    assert "Compare my spending in September" in captured["prompt"]
    assert "monthly_summary" in captured["prompt"]
    assert "Bookshop" in captured["prompt"]


def test_chat_fallback_shares_grounded_facts_for_different_finance_topics(
    chatbot_database,
):
    db = chatbot_database()
    _add_transaction(
        db,
        "topic-user",
        date_value=date(2026, 10, 4),
        description="Video Subscription",
        amount=12,
        tx_type="debit",
        category="Entertainment",
        currency="USD",
    )
    db.close()

    for question in (
        "Which categories are my biggest expenses?",
        "Do I have any recurring subscriptions?",
        "Are any transactions unusual?",
        "How can I improve my savings?",
        "Show my transaction history.",
    ):
        response = client.post(
            "/chat",
            json={"message": question, "user_id": "topic-user"},
        )
        assert response.status_code == 200
        assert response.json()["source"] == "financial_data"
        assert "Video Subscription" in response.json()["response"]


def test_rag_prompt_includes_full_verified_financial_context():
    context = FinancialContext(
        user_id="prompt-user",
        has_data=True,
        currency="USD",
        financial_data={
            "currencies": {
                "USD": {
                    "monthly_summary": [{"month": "2026-09", "expenses": 120.0}],
                    "merchant_spending": [{"name": "Bookshop", "amount": 20.0}],
                    "overspending": {"overall_status": "warning"},
                    "savings": {
                        "recurring_subscriptions": [{"description": "Video Subscription"}]
                    },
                    "anomalies": {"anomalies": [{"description": "Unusual card charge"}]},
                }
            },
            "unavailable_information": ["No explicit budget-plan records are stored."],
        },
    )

    prompt = LLMService().construct_rag_prompt(
        "Compare my spending and explain unusual transactions.", context
    )

    for fact in (
        "monthly_summary",
        "merchant_spending",
        "overspending",
        "recurring_subscriptions",
        "Unusual card charge",
        "No explicit budget-plan records",
        "Compare my spending",
    ):
        assert fact in prompt


def test_chat_empty_message_rejected():
    """Verify empty string or whitespace-only message returns 422 validation error."""
    response_empty = client.post("/chat", json={"message": "", "user_id": "test_user"})
    assert response_empty.status_code == 422

    response_spaces = client.post("/chat", json={"message": "   ", "user_id": "test_user"})
    assert response_spaces.status_code == 422


def test_chat_with_grounded_financial_context():
    """Verify chatbot answers using actual financial metrics when context is present."""
    class MockContextProvider(FinancialContextProvider):
        def get_user_context(self, user_id: str) -> FinancialContext:
            return FinancialContext(
                user_id=user_id,
                has_data=True,
                monthly_income=5000.0,
                monthly_expenses=2150.50,
                category_spending={"Groceries": 650.0, "Rent": 1200.0, "Utilities": 300.50},
            )

    original_provider = set_context_provider(MockContextProvider())
    try:
        response = client.post(
            "/chat",
            json={"message": "How much did I spend this month?", "user_id": "john_doe"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "financial_data"
        assert "2,150.50" in data["response"]
    finally:
        set_context_provider(FinancialContextProvider())


def test_chat_general_finance_question():
    """Verify general finance guidance questions do not require personal financial data."""
    response = client.post(
        "/chat",
        json={"message": "What is the recommended budget rule?", "user_id": "test_user"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "general_knowledge"
    assert "50/30/20" in data["response"]


def test_llm_service_disabled_mode():
    """Verify LLMService operates correctly in disabled mode without throwing."""
    service = LLMService()
    service.enabled = False
    assert service.init_model() is False
    assert service.is_fine_tuned is False

    context = FinancialContext(user_id="test", has_data=False)
    resp, source = service.generate_response("Tell me about savings", context)
    assert "emergency fund" in resp.lower()
    assert source == "general_knowledge"


def test_llm_service_handles_missing_model_path_gracefully():
    """Verify that a missing or invalid model path does not crash the application."""
    service = LLMService()
    service.enabled = True
    service.model_path = "/non/existent/gemma/path"

    init_success = service.init_model()
    assert init_success is False
    assert service._load_error is not None
    assert "does not exist" in service._load_error

    # Fallback response still functions
    context = FinancialContext(user_id="test", has_data=False)
    resp, source = service.generate_response("Hello", context)
    assert source == "general_knowledge"
