import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.schemas.chatbot import FinancialContext
from app.services.context_service import FinancialContextProvider, set_context_provider
from app.services.llm_service import LLMService

client = TestClient(app)


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

