from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Chat message request payload."""

    message: str = Field(
        ...,
        description="User financial question or prompt",
        examples=["How much did I spend this month?"],
    )
    user_id: str = Field(
        default="demo_user",
        description="Identifier used to retrieve user-specific financial context",
        examples=["demo_user"],
    )

    @field_validator("message")
    @classmethod
    def validate_message_not_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty or whitespace only.")
        return cleaned


class ChatResponse(BaseModel):
    """Chatbot output response."""

    response: str = Field(..., description="Generated answer from financial data or LLM")
    source: str = Field(
        ...,
        description="Source of the answer (e.g. 'financial_data', 'general_knowledge', 'system')",
    )


class FinancialContext(BaseModel):
    """Standardized financial context container passed to LLM / RAG."""

    user_id: str
    has_data: bool = False
    monthly_income: Optional[float] = None
    monthly_expenses: Optional[float] = None
    category_spending: Optional[Dict[str, float]] = None
    recent_transactions: Optional[List[Dict[str, Any]]] = None
    budget_recommendations: Optional[Dict[str, Any]] = None
    savings_recommendations: Optional[Dict[str, Any]] = None

