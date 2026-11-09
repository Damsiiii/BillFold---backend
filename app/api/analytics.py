"""
FastAPI Router for Financial Analytics endpoints.

Thin router with no business logic. Delegates computation to the analytics engine.
Endpoints:
- GET /analytics/summary: Total income, expenses, net savings, and savings rate.
- GET /analytics/categories: Category-wise expense breakdown (% share, count, avg, total).

Query Parameters:
- month (YYYY-MM): Optional month filter
- start_date (YYYY-MM-DD): Optional start date filter
- end_date (YYYY-MM-DD): Optional end date filter
- currency: Optional currency filter (defaults to LKR)
"""

from typing import Any
from fastapi import APIRouter, Query
from app.services.analytics_engine import (
    compute_category_spending,
    compute_financial_summary,
    load_transactions,
)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/summary", summary="Get financial summary")
def get_financial_summary_endpoint(
    month: str | None = Query(None, description="Filter by month (YYYY-MM)"),
    start_date: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    currency: str | None = Query(None, description="Currency filter (default LKR)"),
) -> dict[str, Any]:
    """Retrieve overall financial summary including total income, total expenses, net savings, and savings rate."""
    df = load_transactions()
    return compute_financial_summary(
        df, currency=currency, month=month, start_date=start_date, end_date=end_date
    )


@router.get("/categories", summary="Get category-wise spending breakdown")
def get_category_spending_endpoint(
    month: str | None = Query(None, description="Filter by month (YYYY-MM)"),
    start_date: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    currency: str | None = Query(None, description="Currency filter (default LKR)"),
) -> list[dict[str, Any]]:
    """Retrieve category-wise spending breakdown for expenses only, sorted descending by spend."""
    df = load_transactions()
    return compute_category_spending(
        df, currency=currency, month=month, start_date=start_date, end_date=end_date
    )

