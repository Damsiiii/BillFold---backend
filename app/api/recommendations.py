"""
FastAPI Router for Recommendation and Detection endpoints.

Thin router with no business logic. Delegates computation to the budget,
savings, and anomaly detection engines.

Endpoints:
- GET /recommendations/budget: Suggested monthly budget per category.
- GET /recommendations/savings: Savings gap (20% target), ranked discretionary cuts, recurring subscriptions.
- GET /detection/overspending: Overspending detection for a chosen month (<80% ok, 80-100% warning, >100% over).
- GET /detection/anomalies: Statistical outliers (z-score > 3 / IQR), duplicates within 2 days, new merchants.

Query Parameters:
- month (YYYY-MM): Optional month filter
- start_date (YYYY-MM-DD): Optional start date filter
- end_date (YYYY-MM-DD): Optional end date filter
- currency: Optional currency filter (defaults to LKR)
"""

from typing import Any
from fastapi import APIRouter, Query
from app.services.analytics_engine import load_transactions
from app.services.anomaly_detection import detect_anomalies
from app.services.budget_engine import detect_overspending, generate_suggested_budgets
from app.services.savings_engine import calculate_savings_recommendations

recommendations_router = APIRouter(prefix="/recommendations", tags=["Recommendations"])
detection_router = APIRouter(prefix="/detection", tags=["Detection"])

# Alias primary router
router = recommendations_router


# ==============================================================================
# Recommendations Endpoints
# ==============================================================================

@recommendations_router.get("/budget", summary="Get suggested monthly budget")
def get_suggested_budget_endpoint(
    month: str | None = Query(None, description="Target month for budget (YYYY-MM)"),
    currency: str | None = Query(None, description="Currency filter (default LKR)"),
) -> dict[str, Any]:
    """Retrieve suggested monthly budget per category based on median of last 3 months.

    Discretionary categories get a 10% reduction target; essentials stay at 100% of historical median.
    """
    df = load_transactions()
    return generate_suggested_budgets(df, target_month=month, currency=currency)


@recommendations_router.get("/savings", summary="Get savings recommendations and recurring subscriptions")
def get_savings_recommendations_endpoint(
    month: str | None = Query(None, description="Month to evaluate (YYYY-MM)"),
    target_rate: float = Query(0.20, description="Target savings rate (default 0.20 for 20%)"),
    currency: str | None = Query(None, description="Currency filter (default LKR)"),
) -> dict[str, Any]:
    """Retrieve savings gap analysis against a 20% target, ranked discretionary cuts, and detected subscriptions."""
    df = load_transactions()
    return calculate_savings_recommendations(
        df, target_rate=target_rate, month=month, currency=currency
    )


# ==============================================================================
# Detection Endpoints
# ==============================================================================

@detection_router.get("/overspending", summary="Detect category overspending")
def get_overspending_detection_endpoint(
    month: str | None = Query(None, description="Chosen month to detect overspending (YYYY-MM)"),
    currency: str | None = Query(None, description="Currency filter (default LKR)"),
) -> dict[str, Any]:
    """Detect overspending for a chosen month against suggested budgets.

    Status flags: ok (<80% of budget), warning (80-100%), over (>100%), plus % change vs historical average.
    """
    df = load_transactions()
    return detect_overspending(df, target_month=month, currency=currency)


@detection_router.get("/anomalies", summary="Detect unusual transactions")
def get_anomalies_endpoint(
    month: str | None = Query(None, description="Filter anomalies by month (YYYY-MM)"),
    start_date: str | None = Query(None, description="Start date filter (YYYY-MM-DD)"),
    end_date: str | None = Query(None, description="End date filter (YYYY-MM-DD)"),
    currency: str | None = Query(None, description="Currency filter (default LKR)"),
) -> dict[str, Any]:
    """Detect unusual transactions using z-score > 3 / IQR fences, duplicate detection, and new merchant rules."""
    df = load_transactions()
    return detect_anomalies(
        df, currency=currency, month=month, start_date=start_date, end_date=end_date
    )

