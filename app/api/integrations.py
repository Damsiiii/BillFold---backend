"""Integration endpoints for Members 1, 2, and 3.

Provides structured route registration and integration points.
Business logic is deferred to the respective member modules via pluggable dependency handlers.
"""

from typing import Any, Callable, Dict, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

router = APIRouter(tags=["Team Integrations"])


class ApiResponseEnvelope(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# Pluggable handler registries for Members 1, 2, and 3
# Other members plug in their business logic functions here without modifying route code.
class ModuleRegistry:
    member1_upload_handler: Optional[Callable] = None
    member2_transactions_handler: Optional[Callable] = None
    member3_summary_handler: Optional[Callable] = None
    member3_categories_handler: Optional[Callable] = None
    member3_budget_handler: Optional[Callable] = None
    member3_savings_handler: Optional[Callable] = None
    member3_overspending_handler: Optional[Callable] = None
    member3_anomalies_handler: Optional[Callable] = None


registry = ModuleRegistry()


def _not_implemented_response(module_name: str, responsible_member: str) -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"{module_name} is awaiting integration from {responsible_member}.",
    )


# Member 1: Statement Upload and Extraction
@router.post(
    "/upload-statement",
    response_model=ApiResponseEnvelope,
    summary="Upload Statement (Member 1)",
    description="Accepts PDF, CSV, or Excel bank statements for extraction.",
)
async def upload_statement(
    file: UploadFile = File(..., description="Bank statement file (.pdf, .csv, .xlsx)"),
    user_id: str = Query(default="demo_user", description="User ID associated with the statement"),
):
    if registry.member1_upload_handler:
        result = await registry.member1_upload_handler(file=file, user_id=user_id)
        return ApiResponseEnvelope(
            status="success",
            message="Statement processed successfully",
            data=result,
        )
    _not_implemented_response("Statement Upload & Extraction", "Member 1")


# Member 2: Transactions Retrieval
@router.get(
    "/transactions",
    response_model=ApiResponseEnvelope,
    summary="Get Transactions (Member 2)",
    description="Retrieves processed and categorized user transactions from SQLite database.",
)
async def get_transactions(
    user_id: str = Query(default="demo_user", description="User ID"),
    limit: int = Query(default=50, ge=1, le=500),
):
    if registry.member2_transactions_handler:
        result = await registry.member2_transactions_handler(user_id=user_id, limit=limit)
        return ApiResponseEnvelope(
            status="success",
            message="Transactions retrieved successfully",
            data=result,
        )
    _not_implemented_response("Transaction Storage & Retrieval", "Member 2")


# Member 3: Analytics and Recommendations
@router.get(
    "/analytics/summary",
    response_model=ApiResponseEnvelope,
    summary="Financial Summary (Member 3)",
    description="Provides total income, expenses, and net cash flow.",
)
async def get_financial_summary(
    user_id: str = Query(default="demo_user", description="User ID"),
):
    if registry.member3_summary_handler:
        result = await registry.member3_summary_handler(user_id=user_id)
        return ApiResponseEnvelope(
            status="success",
            message="Financial summary computed successfully",
            data=result,
        )
    _not_implemented_response("Financial Summary Analytics", "Member 3")


@router.get(
    "/analytics/categories",
    response_model=ApiResponseEnvelope,
    summary="Category Breakdown (Member 3)",
    description="Provides category-wise spending breakdown and percentages.",
)
async def get_category_analytics(
    user_id: str = Query(default="demo_user", description="User ID"),
):
    if registry.member3_categories_handler:
        result = await registry.member3_categories_handler(user_id=user_id)
        return ApiResponseEnvelope(
            status="success",
            message="Category breakdown computed successfully",
            data=result,
        )
    _not_implemented_response("Category Analytics", "Member 3")


@router.get(
    "/recommendations/budget",
    response_model=ApiResponseEnvelope,
    summary="Budget Recommendations (Member 3)",
    description="Provides personalized monthly budget recommendations.",
)
async def get_budget_recommendations(
    user_id: str = Query(default="demo_user", description="User ID"),
):
    if registry.member3_budget_handler:
        result = await registry.member3_budget_handler(user_id=user_id)
        return ApiResponseEnvelope(
            status="success",
            message="Budget recommendations generated successfully",
            data=result,
        )
    _not_implemented_response("Budget Recommendations", "Member 3")


@router.get(
    "/recommendations/savings",
    response_model=ApiResponseEnvelope,
    summary="Savings Recommendations (Member 3)",
    description="Provides actionable opportunities for monthly savings.",
)
async def get_savings_recommendations(
    user_id: str = Query(default="demo_user", description="User ID"),
):
    if registry.member3_savings_handler:
        result = await registry.member3_savings_handler(user_id=user_id)
        return ApiResponseEnvelope(
            status="success",
            message="Savings recommendations generated successfully",
            data=result,
        )
    _not_implemented_response("Savings Recommendations", "Member 3")


@router.get(
    "/detection/overspending",
    response_model=ApiResponseEnvelope,
    summary="Overspending Detection (Member 3)",
    description="Detects categories exceeding expected budget limits.",
)
async def detect_overspending(
    user_id: str = Query(default="demo_user", description="User ID"),
):
    if registry.member3_overspending_handler:
        result = await registry.member3_overspending_handler(user_id=user_id)
        return ApiResponseEnvelope(
            status="success",
            message="Overspending detection completed successfully",
            data=result,
        )
    _not_implemented_response("Overspending Detection", "Member 3")


@router.get(
    "/detection/anomalies",
    response_model=ApiResponseEnvelope,
    summary="Anomaly Detection (Member 3)",
    description="Identifies unusual or suspicious transactions.",
)
async def detect_anomalies(
    user_id: str = Query(default="demo_user", description="User ID"),
):
    if registry.member3_anomalies_handler:
        result = await registry.member3_anomalies_handler(user_id=user_id)
        return ApiResponseEnvelope(
            status="success",
            message="Anomaly detection completed successfully",
            data=result,
        )
    _not_implemented_response("Anomaly Detection", "Member 3")

