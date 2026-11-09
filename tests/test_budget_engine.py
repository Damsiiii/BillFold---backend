"""
Unit tests for the Budget Engine (app/services/budget_engine.py).
Tests:
- Suggested monthly budget per category = median of last 3 months.
- 10% reduction target for discretionary vs 0% for essentials.
- Overspending detection status thresholds: 'ok' (<80%), 'warning' (80-100%), 'over' (>100%).
- Planted overspent category (Shopping in October 2026).
- Edge cases (empty df, single month, transfers excluded).
"""

import pandas as pd
import pytest
from app.services.budget_engine import (
    detect_overspending,
    generate_suggested_budgets,
    is_discretionary_category,
)
from data.generate_mock_data import generate_mock_transactions


@pytest.fixture
def mock_df() -> pd.DataFrame:
    """Fixture providing 6 months of generated mock transactions."""
    return generate_mock_transactions(seed=42)


# ==============================================================================
# Suggested Budget & Category Reduction Rules
# ==============================================================================

def test_discretionary_classification():
    """Verify classification of categories into discretionary and essentials."""
    assert is_discretionary_category("Entertainment") is True
    assert is_discretionary_category("Shopping") is True
    assert is_discretionary_category("Dining") is True
    assert is_discretionary_category("Other") is True
    assert is_discretionary_category("Uncategorized") is True

    assert is_discretionary_category("Rent") is False
    assert is_discretionary_category("Utilities") is False
    assert is_discretionary_category("Food") is False
    assert is_discretionary_category("Transport") is False
    assert is_discretionary_category("Health") is False
    assert is_discretionary_category("Education") is False


def test_suggested_budget_median_of_last_3_months(mock_df: pd.DataFrame):
    """Verify that suggested budget computes the median of the 3 baseline months with appropriate reduction."""
    # When targeting October 2026, baseline months should be July, August, September (2026-07, 2026-08, 2026-09)
    result = generate_suggested_budgets(mock_df, target_month="2026-10")
    assert result["baseline_months"] == ["2026-07", "2026-08", "2026-09"]

    budgets = {b["category"]: b for b in result["category_budgets"]}

    # Check Rent (Essential: 0% reduction)
    rent_budget = budgets["Rent"]
    assert rent_budget["category_type"] == "essential"
    assert rent_budget["reduction_target_percentage"] == 0.0
    assert rent_budget["suggested_budget"] == rent_budget["historical_median"]

    # Check Entertainment (Discretionary: 10% reduction)
    ent_budget = budgets["Entertainment"]
    assert ent_budget["category_type"] == "discretionary"
    assert ent_budget["reduction_target_percentage"] == 10.0
    expected_ent = round(ent_budget["historical_median"] * 0.90, 2)
    assert ent_budget["suggested_budget"] == expected_ent


# ==============================================================================
# Overspending Detection & Planted Problem Verification
# ==============================================================================

def test_overspending_detection_planted_october_shopping(mock_df: pd.DataFrame):
    """Verify that planted October 2026 Shopping splurge is detected as 'over' budget."""
    result = detect_overspending(mock_df, target_month="2026-10")

    assert result["overall_status"] == "over"
    assert result["summary"]["over_count"] >= 1

    # Locate Shopping category detail
    categories = {c["category"]: c for c in result["categories"]}
    assert "Shopping" in categories
    shop = categories["Shopping"]

    # Verify status is 'over' (>100% of budget)
    assert shop["status"] == "over"
    assert shop["actual_spending"] > shop["suggested_budget"]
    assert shop["percentage_of_budget"] > 100.0
    assert shop["overspend_amount"] > 100000.0  # Spent ~190,000 vs budget ~25,000
    assert shop["percentage_change_vs_historical_avg"] > 100.0


def test_overspending_status_thresholds():
    """Verify exact status thresholds: ok (<80%), warning (80-100%), over (>100%)."""
    # Create 3 baseline months (May, June, July) where each month category has 1,000 spend
    # So historical median = 1,000. For an essential category, suggested budget = 1,000.
    baseline_records = []
    for month in ["2026-05", "2026-06", "2026-07"]:
        for cat in ["CatOK", "CatWarn", "CatOver"]:
            baseline_records.append(
                {
                    "transaction_id": f"T-{month}-{cat}",
                    "transaction_date": f"{month}-10",
                    "description": "Baseline Spend",
                    "amount": 1000.0,
                    "currency": "LKR",
                    "transaction_type": "expense",
                    "category": cat,
                    "source_dataset": "test",
                }
            )

    # In target month August 2026:
    # CatOK: spend 700 (<80% of 900 discretionary budget = 720) -> ok
    # CatWarn: spend 850 (between 80% and 100% of 900) -> warning
    # CatOver: spend 1200 (>100% of 900) -> over
    target_records = [
        {
            "transaction_id": "T-AUG-OK",
            "transaction_date": "2026-08-10",
            "description": "August Spend",
            "amount": 700.0,
            "currency": "LKR",
            "transaction_type": "expense",
            "category": "CatOK",
            "source_dataset": "test",
        },
        {
            "transaction_id": "T-AUG-WARN",
            "transaction_date": "2026-08-10",
            "description": "August Spend",
            "amount": 850.0,
            "currency": "LKR",
            "transaction_type": "expense",
            "category": "CatWarn",
            "source_dataset": "test",
        },
        {
            "transaction_id": "T-AUG-OVER",
            "transaction_date": "2026-08-10",
            "description": "August Spend",
            "amount": 1200.0,
            "currency": "LKR",
            "transaction_type": "expense",
            "category": "CatOver",
            "source_dataset": "test",
        },
    ]

    df = pd.DataFrame(baseline_records + target_records)
    res = detect_overspending(df, target_month="2026-08")

    cat_map = {c["category"]: c for c in res["categories"]}
    assert cat_map["CatOK"]["status"] == "ok"
    assert cat_map["CatWarn"]["status"] == "warning"
    assert cat_map["CatOver"]["status"] == "over"


# ==============================================================================
# Edge Cases
# ==============================================================================

def test_budget_engine_empty_dataframe():
    """Empty DataFrame returns zero budget and 'ok' status."""
    res = detect_overspending(pd.DataFrame())
    assert res["total_budget"] == 0.0
    assert res["total_actual_spending"] == 0.0
    assert res["overall_status"] == "ok"
    assert res["categories"] == []


def test_budget_engine_single_month():
    """Handles single month history without crashing."""
    df = pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "transaction_date": "2026-05-15",
                "description": "Dining",
                "amount": 5000.0,
                "currency": "LKR",
                "transaction_type": "expense",
                "category": "Dining",
                "source_dataset": "test",
            }
        ]
    )
    res = detect_overspending(df, target_month="2026-05")
    assert len(res["baseline_months"]) == 1
    assert res["categories"][0]["category"] == "Dining"


def test_budget_engine_excludes_transfers():
    """Transfers must never be counted as expense budgets or actual spending."""
    df = pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "transaction_date": "2026-05-15",
                "description": "Pot transfer",
                "amount": 50000.0,
                "currency": "LKR",
                "transaction_type": "transfer",
                "category": "Transfer",
                "source_dataset": "test",
            }
        ]
    )
    res = detect_overspending(df, target_month="2026-05")
    assert res["total_budget"] == 0.0
    assert res["total_actual_spending"] == 0.0
    assert res["categories"] == []
