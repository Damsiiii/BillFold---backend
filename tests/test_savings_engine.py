"""
Unit tests for the Savings Engine (app/services/savings_engine.py).
Tests:
- Gap between current savings rate and 20% target.
- Discretionary cuts ranking and concrete saving amount calculations.
- Subscription detection (identifying planted Netflix subscription).
- Edge cases (zero income, target exceeded, empty df, transfers excluded).
"""

import pandas as pd
import pytest
from app.services.savings_engine import (
    analyze_savings_gap,
    calculate_savings_recommendations,
    detect_recurring_subscriptions,
    rank_discretionary_cuts,
)
from data.generate_mock_data import generate_mock_transactions


@pytest.fixture
def mock_df() -> pd.DataFrame:
    """Fixture providing 6 months of generated mock transactions."""
    return generate_mock_transactions(seed=42)


# ==============================================================================
# Savings Gap Analysis Tests
# ==============================================================================

def test_savings_gap_analysis(mock_df: pd.DataFrame):
    """Verify savings gap calculation against the 20% target."""
    gap = analyze_savings_gap(mock_df, target_rate=0.20)

    assert gap["target_savings_rate"] == 0.20
    assert gap["target_savings_rate_pct"] == 20.0
    assert gap["total_income"] > 0
    assert gap["total_expenses"] > 0

    # Mathematical identity
    expected_gap = round(0.20 - gap["current_savings_rate"], 4)
    assert gap["savings_rate_gap"] == expected_gap
    assert gap["savings_rate_gap_pct"] == round(expected_gap * 100, 2)

    expected_target_amount = round(gap["total_income"] * 0.20, 2)
    assert gap["target_savings_amount"] == expected_target_amount

    if gap["current_savings_rate"] >= 0.20:
        assert gap["meets_target"] is True
        assert gap["savings_shortfall"] == 0.0
    else:
        assert gap["meets_target"] is False
        expected_shortfall = round(expected_target_amount - gap["net_savings"], 2)
        assert gap["savings_shortfall"] == expected_shortfall


# ==============================================================================
# Discretionary Cuts Ranking Tests
# ==============================================================================

def test_rank_discretionary_cuts(mock_df: pd.DataFrame):
    """Verify discretionary categories are ranked by spend with computed saving amounts."""
    cuts, total_savings = rank_discretionary_cuts(mock_df)

    assert len(cuts) > 0
    assert total_savings > 0

    # Ensure ranked in descending order of current_spend
    spends = [c["current_spend"] for c in cuts]
    assert spends == sorted(spends, reverse=True)

    # Essentials like Rent, Food, Utilities must NOT be in discretionary cuts
    essential_categories = {"Rent", "Food", "Utilities", "Transport", "Health", "Education"}
    cut_categories = {c["category"] for c in cuts}
    assert cut_categories.isdisjoint(essential_categories)

    # Verify calculation of potential savings
    for c in cuts:
        assert c["rank"] >= 1
        expected_saving = round(c["current_spend"] * (c["recommended_cut_percentage"] / 100.0), 2)
        assert c["potential_monthly_saving"] == expected_saving
        assert c["suggested_new_monthly_spend"] == round(c["current_spend"] - expected_saving, 2)
        assert "Reduce" in c["reason"]


# ==============================================================================
# Subscription Detection & Planted Netflix Test
# ==============================================================================

def test_detect_planted_netflix_subscription(mock_df: pd.DataFrame):
    """Verify recurring subscription detection correctly identifies Netflix across 6 months."""
    subscriptions = detect_recurring_subscriptions(mock_df)

    assert len(subscriptions) >= 1
    netflix = next(
        (s for s in subscriptions if "Netflix" in s["merchant"]), None
    )
    assert netflix is not None, "Planted Netflix subscription was not detected"

    assert netflix["occurrence_count"] == 6
    assert netflix["average_amount"] == 2200.0
    assert netflix["estimated_monthly_cost"] == 2200.0
    assert netflix["estimated_annual_cost"] == 26400.0
    # Roughly monthly: ~30 days average interval
    assert 25.0 <= netflix["average_interval_days"] <= 35.0
    assert netflix["category"] == "Entertainment"
    assert "2,200.00" in netflix["savings_tip"]
    assert "26,400.00" in netflix["savings_tip"]


def test_subscription_ignores_sporadic_or_variable_charges():
    """Sporadic or highly variable charges should not be classified as monthly subscriptions."""
    # Irregular cafe visits: different amounts, irregular dates
    df = pd.DataFrame(
        [
            {
                "transaction_id": "C1",
                "transaction_date": "2026-05-01",
                "description": "City Cafe",
                "amount": 500.0,
                "currency": "LKR",
                "transaction_type": "expense",
                "category": "Food",
                "source_dataset": "test",
            },
            {
                "transaction_id": "C2",
                "transaction_date": "2026-05-04",  # 3 days later
                "description": "City Cafe",
                "amount": 2500.0,  # 5x amount
                "currency": "LKR",
                "transaction_type": "expense",
                "category": "Food",
                "source_dataset": "test",
            },
            {
                "transaction_id": "C3",
                "transaction_date": "2026-05-09",  # 5 days later
                "description": "City Cafe",
                "amount": 800.0,
                "currency": "LKR",
                "transaction_type": "expense",
                "category": "Food",
                "source_dataset": "test",
            },
        ]
    )
    subs = detect_recurring_subscriptions(df)
    assert subs == []


# ==============================================================================
# Edge Cases
# ==============================================================================

def test_savings_engine_empty_dataframe():
    """Empty DataFrame returns zero values gracefully without error."""
    res = calculate_savings_recommendations(pd.DataFrame())

    summary = res["savings_summary"]
    assert summary["total_income"] == 0.0
    assert summary["total_expenses"] == 0.0
    assert summary["current_savings_rate"] == 0.0
    assert summary["savings_shortfall"] == 0.0
    assert res["discretionary_cuts"] == []
    assert res["recurring_subscriptions"] == []


def test_savings_engine_zero_income():
    """Zero income avoids divide-by-zero, sets savings rate to 0.0."""
    expense_only = pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "transaction_date": "2026-06-01",
                "description": "Cinema",
                "amount": 3000.0,
                "currency": "LKR",
                "transaction_type": "expense",
                "category": "Entertainment",
                "source_dataset": "test",
            }
        ]
    )
    res = calculate_savings_recommendations(expense_only)
    summary = res["savings_summary"]
    assert summary["total_income"] == 0.0
    assert summary["current_savings_rate"] == 0.0
    assert summary["target_savings_amount"] == 0.0


def test_savings_engine_target_exceeded():
    """High savings rate (>20%) flags meets_target = True and shortfall = 0."""
    high_savings_df = pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "transaction_date": "2026-06-01",
                "description": "Income",
                "amount": 200000.0,
                "currency": "LKR",
                "transaction_type": "income",
                "category": "Salary",
                "source_dataset": "test",
            },
            {
                "transaction_id": "T2",
                "transaction_date": "2026-06-05",
                "description": "Rent",
                "amount": 50000.0,
                "currency": "LKR",
                "transaction_type": "expense",
                "category": "Rent",
                "source_dataset": "test",
            },
        ]
    )
    # Income = 200,000, Expense = 50,000, Net = 150,000, Savings rate = 75% > 20%
    gap = analyze_savings_gap(high_savings_df, target_rate=0.20)
    assert gap["meets_target"] is True
    assert gap["current_savings_rate"] == 0.75
    assert gap["savings_shortfall"] == 0.0


def test_savings_engine_excludes_transfers():
    """Transfers must not count as income, expense, or subscriptions."""
    transfer_df = pd.DataFrame(
        [
            {
                "transaction_id": "TR1",
                "transaction_date": "2026-05-15",
                "description": "Transfer to Pot",
                "amount": 50000.0,
                "currency": "LKR",
                "transaction_type": "transfer",
                "category": "Transfer",
                "source_dataset": "test",
            },
            {
                "transaction_id": "TR2",
                "transaction_date": "2026-06-15",
                "description": "Transfer to Pot",
                "amount": 50000.0,
                "currency": "LKR",
                "transaction_type": "transfer",
                "category": "Transfer",
                "source_dataset": "test",
            },
            {
                "transaction_id": "TR3",
                "transaction_date": "2026-07-15",
                "description": "Transfer to Pot",
                "amount": 50000.0,
                "currency": "LKR",
                "transaction_type": "transfer",
                "category": "Transfer",
                "source_dataset": "test",
            },
        ]
    )
    subs = detect_recurring_subscriptions(transfer_df)
    assert subs == []
    gap = analyze_savings_gap(transfer_df)
    assert gap["total_income"] == 0.0
    assert gap["total_expenses"] == 0.0

