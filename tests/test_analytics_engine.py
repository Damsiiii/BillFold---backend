"""
Unit tests for the Financial Analytics Engine (app/services/analytics_engine.py)
and Mock Data Generator (data/generate_mock_data.py).
"""

import pandas as pd
import pytest
from app.services.analytics_engine import (
    calculate_analytics,
    compute_category_spending,
    compute_financial_summary,
    compute_monthly_summary,
    prepare_transactions,
)
from data.generate_mock_data import generate_mock_transactions


@pytest.fixture
def mock_df() -> pd.DataFrame:
    """Fixture providing 6 months of generated mock transactions."""
    return generate_mock_transactions(seed=42)


# ==============================================================================
# Mock Data Contract & Planted Problems Tests
# ==============================================================================

def test_mock_data_contract(mock_df: pd.DataFrame):
    """Verify that generated mock data strictly conforms to the transaction contract."""
    required_cols = [
        "transaction_id",
        "transaction_date",
        "description",
        "amount",
        "currency",
        "transaction_type",
        "category",
        "source_dataset",
    ]
    for col in required_cols:
        assert col in mock_df.columns, f"Missing required contract column: {col}"

    assert len(mock_df) > 0
    # Amount must always be positive
    assert (mock_df["amount"] > 0).all(), "Found non-positive amounts"
    # Currency should default to LKR
    assert (mock_df["currency"] == "LKR").all()
    # Transaction types should only be income, expense, transfer, unknown
    valid_types = {"income", "expense", "transfer", "unknown"}
    assert set(mock_df["transaction_type"].unique()).issubset(valid_types)


def test_mock_data_planted_problems(mock_df: pd.DataFrame):
    """Verify that all four required planted anomalies and problems are present in mock data."""
    # 1. Overspent category: Shopping in 2026-10 should be substantially higher than average
    shopping_df = mock_df[(mock_df["category"] == "Shopping") & (mock_df["transaction_type"] == "expense")].copy()
    shopping_df["month"] = shopping_df["transaction_date"].str.slice(0, 7)
    monthly_shopping = shopping_df.groupby("month")["amount"].sum()
    assert "2026-10" in monthly_shopping.index
    assert monthly_shopping["2026-10"] > 100000.0, "Planted October shopping overspending not found"
    earlier_mean = monthly_shopping.drop("2026-10").mean()
    assert monthly_shopping["2026-10"] > (2 * earlier_mean), "October shopping not significantly overspent"

    # 2. Very large transaction: Single transaction > 400,000 LKR
    large_txns = mock_df[mock_df["amount"] >= 400000.0]
    assert len(large_txns) == 1, "Planted very large transaction not found or multiple found"
    assert "SriLankan Airlines" in large_txns.iloc[0]["description"]
    assert large_txns.iloc[0]["category"] == "Transport"

    # 3. Duplicate charge: Same merchant and amount within 2 days
    dinner_txns = mock_df[mock_df["description"] == "Ministry of Crab Seafood Dinner"]
    assert len(dinner_txns) == 2, "Duplicate charges not found"
    dates = pd.to_datetime(dinner_txns["transaction_date"]).sort_values()
    diff_days = (dates.iloc[1] - dates.iloc[0]).days
    assert diff_days <= 2, f"Planted duplicate charges are {diff_days} days apart (expected <= 2)"
    assert dinner_txns.iloc[0]["amount"] == dinner_txns.iloc[1]["amount"]

    # 4. Recurring monthly subscription: Netflix across all 6 months
    sub_txns = mock_df[mock_df["description"] == "Netflix Monthly Subscription"]
    assert len(sub_txns) == 6, f"Expected 6 monthly subscription entries, got {len(sub_txns)}"
    assert sub_txns["amount"].nunique() == 1
    assert sub_txns.iloc[0]["amount"] == 2200.00


# ==============================================================================
# Analytics Engine Mathematical & Core Functionality Tests
# ==============================================================================

def test_financial_summary_calculation(mock_df: pd.DataFrame):
    """Test overall income, expenses, net savings, and savings rate calculations."""
    summary = compute_financial_summary(mock_df)

    assert summary["total_income"] > 0
    assert summary["total_expenses"] > 0
    # Mathematical identities
    expected_net = round(summary["total_income"] - summary["total_expenses"], 2)
    assert summary["net_savings"] == expected_net

    expected_rate = round(expected_net / summary["total_income"], 4)
    assert summary["savings_rate"] == expected_rate
    assert summary["savings_rate_percentage"] == round(expected_rate * 100, 2)

    # Exclude transfers check: ensure transfer amounts are not added to income or expenses
    transfer_amount = mock_df[mock_df["transaction_type"] == "transfer"]["amount"].sum()
    raw_income = mock_df[mock_df["transaction_type"] == "income"]["amount"].sum()
    raw_expense = mock_df[mock_df["transaction_type"] == "expense"]["amount"].sum()

    assert summary["total_income"] == round(raw_income, 2)
    assert summary["total_expenses"] == round(raw_expense, 2)
    assert summary["total_income"] != round(raw_income + transfer_amount, 2)


def test_monthly_summary(mock_df: pd.DataFrame):
    """Test monthly summary aggregation."""
    monthly = compute_monthly_summary(mock_df)

    assert len(monthly) == 6  # 6 months of data
    # Check chronological ordering
    months = [item["month"] for item in monthly]
    assert months == sorted(months)
    assert months[0] == "2026-05"
    assert months[-1] == "2026-10"

    for entry in monthly:
        assert "month" in entry
        assert "income" in entry
        assert "expenses" in entry
        assert "net" in entry
        assert "savings_rate" in entry
        assert entry["net"] == round(entry["income"] - entry["expenses"], 2)


def test_category_spending_breakdown(mock_df: pd.DataFrame):
    """Test category spending breakdown for expenses only."""
    categories = compute_category_spending(mock_df)

    assert len(categories) > 0
    # Must be sorted descending by total_amount
    amounts = [c["total_amount"] for c in categories]
    assert amounts == sorted(amounts, reverse=True)

    # Percentage shares must sum to ~100%
    total_pct = sum(c["percentage_share"] for c in categories)
    assert pytest.approx(total_pct, rel=1e-2) == 100.0

    # Ensure each item has required metrics
    for cat in categories:
        assert cat["total_amount"] > 0
        assert cat["transaction_count"] > 0
        assert cat["average_amount"] == round(cat["total_amount"] / cat["transaction_count"], 2)
        assert 0.0 <= cat["percentage_share"] <= 100.0

    # Ensure "Salary" (income) or "Transfer" are not in category spending
    cat_names = [c["category"] for c in categories]
    assert "Salary" not in cat_names
    assert "Transfer" not in cat_names


# ==============================================================================
# Edge Cases Handling Tests
# ==============================================================================

def test_edge_case_empty_dataframe():
    """Empty DataFrame must return graceful zero-values and empty lists without error."""
    empty_df = pd.DataFrame()
    res = calculate_analytics(empty_df)

    assert res["summary"]["total_income"] == 0.0
    assert res["summary"]["total_expenses"] == 0.0
    assert res["summary"]["net_savings"] == 0.0
    assert res["summary"]["savings_rate"] == 0.0
    assert res["summary"]["savings_rate_percentage"] == 0.0
    assert res["monthly_summary"] == []
    assert res["category_spending"] == []


def test_edge_case_only_transfers():
    """DataFrame containing only transfer transactions must result in 0 income and 0 expenses."""
    transfer_df = pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "transaction_date": "2026-05-10",
                "description": "Pot transfer",
                "amount": 50000.0,
                "currency": "LKR",
                "transaction_type": "transfer",
                "category": "Transfer",
                "source_dataset": "test",
            }
        ]
    )
    summary = compute_financial_summary(transfer_df)
    assert summary["total_income"] == 0.0
    assert summary["total_expenses"] == 0.0
    assert summary["net_savings"] == 0.0
    assert summary["savings_rate"] == 0.0

    categories = compute_category_spending(transfer_df)
    assert categories == []


def test_edge_case_one_month_history():
    """Handles single month history gracefully."""
    single_month_df = pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "transaction_date": "2026-07-01",
                "description": "Salary",
                "amount": 200000.0,
                "currency": "LKR",
                "transaction_type": "income",
                "category": "Salary",
                "source_dataset": "test",
            },
            {
                "transaction_id": "T2",
                "transaction_date": "2026-07-05",
                "description": "Groceries",
                "amount": 50000.0,
                "currency": "LKR",
                "transaction_type": "expense",
                "category": "Food",
                "source_dataset": "test",
            },
        ]
    )
    monthly = compute_monthly_summary(single_month_df)
    assert len(monthly) == 1
    assert monthly[0]["month"] == "2026-07"
    assert monthly[0]["income"] == 200000.0
    assert monthly[0]["expenses"] == 50000.0
    assert monthly[0]["net"] == 150000.0
    assert monthly[0]["savings_rate"] == 0.75


def test_edge_case_zero_income_no_division_by_zero():
    """When income is zero, savings_rate must be 0.0 and not raise ZeroDivisionError."""
    expense_only_df = pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "transaction_date": "2026-08-01",
                "description": "Rent",
                "amount": 60000.0,
                "currency": "LKR",
                "transaction_type": "expense",
                "category": "Rent",
                "source_dataset": "test",
            }
        ]
    )
    summary = compute_financial_summary(expense_only_df)
    assert summary["total_income"] == 0.0
    assert summary["total_expenses"] == 60000.0
    assert summary["net_savings"] == -60000.0
    assert summary["savings_rate"] == 0.0
    assert summary["savings_rate_percentage"] == 0.0


def test_edge_case_missing_category_treated_as_uncategorized():
    """Transactions with missing/NaN/empty category must be categorized as 'Uncategorized'."""
    df = pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "transaction_date": "2026-09-01",
                "description": "Mystery Shop",
                "amount": 1500.0,
                "currency": "LKR",
                "transaction_type": "expense",
                "category": None,
                "source_dataset": "test",
            },
            {
                "transaction_id": "T2",
                "transaction_date": "2026-09-02",
                "description": "Cash Withdrawal",
                "amount": 2500.0,
                "currency": "LKR",
                "transaction_type": "expense",
                "category": "",
                "source_dataset": "test",
            },
        ]
    )
    categories = compute_category_spending(df)
    assert len(categories) == 1
    assert categories[0]["category"] == "Uncategorized"
    assert categories[0]["total_amount"] == 4000.0
    assert categories[0]["transaction_count"] == 2


def test_edge_case_mixed_currency():
    """When mixed currencies exist, filter by target_currency or default to LKR."""
    df = pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "transaction_date": "2026-05-01",
                "description": "Local Income",
                "amount": 100000.0,
                "currency": "LKR",
                "transaction_type": "income",
                "category": "Salary",
                "source_dataset": "test",
            },
            {
                "transaction_id": "T2",
                "transaction_date": "2026-05-02",
                "description": "Foreign Freelance",
                "amount": 500.0,
                "currency": "USD",
                "transaction_type": "income",
                "category": "Salary",
                "source_dataset": "test",
            },
        ]
    )
    # Default behavior: filters to LKR
    summary_default = compute_financial_summary(df)
    assert summary_default["currency"] == "LKR"
    assert summary_default["total_income"] == 100000.0

    # Explicit USD filter
    summary_usd = compute_financial_summary(df, currency="USD")
    assert summary_usd["currency"] == "USD"
    assert summary_usd["total_income"] == 500.0


def test_filters_by_month_and_dates(mock_df: pd.DataFrame):
    """Test filtering by month, start_date, and end_date."""
    # Month filter
    june_summary = compute_financial_summary(mock_df, month="2026-06")
    assert june_summary["total_income"] == 275000.0

    # Date range filter
    range_summary = compute_financial_summary(
        mock_df, start_date="2026-06-01", end_date="2026-06-30"
    )
    assert range_summary["total_income"] == june_summary["total_income"]
    assert range_summary["total_expenses"] == june_summary["total_expenses"]
