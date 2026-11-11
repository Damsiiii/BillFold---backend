"""
Analytics Engine for Personal Finance Recommendation System.

Provides pure functions to calculate financial analytics from transaction DataFrames:
- Overall summary (income, expenses, net savings, savings rate)
- Monthly summaries (income, expenses, net per month)
- Category-wise spending breakdown (total, count, average, % share for expenses)

Rules:
- Pure functions: takes a pandas DataFrame, returns a plain Python dict.
- No database or LLM calls.
- All numbers computed in code.
- Contract enforcement: excludes 'transfer' transactions from income/expense metrics.
- Edge case handling: empty data, 1 month of history, missing category -> 'Uncategorized',
  zero income (no divide-by-zero), mixed currencies.
"""

import os
from typing import Any
import numpy as np
import pandas as pd


def load_transactions(csv_path: str = "data/mock_transactions.csv") -> pd.DataFrame:
    """Load transactions DataFrame from CSV file.

    Central data loader helper function designed to be swapped later
    for Thamindu's database function get_transactions().

    Args:
        csv_path: Path to transactions CSV file.

    Returns:
        pd.DataFrame containing transactions.
    """
    if not os.path.exists(csv_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        alt_path = os.path.join(base_dir, "data", "mock_transactions.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            from data.generate_mock_data import save_mock_data
            return save_mock_data(csv_path)

    return pd.read_csv(csv_path)


def prepare_transactions(
    df: pd.DataFrame,
    currency: str | None = None,
    month: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Clean and filter transaction DataFrame adhering to the system contract.

    Args:
        df: Raw transactions DataFrame.
        currency: Optional ISO currency code filter. Defaults to 'LKR' if mixed.
        month: Optional filter by month in 'YYYY-MM' format.
        start_date: Optional filter for transaction_date >= start_date ('YYYY-MM-DD').
        end_date: Optional filter for transaction_date <= end_date ('YYYY-MM-DD').

    Returns:
        tuple[pd.DataFrame, str]: Cleaned DataFrame and active currency string.
    """
    if df is None or df.empty:
        active_curr = currency or "LKR"
        empty_df = pd.DataFrame(
            columns=[
                "transaction_id",
                "transaction_date",
                "description",
                "amount",
                "currency",
                "transaction_type",
                "category",
                "source_dataset",
                "year_month",
            ]
        )
        return empty_df, active_curr

    # Work on a copy to ensure function purity
    clean_df = df.copy()

    # 1. Standardize category: treat missing / null / blank as 'Uncategorized'
    if "category" in clean_df.columns:
        clean_df["category"] = (
            clean_df["category"]
            .fillna("Uncategorized")
            .astype(str)
            .str.strip()
            .replace(
                {
                    "": "Uncategorized",
                    "nan": "Uncategorized",
                    "None": "Uncategorized",
                    "Unknown": "Uncategorized",
                    "unknown": "Uncategorized",
                }
            )
        )
    else:
        clean_df["category"] = "Uncategorized"

    # 2. Standardize transaction_type (income | expense | transfer | unknown)
    if "transaction_type" in clean_df.columns:
        clean_df["transaction_type"] = (
            clean_df["transaction_type"].fillna("unknown").astype(str).str.strip().str.lower()
        )
        clean_df["transaction_type"] = clean_df["transaction_type"].replace(
            {"debit": "expense", "credit": "income"}
        )
    else:
        clean_df["transaction_type"] = "unknown"
    clean_df = clean_df[
        clean_df["transaction_type"].isin({"income", "expense", "transfer", "unknown"})
    ].copy()

    # 3. Reject malformed, non-finite, zero, and negative amounts instead of
    # turning them into zero or silently reversing their financial meaning.
    if "amount" in clean_df.columns:
        numeric_amount = pd.to_numeric(clean_df["amount"], errors="coerce")
        valid_amount = np.isfinite(numeric_amount) & (numeric_amount > 0)
        clean_df = clean_df.loc[valid_amount].copy()
        clean_df["amount"] = numeric_amount.loc[valid_amount].astype(float)
    else:
        clean_df = clean_df.iloc[0:0].copy()
        clean_df["amount"] = pd.Series(dtype=float)

    # 4. Preserve explicit unknown currency rather than relabeling it as LKR.
    if "currency" in clean_df.columns:
        clean_df["currency"] = (
            clean_df["currency"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"": "UNKNOWN", "NAN": "UNKNOWN", "NONE": "UNKNOWN"})
        )
    else:
        clean_df["currency"] = "LKR"

    unique_currencies = [c for c in clean_df["currency"].unique() if c]
    if currency:
        active_curr = currency.upper().strip()
        clean_df = clean_df[clean_df["currency"] == active_curr]
    else:
        if len(unique_currencies) == 1:
            active_curr = unique_currencies[0]
        else:
            # Rule: Mixed currencies default to LKR
            active_curr = "LKR"
            clean_df = clean_df[clean_df["currency"] == active_curr]

    # 5. Standardize transaction_date (ISO YYYY-MM-DD) and extract year_month
    if "transaction_date" in clean_df.columns:
        clean_df["transaction_date"] = pd.to_datetime(
            clean_df["transaction_date"], errors="coerce"
        )
        clean_df = clean_df.dropna(subset=["transaction_date"])
        clean_df["year_month"] = clean_df["transaction_date"].dt.strftime("%Y-%m")
        clean_df["transaction_date"] = clean_df["transaction_date"].dt.strftime("%Y-%m-%d")
    else:
        clean_df = clean_df.iloc[0:0].copy()
        clean_df["year_month"] = pd.Series(dtype=str)
        clean_df["transaction_date"] = pd.Series(dtype=str)

    # 6. Apply optional date filters
    if month:
        clean_df = clean_df[clean_df["year_month"] == month]
    if start_date:
        clean_df = clean_df[clean_df["transaction_date"] >= start_date]
    if end_date:
        clean_df = clean_df[clean_df["transaction_date"] <= end_date]

    return clean_df.reset_index(drop=True), active_curr


def compute_financial_summary(
    df: pd.DataFrame,
    currency: str | None = None,
    month: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Compute overall financial summary: total income, expenses, net savings, and savings rate.

    Rule: Exclude 'transfer' transactions from all calculations.
    Rule: Zero income edge case -> savings rate is 0.0 (no divide-by-zero).

    Args:
        df: Transactions DataFrame.
        currency: Optional currency code.
        month: Optional month filter (YYYY-MM).
        start_date: Optional start date filter (YYYY-MM-DD).
        end_date: Optional end date filter (YYYY-MM-DD).

    Returns:
        dict with total_income, total_expenses, net_savings, savings_rate, savings_rate_percentage.
    """
    clean_df, active_curr = prepare_transactions(
        df, currency=currency, month=month, start_date=start_date, end_date=end_date
    )

    if clean_df.empty:
        return {
            "total_income": 0.0,
            "total_expenses": 0.0,
            "net_savings": 0.0,
            "savings_rate": 0.0,
            "savings_rate_percentage": 0.0,
            "currency": active_curr,
            "income_transaction_count": 0,
            "expense_transaction_count": 0,
            "total_transaction_count": 0,
        }

    # Contract Rule: Exclude transfer transactions
    income_mask = clean_df["transaction_type"] == "income"
    expense_mask = clean_df["transaction_type"] == "expense"

    total_income = float(clean_df.loc[income_mask, "amount"].sum())
    total_expenses = float(clean_df.loc[expense_mask, "amount"].sum())
    net_savings = float(total_income - total_expenses)

    # Rule: Zero income edge case -> savings rate = 0.0 (no division by zero)
    if total_income > 0:
        savings_rate = float(net_savings / total_income)
    else:
        savings_rate = 0.0

    income_count = int(income_mask.sum())
    expense_count = int(expense_mask.sum())
    total_count = int(len(clean_df))

    return {
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "net_savings": round(net_savings, 2),
        "savings_rate": round(savings_rate, 4),
        "savings_rate_percentage": round(savings_rate * 100, 2),
        "currency": active_curr,
        "income_transaction_count": income_count,
        "expense_transaction_count": expense_count,
        "total_transaction_count": total_count,
    }


def compute_monthly_summary(
    df: pd.DataFrame,
    currency: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Compute monthly breakdown of income, expenses, and net savings.

    Rule: Exclude 'transfer' transactions.
    Rule: Handles single month and multi-month history properly.

    Args:
        df: Transactions DataFrame.
        currency: Optional currency code.
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        list of dicts, each representing a month sorted chronologically.
    """
    clean_df, active_curr = prepare_transactions(
        df, currency=currency, start_date=start_date, end_date=end_date
    )

    if clean_df.empty:
        return []

    # Exclude transfers
    valid_df = clean_df[clean_df["transaction_type"].isin(["income", "expense"])]
    if valid_df.empty:
        return []

    all_months = sorted(clean_df["year_month"].unique())
    monthly_results = []

    for m in all_months:
        m_df = clean_df[clean_df["year_month"] == m]
        m_income = float(m_df.loc[m_df["transaction_type"] == "income", "amount"].sum())
        m_expense = float(m_df.loc[m_df["transaction_type"] == "expense", "amount"].sum())
        m_net = float(m_income - m_expense)

        m_savings_rate = float(m_net / m_income) if m_income > 0 else 0.0

        monthly_results.append(
            {
                "month": str(m),
                "income": round(m_income, 2),
                "expenses": round(m_expense, 2),
                "net": round(m_net, 2),
                "savings_rate": round(m_savings_rate, 4),
                "savings_rate_percentage": round(m_savings_rate * 100, 2),
                "transaction_count": int(len(m_df)),
                "currency": active_curr,
            }
        )

    return monthly_results


def compute_category_spending(
    df: pd.DataFrame,
    currency: str | None = None,
    month: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Compute category-wise spending breakdown for expense transactions only.

    Metrics per category:
    - total: Total amount spent in category
    - count: Number of transactions in category
    - average: Average transaction amount in category
    - percentage_share: % of total expenses

    Rule: Expenses only (exclude income, transfer, unknown).
    Rule: Missing categories are grouped under 'Uncategorized'.

    Args:
        df: Transactions DataFrame.
        currency: Optional currency filter.
        month: Optional month filter (YYYY-MM).
        start_date: Optional start date filter.
        end_date: Optional end date filter.

    Returns:
        list of dicts sorted descending by total spending amount.
    """
    clean_df, active_curr = prepare_transactions(
        df, currency=currency, month=month, start_date=start_date, end_date=end_date
    )

    if clean_df.empty:
        return []

    # Filter for expenses only
    expenses_df = clean_df[clean_df["transaction_type"] == "expense"]
    if expenses_df.empty:
        return []

    total_expense_sum = float(expenses_df["amount"].sum())

    # Group by category
    grouped = (
        expenses_df.groupby("category")
        .agg(
            total_amount=("amount", "sum"),
            transaction_count=("amount", "count"),
            average_amount=("amount", "mean"),
        )
        .reset_index()
    )

    # Sort descending by total amount
    grouped = grouped.sort_values(by="total_amount", ascending=False).reset_index(drop=True)

    category_results = []
    for _, row in grouped.iterrows():
        cat_total = float(row["total_amount"])
        cat_count = int(row["transaction_count"])
        cat_avg = float(row["average_amount"])
        pct_share = (cat_total / total_expense_sum * 100.0) if total_expense_sum > 0 else 0.0

        category_results.append(
            {
                "category": str(row["category"]),
                "total_amount": round(cat_total, 2),
                "transaction_count": cat_count,
                "average_amount": round(cat_avg, 2),
                "percentage_share": round(pct_share, 2),
                "currency": active_curr,
            }
        )

    return category_results


def calculate_analytics(
    df: pd.DataFrame,
    currency: str | None = None,
    month: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Execute the full analytics engine pipeline and return a structured dictionary.

    Pure function combining summary, monthly breakdown, and category spending.

    Args:
        df: Input transactions DataFrame.
        currency: Optional currency code.
        month: Optional month filter ('YYYY-MM').
        start_date: Optional start date filter ('YYYY-MM-DD').
        end_date: Optional end date filter ('YYYY-MM-DD').

    Returns:
        dict containing:
        - summary: Overall metrics (total_income, total_expenses, net_savings, savings_rate)
        - monthly_summary: Month-by-month trajectory
        - category_spending: Category-wise breakdown (expenses only)
        - metadata: Active currency, record counts, active filters
    """
    summary = compute_financial_summary(
        df, currency=currency, month=month, start_date=start_date, end_date=end_date
    )
    monthly = compute_monthly_summary(
        df, currency=currency, start_date=start_date, end_date=end_date
    )
    categories = compute_category_spending(
        df, currency=currency, month=month, start_date=start_date, end_date=end_date
    )

    return {
        "summary": summary,
        "monthly_summary": monthly,
        "category_spending": categories,
        "metadata": {
            "currency": summary.get("currency", "LKR"),
            "filter_month": month,
            "filter_start_date": start_date,
            "filter_end_date": end_date,
            "total_records_analyzed": summary.get("total_transaction_count", 0),
        },
    }
