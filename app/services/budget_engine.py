"""
Budget Engine for Personal Finance Recommendation System.

Provides pure functions to calculate suggested monthly budgets and detect overspending:
- Suggested monthly budget per category = median of the last 3 months.
- Category reduction rule:
  - Discretionary (Entertainment, Shopping, Dining, Other, Uncategorized): 10% reduction target.
  - Essentials (Rent, Utilities, Food, Transport, Health, Education): stay at 100% of historical level.
- Overspending detection for a chosen month:
  - status: 'ok' (< 80% of budget), 'warning' (80% - 100%), 'over' (> 100% of budget).
  - % change vs historical average.

Rules:
- Pure functions: takes a pandas DataFrame, returns a plain Python dict.
- No database or LLM calls.
- All numbers computed in code.
- Contract enforcement: excludes 'transfer' and non-expense transactions from budget calculations.
- Edge case handling: empty data, one month of history, missing category -> 'Uncategorized',
  mixed currencies, zero spending.
"""

from typing import Any
import numpy as np
import pandas as pd
from app.services.analytics_engine import prepare_transactions

# Discretionary categories get a 10% reduction target
DISCRETIONARY_CATEGORIES = {
    "entertainment",
    "shopping",
    "dining",
    "other",
    "uncategorized",
}

# Essential categories stay at 100% of historical median
ESSENTIAL_CATEGORIES = {
    "rent",
    "utilities",
    "food",
    "transport",
    "health",
    "healthcare",
    "education",
}


def is_discretionary_category(category_name: str) -> bool:
    """Classify a category as discretionary or essential.

    Rule: Categories matching Entertainment, Shopping, Dining, Other,
    or Uncategorized are discretionary. Essentials include Rent, Utilities,
    Food, Transport, Health, Education. Unrecognized categories default to discretionary.
    """
    cat_norm = category_name.strip().lower()
    if cat_norm in ESSENTIAL_CATEGORIES:
        return False
    return True


def get_baseline_months(available_months: list[str], target_month: str | None = None) -> list[str]:
    """Determine the up to 3 baseline historical months for budget calculations.

    Args:
        available_months: Sorted list of all months ('YYYY-MM') in dataset.
        target_month: The month being analyzed or budgeted for.

    Returns:
        list of up to 3 baseline month strings sorted chronologically.
    """
    if not available_months:
        return []

    sorted_months = sorted(available_months)

    if target_month:
        # Months strictly prior to target_month
        prior_months = [m for m in sorted_months if m < target_month]
        if prior_months:
            return prior_months[-3:]
        # Edge case: target_month is the earliest month or only month
        if target_month in sorted_months:
            return [target_month]
        return sorted_months[-3:]

    # Default: last 3 available months
    return sorted_months[-3:]


def generate_suggested_budgets(
    df: pd.DataFrame,
    target_month: str | None = None,
    currency: str | None = None,
) -> dict[str, Any]:
    """Compute suggested monthly budget per category based on historical median.

    Rule: Median of last 3 months.
    Rule: 10% reduction for discretionary, 0% reduction for essentials.

    Args:
        df: Input transactions DataFrame.
        target_month: The target month ('YYYY-MM') for which budget is prepared.
        currency: Optional currency code filter.

    Returns:
        dict containing suggested budget per category and metadata.
    """
    clean_df, active_curr = prepare_transactions(df, currency=currency)

    # Exclude non-expenses (transfers, income, unknown)
    expenses_df = clean_df[clean_df["transaction_type"] == "expense"].copy()

    if expenses_df.empty:
        return {
            "target_month": target_month,
            "currency": active_curr,
            "baseline_months": [],
            "total_suggested_budget": 0.0,
            "category_budgets": [],
        }

    all_months = sorted(clean_df["year_month"].unique())
    baseline_months = get_baseline_months(all_months, target_month)

    if not baseline_months:
        return {
            "target_month": target_month,
            "currency": active_curr,
            "baseline_months": [],
            "total_suggested_budget": 0.0,
            "category_budgets": [],
        }

    # Filter expenses to baseline months
    baseline_df = expenses_df[expenses_df["year_month"].isin(baseline_months)]
    all_categories = sorted(expenses_df["category"].unique())

    # Build monthly spend table: row = month, col = category, filled with 0.0
    monthly_cat_spend = (
        baseline_df.groupby(["year_month", "category"])["amount"]
        .sum()
        .unstack(fill_value=0.0)
    )

    # Ensure all baseline months and categories exist in the grid
    for m in baseline_months:
        if m not in monthly_cat_spend.index:
            monthly_cat_spend.loc[m] = 0.0
    for c in all_categories:
        if c not in monthly_cat_spend.columns:
            monthly_cat_spend[c] = 0.0

    monthly_cat_spend = monthly_cat_spend.loc[baseline_months]

    category_budgets = []
    total_suggested = 0.0

    for cat in all_categories:
        cat_series = monthly_cat_spend[cat]
        hist_median = float(cat_series.median())
        hist_average = float(cat_series.mean())

        is_disc = is_discretionary_category(cat)
        # 10% reduction for discretionary, 0% for essentials
        reduction_rate = 0.10 if is_disc else 0.0
        suggested_budget = round(hist_median * (1.0 - reduction_rate), 2)
        total_suggested += suggested_budget

        category_budgets.append(
            {
                "category": cat,
                "category_type": "discretionary" if is_disc else "essential",
                "reduction_target_percentage": round(reduction_rate * 100, 1),
                "historical_median": round(hist_median, 2),
                "historical_average": round(hist_average, 2),
                "suggested_budget": suggested_budget,
            }
        )

    # Sort descending by suggested budget
    category_budgets.sort(key=lambda x: x["suggested_budget"], reverse=True)

    return {
        "target_month": target_month,
        "currency": active_curr,
        "baseline_months": baseline_months,
        "total_suggested_budget": round(total_suggested, 2),
        "category_budgets": category_budgets,
    }


def detect_overspending(
    df: pd.DataFrame,
    target_month: str | None = None,
    currency: str | None = None,
) -> dict[str, Any]:
    """Detect category overspending for a chosen month against suggested budgets.

    Status rules:
    - 'ok': actual < 80% of budget
    - 'warning': 80% <= actual <= 100% of budget
    - 'over': actual > 100% of budget

    Also calculates percentage change vs historical average.

    Args:
        df: Input transactions DataFrame.
        target_month: Month to inspect ('YYYY-MM'). Defaults to latest month.
        currency: Optional currency code.

    Returns:
        dict with overspending status per category and overall summary.
    """
    clean_df, active_curr = prepare_transactions(df, currency=currency)
    expenses_df = clean_df[clean_df["transaction_type"] == "expense"].copy()

    if expenses_df.empty:
        return {
            "target_month": target_month,
            "currency": active_curr,
            "baseline_months": [],
            "total_budget": 0.0,
            "total_actual_spending": 0.0,
            "overall_status": "ok",
            "categories": [],
            "summary": {"ok_count": 0, "warning_count": 0, "over_count": 0},
        }

    all_months = sorted(clean_df["year_month"].unique())
    chosen_month = target_month or all_months[-1]

    # Generate baseline budgets for chosen_month
    budget_result = generate_suggested_budgets(df, target_month=chosen_month, currency=currency)
    baseline_months = budget_result["baseline_months"]
    budgets_map = {b["category"]: b for b in budget_result["category_budgets"]}

    # Actual spending in the target month
    target_month_expenses = expenses_df[expenses_df["year_month"] == chosen_month]
    actual_spending_map = (
        target_month_expenses.groupby("category")["amount"].sum().to_dict()
    )

    # Union of all categories
    all_categories = sorted(set(budgets_map.keys()).union(actual_spending_map.keys()))

    categories_detail = []
    total_budget_sum = 0.0
    total_actual_sum = 0.0
    status_counts = {"ok": 0, "warning": 0, "over": 0}

    for cat in all_categories:
        b_info = budgets_map.get(cat, {})
        suggested_budget = float(b_info.get("suggested_budget", 0.0))
        hist_median = float(b_info.get("historical_median", 0.0))
        hist_average = float(b_info.get("historical_average", 0.0))
        cat_type = b_info.get(
            "category_type", "discretionary" if is_discretionary_category(cat) else "essential"
        )
        reduction_rate = float(b_info.get("reduction_target_percentage", 10.0 if cat_type == "discretionary" else 0.0))

        actual = float(actual_spending_map.get(cat, 0.0))
        total_budget_sum += suggested_budget
        total_actual_sum += actual

        # Determine status
        if suggested_budget > 0:
            pct_of_budget = (actual / suggested_budget) * 100.0
            if pct_of_budget < 80.0:
                status = "ok"
            elif pct_of_budget <= 100.0:
                status = "warning"
            else:
                status = "over"
        else:
            # Edge case: No historical budget
            if actual > 0:
                pct_of_budget = 100.0
                status = "over"
            else:
                pct_of_budget = 0.0
                status = "ok"

        status_counts[status] += 1

        # Percentage change vs historical average
        if hist_average > 0:
            pct_change_vs_avg = ((actual - hist_average) / hist_average) * 100.0
        elif actual > 0:
            pct_change_vs_avg = 100.0
        else:
            pct_change_vs_avg = 0.0

        overspend_amt = max(0.0, actual - suggested_budget)

        categories_detail.append(
            {
                "category": cat,
                "category_type": cat_type,
                "suggested_budget": round(suggested_budget, 2),
                "actual_spending": round(actual, 2),
                "historical_median": round(hist_median, 2),
                "historical_average": round(hist_average, 2),
                "reduction_target_percentage": reduction_rate,
                "percentage_of_budget": round(pct_of_budget, 2),
                "percentage_change_vs_historical_avg": round(pct_change_vs_avg, 2),
                "status": status,
                "overspend_amount": round(overspend_amt, 2),
            }
        )

    # Sort categories: 'over' first, then 'warning', then 'ok', and within status by overspend_amount desc
    status_priority = {"over": 0, "warning": 1, "ok": 2}
    categories_detail.sort(
        key=lambda x: (status_priority[x["status"]], -x["overspend_amount"], -x["actual_spending"])
    )

    # Overall month status
    if status_counts["over"] > 0:
        overall_status = "over"
    elif status_counts["warning"] > 0:
        overall_status = "warning"
    else:
        overall_status = "ok"

    return {
        "target_month": chosen_month,
        "currency": active_curr,
        "baseline_months": baseline_months,
        "total_budget": round(total_budget_sum, 2),
        "total_actual_spending": round(total_actual_sum, 2),
        "overall_status": overall_status,
        "summary": {
            "ok_count": status_counts["ok"],
            "warning_count": status_counts["warning"],
            "over_count": status_counts["over"],
        },
        "categories": categories_detail,
    }
