"""
Savings Engine for Personal Finance Recommendation System.

Provides pure functions to analyze savings performance and generate recommendations:
1. Savings Gap Analysis:
   - Evaluates current savings rate vs a 20% target.
   - Computes savings rate gap and numerical monetary shortfall.
2. Discretionary Spending Cuts:
   - Identifies and ranks discretionary categories by spend.
   - Computes realistic reductions and concrete monthly savings in code.
3. Subscription Detection:
   - Detects recurring subscriptions (same merchant, similar amount, roughly monthly).
   - Computes annualized costs and cancellation savings.

Rules:
- Pure functions: takes a pandas DataFrame, returns a plain Python dict.
- No database or LLM calls.
- All numbers computed in code.
- Contract enforcement: excludes 'transfer' and non-expense transactions from calculations.
- Edge case handling: empty data, zero income (no divide-by-zero), missing category -> 'Uncategorized',
  mixed currency.
"""

from typing import Any
import numpy as np
import pandas as pd
from app.services.analytics_engine import compute_financial_summary, prepare_transactions
from app.services.budget_engine import is_discretionary_category

DEFAULT_TARGET_SAVINGS_RATE = 0.20  # 20% savings target


def analyze_savings_gap(
    df: pd.DataFrame,
    target_rate: float = DEFAULT_TARGET_SAVINGS_RATE,
    month: str | None = None,
    currency: str | None = None,
) -> dict[str, Any]:
    """Calculate the gap between current savings rate and target rate (default 20%).

    Rule: Avoid division by zero when income is zero or negative.
    Rule: Exclude transfers from income and expense calculations.

    Args:
        df: Input transactions DataFrame.
        target_rate: Target savings rate as a decimal (e.g. 0.20 for 20%).
        month: Optional month filter ('YYYY-MM').
        currency: Optional currency code.

    Returns:
        dict containing current rate, target rate, gap, and monetary shortfall.
    """
    summary = compute_financial_summary(df, currency=currency, month=month)

    total_income = float(summary["total_income"])
    total_expenses = float(summary["total_expenses"])
    net_savings = float(summary["net_savings"])
    current_savings_rate = float(summary["savings_rate"])
    active_curr = str(summary["currency"])

    # Target savings in currency
    target_savings_amount = round(total_income * target_rate, 2) if total_income > 0 else 0.0

    # Rate gap: target - current
    rate_gap = round(target_rate - current_savings_rate, 4)
    rate_gap_pct = round(rate_gap * 100, 2)

    # Monetary shortfall needed to hit the target
    if current_savings_rate >= target_rate:
        meets_target = True
        savings_shortfall = 0.0
    else:
        meets_target = False
        savings_shortfall = max(0.0, round(target_savings_amount - net_savings, 2))

    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "net_savings": net_savings,
        "current_savings_rate": current_savings_rate,
        "current_savings_rate_pct": round(current_savings_rate * 100, 2),
        "target_savings_rate": target_rate,
        "target_savings_rate_pct": round(target_rate * 100, 2),
        "savings_rate_gap": rate_gap,
        "savings_rate_gap_pct": rate_gap_pct,
        "target_savings_amount": target_savings_amount,
        "savings_shortfall": savings_shortfall,
        "meets_target": meets_target,
        "currency": active_curr,
        "month": month,
    }


def rank_discretionary_cuts(
    df: pd.DataFrame,
    target_rate: float = DEFAULT_TARGET_SAVINGS_RATE,
    month: str | None = None,
    currency: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Rank discretionary categories by spend and compute realistic spending cuts.

    Rule: Discretionary categories (Entertainment, Shopping, Dining, Other, Uncategorized).
    Rule: Proposes realistic reduction targets (e.g. 10% - 20%) with exact rupee savings computed in code.

    Args:
        df: Input transactions DataFrame.
        target_rate: Target savings rate.
        month: Optional month filter.
        currency: Optional currency code.

    Returns:
        tuple[list[dict], float]: List of ranked cut recommendations and total potential monthly savings.
    """
    clean_df, active_curr = prepare_transactions(df, currency=currency, month=month)
    expenses_df = clean_df[clean_df["transaction_type"] == "expense"]

    if expenses_df.empty:
        return [], 0.0

    # Group by category
    cat_spend = expenses_df.groupby("category")["amount"].sum().reset_index()

    # Filter for discretionary categories
    cat_spend["is_discretionary"] = cat_spend["category"].apply(is_discretionary_category)
    disc_spend = cat_spend[cat_spend["is_discretionary"]].copy()

    if disc_spend.empty:
        return [], 0.0

    # Sort descending by spend
    disc_spend = disc_spend.sort_values(by="amount", ascending=False).reset_index(drop=True)
    total_disc_spend = float(disc_spend["amount"].sum())

    recommendations = []
    total_potential_savings = 0.0

    for rank_idx, row in disc_spend.iterrows():
        cat = str(row["category"])
        spend = float(row["amount"])
        share_of_disc = (spend / total_disc_spend * 100.0) if total_disc_spend > 0 else 0.0

        # Tiered realistic cut: top discretionary categories get 15% cut, others 10%
        cut_pct = 15.0 if rank_idx < 2 else 10.0
        potential_saving = round(spend * (cut_pct / 100.0), 2)
        new_spend_target = round(spend - potential_saving, 2)
        total_potential_savings += potential_saving

        recommendations.append(
            {
                "rank": int(rank_idx + 1),
                "category": cat,
                "current_spend": round(spend, 2),
                "percentage_of_discretionary": round(share_of_disc, 2),
                "recommended_cut_percentage": cut_pct,
                "potential_monthly_saving": potential_saving,
                "suggested_new_monthly_spend": new_spend_target,
                "reason": (
                    f"Reduce {cat} spend by {cut_pct:.1f}% to save "
                    f"{potential_saving:,.2f} {active_curr}/month (new target: {new_spend_target:,.2f} {active_curr})"
                ),
            }
        )

    return recommendations, round(total_potential_savings, 2)


def detect_recurring_subscriptions(
    df: pd.DataFrame,
    currency: str | None = None,
    min_occurrences: int = 3,
    min_interval_days: int = 20,
    max_interval_days: int = 40,
    max_amount_cv: float = 0.15,
) -> list[dict[str, Any]]:
    """Detect recurring subscriptions based on same merchant, similar amount, and roughly monthly interval.

    Rule: Same merchant/description.
    Rule: Similar amount (coefficient of variation <= 0.15).
    Rule: Roughly monthly interval (average days between charges 20 to 40 days).
    Rule: At least min_occurrences charges across different dates.

    Args:
        df: Input transactions DataFrame.
        currency: Optional currency code.
        min_occurrences: Minimum charges required (default 3).
        min_interval_days: Minimum days between charges to qualify as monthly (default 20).
        max_interval_days: Maximum days between charges to qualify as monthly (default 40).
        max_amount_cv: Max coefficient of variation for amount similarity (default 0.15).

    Returns:
        list of dicts, each representing a detected subscription.
    """
    clean_df, active_curr = prepare_transactions(df, currency=currency)
    expenses_df = clean_df[clean_df["transaction_type"] == "expense"].copy()

    if expenses_df.empty or len(expenses_df) < min_occurrences:
        return []

    expenses_df["dt"] = pd.to_datetime(expenses_df["transaction_date"])
    expenses_df = expenses_df.sort_values(by=["dt", "transaction_id"]).reset_index(drop=True)

    subscriptions = []

    # Group by normalized merchant description
    for desc, group in expenses_df.groupby("description"):
        if len(group) < min_occurrences:
            continue

        amounts = group["amount"]
        mean_amt = float(amounts.mean())
        std_amt = float(amounts.std(ddof=0)) if len(amounts) > 1 else 0.0

        # Check amount similarity (low coefficient of variation)
        cv = (std_amt / mean_amt) if mean_amt > 0 else 0.0
        if cv > max_amount_cv:
            continue

        # Check date intervals
        unique_dates = group["dt"].drop_duplicates().sort_values()
        if len(unique_dates) < min_occurrences:
            continue

        intervals = (unique_dates.diff().dt.days).dropna().tolist()
        if not intervals:
            continue

        avg_interval = float(np.mean(intervals))

        # Check roughly monthly: average interval between 20 and 40 days
        # and at least 60% of intervals fall within roughly monthly window (25-35 days)
        if not (min_interval_days <= avg_interval <= max_interval_days):
            continue

        monthly_cost = round(mean_amt, 2)
        annual_cost = round(monthly_cost * 12.0, 2)
        cat = str(group["category"].iloc[0])
        first_date = str(group["transaction_date"].min())
        last_date = str(group["transaction_date"].max())

        subscriptions.append(
            {
                "merchant": str(desc),
                "category": cat,
                "occurrence_count": int(len(group)),
                "average_amount": monthly_cost,
                "estimated_monthly_cost": monthly_cost,
                "estimated_annual_cost": annual_cost,
                "average_interval_days": round(avg_interval, 1),
                "first_seen_date": first_date,
                "last_seen_date": last_date,
                "currency": active_curr,
                "savings_tip": (
                    f"Canceling or pausing '{desc}' would save {monthly_cost:,.2f} {active_curr}/month "
                    f"({annual_cost:,.2f} {active_curr}/year)"
                ),
            }
        )

    # Sort descending by monthly cost
    subscriptions.sort(key=lambda s: s["estimated_monthly_cost"], reverse=True)
    return subscriptions


def calculate_savings_recommendations(
    df: pd.DataFrame,
    target_rate: float = DEFAULT_TARGET_SAVINGS_RATE,
    month: str | None = None,
    currency: str | None = None,
) -> dict[str, Any]:
    """Unified savings engine entry point combining gap analysis, spending cuts, and subscriptions.

    Args:
        df: Input transactions DataFrame.
        target_rate: Target savings rate (default 0.20 for 20%).
        month: Optional month filter ('YYYY-MM').
        currency: Optional currency code.

    Returns:
        dict containing gap analysis, ranked discretionary cuts, and detected subscriptions.
    """
    gap_info = analyze_savings_gap(df, target_rate=target_rate, month=month, currency=currency)
    cuts, potential_savings = rank_discretionary_cuts(
        df, target_rate=target_rate, month=month, currency=currency
    )
    subscriptions = detect_recurring_subscriptions(df, currency=currency)

    total_sub_monthly = round(sum(s["estimated_monthly_cost"] for s in subscriptions), 2)
    shortfall = float(gap_info["savings_shortfall"])

    if shortfall > 0:
        shortfall_covered_pct = round((potential_savings / shortfall) * 100.0, 2)
    else:
        shortfall_covered_pct = 100.0

    return {
        "savings_summary": gap_info,
        "discretionary_cuts": cuts,
        "total_potential_monthly_savings": potential_savings,
        "shortfall_covered_percentage": shortfall_covered_pct,
        "recurring_subscriptions": subscriptions,
        "total_subscription_monthly_spend": total_sub_monthly,
        "currency": gap_info["currency"],
    }

