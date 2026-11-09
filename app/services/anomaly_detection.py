"""
Anomaly Detection Engine for Personal Finance Recommendation System.

Provides pure functions to detect unusual transactions in transaction DataFrames:
- Statistical outliers:
  - Per category: flag z-score > 3 (or IQR fences) only if category has >= 5 transactions.
  - Otherwise, fallback to global threshold across all expense transactions.
- Possible duplicate charges:
  - Flag transactions with identical description and amount occurring within 2 days.
- Large transaction at a new merchant:
  - Flag high-value transactions at merchants/descriptions not seen before in history.
- Wording policy:
  - All reasons and descriptions use 'unusual transaction', NEVER 'fraud'.

Rules:
- Pure functions: takes a pandas DataFrame, returns a plain Python dict.
- No database or LLM calls.
- All numbers computed in code.
- Contract enforcement: excludes 'transfer' and non-expense transactions from expense anomaly detection.
- Edge case handling: empty data, single month, category < 5 transactions, missing category -> 'Uncategorized',
  mixed currency.
"""

from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from app.services.analytics_engine import prepare_transactions


def detect_anomalies(
    df: pd.DataFrame,
    currency: str | None = None,
    month: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    z_threshold: float = 3.0,
    min_category_samples: int = 5,
    duplicate_window_days: int = 2,
    large_new_merchant_multiplier: float = 2.5,
) -> dict[str, Any]:
    """Detect unusual transactions based on statistical outliers, duplicates, and new merchants.

    Args:
        df: Input transactions DataFrame.
        currency: Optional currency code.
        month: Optional month filter ('YYYY-MM') for reported anomalies.
        start_date: Optional start date filter for reported anomalies.
        end_date: Optional end date filter for reported anomalies.
        z_threshold: Z-score cutoff for statistical outliers (default 3.0).
        min_category_samples: Minimum category sample size to use category stats (default 5).
        duplicate_window_days: Max days between identical charges to flag duplicate (default 2).
        large_new_merchant_multiplier: Multiplier over category/global avg to flag new merchant (default 2.5).

    Returns:
        dict containing flagged unusual transactions, counts by rule, and metadata.
    """
    clean_df, active_curr = prepare_transactions(df, currency=currency)

    # Filter for expense transactions (exclude transfers, income, unknown)
    expenses_df = clean_df[clean_df["transaction_type"] == "expense"].copy()

    if expenses_df.empty:
        return {
            "anomalies": [],
            "total_anomalies": 0,
            "currency": active_curr,
            "filter_month": month,
            "summary_by_type": {
                "statistical_outlier": 0,
                "duplicate_charge": 0,
                "new_merchant_large_spend": 0,
            },
        }

    # Sort chronologically for sequence and merchant history analysis
    expenses_df = expenses_df.sort_values(
        by=["transaction_date", "transaction_id"]
    ).reset_index(drop=True)

    # --------------------------------------------------------------------------
    # 1. Global statistics calculation (used when category has < min_category_samples)
    # --------------------------------------------------------------------------
    global_amounts = expenses_df["amount"]
    global_mean = float(global_amounts.mean())
    global_std = float(global_amounts.std(ddof=0)) if len(global_amounts) > 1 else 0.0
    global_median = float(global_amounts.median())

    # --------------------------------------------------------------------------
    # 2. Per-category statistics calculation
    # --------------------------------------------------------------------------
    category_stats: dict[str, dict[str, float]] = {}
    for cat, group in expenses_df.groupby("category"):
        c_amounts = group["amount"]
        c_count = len(c_amounts)
        c_mean = float(c_amounts.mean())
        c_std = float(c_amounts.std(ddof=0)) if c_count > 1 else 0.0
        c_median = float(c_amounts.median())
        q25 = float(c_amounts.quantile(0.25))
        q75 = float(c_amounts.quantile(0.75))
        iqr = q75 - q25

        category_stats[cat] = {
            "count": c_count,
            "mean": c_mean,
            "std": c_std,
            "median": c_median,
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
        }

    # --------------------------------------------------------------------------
    # 3. Duplicate Detection (same description and amount within duplicate_window_days)
    # --------------------------------------------------------------------------
    # Parse dates for datetime arithmetic
    parsed_dates = pd.to_datetime(expenses_df["transaction_date"])
    duplicate_flags: dict[str, list[dict[str, Any]]] = {}

    n_rows = len(expenses_df)
    for i in range(n_rows):
        row_i = expenses_df.iloc[i]
        date_i = parsed_dates.iloc[i]
        desc_i = str(row_i["description"]).strip().lower()
        amt_i = float(row_i["amount"])
        id_i = str(row_i["transaction_id"])

        for j in range(i + 1, n_rows):
            row_j = expenses_df.iloc[j]
            date_j = parsed_dates.iloc[j]
            day_diff = abs((date_j - date_i).days)

            # If date difference exceeds window, since df is sorted, check next or stop
            if day_diff > duplicate_window_days:
                break

            desc_j = str(row_j["description"]).strip().lower()
            amt_j = float(row_j["amount"])
            id_j = str(row_j["transaction_id"])

            if desc_i == desc_j and abs(amt_i - amt_j) < 0.01:
                # Duplicate pair found!
                duplicate_flags.setdefault(id_i, []).append(
                    {
                        "matched_txn_id": id_j,
                        "matched_date": str(row_j["transaction_date"]),
                        "days_apart": int(day_diff),
                    }
                )
                duplicate_flags.setdefault(id_j, []).append(
                    {
                        "matched_txn_id": id_i,
                        "matched_date": str(row_i["transaction_date"]),
                        "days_apart": int(day_diff),
                    }
                )

    # --------------------------------------------------------------------------
    # 4. Merchant History & Anomaly Evaluation per Transaction
    # --------------------------------------------------------------------------
    seen_merchants: set[str] = set()
    anomalies = []
    type_counts = {
        "statistical_outlier": 0,
        "duplicate_charge": 0,
        "new_merchant_large_spend": 0,
    }

    for idx in range(n_rows):
        row = expenses_df.iloc[idx]
        t_id = str(row["transaction_id"])
        t_date = str(row["transaction_date"])
        t_desc = str(row["description"]).strip()
        t_cat = str(row["category"])
        t_amt = float(row["amount"])
        t_month = str(row.get("year_month", t_date[:7]))

        norm_desc = t_desc.lower()
        is_first_time_merchant = norm_desc not in seen_merchants
        seen_merchants.add(norm_desc)

        rules_triggered = []
        reasons = []
        metrics: dict[str, Any] = {}

        # Rule A: Statistical Outlier (Z-Score & IQR)
        c_stat = category_stats.get(t_cat, {})
        c_count = c_stat.get("count", 0)

        if c_count >= min_category_samples:
            # Category has sufficient transactions (>= 5)
            c_mean = c_stat["mean"]
            c_std = c_stat["std"]
            z_score = float((t_amt - c_mean) / c_std) if c_std > 0 else 0.0
            iqr_fence = c_stat["q75"] + (3.0 * c_stat["iqr"])

            # Flag if z-score > threshold OR exceeds IQR extreme fence (Tukey 3x IQR)
            is_z_outlier = z_score > z_threshold
            is_iqr_outlier = (c_stat["iqr"] > 0 and t_amt > iqr_fence and t_amt > c_mean) or (c_std == 0 and t_amt > c_mean)
            is_outlier = is_z_outlier or is_iqr_outlier
            ratio = (t_amt / c_mean) if c_mean > 0 else 1.0

            if is_outlier:
                rules_triggered.append("statistical_outlier")
                type_counts["statistical_outlier"] += 1
                threshold_name = "category_z_score" if is_z_outlier else "category_iqr_fence"
                reasons.append(
                    f"Unusual transaction: {ratio:.1f}x larger than your usual {t_cat} spend "
                    f"({t_amt:,.2f} {active_curr} vs avg {c_mean:,.2f} {active_curr})"
                )
                metrics.update(
                    {
                        "z_score": round(z_score, 2),
                        "category_avg": round(c_mean, 2),
                        "ratio_to_avg": round(ratio, 2),
                        "threshold_used": threshold_name,
                        "category_sample_count": c_count,
                    }
                )
            baseline_mean = c_mean
        else:
            # Rule fallback: Global threshold when category < 5 transactions
            z_score = float((t_amt - global_mean) / global_std) if global_std > 0 else 0.0
            ratio = (t_amt / global_mean) if global_mean > 0 else 1.0
            g_q25 = float(expenses_df["amount"].quantile(0.25))
            g_q75 = float(expenses_df["amount"].quantile(0.75))
            global_iqr_fence = g_q75 + (3.0 * (g_q75 - g_q25))

            is_z_outlier = z_score > z_threshold
            is_iqr_outlier = (g_q75 > g_q25 and t_amt > global_iqr_fence and t_amt > global_mean) or (global_std == 0 and t_amt > global_mean)
            is_outlier = is_z_outlier or is_iqr_outlier

            if is_outlier:
                rules_triggered.append("statistical_outlier")
                type_counts["statistical_outlier"] += 1
                threshold_name = "global_z_score" if is_z_outlier else "global_iqr_fence"
                reasons.append(
                    f"Unusual transaction: {ratio:.1f}x larger than your overall typical spend "
                    f"({t_amt:,.2f} {active_curr} vs global avg {global_mean:,.2f} {active_curr})"
                )
                metrics.update(
                    {
                        "z_score": round(z_score, 2),
                        "global_avg": round(global_mean, 2),
                        "ratio_to_avg": round(ratio, 2),
                        "threshold_used": threshold_name,
                        "category_sample_count": c_count,
                    }
                )
            baseline_mean = global_mean

        # Rule B: Possible Duplicate (same description and amount within duplicate_window_days)
        if t_id in duplicate_flags:
            dup_matches = duplicate_flags[t_id]
            rules_triggered.append("duplicate_charge")
            type_counts["duplicate_charge"] += 1
            match_details = ", ".join(
                [f"{m['matched_txn_id']} on {m['matched_date']}" for m in dup_matches]
            )
            reasons.append(
                f"Unusual transaction: Possible duplicate charge of {t_amt:,.2f} {active_curr} "
                f"matching {match_details} within {duplicate_window_days} days"
            )
            metrics["duplicate_matches"] = dup_matches

        # Rule C: Large transaction at a merchant not seen before
        if is_first_time_merchant and baseline_mean > 0:
            is_large = t_amt >= (large_new_merchant_multiplier * baseline_mean)
            if is_large:
                rules_triggered.append("new_merchant_large_spend")
                type_counts["new_merchant_large_spend"] += 1
                ratio_new = t_amt / baseline_mean
                reasons.append(
                    f"Unusual transaction: Large spend of {t_amt:,.2f} {active_curr} at a new merchant "
                    f"not seen before ('{t_desc}'), {ratio_new:.1f}x higher than benchmark"
                )
                metrics["new_merchant"] = True
                metrics["new_merchant_ratio"] = round(ratio_new, 2)

        # If any anomaly rules were triggered, record the anomaly
        if rules_triggered:
            # Combine reasons concisely
            combined_reason = "; ".join(reasons)
            anomalies.append(
                {
                    "transaction_id": t_id,
                    "transaction_date": t_date,
                    "month": t_month,
                    "description": t_desc,
                    "amount": round(t_amt, 2),
                    "currency": active_curr,
                    "category": t_cat,
                    "rules_triggered": rules_triggered,
                    "reason": combined_reason,
                    "metrics": metrics,
                }
            )

    # --------------------------------------------------------------------------
    # 5. Apply optional filters to returned anomalies
    # --------------------------------------------------------------------------
    filtered_anomalies = anomalies
    if month:
        filtered_anomalies = [a for a in filtered_anomalies if a["month"] == month]
    if start_date:
        filtered_anomalies = [a for a in filtered_anomalies if a["transaction_date"] >= start_date]
    if end_date:
        filtered_anomalies = [a for a in filtered_anomalies if a["transaction_date"] <= end_date]

    # Recalculate summary counts for the filtered output
    filtered_type_counts = {
        "statistical_outlier": sum(
            1 for a in filtered_anomalies if "statistical_outlier" in a["rules_triggered"]
        ),
        "duplicate_charge": sum(
            1 for a in filtered_anomalies if "duplicate_charge" in a["rules_triggered"]
        ),
        "new_merchant_large_spend": sum(
            1 for a in filtered_anomalies if "new_merchant_large_spend" in a["rules_triggered"]
        ),
    }

    return {
        "anomalies": filtered_anomalies,
        "total_anomalies": len(filtered_anomalies),
        "currency": active_curr,
        "filter_month": month,
        "filter_start_date": start_date,
        "filter_end_date": end_date,
        "summary_by_type": filtered_type_counts,
    }
