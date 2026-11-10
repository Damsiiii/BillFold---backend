"""Load user-scoped financial data for chatbot grounding."""

from collections import defaultdict
from typing import Optional

import pandas as pd

from app.database import crud
from app.database.database import SessionLocal
from app.schemas.chatbot import FinancialContext
from app.services.analytics_engine import calculate_analytics
from app.services.anomaly_detection import detect_anomalies
from app.services.budget_engine import detect_overspending, generate_suggested_budgets
from app.services.savings_engine import calculate_savings_recommendations

CHAT_CONTEXT_TRANSACTION_LIMIT = 100


class FinancialContextProvider:
    """Fetch transactions and derive grounded financial insights for one user."""

    def __init__(
        self,
        db_service: Optional[object] = None,
        analytics_service: Optional[object] = None,
    ):
        self.db_service = db_service
        self.analytics_service = analytics_service

    def get_user_context(self, user_id: str) -> FinancialContext:
        db = SessionLocal()
        try:
            rows = crud.get_transactions(db, user_id=user_id, limit=None)
            records = [
                {
                    "transaction_id": row.id,
                    "transaction_date": row.date.isoformat(),
                    "description": row.description,
                    "amount": row.amount,
                    "currency": row.currency,
                    "transaction_type": _normalize_type(row.type),
                    "category": row.category or "Uncategorized",
                    "balance": row.balance,
                }
                for row in rows
            ]
        finally:
            db.close()

        if not records:
            return FinancialContext(user_id=user_id, has_data=False)

        by_currency: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            by_currency[record["currency"] or "Unknown"].append(record)

        currency_insights: dict[str, dict] = {}
        for currency, currency_records in by_currency.items():
            currency_insights[currency] = self._build_currency_insights(
                currency, currency_records
            )

        latest_month = max(record["transaction_date"][:7] for record in records)
        currencies = list(by_currency)
        legacy_insights = (
            currency_insights[currencies[0]]
            if len(currencies) == 1 and currencies[0] != "Unknown"
            else {}
        )
        recent_transactions = records[:CHAT_CONTEXT_TRANSACTION_LIMIT]
        financial_data = {
            "transaction_count": len(records),
            "history_start": min(record["transaction_date"] for record in records),
            "history_end": max(record["transaction_date"] for record in records),
            "transaction_history_truncated": len(records) > len(recent_transactions),
            "recent_transactions": recent_transactions,
            "currencies": currency_insights,
            "unavailable_information": [
                "The database has no separate budget-plan records; budget results are suggestions inferred from transaction history.",
                "The database has no explicit savings-goal records; savings insights are calculated from transaction history.",
                "Merchant spending is grouped by stored transaction description because no normalized merchant field is available.",
            ],
        }

        return FinancialContext(
            user_id=user_id,
            has_data=True,
            monthly_income=legacy_insights.get("latest_month_summary", {}).get(
                "total_income"
            ),
            monthly_expenses=legacy_insights.get("latest_month_summary", {}).get(
                "total_expenses"
            ),
            currency=currencies[0] if len(currencies) == 1 and currencies[0] != "Unknown" else None,
            category_spending=legacy_insights.get("latest_month_categories"),
            recent_transactions=recent_transactions,
            budget_recommendations=legacy_insights.get("budgets"),
            savings_recommendations=legacy_insights.get("savings"),
            financial_data=financial_data,
        )

    @staticmethod
    def _build_currency_insights(currency: str, records: list[dict]) -> dict:
        frame = pd.DataFrame(records)
        latest_month = max(record["transaction_date"][:7] for record in records)
        balances = [
            record
            for record in records
            if record["balance"] is not None
        ]
        latest_balance = balances[0] if balances else None
        category_totals: dict[str, float] = defaultdict(float)
        merchant_totals: dict[str, float] = defaultdict(float)
        for record in records:
            if record["transaction_type"] == "expense":
                category_totals[record["category"]] += record["amount"]
                merchant_totals[record["description"]] += record["amount"]

        weekly_totals: dict[str, dict[str, float | int]] = {}
        for record in records:
            transaction_date = pd.Timestamp(record["transaction_date"])
            week_key = transaction_date.strftime("%G-W%V")
            bucket = weekly_totals.setdefault(
                week_key,
                {"income": 0.0, "expenses": 0.0, "net_savings": 0.0, "count": 0},
            )
            if record["transaction_type"] == "income":
                bucket["income"] += record["amount"]
            elif record["transaction_type"] == "expense":
                bucket["expenses"] += record["amount"]
            bucket["net_savings"] = bucket["income"] - bucket["expenses"]
            bucket["count"] += 1

        currency_insights: dict = {
            "transaction_count": len(records),
            "latest_balance": (
                {
                    "amount": latest_balance["balance"],
                    "date": latest_balance["transaction_date"],
                }
                if latest_balance
                else None
            ),
            "category_spending": _ranked_totals(category_totals, currency),
            "merchant_spending": _ranked_totals(merchant_totals, currency),
            "weekly_summary": [
                {"week": week, **_rounded_bucket(values)}
                for week, values in sorted(weekly_totals.items())
            ],
        }

        if currency == "Unknown":
            income = sum(
                record["amount"]
                for record in records
                if record["transaction_type"] == "income"
            )
            expenses = sum(
                record["amount"]
                for record in records
                if record["transaction_type"] == "expense"
            )
            currency_insights.update(
                {
                    "summary": {
                        "income": round(income, 2),
                        "expenses": round(expenses, 2),
                        "net_savings": round(income - expenses, 2),
                        "currency": None,
                    },
                    "monthly_summary": _manual_monthly_summary(records),
                    "budget_recommendations": None,
                    "overspending": None,
                    "savings_recommendations": None,
                    "anomalies": None,
                    "limitations": [
                        "Currency is unknown, so currency-dependent analytics and recommendations were not calculated."
                    ],
                }
            )
            return currency_insights

        analytics = calculate_analytics(frame, currency=currency)
        latest_month_summary = calculate_analytics(
            frame, currency=currency, month=latest_month
        )["summary"]
        latest_month_categories = {
            item["category"]: item["total_amount"]
            for item in calculate_analytics(
                frame, currency=currency, month=latest_month
            )["category_spending"]
            if item.get("category")
        }
        currency_insights.update(
            {
                "summary": analytics["summary"],
                "latest_month": latest_month,
                "latest_month_summary": latest_month_summary,
                "monthly_summary": analytics["monthly_summary"],
                "category_spending": analytics["category_spending"],
                "latest_month_categories": latest_month_categories,
                "budgets": generate_suggested_budgets(frame),
                "overspending": detect_overspending(frame),
                "savings": calculate_savings_recommendations(frame),
                "anomalies": detect_anomalies(frame),
                "limitations": [],
            }
        )
        return currency_insights


def _normalize_type(transaction_type: str) -> str:
    normalized = transaction_type.strip().lower()
    if normalized == "debit":
        return "expense"
    if normalized == "credit":
        return "income"
    return normalized


def _ranked_totals(totals: dict[str, float], currency: str) -> list[dict]:
    return [
        {"name": name, "amount": round(amount, 2), "currency": currency}
        for name, amount in sorted(
            totals.items(), key=lambda item: item[1], reverse=True
        )
    ]


def _rounded_bucket(bucket: dict[str, float | int]) -> dict[str, float | int]:
    return {
        key: round(value, 2) if isinstance(value, float) else value
        for key, value in bucket.items()
    }


def _manual_monthly_summary(records: list[dict]) -> list[dict]:
    buckets: dict[str, dict[str, float | int]] = {}
    for record in records:
        month = record["transaction_date"][:7]
        bucket = buckets.setdefault(
            month, {"income": 0.0, "expenses": 0.0, "count": 0}
        )
        if record["transaction_type"] == "income":
            bucket["income"] += record["amount"]
        elif record["transaction_type"] == "expense":
            bucket["expenses"] += record["amount"]
        bucket["count"] += 1
    return [
        {
            "month": month,
            **_rounded_bucket(values),
            "net_savings": round(values["income"] - values["expenses"], 2),
            "currency": None,
        }
        for month, values in sorted(buckets.items())
    ]


_default_context_provider = FinancialContextProvider()


def get_context_provider() -> FinancialContextProvider:
    return _default_context_provider


def set_context_provider(provider: FinancialContextProvider) -> None:
    global _default_context_provider
    _default_context_provider = provider
