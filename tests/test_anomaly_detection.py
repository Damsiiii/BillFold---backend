"""
Unit tests for the Anomaly Detection Engine (app/services/anomaly_detection.py).
Verifies detection of planted anomalies (huge transaction, duplicate charge),
statistical rules (category vs global thresholds), and strict wording compliance.
"""

import pandas as pd
import pytest
from app.services.anomaly_detection import detect_anomalies
from data.generate_mock_data import generate_mock_transactions


@pytest.fixture
def mock_df() -> pd.DataFrame:
    """Fixture providing 6 months of generated mock transactions."""
    return generate_mock_transactions(seed=42)


# ==============================================================================
# Planted Anomalies Verification Tests
# ==============================================================================

def test_detect_planted_huge_transaction(mock_df: pd.DataFrame):
    """Verify that the planted very large transaction (485,000 LKR flight booking) is flagged."""
    result = detect_anomalies(mock_df)
    anomalies = result["anomalies"]

    # Locate the huge transaction
    huge_txns = [a for a in anomalies if a["amount"] >= 400000.0]
    assert len(huge_txns) == 1, "Planted very large transaction was not flagged"

    huge_txn = huge_txns[0]
    assert "SriLankan Airlines" in huge_txn["description"]
    assert huge_txn["category"] == "Transport"
    assert "statistical_outlier" in huge_txn["rules_triggered"]
    assert "new_merchant_large_spend" in huge_txn["rules_triggered"]

    # Verify reason formatting and ratio
    reason = huge_txn["reason"]
    assert "Unusual transaction" in reason
    assert "larger than your usual Transport spend" in reason
    assert "SriLankan Airlines" in reason

    # Verify metrics
    metrics = huge_txn["metrics"]
    assert metrics["z_score"] > 3.0
    assert metrics["threshold_used"] == "category_z_score"
    assert metrics["category_sample_count"] >= 5


def test_detect_planted_duplicate_charge(mock_df: pd.DataFrame):
    """Verify that planted duplicate charges (Ministry of Crab within 1 day) are flagged."""
    result = detect_anomalies(mock_df)
    anomalies = result["anomalies"]

    # Find the duplicate charges
    dup_txns = [
        a for a in anomalies if "Ministry of Crab" in a["description"]
    ]
    assert len(dup_txns) == 2, f"Expected 2 duplicate charges flagged, found {len(dup_txns)}"

    for txn in dup_txns:
        assert txn["amount"] == 14500.0
        assert txn["category"] == "Food"
        assert "duplicate_charge" in txn["rules_triggered"]
        assert "Unusual transaction: Possible duplicate charge" in txn["reason"]
        assert "duplicate_matches" in txn["metrics"]
        assert len(txn["metrics"]["duplicate_matches"]) >= 1


def test_wording_never_contains_fraud(mock_df: pd.DataFrame):
    """Strict Rule: Wording must always be 'unusual transaction', NEVER 'fraud'."""
    result = detect_anomalies(mock_df)

    for anomaly in result["anomalies"]:
        full_text = f"{anomaly['reason']} {anomaly['rules_triggered']}".lower()
        assert "fraud" not in full_text, f"Found forbidden word 'fraud' in anomaly: {anomaly}"

    # Also test an explicitly synthesized edge case
    edge_df = pd.DataFrame(
        [
            {
                "transaction_id": "T-01",
                "transaction_date": "2026-05-01",
                "description": "Suspicious Merchant",
                "amount": 999999.0,
                "currency": "LKR",
                "transaction_type": "expense",
                "category": "Shopping",
                "source_dataset": "test",
            }
        ]
    )
    edge_res = detect_anomalies(edge_df)
    for a in edge_res["anomalies"]:
        assert "fraud" not in a["reason"].lower()
        assert "unusual transaction" in a["reason"].lower()


# ==============================================================================
# Statistical Rules: Category (>= 5 txns) vs Global (< 5 txns) Thresholds
# ==============================================================================

def test_category_with_at_least_5_txns_uses_category_stats():
    """Categories with >= 5 transactions must use category-specific z-score."""
    # Build category with 6 normal txns (~1,000 LKR) and 1 extreme outlier (50,000 LKR)
    normal_txns = [
        {
            "transaction_id": f"T-{i}",
            "transaction_date": f"2026-06-{i+1:02d}",
            "description": f"Normal Spend {i}",
            "amount": 1000.0 + (i * 10),
            "currency": "LKR",
            "transaction_type": "expense",
            "category": "Gym",
            "source_dataset": "test",
        }
        for i in range(6)
    ]
    outlier_txn = {
        "transaction_id": "T-OUTLIER",
        "transaction_date": "2026-06-15",
        "description": "Normal Spend 1",  # Repeat description to not trigger new merchant
        "amount": 50000.0,
        "currency": "LKR",
        "transaction_type": "expense",
        "category": "Gym",
        "source_dataset": "test",
    }
    df = pd.DataFrame(normal_txns + [outlier_txn])

    res = detect_anomalies(df)
    gym_anomalies = [a for a in res["anomalies"] if a["transaction_id"] == "T-OUTLIER"]
    assert len(gym_anomalies) == 1
    assert gym_anomalies[0]["metrics"]["threshold_used"] in ["category_z_score", "category_iqr_fence"]
    assert gym_anomalies[0]["metrics"]["category_sample_count"] == 7
    assert "larger than your usual Gym spend" in gym_anomalies[0]["reason"]


def test_category_with_fewer_than_5_txns_uses_global_threshold():
    """Categories with < 5 transactions must fall back to global threshold."""
    # Global transactions in Food (10 transactions around 2,000 LKR, std ~100)
    food_txns = [
        {
            "transaction_id": f"FOOD-{i}",
            "transaction_date": f"2026-06-{i+1:02d}",
            "description": "Grocery",
            "amount": 2000.0 + (i * 20),
            "currency": "LKR",
            "transaction_type": "expense",
            "category": "Food",
            "source_dataset": "test",
        }
        for i in range(10)
    ]
    # Small category 'Art' with only 2 transactions
    art_txns = [
        {
            "transaction_id": "ART-1",
            "transaction_date": "2026-06-05",
            "description": "Grocery",  # Seen merchant
            "amount": 1500.0,
            "currency": "LKR",
            "transaction_type": "expense",
            "category": "Art",
            "source_dataset": "test",
        },
        {
            "transaction_id": "ART-2",
            "transaction_date": "2026-06-12",
            "description": "Grocery",  # Seen merchant
            "amount": 35000.0,  # Global outlier
            "currency": "LKR",
            "transaction_type": "expense",
            "category": "Art",
            "source_dataset": "test",
        },
    ]
    df = pd.DataFrame(food_txns + art_txns)

    res = detect_anomalies(df)
    art_anomalies = [a for a in res["anomalies"] if a["transaction_id"] == "ART-2"]
    assert len(art_anomalies) == 1
    assert art_anomalies[0]["metrics"]["threshold_used"] == "global_z_score"
    assert art_anomalies[0]["metrics"]["category_sample_count"] == 2
    assert "overall typical spend" in art_anomalies[0]["reason"]


# ==============================================================================
# Duplicate Window Boundary Tests
# ==============================================================================

def test_duplicate_window_boundaries():
    """Identical charges <= 2 days apart are flagged; >= 3 days apart are NOT flagged."""
    # Within 2 days (Day 1 and Day 3 = 2 days apart)
    df_dup = pd.DataFrame(
        [
            {
                "transaction_id": "D1",
                "transaction_date": "2026-05-01",
                "description": "Cafe Mocha",
                "amount": 1200.0,
                "currency": "LKR",
                "transaction_type": "expense",
                "category": "Food",
                "source_dataset": "test",
            },
            {
                "transaction_id": "D2",
                "transaction_date": "2026-05-03",
                "description": "Cafe Mocha",
                "amount": 1200.0,
                "currency": "LKR",
                "transaction_type": "expense",
                "category": "Food",
                "source_dataset": "test",
            },
            # 4 days later (Day 7) - outside the 2-day duplicate window from D2
            {
                "transaction_id": "D3",
                "transaction_date": "2026-05-07",
                "description": "Cafe Mocha",
                "amount": 1200.0,
                "currency": "LKR",
                "transaction_type": "expense",
                "category": "Food",
                "source_dataset": "test",
            },
        ]
    )

    res = detect_anomalies(df_dup)
    dup_ids = {
        a["transaction_id"] for a in res["anomalies"] if "duplicate_charge" in a["rules_triggered"]
    }
    assert "D1" in dup_ids
    assert "D2" in dup_ids
    assert "D3" not in dup_ids, "Transaction 4 days apart should not be flagged as duplicate"


# ==============================================================================
# Edge Cases & Filtering Tests
# ==============================================================================

def test_anomaly_detection_empty_df():
    """Empty DataFrame returns zero anomalies gracefully."""
    empty_res = detect_anomalies(pd.DataFrame())
    assert empty_res["total_anomalies"] == 0
    assert empty_res["anomalies"] == []
    assert empty_res["summary_by_type"] == {
        "statistical_outlier": 0,
        "duplicate_charge": 0,
        "new_merchant_large_spend": 0,
    }


def test_anomaly_detection_excludes_transfers():
    """Transfers and income must not be flagged by expense anomaly detection."""
    df = pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "transaction_date": "2026-05-01",
                "description": "Transfer to Pot",
                "amount": 10000000.0,  # Huge transfer
                "currency": "LKR",
                "transaction_type": "transfer",
                "category": "Transfer",
                "source_dataset": "test",
            },
            {
                "transaction_id": "T2",
                "transaction_date": "2026-05-02",
                "description": "Transfer to Pot",
                "amount": 10000000.0,  # Duplicate transfer
                "currency": "LKR",
                "transaction_type": "transfer",
                "category": "Transfer",
                "source_dataset": "test",
            },
        ]
    )
    res = detect_anomalies(df)
    assert res["total_anomalies"] == 0


def test_anomaly_detection_filtering_by_month(mock_df: pd.DataFrame):
    """Filtering by month returns only anomalies occurring within that month."""
    # August should contain the Ministry of Crab duplicates
    aug_res = detect_anomalies(mock_df, month="2026-08")
    aug_anomalies = aug_res["anomalies"]
    assert len(aug_anomalies) == 2
    assert all("Ministry of Crab" in a["description"] for a in aug_anomalies)

    # September should contain the flight booking outlier
    sep_res = detect_anomalies(mock_df, month="2026-09")
    sep_anomalies = sep_res["anomalies"]
    assert any("SriLankan Airlines" in a["description"] for a in sep_anomalies)
    assert not any("Ministry of Crab" in a["description"] for a in sep_anomalies)
