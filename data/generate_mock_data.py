"""
Mock Data Generator for Personal Finance Recommendation System.

Generates ~6 months of realistic synthetic financial transactions (May 2026 - October 2026)
following the transaction contract:
- transaction_id: Unique string identifier
- transaction_date: ISO format YYYY-MM-DD
- description: Transaction description / merchant name
- amount: Always positive float
- currency: ISO currency code (default LKR)
- transaction_type: income | expense | transfer | unknown
- category: Expense/Income category
- source_dataset: Identifier for data provenance

Planted problems included:
1. Overspent category: "Shopping" in October 2026 (~185,000 LKR vs normal ~25,000-35,000 LKR)
2. Very large transaction: Apple Store MacBook Pro purchase (485,000 LKR on 2026-09-18)
3. Duplicate charge: "Ministry of Crab Dinner" (14,500 LKR on 2026-08-14 and 2026-08-15)
4. Monthly recurring subscription: "Netflix Monthly Subscription" (2,200 LKR on the 15th of each month)
Also includes:
- Monthly transfers (to test transfer exclusion from income/expense calculations)
- One uncategorized transaction (missing category)
- One unknown transaction type
"""

import os
from datetime import date, timedelta
import random
import pandas as pd


def generate_mock_transactions(seed: int = 42) -> pd.DataFrame:
    """Generate ~6 months of realistic synthetic transactions with planted anomalies.

    Args:
        seed: Random seed for deterministic generation.

    Returns:
        pd.DataFrame adhering to the transaction contract.
    """
    random.seed(seed)
    transactions = []
    txn_counter = 1

    def make_txn(
        txn_date: str,
        desc: str,
        amount: float,
        txn_type: str,
        category: str | None,
        currency: str = "LKR",
        source: str = "synthetic_mock_v1",
    ) -> dict:
        nonlocal txn_counter
        tid = f"TXN-{txn_counter:04d}"
        txn_counter += 1
        return {
            "transaction_id": tid,
            "transaction_date": txn_date,
            "description": desc,
            "amount": round(float(amount), 2),
            "currency": currency,
            "transaction_type": txn_type,
            "category": category,
            "source_dataset": source,
        }

    # Month definitions: May 2026 to October 2026 (6 months)
    months = [
        ("2026-05", date(2026, 5, 1), 31),
        ("2026-06", date(2026, 6, 1), 30),
        ("2026-07", date(2026, 7, 1), 31),
        ("2026-08", date(2026, 8, 1), 31),
        ("2026-09", date(2026, 9, 1), 30),
        ("2026-10", date(2026, 10, 1), 31),
    ]

    for month_str, start_dt, days_in_month in months:
        # 1. Salary (Income) on the 25th of each month
        salary_date = f"{month_str}-25"
        transactions.append(
            make_txn(
                txn_date=salary_date,
                desc="Monthly Salary - Tech Lead",
                amount=275000.00,
                txn_type="income",
                category="Salary",
            )
        )

        # 2. Rent (Expense) on the 1st of each month
        rent_date = f"{month_str}-01"
        transactions.append(
            make_txn(
                txn_date=rent_date,
                desc="Apartment Rental Payment",
                amount=65000.00,
                txn_type="expense",
                category="Rent",
            )
        )

        # 3. Monthly Transfer to Savings Pot (Transfer - must be excluded from income/expense)
        transfer_date = f"{month_str}-28"
        transactions.append(
            make_txn(
                txn_date=transfer_date,
                desc="Transfer to High-Yield Savings Pot",
                amount=35000.00,
                txn_type="transfer",
                category="Transfer",
            )
        )

        # 4. Planted Problem 4: Monthly Subscription (Netflix) on the 15th
        sub_date = f"{month_str}-15"
        transactions.append(
            make_txn(
                txn_date=sub_date,
                desc="Netflix Monthly Subscription",
                amount=2200.00,
                txn_type="expense",
                category="Entertainment",
            )
        )

        # 5. Utilities (Electricity, Water, Home Broadband)
        elec_day = random.randint(5, 8)
        transactions.append(
            make_txn(
                txn_date=f"{month_str}-{elec_day:02d}",
                desc="CEB Electricity Bill",
                amount=random.uniform(9500, 13800),
                txn_type="expense",
                category="Utilities",
            )
        )

        water_day = random.randint(9, 12)
        transactions.append(
            make_txn(
                txn_date=f"{month_str}-{water_day:02d}",
                desc="NWSDB Water Supply Bill",
                amount=random.uniform(1800, 2900),
                txn_type="expense",
                category="Utilities",
            )
        )

        broadband_day = random.randint(11, 14)
        transactions.append(
            make_txn(
                txn_date=f"{month_str}-{broadband_day:02d}",
                desc="Dialog Home Broadband Bill",
                amount=random.uniform(4900, 5800),
                txn_type="expense",
                category="Utilities",
            )
        )

        # 6. Food & Groceries (12 to 18 transactions per month)
        grocery_merchants = [
            "Keells Super - Havelock",
            "Cargills Food City - Rajagiriya",
            "Arpico Supercentre",
            "PickMe Food Delivery",
            "Uber Eats Restaurant Order",
            "Java Lounge Coffee",
            "BreadTalk Pastry & Bakery",
            "Green Cabin Cafe",
        ]
        num_food_txns = random.randint(14, 18)
        for _ in range(num_food_txns):
            f_day = random.randint(1, days_in_month)
            merchant = random.choice(grocery_merchants)
            if "Super" in merchant or "Supercentre" in merchant:
                amt = random.uniform(4500, 16500)
            elif "Delivery" in merchant or "Order" in merchant:
                amt = random.uniform(1600, 4800)
            else:
                amt = random.uniform(800, 2800)
            transactions.append(
                make_txn(
                    txn_date=f"{month_str}-{f_day:02d}",
                    desc=merchant,
                    amount=amt,
                    txn_type="expense",
                    category="Food",
                )
            )

        # 7. Transport (8 to 12 transactions per month)
        transport_merchants = [
            "PickMe Ride - Colombo",
            "Uber Taxi Ride",
            "Lanka IOC Fuel Station",
            "Ceypetco Fuel Station",
        ]
        num_trans_txns = random.randint(8, 12)
        for _ in range(num_trans_txns):
            t_day = random.randint(1, days_in_month)
            merchant = random.choice(transport_merchants)
            amt = random.uniform(3500, 7500) if "Fuel" in merchant else random.uniform(650, 2200)
            transactions.append(
                make_txn(
                    txn_date=f"{month_str}-{t_day:02d}",
                    desc=merchant,
                    amount=amt,
                    txn_type="expense",
                    category="Transport",
                )
            )

        # 8. Entertainment (2 to 4 transactions per month)
        ent_merchants = [
            "Scope Cinemas - Colombo City Centre",
            "PVR Cinemas One Galle Face",
            "Strikes Bowling Alley",
            "Colombo Comedy Club",
        ]
        for _ in range(random.randint(2, 4)):
            e_day = random.randint(1, days_in_month)
            merchant = random.choice(ent_merchants)
            transactions.append(
                make_txn(
                    txn_date=f"{month_str}-{e_day:02d}",
                    desc=merchant,
                    amount=random.uniform(2200, 6800),
                    txn_type="expense",
                    category="Entertainment",
                )
            )

        # 9. Shopping
        # For months May - September, typical shopping is ~20,000 - 32,000 LKR
        # Planted Problem 1: Overspent category in October 2026 (~185,000 LKR)
        if month_str == "2026-10":
            # Planted overspending: massive shopping splurge
            october_shopping_sprees = [
                ("ODEL Department Store - Wardrobe Refresh", 68500.00, 4),
                ("Glitz Luxury Perfumes & Watches", 45000.00, 11),
                ("Singer Mega - Home Appliance Upgrade", 52000.00, 19),
                ("Daraz Online Shopping - Electronics & Gear", 24500.00, 24),
            ]
            for desc, amt, day in october_shopping_sprees:
                transactions.append(
                    make_txn(
                        txn_date=f"{month_str}-{day:02d}",
                        desc=desc,
                        amount=amt,
                        txn_type="expense",
                        category="Shopping",
                    )
                )
        else:
            # Normal shopping
            shop_merchants = [
                "ODEL Flagship Store",
                "Daraz Online Marketplace",
                "Cotton Collection",
                "Miniso Lifestyle",
            ]
            for _ in range(random.randint(2, 4)):
                s_day = random.randint(1, days_in_month)
                merchant = random.choice(shop_merchants)
                transactions.append(
                    make_txn(
                        txn_date=f"{month_str}-{s_day:02d}",
                        desc=merchant,
                        amount=random.uniform(3200, 12500),
                        txn_type="expense",
                        category="Shopping",
                    )
                )

    # Planted Problem 2: Very large transaction (Outlier in Transport at a new merchant on 2026-09-18)
    transactions.append(
        make_txn(
            txn_date="2026-09-18",
            desc="SriLankan Airlines First Class Flight Booking",
            amount=485000.00,
            txn_type="expense",
            category="Transport",
        )
    )

    # Planted Problem 3: Duplicate charge (same merchant & amount within 2 days)
    # Dinner charged on 2026-08-14 and duplicate on 2026-08-15
    transactions.append(
        make_txn(
            txn_date="2026-08-14",
            desc="Ministry of Crab Seafood Dinner",
            amount=14500.00,
            txn_type="expense",
            category="Food",
        )
    )
    transactions.append(
        make_txn(
            txn_date="2026-08-15",
            desc="Ministry of Crab Seafood Dinner",
            amount=14500.00,
            txn_type="expense",
            category="Food",
        )
    )

    # Planted Edge Case 1: Missing category (None) to test fallback to "Uncategorized"
    transactions.append(
        make_txn(
            txn_date="2026-07-20",
            desc="Boutique Gift Shop Cash Sale",
            amount=3850.00,
            txn_type="expense",
            category=None,
        )
    )

    # Planted Edge Case 2: Unknown transaction type
    transactions.append(
        make_txn(
            txn_date="2026-08-22",
            desc="Pending Bank Fee Hold",
            amount=500.00,
            txn_type="unknown",
            category="Other",
        )
    )

    df = pd.DataFrame(transactions)
    # Sort chronologically by date
    df = df.sort_values(by=["transaction_date", "transaction_id"]).reset_index(drop=True)
    return df


def save_mock_data(output_path: str = "data/mock_transactions.csv") -> pd.DataFrame:
    """Generate and save mock transactions to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_mock_transactions()
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = save_mock_data()
    print("=" * 60)
    print(f"Successfully generated {len(df)} transactions.")
    print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
    print("\nTransaction Type Breakdown:")
    print(df["transaction_type"].value_counts())
    print("\nCategory Breakdown (Expenses):")
    print(df[df["transaction_type"] == "expense"]["category"].value_counts(dropna=False))
    print("\nPlanted Problems Summary:")
    print("1. Overspent Category: Shopping in 2026-10")
    print("2. Very Large Transaction: SriLankan Airlines First Class Flight (485,000 LKR in Transport) on 2026-09-18")
    print("3. Duplicate Charge: 'Ministry of Crab Seafood Dinner' (14,500 LKR) on 2026-08-14 and 2026-08-15")
    print("4. Monthly Subscription: 'Netflix Monthly Subscription' (2,200 LKR) on 15th of each month")
    print("=" * 60)
