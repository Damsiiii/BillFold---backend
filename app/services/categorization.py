import re
from typing import Optional

CATEGORY_KEYWORDS = {
    "Salary": [
        "salary", "payroll", "wages", "monthly pay", "paycheck", "pay cheque",
        "direct deposit", "employer",
    ],
    "Food": [
        "keells", "cargills", "arpico", "sathosa", "glomark", "food city",
        "supermarket", "grocery", "groceries", "restaurant", "cafe", "coffee",
        "kfc", "pizza", "uber eats", "foodpanda", "pickme food", "bakery",
        "dining", "meal",
    ],
    "Transport": [
        "uber", "pickme", "fuel", "petrol", "diesel", "ceypetco", "lanka ioc",
        "bus", "train", "taxi", "parking", "toll", "airline", "transport",
        "ride fare",
    ],
    "Utilities": [
        "dialog", "mobitel", "slt", "ceb", "leco", "water board",
        "electricity", "internet", "utility", "utilities", "telephone",
        "mobile reload",
    ],
    "Entertainment": [
        "netflix", "spotify", "cinema", "youtube", "gaming", "concert",
        "disney+", "prime video", "subscription",
    ],
    "Healthcare": [
        "pharmacy", "hospital", "clinic", "medical", "healthcare", "doctor",
        "dental", "laboratory", "health insurance",
    ],
    "Shopping": [
        "odel", "fashion", "amazon", "daraz", "shopping", "retail",
        "clothing", "bookstore", "online purchase",
    ],
    "Fees": [
        "bank fee", "service fee", "maintenance fee", "transaction fee",
        "late fee", "atm fee", "commission", "charge", "levy",
    ],
    "Rent": ["rent", "lease payment", "landlord"],
    "Education": ["tuition", "school fee", "university", "education"],
    "Cash": ["atm withdrawal", "cash withdrawal"],
    "Transfer": ["transfer", "xfer", "wire"],
}


def categorize(description: str, tx_type: Optional[str] = None) -> str:
    text = " ".join((description or "").casefold().split())
    transaction_type = (tx_type or "").strip().casefold()

    if transaction_type in {"transfer", "xfer"}:
        return "Transfer"

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(
            re.search(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", text)
            for keyword in keywords
        ):
            if category == "Salary" and transaction_type in {"debit", "expense"}:
                continue
            return category

    if transaction_type in {"credit", "income"}:
        return "Income"
    return "Uncategorized"