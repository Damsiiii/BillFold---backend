from typing import Optional

CATEGORY_KEYWORDS = {
    "Salary": ["salary", "payroll", "wages"],
    "Food": ["keells", "cargills", "food city", "restaurant", "cafe", "kfc", "pizza", "uber eats", "pickme food"],
    "Transport": ["uber", "pickme", "fuel", "petrol", "ceypetco", "lanka ioc", "bus", "train"],
    "Bills": ["dialog", "mobitel", "slt", "ceb", "leco", "water board", "electricity", "internet"],
    "Entertainment": ["netflix", "spotify", "cinema", "youtube"],
    "Health": ["pharmacy", "hospital", "clinic", "medical"],
    "Shopping": ["odel", "fashion", "amazon", "daraz"],
    "Transfer": ["transfer", "atm", "cash withdrawal"],
}


def categorize(description: str, tx_type: Optional[str] = None) -> str:
    text = (description or "").lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(word in text for word in keywords):
            if category == "Salary" and tx_type == "debit":
                continue
            return category
    return "Income" if tx_type == "credit" else "Uncategorized"