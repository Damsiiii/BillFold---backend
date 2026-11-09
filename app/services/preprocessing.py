import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d %b %Y", "%d %B %Y", "%d/%m/%y", "%Y/%m/%d",
]  # day-first formats, since Sri Lankan statements use dd/mm/yyyy

DEBIT_WORDS = {"debit", "dr", "withdrawal", "expense"}
CREDIT_WORDS = {"credit", "cr", "deposit", "income"}


def parse_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def parse_amount(value: Any) -> Optional[float]:
    """Returns a signed float. Negative means money going out (minus sign, brackets or 'Dr')."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    negative = (
        text.startswith("(") and text.endswith(")")
    ) or bool(re.search(r"(?i)\bdr\.?$", text))

    text = re.sub(r"(?i)lkr|rs\.?|usd|dr\.?|cr\.?", "", text)
    text = re.sub(r"[^0-9.\-]", "", text)
    if text.startswith("-"):
        negative = True
        text = text.lstrip("-")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def clean_description(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if text else "Unknown"


def normalize_transaction(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Turns one messy raw row into the standard format, or None if it is unusable."""
    tx_date = parse_date(raw.get("date"))
    if tx_date is None:
        return None

    debit = parse_amount(raw.get("debit"))
    credit = parse_amount(raw.get("credit"))
    amount = parse_amount(raw.get("amount"))
    declared = str(raw.get("type") or "").strip().lower()

    if debit:
        value, tx_type = abs(debit), "debit"
    elif credit:
        value, tx_type = abs(credit), "credit"
    elif amount is not None and amount != 0:
        if declared in DEBIT_WORDS:
            tx_type = "debit"
        elif declared in CREDIT_WORDS:
            tx_type = "credit"
        else:
            tx_type = "debit" if amount < 0 else "credit"
        value = abs(amount)
    else:
        return None

    return {
        "date": tx_date,
        "description": clean_description(raw.get("description")),
        "amount": round(value, 2),
        "type": tx_type,
        "balance": parse_amount(raw.get("balance")),
    }


def clean_transactions(
    raw_rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (clean_rows, rejected_rows). Removes duplicates inside the same upload."""
    clean, rejected, seen = [], [], set()
    for raw in raw_rows:
        row = normalize_transaction(raw)
        if row is None:
            rejected.append(raw)
            continue
        key = (row["date"], row["description"].lower(), row["amount"], row["type"], row["balance"])
        if key in seen:
            continue
        seen.add(key)
        clean.append(row)
    return clean, rejected