"""
Transaction normalization service.
Standardizes dates, descriptions, numeric amounts, transaction types, and currencies
into the BillFold canonical transaction schema.
"""
from typing import Any, Optional, Tuple, List
import datetime
import hashlib
import html
import re
import math
import dateutil.parser


# Currency symbols and their corresponding standard ISO codes
CURRENCY_SYMBOL_MAP = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "Rs": "LKR",
    "Rs.": "LKR",
    "LKR": "LKR",
    "USD": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
    "INR": "INR",
    "AUD": "AUD",
    "CAD": "CAD",
    "SGD": "SGD",
}

# Transaction type indicator keywords
EXPENSE_KEYWORDS = {
    "debit", "dr", "withdrawal", "withdraw", "expense", "payment",
    "charge", "fee", "transfer out", "pos", "bill", "payout"
}
INCOME_KEYWORDS = {
    "credit", "cr", "deposit", "income", "refund", "salary",
    "transfer in", "interest", "dividend", "payin", "credit card payment"
}
TRANSFER_KEYWORDS = {
    "transfer", "xfer", "internal transfer", "wire"
}


def infer_date_convention(
    raw_dates: list,
    currency: Optional[str] = None,
    explicit_dayfirst: Optional[bool] = None
) -> Tuple[bool, List[str]]:
    """
    Analyzes raw date representations across a statement to infer whether dates are
    DD/MM/YYYY (dayfirst=True) or MM/DD/YYYY (dayfirst=False).

    If an explicit dayfirst is supplied, it strictly takes priority over automatic inference.
    If contradictory dates appear (both DD > 12 and MM > 12), conflicting rows are identified
    and flagged with an explicit warning, and will be rejected during row parsing.
    """
    warnings: List[str] = []
    dmy_samples: List[Tuple[int, str]] = []  # p1 > 12
    mdy_samples: List[Tuple[int, str]] = []  # p2 > 12
    ambiguous_samples: List[Tuple[int, str]] = []  # both <= 12

    for idx, item in enumerate(raw_dates):
        if item is None:
            continue
        s = str(item).strip().strip("'\"`")
        m = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})$", s)
        if m:
            v1, v2 = int(m.group(1)), int(m.group(2))
            if v1 > 12 and v2 <= 12:
                dmy_samples.append((idx + 1, s))
            elif v2 > 12 and v1 <= 12:
                mdy_samples.append((idx + 1, s))
            elif v1 <= 12 and v2 <= 12:
                ambiguous_samples.append((idx + 1, s))

    # 1. Explicit dayfirst takes absolute priority over automatic inference
    if explicit_dayfirst is not None:
        if explicit_dayfirst is False and dmy_samples:
            row_num, sample_str = dmy_samples[0]
            warnings.append(
                f"Explicit dayfirst=False specified, but contradictory date '{sample_str}' "
                f"(row {row_num}) was found. Contradictory row(s) will be rejected."
            )
        elif explicit_dayfirst is True and mdy_samples:
            row_num, sample_str = mdy_samples[0]
            warnings.append(
                f"Explicit dayfirst=True specified, but contradictory date '{sample_str}' "
                f"(row {row_num}) was found. Contradictory row(s) will be rejected."
            )
        return explicit_dayfirst, warnings

    # 2. Conflicting conventions present (both DD > 12 and MM > 12)
    if dmy_samples and mdy_samples:
        if len(dmy_samples) >= len(mdy_samples):
            chosen = True
            conflicts = mdy_samples
            chosen_name = "DD/MM/YYYY"
        else:
            chosen = False
            conflicts = dmy_samples
            chosen_name = "MM/DD/YYYY"
        conflict_list = ", ".join(f"'{s}' (row {r})" for r, s in conflicts[:3])
        warnings.append(
            f"Mixed date conventions detected: statement follows {chosen_name}, "
            f"and conflicting date(s) {conflict_list} will be rejected."
        )
        return chosen, warnings

    # 3. Unambiguous single convention
    if dmy_samples:
        return True, warnings
    if mdy_samples:
        return False, warnings

    # 4. Ambiguous samples only
    if ambiguous_samples:
        if currency in {"GBP", "EUR", "LKR", "INR", "AUD", "NZD", "SGD"}:
            chosen = True
            format_name = "DD/MM/YYYY"
        else:
            chosen = False
            format_name = "MM/DD/YYYY"
        row_num, sample_str = ambiguous_samples[0]
        warnings.append(
            f"Ambiguous date detected: '{sample_str}' (row {row_num}). "
            f"Parsed using {format_name} convention based on statement context."
        )
        return chosen, warnings

    return False, warnings


def normalize_date(raw_date: Any, dayfirst: bool = False) -> Optional[str]:
    """
    Strictly normalizes date representations into canonical ISO format (YYYY-MM-DD).
    Enforces calendar validity (rejecting invalid dates like 31/02/2026) and strictly
    validates 3-part numeric dates against the specified dayfirst convention, rejecting
    contradictory rows without lenient fallback.
    """
    if raw_date is None:
        return None

    # Handle float NaN
    if isinstance(raw_date, float) and math.isnan(raw_date):
        return None

    # Handle datetime and date objects directly
    if isinstance(raw_date, (datetime.datetime, datetime.date)):
        return raw_date.strftime("%Y-%m-%d")

    date_str = str(raw_date).strip().strip("'\"`")
    if not date_str or date_str.lower() in {"nan", "nat", "none", "null"}:
        return None

    # 1. Standard ISO format: YYYY-MM-DD or YYYY/MM/DD
    m_iso = re.match(r"^(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})$", date_str)
    if m_iso:
        yr, mo, da = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
        try:
            return datetime.date(yr, mo, da).strftime("%Y-%m-%d")
        except ValueError:
            return None

    # 2. Compact 8-digit string: YYYYMMDD
    if re.fullmatch(r"\d{8}", date_str):
        try:
            return datetime.datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            return None

    # 3. Strict 3-part numeric date: d1/d2/d3 or d1-d2-d3 or d1.d2.d3
    m_num = re.match(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})$", date_str)
    if m_num:
        p1, p2, yr = int(m_num.group(1)), int(m_num.group(2)), int(m_num.group(3))
        if yr < 100:
            yr += 2000 if yr < 70 else 1900
        if dayfirst:
            day, month = p1, p2
        else:
            month, day = p1, p2

        # Strict month bounds check
        if not (1 <= month <= 12):
            return None

        # Calendar check (rejects invalid days such as Feb 31 or Feb 29 in non-leap year)
        try:
            return datetime.date(yr, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    # 4. Text month formats (e.g. '15 Jan 2024', 'January 15, 2024')
    try:
        parsed = dateutil.parser.parse(date_str, dayfirst=dayfirst)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return None


def normalize_description(raw_desc: Any) -> str:
    """
    Cleans and standardizes transaction description strings.
    Strips excess whitespace, unescapes HTML, and replaces newlines/tabs.
    """
    if raw_desc is None:
        return "Unknown Transaction"

    if isinstance(raw_desc, float) and math.isnan(raw_desc):
        return "Unknown Transaction"

    text = str(raw_desc)
    # Unescape HTML entities
    text = html.unescape(text)
    # Replace non-breaking spaces and clean whitespace
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    cleaned = text.strip()

    if not cleaned or cleaned.lower() in {"nan", "none", "null"}:
        return "Unknown Transaction"

    return cleaned


SUMMARY_OR_BALANCE_PATTERNS = [
    re.compile(r"\b(?:opening|beginning|initial|start(?:ing)?)\s+bal(?:ance)?\b", re.IGNORECASE),
    re.compile(r"\b(?:closing|ending|final)\s+bal(?:ance)?\b", re.IGNORECASE),
    re.compile(r"\bbal(?:ance)?\s+(?:brought\s+forward|b/f(?:wd)?)\b", re.IGNORECASE),
    re.compile(r"^(?:brought\s+forward|b/fwd|b/f)\b", re.IGNORECASE),
    re.compile(r"\bbal(?:ance)?\s+(?:carried\s+forward|c/f(?:wd)?)\b", re.IGNORECASE),
    re.compile(r"^(?:carried\s+forward|c/fwd|c/f)\b", re.IGNORECASE),
    re.compile(r"\b(?:total|subtotal|sub-total|sub\s+total)\s+(?:debits?|credits?|withdrawals?|deposits?|payments?|charges?|fees?|transactions?)\b", re.IGNORECASE),
    re.compile(r"^(?:total|subtotal|sub-total|sub\s+total|grand\s+total)$", re.IGNORECASE),
]


def is_summary_or_balance_row(description: Optional[str]) -> bool:
    """
    Checks if a row description corresponds to an opening/closing balance or summary total row
    rather than an actual financial transaction.
    """
    if not description:
        return False
    cleaned = description.strip()
    return any(p.search(cleaned) for p in SUMMARY_OR_BALANCE_PATTERNS)


def extract_currency(text: str) -> Optional[str]:
    """
    Attempts to identify a currency symbol or ISO 3-letter currency code from a string.
    """
    if not text:
        return None

    # Check non-letter symbols first (exact character in text)
    symbol_pairs = [("$", "USD"), ("€", "EUR"), ("£", "GBP"), ("¥", "JPY"), ("₹", "INR")]
    for symbol, code in symbol_pairs:
        if symbol in text:
            return code

    # Check 3-letter currency regex with word boundary
    match = re.search(r"\b([A-Z]{3})\b", text, re.IGNORECASE)
    if match:
        curr = match.group(1).upper()
        if curr in {"USD", "EUR", "GBP", "LKR", "INR", "AUD", "CAD", "SGD", "JPY", "CHF", "NZD"}:
            return curr

    # Check word symbols e.g. Rs, Rs., Rupees
    if re.search(r"\b(Rs\.?|Rupees?)\b", text, re.IGNORECASE):
        return "LKR"

    return None


def parse_numeric_amount(raw_val: Any) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Parses a raw amount representation into:
    (signed_float_value, detected_currency, detected_direction)
    detected_direction is 'expense' (debit) or 'income' (credit) if indicated by syntax (e.g., '(100)' or '100 DR').
    """
    if raw_val is None:
        return None, None, None

    if isinstance(raw_val, (int, float)):
        if isinstance(raw_val, float) and math.isnan(raw_val):
            return None, None, None
        return float(raw_val), None, None

    text = str(raw_val).strip()
    if not text or text.lower() in {"nan", "none", "null", "-", "--", ""}:
        return None, None, None

    # Detect currency
    currency = extract_currency(text)

    # Detect direction suffix or indicator (DR / CR / DB)
    direction: Optional[str] = None
    if re.search(r"\b(DR|DEBIT|DB)\b", text, re.IGNORECASE):
        direction = "expense"
    elif re.search(r"\b(CR|CREDIT)\b", text, re.IGNORECASE):
        direction = "income"

    # Strip currency symbols and letters
    clean_str = re.sub(r"[^\d.,()\-+]", "", text).strip()
    if not clean_str:
        return None, currency, direction

    # Check for parentheses indicating negative/debit e.g. (1,234.56)
    is_negative = False
    if clean_str.startswith("(") and clean_str.endswith(")"):
        is_negative = True
        clean_str = clean_str[1:-1].strip()
    elif clean_str.startswith("-") or clean_str.endswith("-"):
        is_negative = True
        clean_str = clean_str.replace("-", "").strip()
    elif clean_str.startswith("+") or clean_str.endswith("+"):
        clean_str = clean_str.replace("+", "").strip()

    if direction == "expense":
        is_negative = True
    elif direction == "income":
        is_negative = False

    # Handle European vs US number formatting
    # European: 1.234,56 -> period thousands, comma decimal
    # US: 1,234.56 -> comma thousands, period decimal
    if "," in clean_str and "." in clean_str:
        if clean_str.rfind(",") > clean_str.rfind("."):
            # European format
            clean_str = clean_str.replace(".", "").replace(",", ".")
        else:
            # US format
            clean_str = clean_str.replace(",", "")
    elif "," in clean_str:
        # Could be comma as decimal e.g. "1234,56" or thousands e.g. "1,000"
        parts = clean_str.split(",")
        if len(parts) == 2 and len(parts[1]) in {1, 2}:
            clean_str = clean_str.replace(",", ".")
        else:
            clean_str = clean_str.replace(",", "")
    elif "." in clean_str:
        # Standard decimal or thousands
        parts = clean_str.split(".")
        if len(parts) > 2:
            clean_str = "".join(parts[:-1]) + "." + parts[-1]

    try:
        val = float(clean_str)
        if is_negative:
            val = -abs(val)
        return val, currency, direction
    except ValueError:
        return None, currency, direction


def determine_amount_and_type(
    debit_val: Any = None,
    credit_val: Any = None,
    amount_val: Any = None,
    type_hint: Optional[str] = None
) -> Tuple[Optional[float], str, Optional[str]]:
    """
    Reconciles debit, credit, amount, and type columns to produce:
    (positive_amount, transaction_type, detected_currency)
    Where transaction_type is strictly one of: 'income', 'expense', 'transfer', 'unknown'.
    """
    detected_currency = None

    parsed_debit, curr_deb, dir_deb = parse_numeric_amount(debit_val)
    parsed_credit, curr_cred, dir_cred = parse_numeric_amount(credit_val)
    parsed_amount, curr_amt, dir_amt = parse_numeric_amount(amount_val)

    detected_currency = curr_deb or curr_cred or curr_amt

    # Check explicit debit / credit columns
    has_debit = parsed_debit is not None and abs(parsed_debit) > 1e-6
    has_credit = parsed_credit is not None and abs(parsed_credit) > 1e-6

    if has_debit and not has_credit:
        return round(abs(parsed_debit), 2), "expense", detected_currency
    if has_credit and not has_debit:
        return round(abs(parsed_credit), 2), "income", detected_currency
    if has_debit and has_credit:
        # Both populated: use net or credit/debit preference
        if abs(parsed_debit) > abs(parsed_credit):
            return round(abs(parsed_debit), 2), "expense", detected_currency
        else:
            return round(abs(parsed_credit), 2), "income", detected_currency

    # Check type_hint from a dedicated column
    hint_lower = str(type_hint or "").strip().lower()
    inferred_type = "unknown"

    if hint_lower in EXPENSE_KEYWORDS or dir_amt == "expense":
        inferred_type = "expense"
    elif hint_lower in INCOME_KEYWORDS or dir_amt == "income":
        inferred_type = "income"
    elif hint_lower in TRANSFER_KEYWORDS:
        inferred_type = "transfer"

    # Fall back to single amount column
    if parsed_amount is not None:
        amt_abs = round(abs(parsed_amount), 2)
        if inferred_type != "unknown":
            return amt_abs, inferred_type, detected_currency
        if parsed_amount < 0:
            return amt_abs, "expense", detected_currency
        elif parsed_amount > 0:
            # Positive amounts without explicit type default to income if signed convention used
            return amt_abs, "income", detected_currency
        else:
            return 0.0, "unknown", detected_currency

    return None, "unknown", detected_currency


def generate_transaction_id(
    source_dataset: Optional[str],
    index: int,
    date_str: str,
    description: str,
    amount: float
) -> str:
    """
    Generates a deterministic unique identifier for the transaction record.
    """
    raw_key = f"{source_dataset or 'statement'}_{index}_{date_str}_{description}_{amount:.2f}"
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    return f"txn_{digest}"
