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


# ISO currency codes recognized in statement text and amount values.
CURRENCY_CODES = {
    "AED", "AFN", "ALL", "AMD", "ANG", "AOA", "ARS", "AUD", "AWG", "AZN",
    "BAM", "BBD", "BDT", "BGN", "BHD", "BIF", "BMD", "BND", "BOB", "BRL",
    "BSD", "BTN", "BWP", "BYN", "BZD", "CAD", "CDF", "CHF", "CLP", "CNY",
    "COP", "CRC", "CUP", "CVE", "CZK", "DJF", "DKK", "DOP", "DZD", "EGP",
    "ERN", "ETB", "EUR", "FJD", "FKP", "GBP", "GEL", "GHS", "GIP", "GMD",
    "GNF", "GTQ", "GYD", "HKD", "HNL", "HTG", "HUF", "IDR", "ILS", "INR",
    "IQD", "IRR", "ISK", "JMD", "JOD", "JPY", "KES", "KGS", "KHR", "KMF",
    "KPW", "KRW", "KWD", "KYD", "KZT", "LAK", "LBP", "LKR", "LRD", "LSL",
    "LYD", "MAD", "MDL", "MGA", "MKD", "MMK", "MNT", "MOP", "MRU", "MUR",
    "MVR", "MWK", "MXN", "MYR", "MZN", "NAD", "NGN", "NIO", "NOK", "NPR",
    "NZD", "OMR", "PAB", "PEN", "PGK", "PHP", "PKR", "PLN", "PYG", "QAR",
    "RON", "RSD", "RUB", "RWF", "SAR", "SBD", "SCR", "SDG", "SEK", "SGD",
    "SHP", "SLE", "SOS", "SRD", "SSP", "STN", "SVC", "SYP", "SZL", "THB",
    "TJS", "TMT", "TND", "TOP", "TRY", "TTD", "TWD", "TZS", "UAH", "UGX",
    "USD", "UYU", "UZS", "VES", "VND", "VUV", "WST", "XAF", "XCD", "XOF",
    "XPF", "YER", "ZAR", "ZMW", "ZWL",
}

# Only unambiguous symbols map directly to an ISO code. Dollar, yen, and
# rupee symbols are shared by multiple currencies and require an explicit code.
CURRENCY_SYMBOL_MAP = {
    "€": "EUR",
    "£": "GBP",
    "₹": "INR",
    "US$": "USD",
    "A$": "AUD",
    "C$": "CAD",
    "NZ$": "NZD",
    "S$": "SGD",
    "HK$": "HKD",
    "د.إ": "AED",
}

CURRENCY_CODE_PATTERN = "|".join(sorted(CURRENCY_CODES, key=len, reverse=True))

# Transaction type indicator keywords
EXPENSE_KEYWORDS = {
    "debit", "dr", "withdrawal", "withdraw", "expense", "payment",
    "charge", "fee", "transfer out", "pos", "bill", "payout"
}
INCOME_KEYWORDS = {
    "credit", "cr", "deposit", "income", "refund", "salary",
    "transfer in", "interest", "dividend", "payin", "credit card payment",
    "invoice", "client payment", "paycheck", "pay cheque",
}
TRANSFER_KEYWORDS = {
    "transfer", "xfer", "internal transfer", "wire"
}


def infer_date_convention(
    raw_dates: list,
    currency: Optional[str] = None,
    explicit_dayfirst: Optional[bool] = None
) -> Tuple[Optional[bool], List[str]]:
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
        row_num, sample_str = ambiguous_samples[0]
        warnings.append(
            f"Ambiguous date detected: '{sample_str}' (row {row_num}). "
            "The date convention could not be determined; ambiguous date rows "
            "will be skipped."
        )
        return None, warnings

    return None, warnings


def normalize_date(
    raw_date: Any, dayfirst: Optional[bool] = None
) -> Optional[str]:
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
        if dayfirst is None:
            if p1 > 12 and p2 <= 12:
                dayfirst = True
            elif p2 > 12 and p1 <= 12:
                dayfirst = False
            else:
                return None
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

    numeric_date_time = re.match(
        r"^((?:\d{4}[/.-]\d{1,2}[/.-]\d{1,2}|"
        r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}))([ T].+)$",
        date_str,
    )
    if numeric_date_time:
        normalized_date = normalize_date(
            numeric_date_time.group(1), dayfirst=dayfirst
        )
        if not normalized_date:
            return None
        try:
            dateutil.parser.parse(date_str, dayfirst=dayfirst)
            return normalized_date
        except (OverflowError, TypeError, ValueError):
            return None

    # 4. Text month formats require an explicit year and day; dateutil otherwise
    # fills missing components from the current date.
    numeric_tokens = re.findall(r"(?<!\d)\d{1,4}(?!\d)", date_str)
    four_digit_years = [token for token in numeric_tokens if len(token) == 4]
    year_tokens = four_digit_years or (
        [numeric_tokens[-1]]
        if len(numeric_tokens) > 1 and len(numeric_tokens[-1]) == 2
        else []
    )
    has_year = bool(year_tokens)
    has_month_name = bool(
        re.search(
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
            r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
            r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
            date_str,
            re.IGNORECASE,
        )
    )
    has_day = any(
        token not in year_tokens and 1 <= int(token) <= 31
        for token in numeric_tokens
    )
    if not (has_year and has_month_name and has_day):
        return None

    try:
        parsed = dateutil.parser.parse(date_str, dayfirst=dayfirst)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return None


LEADING_DESCRIPTION_DATE_PATTERN = re.compile(
    r"^\s*((?:\d{4}[/.-]\d{1,2}[/.-]\d{1,2}|"
    r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|"
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|"
    r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}))(?=$|[\s,;|])",
    re.IGNORECASE,
)


def correct_dates_from_leading_descriptions(
    transactions: List[Any], dayfirst: Optional[bool] = None
) -> None:
    """Correct an extracted date only when a reliable leading description date disagrees."""
    def get_value(transaction: Any, field: str) -> Any:
        if isinstance(transaction, dict):
            return transaction.get(field)
        return getattr(transaction, field, None)

    leading_dates = []
    for transaction in transactions:
        match = LEADING_DESCRIPTION_DATE_PATTERN.match(
            get_value(transaction, "description") or ""
        )
        if match:
            leading_dates.append(match.group(1))

    description_dayfirst, _ = infer_date_convention(
        leading_dates, explicit_dayfirst=dayfirst
    )
    if description_dayfirst is None:
        description_dayfirst = dayfirst

    for transaction in transactions:
        match = LEADING_DESCRIPTION_DATE_PATTERN.match(
            get_value(transaction, "description") or ""
        )
        if not match:
            continue
        description_date = normalize_date(
            match.group(1), dayfirst=description_dayfirst
        )
        if (
            description_date
            and description_date
            != normalize_date(get_value(transaction, "transaction_date"))
        ):
            if isinstance(transaction, dict):
                transaction["transaction_date"] = description_date
            else:
                transaction.transaction_date = description_date


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

    match = re.search(
        rf"(?<![A-Za-z])({CURRENCY_CODE_PATTERN})(?![A-Za-z])",
        text,
    )
    if match:
        return match.group(1).upper()

    aliases = (
        (r"\b(?:united arab emirates|uae)\s+dirham\b", "AED"),
        (r"\bsri lankan rupee\b", "LKR"),
        (r"\bindian rupee\b", "INR"),
        (r"\bunited states dollar\b|\bus dollar\b", "USD"),
        (r"\baustralian dollar\b", "AUD"),
        (r"\bcanadian dollar\b", "CAD"),
        (r"\bnew zealand dollar\b", "NZD"),
        (r"\bsingapore dollar\b", "SGD"),
        (r"\beuro\b", "EUR"),
        (r"\bpound sterling\b|\bbritish pound\b", "GBP"),
        (r"\bjapanese yen\b", "JPY"),
        (r"\bchinese yuan\b|\brenminbi\b", "CNY"),
    )
    for pattern, code in aliases:
        if re.search(pattern, text, re.IGNORECASE):
            return code

    for symbol, code in CURRENCY_SYMBOL_MAP.items():
        if symbol in text:
            return code

    return None


def extract_statement_currency(text: str) -> Optional[str]:
    """Detect currency from statement context, excluding dated transaction rows."""
    for line in (text or "").splitlines():
        if re.match(
            r"^\s*(?:\d{4}[/.-]\d{1,2}[/.-]\d{1,2}|"
            r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|"
            r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|"
            r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})\b",
            line,
        ):
            continue
        currency = extract_currency(line)
        if currency:
            return currency
    return None


def parse_numeric_amount(raw_val: Any) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Parses a raw amount representation into:
    (signed_float_value, detected_currency, detected_direction)
    detected_direction is 'expense' (debit) or 'income' (credit) when indicated
    by a sign, accounting parentheses, or DR/CR suffix.
    """
    if raw_val is None:
        return None, None, None

    if isinstance(raw_val, bool):
        return None, None, None

    if isinstance(raw_val, (int, float)):
        if isinstance(raw_val, float) and math.isnan(raw_val):
            return None, None, None
        value = float(raw_val)
        return (value, None, None) if math.isfinite(value) else (None, None, None)

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

    # Remove only recognized currency and direction markers. Unknown text is
    # invalid financial data, not something to strip from an amount cell.
    numeric_text = re.sub(
        rf"(?<![A-Za-z])(?:{CURRENCY_CODE_PATTERN}|Rs\.?|Rupees?)(?![A-Za-z])",
        "",
        text,
        flags=re.IGNORECASE,
    )
    numeric_text = re.sub(
        r"\b(?:DR|DEBIT|DB|CR|CREDIT)\b",
        "",
        numeric_text,
        flags=re.IGNORECASE,
    )
    numeric_text = re.sub(
        r"(?:US\$|NZ\$|HK\$|A\$|C\$|S\$|د\.إ|[$€£¥₹])",
        "",
        numeric_text,
    ).strip()
    if re.search(r"[A-Za-z]", numeric_text) or re.search(r"\d\s+\d", numeric_text):
        return None, currency, direction

    clean_str = re.sub(r"\s+", "", numeric_text)
    if not clean_str:
        return None, currency, direction

    # Check for parentheses indicating negative/debit e.g. (1,234.56)
    is_negative = False
    if clean_str.startswith("(") and clean_str.endswith(")"):
        is_negative = True
        direction = direction or "expense"
        clean_str = clean_str[1:-1].strip()
        if re.search(r"[()+\-]", clean_str):
            return None, currency, direction
    elif "-" in clean_str:
        if clean_str.count("-") != 1 or not (
            clean_str.startswith("-") or clean_str.endswith("-")
        ) or "+" in clean_str:
            return None, currency, direction
        is_negative = True
        direction = direction or "expense"
        clean_str = clean_str.replace("-", "").strip()
    elif "+" in clean_str:
        if clean_str.count("+") != 1 or not (
            clean_str.startswith("+") or clean_str.endswith("+")
        ):
            return None, currency, direction
        direction = direction or "income"
        clean_str = clean_str.replace("+", "").strip()

    if "'" in clean_str:
        if not re.fullmatch(r"\d{1,3}(?:'\d{3})+(?:[.,]\d+)?", clean_str):
            return None, currency, direction
        clean_str = clean_str.replace("'", "")

    if not re.fullmatch(r"\d[\d.,]*", clean_str):
        return None, currency, direction

    if direction == "expense":
        is_negative = True
    elif direction == "income":
        is_negative = False

    # Validate thousands groups before normalizing the decimal separator.
    if "," in clean_str and "." in clean_str:
        if clean_str.rfind(",") > clean_str.rfind("."):
            integer_part, decimal_part = clean_str.rsplit(",", 1)
            valid_groups = bool(
                re.fullmatch(r"\d{1,3}(?:\.\d{3})+", integer_part)
            )
            if not decimal_part.isdigit() or not valid_groups:
                return None, currency, direction
            clean_str = integer_part.replace(".", "") + "." + decimal_part
        else:
            integer_part, decimal_part = clean_str.rsplit(".", 1)
            valid_groups = bool(
                re.fullmatch(r"\d{1,3}(?:,\d{3})+", integer_part)
            )
            if not decimal_part.isdigit() or not valid_groups:
                return None, currency, direction
            clean_str = integer_part.replace(",", "") + "." + decimal_part
    elif "," in clean_str:
        if clean_str.count(",") == 1:
            integer_part, decimal_part = clean_str.split(",", 1)
            if decimal_part.isdigit() and len(decimal_part) in {1, 2}:
                clean_str = integer_part + "." + decimal_part
            elif (
                decimal_part.isdigit()
                and len(decimal_part) == 3
                and 1 <= len(integer_part) <= 3
            ):
                clean_str = integer_part + decimal_part
            else:
                return None, currency, direction
        else:
            if not re.fullmatch(r"\d{1,3}(?:,\d{3})+", clean_str):
                return None, currency, direction
            clean_str = clean_str.replace(",", "")
    elif "." in clean_str:
        if clean_str.count(".") > 1:
            if not re.fullmatch(r"\d{1,3}(?:\.\d{3})+", clean_str):
                return None, currency, direction
            clean_str = clean_str.replace(".", "")
        else:
            integer_part, decimal_part = clean_str.split(".", 1)
            if not integer_part.isdigit() or not decimal_part.isdigit():
                return None, currency, direction
    elif not clean_str.isdigit():
        return None, currency, direction

    try:
        val = float(clean_str)
        if not math.isfinite(val):
            return None, currency, direction
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

    # Separate debit and credit columns should contain at most one populated value.
    # Choosing one when both are populated silently loses financial information.
    if has_debit and has_credit:
        return None, "unknown", detected_currency
    if has_debit and not has_credit:
        amount = round(abs(parsed_debit), 2)
        return (amount, "expense", detected_currency) if amount > 0 else (None, "unknown", detected_currency)
    if has_credit and not has_debit:
        amount = round(abs(parsed_credit), 2)
        return (amount, "income", detected_currency) if amount > 0 else (None, "unknown", detected_currency)

    # Check type_hint from a dedicated column
    hint_lower = str(type_hint or "").strip().lower()
    inferred_type = "unknown"

    if any(
        re.search(r"\b" + re.escape(keyword) + r"\b", hint_lower)
        for keyword in TRANSFER_KEYWORDS
    ):
        inferred_type = "transfer"
    elif any(
        re.search(r"\b" + re.escape(keyword) + r"\b", hint_lower)
        for keyword in INCOME_KEYWORDS
    ):
        inferred_type = "income"
    elif any(
        re.search(r"\b" + re.escape(keyword) + r"\b", hint_lower)
        for keyword in EXPENSE_KEYWORDS
    ):
        inferred_type = "expense"

    # Explicit amount syntax is stronger evidence than words in a description.
    # For example, "Consulting fee 1200.00 CR" is income despite "fee".
    if parsed_amount is not None:
        amt_abs = round(abs(parsed_amount), 2)
        if amt_abs <= 0:
            return None, "unknown", detected_currency
        if dir_amt is not None:
            if (
                parsed_amount < 0
                and inferred_type == "transfer"
                and "internal transfer" in hint_lower
            ):
                return amt_abs, "transfer", detected_currency
            return amt_abs, dir_amt, detected_currency
        if parsed_amount < 0:
            if inferred_type == "transfer" and "internal transfer" in hint_lower:
                return amt_abs, "transfer", detected_currency
            return amt_abs, "expense", detected_currency
        if inferred_type != "unknown":
            return amt_abs, inferred_type, detected_currency
        if parsed_amount > 0:
            # Positive amounts without explicit type default to income if signed convention used
            return amt_abs, "income", detected_currency

    return None, "unknown", detected_currency


def generate_transaction_id(
    source_dataset: Optional[str],
    index: int,
    date_str: str,
    description: str,
    amount: float,
    transaction_type: str = "unknown",
    currency: Optional[str] = None,
) -> str:
    """
    Generates a deterministic transaction fingerprint scoped to the source file.
    The row index preserves identical transactions as separate records.
    """
    normalized_description = " ".join(description.casefold().split())
    raw_key = "\x1f".join(
        (
            (source_dataset or "statement").strip().casefold(),
            str(index),
            date_str,
            normalized_description,
            f"{amount:.2f}",
            transaction_type.strip().lower(),
            (currency or "UNKNOWN").strip().upper(),
        )
    )
    digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
    return f"txn_{digest}"
