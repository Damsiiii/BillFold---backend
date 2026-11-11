"""
CSV Bank Statement Parser.
Extracts and normalizes transaction data from CSV bank statements with support for
diverse delimiters, encodings, custom headers, and multi-column debit/credit layouts.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, BinaryIO, TextIO
from pathlib import Path
import csv
import io
import re
import pandas as pd

from models.transaction import Transaction, ExtractionResult
from app.services.categorization import categorize
from services.transaction_normalizer import (
    normalize_date,
    normalize_description,
    determine_amount_and_type,
    parse_numeric_amount,
    generate_transaction_id,
    extract_currency,
    extract_statement_currency,
    infer_date_convention,
    is_summary_or_balance_row,
)


# Column keyword definitions for header mapping
DATE_KEYWORDS = [
    "date", "transaction date", "txn date", "trans date", "posting date",
    "value date", "booking date", "trans. date", "post date"
]

DESCRIPTION_KEYWORDS = [
    "description", "details", "narration", "particulars", "memo",
    "narrative", "payee", "merchant", "notes", "note", "reference",
    "transaction details", "remarks", "name", "desc"
]

DEBIT_KEYWORDS = [
    "debit", "withdrawal", "dr", "dr amount", "debit amount",
    "money out", "paid out", "withdrawals", "expense", "charges"
]

CREDIT_KEYWORDS = [
    "credit", "deposit", "cr", "cr amount", "credit amount",
    "money in", "paid in", "deposits", "income"
]

AMOUNT_KEYWORDS = [
    "amount", "txn amount", "transaction amount", "trans amount",
    "net amount", "sum", "total", "value"
]

BALANCE_KEYWORDS = [
    "balance", "running balance", "closing balance", "available balance",
    "account balance", "bal", "balance amount"
]

TYPE_KEYWORDS = [
    "type", "transaction type", "txn type", "dr/cr", "cr/dr", "indicator", "d/c"
]

CATEGORY_KEYWORDS = [
    "category", "spending category", "expense category", "classification"
]

CURRENCY_KEYWORDS = [
    "currency", "curr", "ccy"
]


class CSVParser:
    """
    Parser for CSV bank statements.
    """

    SUPPORTED_ENCODINGS = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]
    COMMON_DELIMITERS = [",", ";", "\t", "|"]

    def __init__(
        self,
        max_header_scan_rows: int = 30,
        dayfirst: Optional[bool] = None
    ):
        self.max_header_scan_rows = max_header_scan_rows
        self.dayfirst = dayfirst

    def parse(
        self,
        file_source: Union[str, Path, bytes, BinaryIO, TextIO],
        filename: str = "statement.csv",
        dayfirst: Optional[bool] = None
    ) -> ExtractionResult:
        """
        Parses a CSV bank statement from a file path, raw bytes, or stream.
        """
        warnings: List[str] = []

        # Read content and resolve encoding
        raw_text, encoding_used, read_error = self._read_text(file_source)
        if read_error:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="csv",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error=read_error
            )

        if not raw_text.strip():
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="csv",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error="CSV file is empty."
            )

        # Detect delimiter
        delimiter = self._detect_delimiter(raw_text)

        # Detect header row index and column mapping
        header_row_idx, col_mapping = self._detect_header_and_mapping(raw_text, delimiter)

        if col_mapping is None or "date" not in col_mapping or not (
            "amount" in col_mapping or "debit" in col_mapping or "credit" in col_mapping
        ):
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="csv",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error="Could not identify required transaction columns (Date and Amount/Debit/Credit)."
            )

        # Parse CSV into DataFrame using detected header and delimiter
        try:
            df = pd.read_csv(
                io.StringIO(raw_text),
                delimiter=delimiter,
                skiprows=header_row_idx,
                dtype=str,
                on_bad_lines="skip"
            )
        except Exception as e:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="csv",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error=f"Malformed CSV structure: {str(e)}"
            )

        if df.empty:
            return ExtractionResult(
                success=True,
                filename=filename,
                file_type="csv",
                total_transactions=0,
                transactions=[],
                warnings=["CSV file contains a header but no data rows."],
                error=None
            )

        # Check global statement currency from headers or metadata
        statement_header_text = "\n".join(
            raw_text.splitlines()[: header_row_idx + 1]
        )
        global_currency = self._detect_global_currency(
            statement_header_text, col_mapping, df
        )

        date_col = col_mapping["date"]

        # Infer date convention across statement (resolves ambiguous dates like 03/04/2026)
        date_samples = df[date_col].dropna().tolist()
        resolved_dayfirst, date_warnings = infer_date_convention(
            raw_dates=date_samples,
            currency=global_currency,
            explicit_dayfirst=dayfirst if dayfirst is not None else self.dayfirst
        )
        warnings.extend(date_warnings)

        # Extract transactions row by row
        transactions: List[Transaction] = []
        column_positions = {
            column: position for position, column in enumerate(df.columns)
        }

        def position_for(field: str) -> Optional[int]:
            column = col_mapping.get(field)
            return column_positions.get(column) if column is not None else None

        date_position = position_for("date")
        if date_position is None:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="csv",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error="Could not identify the transaction date column.",
            )
        desc_position = position_for("description")
        debit_position = position_for("debit")
        credit_position = position_for("credit")
        amount_position = position_for("amount")
        balance_position = position_for("balance")
        type_position = position_for("type")
        category_position = position_for("category")
        currency_position = position_for("currency")

        skipped_rows = 0

        for row_idx, row in enumerate(df.itertuples(index=False, name=None)):
            raw_date = row[date_position]
            normalized_dt = normalize_date(raw_date, dayfirst=resolved_dayfirst)

            if not normalized_dt:
                # Likely summary row, subtotal, disclaimer or empty row
                skipped_rows += 1
                continue

            raw_desc = row[desc_position] if desc_position is not None else None
            normalized_desc = normalize_description(raw_desc)

            if is_summary_or_balance_row(normalized_desc):
                skipped_rows += 1
                continue

            raw_debit = row[debit_position] if debit_position is not None else None
            raw_credit = row[credit_position] if credit_position is not None else None
            raw_amt = row[amount_position] if amount_position is not None else None
            raw_type = row[type_position] if type_position is not None else None

            amount, txn_type, row_curr = determine_amount_and_type(
                debit_val=raw_debit,
                credit_val=raw_credit,
                amount_val=raw_amt,
                type_hint=raw_type or normalized_desc,
            )

            if amount is None:
                skipped_rows += 1
                continue

            # Resolve currency
            explicit_curr = None
            if currency_position is not None and row[currency_position]:
                explicit_curr = extract_currency(str(row[currency_position]))
            resolved_currency = explicit_curr or row_curr or global_currency

            # Resolve balance
            resolved_balance = None
            if balance_position is not None and row[balance_position]:
                bal_val, _, _ = parse_numeric_amount(row[balance_position])
                if bal_val is not None:
                    resolved_balance = round(bal_val, 2)

            # Resolve category
            category = "Unknown"
            if category_position is not None and row[category_position]:
                cat_text = str(row[category_position]).strip()
                if cat_text and cat_text.lower() not in {"nan", "none", "null"}:
                    category = cat_text
            if category.casefold() in {"unknown", "uncategorized"}:
                category = categorize(normalized_desc, txn_type)

            # Create canonical transaction
            txn_id = generate_transaction_id(
                filename,
                int(row_idx),
                normalized_dt,
                normalized_desc,
                amount,
                txn_type,
                resolved_currency,
            )

            txn = Transaction(
                transaction_id=txn_id,
                transaction_date=normalized_dt,
                description=normalized_desc,
                amount=amount,
                currency=resolved_currency,
                transaction_type=txn_type,
                category=category,
                source_dataset=filename,
                balance=resolved_balance
            )
            transactions.append(txn)

        if skipped_rows > 0:
            warnings.append(f"Skipped {skipped_rows} non-transaction or malformed row(s).")

        if not transactions:
            warnings.append("No valid transaction records could be extracted from this statement.")

        return ExtractionResult(
            success=True,
            filename=filename,
            file_type="csv",
            total_transactions=len(transactions),
            transactions=transactions,
            warnings=warnings,
            error=None
        )

    def _read_text(
        self,
        file_source: Union[str, Path, bytes, BinaryIO, TextIO]
    ) -> Tuple[str, str, Optional[str]]:
        """
        Reads input source into string, trying candidate encodings.
        """
        raw_bytes: Optional[bytes] = None

        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if not path.exists():
                return "", "", f"File not found: {path.name}"
            try:
                raw_bytes = path.read_bytes()
            except Exception as e:
                return "", "", f"Error reading file: {str(e)}"
        elif isinstance(file_source, bytes):
            raw_bytes = file_source
        elif hasattr(file_source, "read"):
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            content = file_source.read()
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            if isinstance(content, str):
                return content.lstrip("\ufeff"), "utf-8", None
            raw_bytes = content
        else:
            return "", "", "Unsupported file source type."

        if not raw_bytes:
            return "", "", "CSV file is empty."

        for enc in self.SUPPORTED_ENCODINGS:
            try:
                text = raw_bytes.decode(enc)
                return text.lstrip("\ufeff"), enc, None
            except (UnicodeDecodeError, LookupError):
                continue

        return "", "", "Failed to decode CSV file with supported encodings (UTF-8, Latin-1, CP1252)."

    def _detect_delimiter(self, text: str) -> str:
        """
        Detects delimiter using csv.Sniffer or character frequency heuristic.
        """
        sample = "\n".join(text.splitlines()[:20])
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample, delimiters=";,|\t")
            return dialect.delimiter
        except Exception:
            pass

        # Heuristic count of delimiters across non-empty lines
        lines = [line for line in sample.splitlines() if line.strip()][:10]
        if not lines:
            return ","

        scores: Dict[str, int] = {d: 0 for d in self.COMMON_DELIMITERS}
        for d in self.COMMON_DELIMITERS:
            counts = [line.count(d) for line in lines]
            if counts and min(counts) > 0:
                scores[d] = sum(counts)

        best_delimiter = max(scores, key=scores.get)
        return best_delimiter if scores[best_delimiter] > 0 else ","

    def _detect_header_and_mapping(
        self,
        text: str,
        delimiter: str
    ) -> Tuple[int, Optional[Dict[str, str]]]:
        """
        Scans up to max_header_scan_rows lines to find the line that best represents the column headers.
        Returns (header_row_index, column_mapping_dict).
        """
        lines = text.splitlines()[:self.max_header_scan_rows]
        best_row_idx = 0
        best_score = -1
        best_mapping: Optional[Dict[str, str]] = None

        for row_idx, line in enumerate(lines):
            if not line.strip():
                continue

            try:
                reader = csv.reader([line], delimiter=delimiter)
                headers = next(reader)
            except Exception:
                continue

            mapping, score = self._score_headers(headers)
            if score > best_score:
                best_score = score
                best_row_idx = row_idx
                best_mapping = mapping

        # Ensure minimal viable score
        if best_score < 2 or best_mapping is None or "date" not in best_mapping:
            return 0, None

        return best_row_idx, best_mapping

    def _clean_header_name(self, header: str) -> str:
        """
        Standardizes column headers for keyword matching.
        Treats underscores and punctuation as word separators.
        """
        cleaned = re.sub(r"[^\w\s/]", " ", str(header)).strip().lower()
        return re.sub(r"\s+", " ", cleaned)

    @staticmethod
    def _matches_keyword(cleaned: str, keywords: List[str]) -> bool:
        """
        Matches a cleaned header name against keywords using exact match or whole word boundary.
        Prevents short tokens like 'cr' or 'dr' from false-matching substrings in words like 'description'.
        """
        for kw in keywords:
            if cleaned == kw:
                return True
            if re.search(r"\b" + re.escape(kw) + r"\b", cleaned):
                return True
        return False

    def _score_headers(self, headers: List[str]) -> Tuple[Dict[str, str], int]:
        """
        Evaluates a list of header strings and maps them to semantic fields.
        Returns (mapping, confidence_score).
        """
        mapping: Dict[str, str] = {}
        score = 0
        cleaned_headers = [self._clean_header_name(h) for h in headers]

        for orig_header, cleaned in zip(headers, cleaned_headers):
            if not cleaned:
                continue

            # Match date
            if "date" not in mapping and self._matches_keyword(cleaned, DATE_KEYWORDS):
                mapping["date"] = orig_header
                score += 3
                continue

            # Match description
            if "description" not in mapping and self._matches_keyword(cleaned, DESCRIPTION_KEYWORDS):
                mapping["description"] = orig_header
                score += 2
                continue

            # Match debit
            if "debit" not in mapping and self._matches_keyword(cleaned, DEBIT_KEYWORDS):
                mapping["debit"] = orig_header
                score += 2
                continue

            # Match credit
            if "credit" not in mapping and self._matches_keyword(cleaned, CREDIT_KEYWORDS):
                mapping["credit"] = orig_header
                score += 2
                continue

            # Match amount
            if "amount" not in mapping and self._matches_keyword(cleaned, AMOUNT_KEYWORDS):
                mapping["amount"] = orig_header
                score += 2
                continue

            # Match balance
            if "balance" not in mapping and self._matches_keyword(cleaned, BALANCE_KEYWORDS):
                mapping["balance"] = orig_header
                score += 1
                continue

            # Match type
            if "type" not in mapping and self._matches_keyword(cleaned, TYPE_KEYWORDS):
                mapping["type"] = orig_header
                score += 1
                continue

            # Match category
            if "category" not in mapping and self._matches_keyword(cleaned, CATEGORY_KEYWORDS):
                mapping["category"] = orig_header
                score += 1
                continue

            # Match currency
            if "currency" not in mapping and self._matches_keyword(cleaned, CURRENCY_KEYWORDS):
                mapping["currency"] = orig_header
                score += 1
                continue

        return mapping, score

    def _detect_global_currency(
        self,
        header_text: str,
        col_mapping: Dict[str, str],
        df: pd.DataFrame
    ) -> Optional[str]:
        """
        Checks metadata rows, header titles, and top cells for statement-wide currency.
        """
        return extract_statement_currency(header_text)


def parse_csv(
    file_source: Union[str, Path, bytes, BinaryIO, TextIO],
    filename: str = "statement.csv",
    dayfirst: Optional[bool] = None
) -> ExtractionResult:
    """
    Convenience function to parse a CSV statement.
    """
    parser = CSVParser(dayfirst=dayfirst)
    return parser.parse(file_source, filename, dayfirst=dayfirst)
