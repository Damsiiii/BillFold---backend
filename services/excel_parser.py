"""
Excel Bank Statement Parser.
Extracts and normalizes transaction data from Excel bank statements (.xlsx and supported .xls)
with support for multi-sheet workbooks, metadata headers, and diverse statement layouts.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, BinaryIO
from pathlib import Path
import io
import logging
import re
import openpyxl
import openpyxl.utils.datetime as ox_dt

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
logger = logging.getLogger(__name__)


class _LegacyXlsWorksheet:
    def __init__(self, sheet: Any, datemode: int, xlrd_module: Any):
        self._sheet = sheet
        self._datemode = datemode
        self._xlrd = xlrd_module

    def iter_rows(self, max_row: Optional[int] = None, values_only: bool = True):
        row_count = self._sheet.nrows
        if max_row is not None:
            row_count = min(row_count, max_row)
        for row_idx in range(row_count):
            values = []
            for cell in self._sheet.row(row_idx):
                if cell.ctype in {
                    self._xlrd.XL_CELL_EMPTY,
                    self._xlrd.XL_CELL_BLANK,
                    self._xlrd.XL_CELL_ERROR,
                }:
                    values.append(None)
                elif cell.ctype == self._xlrd.XL_CELL_DATE:
                    try:
                        values.append(
                            self._xlrd.xldate.xldate_as_datetime(
                                cell.value, self._datemode
                            )
                        )
                    except (OverflowError, TypeError, ValueError):
                        values.append(None)
                else:
                    values.append(cell.value)
            yield tuple(values)


class _LegacyXlsWorkbook:
    def __init__(self, workbook: Any, xlrd_module: Any):
        self._workbook = workbook
        self._xlrd = xlrd_module
        self.sheetnames = workbook.sheet_names()

    def __getitem__(self, name: str) -> _LegacyXlsWorksheet:
        return _LegacyXlsWorksheet(
            self._workbook.sheet_by_name(name),
            self._workbook.datemode,
            self._xlrd,
        )


class ExcelParser:
    """
    Parser for Excel bank statements (.xlsx and .xls).
    """

    def __init__(
        self,
        max_header_scan_rows: int = 30,
        dayfirst: Optional[bool] = None,
        sheet_name: Optional[Union[str, int]] = None
    ):
        self.max_header_scan_rows = max_header_scan_rows
        self.dayfirst = dayfirst
        self.sheet_name = sheet_name

    def parse(
        self,
        file_source: Union[str, Path, bytes, BinaryIO],
        filename: str = "statement.xlsx",
        sheet_name: Optional[Union[str, int]] = None,
        dayfirst: Optional[bool] = None
    ) -> ExtractionResult:
        """
        Parses an Excel bank statement from a file path, raw bytes, or stream.
        """
        warnings: List[str] = []

        # Read binary data
        raw_bytes, read_error = self._read_bytes(file_source)
        if read_error:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="excel",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error=read_error
            )

        if not raw_bytes:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="excel",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error="Excel file is empty."
            )

        # Load the workbook with a format-specific reader.
        is_legacy = filename.lower().endswith(".xls") and not filename.lower().endswith(".xlsx")
        load_error: Optional[str] = None
        if is_legacy:
            try:
                import xlrd

                wb = _LegacyXlsWorkbook(
                    xlrd.open_workbook(file_contents=raw_bytes, on_demand=True),
                    xlrd,
                )
            except ImportError:
                load_error = "Legacy Excel support is unavailable; install the xlrd dependency."
            except Exception:
                load_error = "Invalid or corrupted legacy Excel (.xls) file."
                logger.exception("Could not load legacy Excel statement %s", filename)
        else:
            try:
                wb = openpyxl.load_workbook(
                    io.BytesIO(raw_bytes), data_only=True, read_only=True
                )
            except Exception as e:
                load_error = f"Invalid or corrupted Excel file: {str(e)}"

        if load_error:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="excel",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error=load_error,
            )

        if not wb.sheetnames:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="excel",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error="Excel workbook contains no worksheets."
            )

        # Resolve target worksheet
        target_sheet_name = sheet_name if sheet_name is not None else self.sheet_name
        chosen_sheet, best_mapping, header_row_idx, sheet_err = self._select_worksheet(
            wb, target_sheet_name
        )

        if sheet_err:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="excel",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error=sheet_err
            )

        # Read all rows from the chosen worksheet
        ws = wb[chosen_sheet]
        all_rows: List[List[Any]] = []
        try:
            for row in ws.iter_rows(values_only=True):
                all_rows.append(list(row))
        except Exception as e:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="excel",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error=f"Error reading rows from worksheet '{chosen_sheet}': {str(e)}"
            )

        if not all_rows:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="excel",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error=f"Worksheet '{chosen_sheet}' is empty."
            )

        if header_row_idx is None or best_mapping is None or "date" not in best_mapping or not (
            "amount" in best_mapping or "debit" in best_mapping or "credit" in best_mapping
        ):
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="excel",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error="Could not identify required transaction columns (Date and Amount/Debit/Credit)."
            )

        # Data rows start immediately after the header row
        data_rows = all_rows[header_row_idx + 1:]
        if not data_rows:
            return ExtractionResult(
                success=True,
                filename=filename,
                file_type="excel",
                total_transactions=0,
                transactions=[],
                warnings=[f"Worksheet '{chosen_sheet}' contains headers but no transaction rows."],
                error=None
            )

        # Detect statement-wide currency from metadata rows and header names
        metadata_text = "\n".join(
            " ".join(str(cell) for cell in row if cell is not None)
            for row in all_rows[: header_row_idx + 1]
        )
        global_currency = self._detect_global_currency(metadata_text, best_mapping)

        # Resolve date convention
        date_col_idx = best_mapping["date"]
        raw_dates_sample = [r[date_col_idx] for r in data_rows if len(r) > date_col_idx and r[date_col_idx] is not None]

        resolved_dayfirst, date_warnings = infer_date_convention(
            raw_dates=raw_dates_sample,
            currency=global_currency,
            explicit_dayfirst=dayfirst if dayfirst is not None else self.dayfirst
        )
        warnings.extend(date_warnings)

        # Extract transactions row by row
        transactions: List[Transaction] = []
        desc_col_idx = best_mapping.get("description")
        debit_col_idx = best_mapping.get("debit")
        credit_col_idx = best_mapping.get("credit")
        amount_col_idx = best_mapping.get("amount")
        balance_col_idx = best_mapping.get("balance")
        type_col_idx = best_mapping.get("type")
        cat_col_idx = best_mapping.get("category")
        curr_col_idx = best_mapping.get("currency")

        skipped_rows = 0

        for row_offset, row in enumerate(data_rows):
            # Safe access to cell by index
            def get_cell(idx: Optional[int]) -> Any:
                if idx is not None and 0 <= idx < len(row):
                    return row[idx]
                return None

            raw_date = get_cell(date_col_idx)
            # Handle Excel date serial numbers
            if isinstance(raw_date, (int, float)) and not isinstance(raw_date, bool):
                if 20000 <= raw_date <= 60000:
                    try:
                        raw_date = ox_dt.from_excel(raw_date)
                    except Exception:
                        pass

            normalized_dt = normalize_date(raw_date, dayfirst=resolved_dayfirst)
            if not normalized_dt:
                skipped_rows += 1
                continue

            raw_desc = get_cell(desc_col_idx)
            normalized_desc = normalize_description(raw_desc)

            if is_summary_or_balance_row(normalized_desc):
                skipped_rows += 1
                continue

            raw_debit = get_cell(debit_col_idx)
            raw_credit = get_cell(credit_col_idx)
            raw_amt = get_cell(amount_col_idx)
            raw_type = get_cell(type_col_idx)

            amount, txn_type, row_curr = determine_amount_and_type(
                debit_val=raw_debit,
                credit_val=raw_credit,
                amount_val=raw_amt,
                type_hint=(
                    str(raw_type) if raw_type is not None else normalized_desc
                ),
            )

            if amount is None:
                skipped_rows += 1
                continue

            # Resolve currency
            explicit_curr = None
            raw_curr_val = get_cell(curr_col_idx)
            if raw_curr_val:
                explicit_curr = extract_currency(str(raw_curr_val))
            resolved_currency = explicit_curr or row_curr or global_currency

            # Resolve balance
            resolved_balance = None
            raw_bal = get_cell(balance_col_idx)
            if raw_bal is not None:
                bal_val, _, _ = parse_numeric_amount(raw_bal)
                if bal_val is not None:
                    resolved_balance = round(bal_val, 2)

            # Resolve category
            category = "Unknown"
            raw_cat = get_cell(cat_col_idx)
            if raw_cat is not None:
                cat_text = str(raw_cat).strip()
                if cat_text and cat_text.lower() not in {"nan", "none", "null"}:
                    category = cat_text
            if category.casefold() in {"unknown", "uncategorized"}:
                category = categorize(normalized_desc, txn_type)

            txn_id = generate_transaction_id(
                filename,
                row_offset,
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
            warnings.append(f"No valid transaction records could be extracted from worksheet '{chosen_sheet}'.")

        return ExtractionResult(
            success=True,
            filename=filename,
            file_type="excel",
            total_transactions=len(transactions),
            transactions=transactions,
            warnings=warnings,
            error=None
        )

    def _read_bytes(
        self,
        file_source: Union[str, Path, bytes, BinaryIO]
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Extracts raw bytes from file path, bytes, or binary stream.
        """
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if not path.exists():
                return None, f"File not found: {path.name}"
            try:
                return path.read_bytes(), None
            except Exception as e:
                return None, f"Error reading file: {str(e)}"
        elif isinstance(file_source, bytes):
            return file_source, None
        elif hasattr(file_source, "read"):
            try:
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                data = file_source.read()
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                if isinstance(data, str):
                    return data.encode("utf-8"), None
                return data, None
            except Exception as e:
                return None, f"Error reading stream: {str(e)}"
        return None, "Unsupported file source type."

    def _clean_header_name(self, header: Any) -> str:
        """
        Standardizes a column header name for fuzzy keyword matching.
        """
        cleaned = re.sub(r"[^\w\s/]", " ", str(header or "")).strip().lower()
        return re.sub(r"\s+", " ", cleaned)

    @staticmethod
    def _matches_keyword(cleaned: str, keywords: List[str]) -> bool:
        """
        Matches a cleaned header name against keywords using exact match or whole word boundary.
        """
        for kw in keywords:
            if cleaned == kw:
                return True
            if re.search(r"\b" + re.escape(kw) + r"\b", cleaned):
                return True
        return False

    def _score_headers(self, headers: List[Any]) -> Tuple[Dict[str, int], int]:
        """
        Evaluates a row of cell values and maps semantic fields to column indices.
        Returns (mapping_field_to_col_idx, confidence_score).
        """
        mapping: Dict[str, int] = {}
        score = 0
        cleaned_headers = [self._clean_header_name(h) for h in headers]

        for col_idx, cleaned in enumerate(cleaned_headers):
            if not cleaned:
                continue

            # Match date
            if "date" not in mapping and self._matches_keyword(cleaned, DATE_KEYWORDS):
                mapping["date"] = col_idx
                score += 3
                continue

            # Match description
            if "description" not in mapping and self._matches_keyword(cleaned, DESCRIPTION_KEYWORDS):
                mapping["description"] = col_idx
                score += 2
                continue

            # Match debit
            if "debit" not in mapping and self._matches_keyword(cleaned, DEBIT_KEYWORDS):
                mapping["debit"] = col_idx
                score += 2
                continue

            # Match credit
            if "credit" not in mapping and self._matches_keyword(cleaned, CREDIT_KEYWORDS):
                mapping["credit"] = col_idx
                score += 2
                continue

            # Match amount
            if "amount" not in mapping and self._matches_keyword(cleaned, AMOUNT_KEYWORDS):
                mapping["amount"] = col_idx
                score += 2
                continue

            # Match balance
            if "balance" not in mapping and self._matches_keyword(cleaned, BALANCE_KEYWORDS):
                mapping["balance"] = col_idx
                score += 1
                continue

            # Match type
            if "type" not in mapping and self._matches_keyword(cleaned, TYPE_KEYWORDS):
                mapping["type"] = col_idx
                score += 1
                continue

            # Match category
            if "category" not in mapping and self._matches_keyword(cleaned, CATEGORY_KEYWORDS):
                mapping["category"] = col_idx
                score += 1
                continue

            # Match currency
            if "currency" not in mapping and self._matches_keyword(cleaned, CURRENCY_KEYWORDS):
                mapping["currency"] = col_idx
                score += 1
                continue

        return mapping, score

    def _scan_sheet_for_header(
        self,
        ws: Any
    ) -> Tuple[Optional[int], Optional[Dict[str, int]], int]:
        """
        Scans top rows of a worksheet to locate the header row.
        Returns (header_row_idx, col_mapping, confidence_score).
        """
        best_row_idx: Optional[int] = None
        best_mapping: Optional[Dict[str, int]] = None
        best_score = -1

        row_idx = 0
        for row in ws.iter_rows(max_row=self.max_header_scan_rows, values_only=True):
            if not any(cell is not None and str(cell).strip() for cell in row):
                row_idx += 1
                continue

            mapping, score = self._score_headers(list(row))
            if score > best_score:
                best_score = score
                best_row_idx = row_idx
                best_mapping = mapping
            row_idx += 1

        if best_score < 2 or best_mapping is None or "date" not in best_mapping:
            return None, None, 0

        return best_row_idx, best_mapping, best_score

    def _select_worksheet(
        self,
        wb: Any,
        requested_sheet: Optional[Union[str, int]]
    ) -> Tuple[Optional[str], Optional[Dict[str, int]], Optional[int], Optional[str]]:
        """
        Selects the worksheet to parse. If requested_sheet is provided, validates and selects it.
        Otherwise, inspects all sheets and selects the one with the highest header score.
        Returns (sheet_name, col_mapping, header_row_idx, error_message).
        """
        if requested_sheet is not None:
            if isinstance(requested_sheet, int):
                if 0 <= requested_sheet < len(wb.sheetnames):
                    chosen = wb.sheetnames[requested_sheet]
                else:
                    return None, None, None, f"Sheet index {requested_sheet} out of range (workbook has {len(wb.sheetnames)} sheets)."
            else:
                if requested_sheet in wb.sheetnames:
                    chosen = requested_sheet
                else:
                    return None, None, None, f"Worksheet '{requested_sheet}' not found in workbook."

            header_idx, mapping, score = self._scan_sheet_for_header(wb[chosen])
            return chosen, mapping, header_idx, None

        # Auto-detect best sheet across workbook
        best_sheet_name: Optional[str] = None
        best_header_idx: Optional[int] = None
        best_mapping: Optional[Dict[str, int]] = None
        highest_score = -1

        for name in wb.sheetnames:
            ws = wb[name]
            header_idx, mapping, score = self._scan_sheet_for_header(ws)
            if score > highest_score:
                highest_score = score
                best_sheet_name = name
                best_header_idx = header_idx
                best_mapping = mapping

        if best_sheet_name is None or highest_score < 2:
            # Fall back to active sheet name
            fallback_sheet = wb.sheetnames[0]
            return fallback_sheet, None, None, "Could not identify required transaction columns (Date and Amount/Debit/Credit) in any worksheet."

        return best_sheet_name, best_mapping, best_header_idx, None

    def _detect_global_currency(
        self,
        metadata_text: str,
        col_mapping: Dict[str, int]
    ) -> Optional[str]:
        """
        Checks metadata rows for statement-wide currency codes.
        """
        return extract_statement_currency(metadata_text)


def parse_excel(
    file_source: Union[str, Path, bytes, BinaryIO],
    filename: str = "statement.xlsx",
    sheet_name: Optional[Union[str, int]] = None,
    dayfirst: Optional[bool] = None
) -> ExtractionResult:
    """
    Convenience function to parse an Excel bank statement.
    """
    parser = ExcelParser(sheet_name=sheet_name, dayfirst=dayfirst)
    return parser.parse(file_source, filename, sheet_name=sheet_name, dayfirst=dayfirst)
