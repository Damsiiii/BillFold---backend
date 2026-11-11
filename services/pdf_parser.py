"""
PDF Bank Statement Parser.
Extracts and normalizes transaction data from PDF bank statements using pdfplumber and pypdf.
Supports both table-structured layouts and formatted text layouts, multi-page statements,
scanned PDF detection, and safe handling of unreadable or encrypted files.
"""
from functools import lru_cache
from contextvars import ContextVar
from collections import Counter
import logging
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, BinaryIO
from pathlib import Path
import io
import re

import numpy as np
import pdfplumber
import pypdf

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
    correct_dates_from_leading_descriptions,
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
ocr_stage_callback: ContextVar[
    Optional[Callable[[str, Optional[int], Optional[float]], None]]
] = ContextVar(
    "ocr_stage_callback", default=None
)
OCR_DPI = 150
MAX_OCR_PAGE_PIXELS = 8_000_000


def _page_needs_ocr(page: Any, text: Optional[str] = None) -> bool:
    if not page.images:
        return False

    if text is None:
        try:
            text = page.extract_text() or ""
        except Exception:
            logger.debug(
                "Could not inspect extracted text on a PDF page",
                exc_info=True,
            )
            return True

    return len(re.sub(r"\W", "", text)) < 40


def _record_ocr_stage(
    stage: str,
    page_number: Optional[int] = None,
    elapsed_seconds: Optional[float] = None,
) -> None:
    callback = ocr_stage_callback.get()
    if callback is not None:
        callback(stage, page_number, elapsed_seconds)


@lru_cache(maxsize=1)
def _get_ocr_engine():
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang="en",
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=True,
        enable_mkldnn=False,
        cpu_threads=min(4, os.cpu_count() or 1),
    )


class PDFParser:
    """
    Parser for PDF bank statements.
    """

    def __init__(self, dayfirst: Optional[bool] = None):
        self.dayfirst = dayfirst

    def parse(
        self,
        file_source: Union[str, Path, bytes, BinaryIO],
        filename: str = "statement.pdf",
        dayfirst: Optional[bool] = None
    ) -> ExtractionResult:
        """
        Parses a PDF bank statement from a file path, raw bytes, or stream.
        """
        warnings: List[str] = []

        # Read binary data
        raw_bytes, read_error = self._read_bytes(file_source)
        if read_error:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="pdf",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error=read_error
            )

        if not raw_bytes:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="pdf",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error="PDF file is empty."
            )

        # Check PDF header signature
        if not raw_bytes.startswith(b"%PDF"):
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="pdf",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error="Invalid or corrupted PDF file: Missing %PDF header signature."
            )

        # Check for encryption using pypdf
        try:
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            if reader.is_encrypted:
                return ExtractionResult(
                    success=False,
                    filename=filename,
                    file_type="pdf",
                    total_transactions=0,
                    transactions=[],
                    warnings=warnings,
                    error="PDF file is password protected. Please remove password encryption before uploading."
                )
            if any(page.rotation for page in reader.pages):
                normalized_writer = pypdf.PdfWriter(clone_from=reader)
                for page in normalized_writer.pages:
                    if page.rotation:
                        page.transfer_rotation_to_content()
                normalized_bytes = io.BytesIO()
                normalized_writer.write(normalized_bytes)
                raw_bytes = normalized_bytes.getvalue()
        except Exception as e:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="pdf",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error=f"Invalid or corrupted PDF file: {str(e)}"
            )

        # Open with pdfplumber
        try:
            pdf = pdfplumber.open(io.BytesIO(raw_bytes))
        except Exception as e:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="pdf",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error=f"Invalid or corrupted PDF structure: {str(e)}"
            )

        with pdf:
            if not pdf.pages:
                return ExtractionResult(
                    success=False,
                    filename=filename,
                    file_type="pdf",
                    total_transactions=0,
                    transactions=[],
                    warnings=warnings,
                    error="PDF document contains no pages."
                )

            # Keep page-level text and image information so mixed scans are OCRed too.
            all_text_chunks: List[str] = []
            ocr_pages: List[Tuple[int, Any]] = []
            try:
                for page_number, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    all_text_chunks.append(text)
                    if _page_needs_ocr(page, text):
                        ocr_pages.append((page_number, page))
            except Exception as exc:
                return ExtractionResult(
                    success=False,
                    filename=filename,
                    file_type="pdf",
                    total_transactions=0,
                    transactions=[],
                    warnings=warnings,
                    error=f"Could not read PDF page content: {exc}",
                )

            full_text = "\n".join(all_text_chunks).strip()

            if not full_text:
                if ocr_pages:
                    return self._attempt_ocr_extraction(
                        pdf, filename, dayfirst, pages=ocr_pages
                    )
                else:
                    return ExtractionResult(
                        success=False,
                        filename=filename,
                        file_type="pdf",
                        total_transactions=0,
                        transactions=[],
                        warnings=warnings,
                        error="PDF document contains no readable text or image content."
                    )

            # Detect statement currency
            global_currency = extract_statement_currency(full_text)

            # Attempt extraction:
            # 1. Try table-based extraction across pages
            transactions, table_warnings, table_pages = self._extract_from_tables(
                pdf, filename, global_currency, dayfirst
            )

            # Structured table rows are authoritative for their pages. Running
            # fallback parsers on those pages can reinterpret reference numbers
            # or balances as separate transactions.
            fallback_page_indexes = [
                page_index
                for page_index in range(len(pdf.pages))
                if page_index not in table_pages
            ]
            positioned_words = [
                pdf.pages[page_index].extract_words(x_tolerance=2, y_tolerance=3)
                for page_index in fallback_page_indexes
            ]
            positioned_transactions, positioned_warnings = (
                self._extract_from_positioned_words(
                    positioned_words, filename, global_currency, dayfirst
                )
            )
            line_transactions, line_warnings = self._extract_from_text(
                [all_text_chunks[index] for index in fallback_page_indexes],
                filename,
                global_currency,
                dayfirst,
            )
            transactions = self._merge_transactions(
                transactions, positioned_transactions, line_transactions
            )
            warnings.extend(table_warnings)
            warnings.extend(positioned_warnings)
            warnings.extend(line_warnings)

            if ocr_pages:
                ocr_result = self._attempt_ocr_extraction(
                    pdf,
                    filename,
                    dayfirst,
                    pages=ocr_pages,
                    allow_empty=True,
                )
                if ocr_result.success:
                    transactions = self._merge_transactions(
                        transactions, ocr_result.transactions
                    )
                    warnings.extend(ocr_result.warnings)
                else:
                    warnings.append(
                        "OCR could not read one or more scanned pages; readable PDF text "
                        "was still processed."
                    )

        correct_dates_from_leading_descriptions(transactions, dayfirst=dayfirst)
        if not transactions:
            warnings.append("No valid transaction records could be extracted from this PDF statement.")

        return ExtractionResult(
            success=True,
            filename=filename,
            file_type="pdf",
            total_transactions=len(transactions),
            transactions=transactions,
            warnings=warnings,
            error=None
        )

    def _read_bytes(
        self,
        file_source: Union[str, Path, bytes, BinaryIO]
    ) -> Tuple[Optional[bytes], Optional[str]]:
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
                    return data.encode("latin-1"), None
                return data, None
            except Exception as e:
                return None, f"Error reading stream: {str(e)}"
        return None, "Unsupported file source type."

    def _clean_header_name(self, header: Any) -> str:
        cleaned = re.sub(r"[^\w\s/]", " ", str(header or "")).strip().lower()
        return re.sub(r"\s+", " ", cleaned)

    @staticmethod
    def _matches_keyword(cleaned: str, keywords: List[str]) -> bool:
        for kw in keywords:
            if cleaned == kw:
                return True
            if re.search(r"\b" + re.escape(kw) + r"\b", cleaned):
                return True
        return False

    def _score_headers(self, headers: List[Any]) -> Tuple[Dict[str, int], int]:
        mapping: Dict[str, int] = {}
        score = 0
        cleaned_headers = [self._clean_header_name(h) for h in headers]

        for col_idx, cleaned in enumerate(cleaned_headers):
            if not cleaned:
                continue

            if "date" not in mapping and self._matches_keyword(cleaned, DATE_KEYWORDS):
                mapping["date"] = col_idx
                score += 3
                continue

            if "description" not in mapping and self._matches_keyword(cleaned, DESCRIPTION_KEYWORDS):
                mapping["description"] = col_idx
                score += 2
                continue

            if "debit" not in mapping and self._matches_keyword(cleaned, DEBIT_KEYWORDS):
                mapping["debit"] = col_idx
                score += 2
                continue

            if "credit" not in mapping and self._matches_keyword(cleaned, CREDIT_KEYWORDS):
                mapping["credit"] = col_idx
                score += 2
                continue

            if "amount" not in mapping and self._matches_keyword(cleaned, AMOUNT_KEYWORDS):
                mapping["amount"] = col_idx
                score += 2
                continue

            if "balance" not in mapping and self._matches_keyword(cleaned, BALANCE_KEYWORDS):
                mapping["balance"] = col_idx
                score += 1
                continue

            if "type" not in mapping and self._matches_keyword(cleaned, TYPE_KEYWORDS):
                mapping["type"] = col_idx
                score += 1
                continue

            if "category" not in mapping and self._matches_keyword(cleaned, CATEGORY_KEYWORDS):
                mapping["category"] = col_idx
                score += 1
                continue

            if "currency" not in mapping and self._matches_keyword(cleaned, CURRENCY_KEYWORDS):
                mapping["currency"] = col_idx
                score += 1
                continue

        return mapping, score

    def _extract_from_tables(
        self,
        pdf: pdfplumber.pdf.PDF,
        filename: str,
        global_currency: Optional[str],
        dayfirst_override: Optional[bool]
    ) -> Tuple[List[Transaction], List[str], set[int]]:
        """
        Extracts transactions from table structures detected by pdfplumber.
        """
        transactions: List[Transaction] = []
        warnings: List[str] = []

        active_col_mapping: Optional[Dict[str, int]] = None
        all_table_data_rows: List[Tuple[int, List[Any], Dict[str, int]]] = []

        for page_index, page in enumerate(pdf.pages):
            try:
                found_tables = page.extract_tables() or []
            except Exception:
                logger.debug("PDF ruled-table extraction failed on a page", exc_info=True)
                found_tables = []
            has_header = any(
                self._has_transaction_headers(row)
                for table in found_tables
                for row in table
            )
            if not has_header:
                try:
                    found_tables = page.extract_tables(
                        table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "intersection_tolerance": 5,
                            "min_words_vertical": 2,
                            "min_words_horizontal": 1,
                        }
                    ) or []
                except Exception:
                    logger.debug("PDF text-table extraction failed on a page", exc_info=True)
                    found_tables = []

            for table in found_tables:
                if not table:
                    continue

                for row_idx, row in enumerate(table):
                    if not row or not any(c for c in row):
                        continue

                    mapping, score = self._score_headers(row)
                    if score >= 2 and "date" in mapping and (
                        "amount" in mapping or "debit" in mapping or "credit" in mapping
                    ):
                        active_col_mapping = mapping
                        continue

                    if active_col_mapping:
                        all_table_data_rows.append(
                            (page_index, row, active_col_mapping.copy())
                        )

        if not active_col_mapping or not all_table_data_rows:
            return [], warnings, set()

        # Infer date convention across table rows
        raw_dates = [
            row[mapping["date"]]
            for _, row, mapping in all_table_data_rows
            if len(row) > mapping["date"] and row[mapping["date"]]
        ]

        resolved_dayfirst, date_warnings = infer_date_convention(
            raw_dates=raw_dates,
            currency=global_currency,
            explicit_dayfirst=dayfirst_override if dayfirst_override is not None else self.dayfirst
        )
        warnings.extend(date_warnings)

        skipped_rows = 0
        extracted_pages: set[int] = set()

        for row_idx, (page_index, row, col_mapping) in enumerate(all_table_data_rows):
            def get_cell(idx: Optional[int]) -> Any:
                if idx is not None and 0 <= idx < len(row):
                    return row[idx]
                return None

            date_col = col_mapping["date"]
            desc_col = col_mapping.get("description")
            debit_col = col_mapping.get("debit")
            credit_col = col_mapping.get("credit")
            amount_col = col_mapping.get("amount")
            balance_col = col_mapping.get("balance")
            type_col = col_mapping.get("type")
            cat_col = col_mapping.get("category")
            curr_col = col_mapping.get("currency")

            raw_date = get_cell(date_col)
            normalized_dt = normalize_date(raw_date, dayfirst=resolved_dayfirst)
            if not normalized_dt:
                skipped_rows += 1
                continue

            raw_desc = get_cell(desc_col)
            normalized_desc = normalize_description(raw_desc)
            if is_summary_or_balance_row(normalized_desc):
                skipped_rows += 1
                continue

            raw_debit = get_cell(debit_col)
            raw_credit = get_cell(credit_col)
            raw_amt = get_cell(amount_col)
            raw_type = get_cell(type_col)

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

            explicit_curr = None
            raw_curr_val = get_cell(curr_col)
            if raw_curr_val:
                explicit_curr = extract_currency(str(raw_curr_val))
            resolved_currency = explicit_curr or row_curr or global_currency

            resolved_balance = None
            raw_bal = get_cell(balance_col)
            if raw_bal is not None:
                bal_val, _, _ = parse_numeric_amount(raw_bal)
                if bal_val is not None:
                    resolved_balance = round(bal_val, 2)

            category = "Unknown"
            raw_cat = get_cell(cat_col)
            if raw_cat is not None:
                cat_text = str(raw_cat).strip()
                if cat_text and cat_text.lower() not in {"nan", "none", "null"}:
                    category = cat_text
            if category.casefold() in {"unknown", "uncategorized"}:
                category = categorize(normalized_desc, txn_type)

            txn_id = generate_transaction_id(
                filename,
                row_idx,
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
            extracted_pages.add(page_index)

        if skipped_rows > 0:
            warnings.append(f"Skipped {skipped_rows} non-transaction or malformed table row(s).")

        return transactions, warnings, extracted_pages

    def _has_transaction_headers(self, row: List[Any]) -> bool:
        mapping, _ = self._score_headers(row)
        return "date" in mapping and any(
            column in mapping for column in ("amount", "debit", "credit")
        )

    @staticmethod
    def _merge_transactions(
        *transaction_groups: List[Transaction],
    ) -> List[Transaction]:
        merged: List[Transaction] = []
        prior_exact_counts: Counter[
            Tuple[str, str, float, str, Optional[str]]
        ] = Counter()
        prior_overlap_counts: Counter[
            Tuple[str, str, float, Optional[str]]
        ] = Counter()
        prior_balance_counts: Counter[
            Tuple[str, str, Optional[str], float]
        ] = Counter()
        reference_directions: Dict[
            Tuple[str, str, float, Optional[str]], set[str]
        ] = {}
        for group in transaction_groups:
            current_group_counts: Counter[
                Tuple[str, str, float, str, Optional[str]]
            ] = Counter()
            current_overlap_counts: Counter[
                Tuple[str, str, float, Optional[str]]
            ] = Counter()
            current_balance_counts: Counter[
                Tuple[str, str, Optional[str], float]
            ] = Counter()
            current_directions: Dict[
                Tuple[str, str, float, Optional[str]], set[str]
            ] = {}
            for transaction in group:
                overlap_key = (
                    transaction.transaction_date,
                    transaction.description.casefold(),
                    round(transaction.amount, 2),
                    transaction.currency.upper() if transaction.currency else None,
                )
                direction = transaction.transaction_type.casefold()
                key = (*overlap_key[:3], direction, overlap_key[3])
                current_group_counts[key] += 1
                current_overlap_counts[overlap_key] += 1
                current_directions.setdefault(overlap_key, set()).add(direction)
                if transaction.balance is not None:
                    balance_key = (
                        transaction.transaction_date,
                        transaction.description.casefold(),
                        transaction.currency.upper()
                        if transaction.currency
                        else None,
                        round(transaction.balance, 2),
                    )
                    current_balance_counts[balance_key] += 1
            processed_counts: Counter[
                Tuple[str, str, float, str, Optional[str]]
            ] = Counter()
            processed_overlap_counts: Counter[
                Tuple[str, str, float, Optional[str]]
            ] = Counter()
            processed_balance_counts: Counter[
                Tuple[str, str, Optional[str], float]
            ] = Counter()
            for transaction in group:
                overlap_key = (
                    transaction.transaction_date,
                    transaction.description.casefold(),
                    round(transaction.amount, 2),
                    transaction.currency.upper() if transaction.currency else None,
                )
                direction = transaction.transaction_type.casefold()
                key = (*overlap_key[:3], direction, overlap_key[3])
                balance_key = None
                if transaction.balance is not None:
                    balance_key = (
                        transaction.transaction_date,
                        transaction.description.casefold(),
                        transaction.currency.upper()
                        if transaction.currency
                        else None,
                        round(transaction.balance, 2),
                    )
                processed_counts[key] += 1
                processed_overlap_counts[overlap_key] += 1
                if balance_key is not None:
                    processed_balance_counts[balance_key] += 1
                conflicting_strategy_match = (
                    processed_overlap_counts[overlap_key]
                    <= prior_overlap_counts[overlap_key]
                    and len(reference_directions.get(overlap_key, ())) == 1
                    and len(current_directions[overlap_key]) == 1
                    and direction not in reference_directions[overlap_key]
                )
                conflicting_amount_match = (
                    balance_key is not None
                    and processed_balance_counts[balance_key]
                    <= prior_balance_counts[balance_key]
                    and processed_counts[key] > prior_exact_counts[key]
                )
                # Balance provides a cross-strategy row identity when amount
                # parsing differs. Keep repeated occurrences within a strategy
                # and preserve collisions without matching balances.
                if (
                    processed_counts[key] > prior_exact_counts[key]
                    and not conflicting_strategy_match
                    and not conflicting_amount_match
                ):
                    merged.append(transaction)
            for key, count in current_group_counts.items():
                prior_exact_counts[key] = max(prior_exact_counts[key], count)
            for key, count in current_overlap_counts.items():
                prior_overlap_counts[key] = max(prior_overlap_counts[key], count)
                reference_directions.setdefault(key, current_directions[key])
            for key, count in current_balance_counts.items():
                prior_balance_counts[key] = max(prior_balance_counts[key], count)
        return merged

    def _header_column_centers(
        self, words: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        columns: Dict[str, float] = {}
        index = 0
        while index < len(words):
            best_match: Optional[Tuple[int, str]] = None
            for end in range(index, min(index + 4, len(words))):
                if end > index:
                    gap = float(words[end]["x0"]) - float(words[end - 1]["x1"])
                    word_height = float(words[end]["bottom"]) - float(words[end]["top"])
                    if gap > max(8.0, word_height * 0.6):
                        break
                phrase = " ".join(str(word["text"]) for word in words[index : end + 1])
                mapping, _ = self._score_headers([phrase])
                if mapping:
                    best_match = (end, next(iter(mapping)))
            if best_match is None:
                index += 1
                continue
            end, column = best_match
            if column not in columns:
                columns[column] = (
                    float(words[index]["x0"]) + float(words[end]["x1"])
                ) / 2
            index = end + 1
        return columns

    @staticmethod
    def _group_positioned_words(
        words: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        ordered_words = sorted(words, key=lambda word: (word["top"], word["x0"]))
        lines: List[List[Dict[str, Any]]] = []
        for word in ordered_words:
            center = (float(word["top"]) + float(word["bottom"])) / 2
            if lines:
                previous = lines[-1]
                previous_center = sum(
                    (float(item["top"]) + float(item["bottom"])) / 2
                    for item in previous
                ) / len(previous)
                tolerance = max(
                    2.0,
                    min(
                        float(word["bottom"]) - float(word["top"]),
                        float(previous[0]["bottom"]) - float(previous[0]["top"]),
                    )
                    * 0.6,
                )
                if abs(center - previous_center) <= tolerance:
                    previous.append(word)
                    continue
            lines.append([word])
        return [sorted(line, key=lambda word: word["x0"]) for line in lines]

    def _extract_from_positioned_words(
        self,
        pages_words: List[List[Dict[str, Any]]],
        filename: str,
        global_currency: Optional[str],
        dayfirst_override: Optional[bool],
    ) -> Tuple[List[Transaction], List[str]]:
        transactions: List[Transaction] = []
        warnings: List[str] = []
        row_index = 0
        candidate_rows: List[Dict[str, str]] = []

        for words in pages_words:
            active_columns: Dict[str, float] = {}
            for line in self._group_positioned_words(words):
                header_columns = self._header_column_centers(line)
                if "date" in header_columns and any(
                    column in header_columns
                    for column in ("amount", "debit", "credit")
                ):
                    active_columns = header_columns
                    continue
                if not active_columns:
                    continue

                cells: Dict[str, List[str]] = {
                    column: [] for column in active_columns
                }
                for word in line:
                    nearest_column = min(
                        active_columns,
                        key=lambda column: abs(
                            float(word["x0"] + word["x1"]) / 2
                            - active_columns[column]
                        ),
                    )
                    cells[nearest_column].append(str(word["text"]))
                values = {
                    column: " ".join(parts).strip()
                    for column, parts in cells.items()
                }
                raw_date = values.get("date")
                if not raw_date:
                    continue
                candidate_rows.append(values)

        raw_dates = [row["date"] for row in candidate_rows]
        resolved_dayfirst, date_warnings = infer_date_convention(
            raw_dates=raw_dates,
            currency=global_currency,
            explicit_dayfirst=(
                dayfirst_override
                if dayfirst_override is not None
                else self.dayfirst
            ),
        )
        warnings.extend(date_warnings)
        for values in candidate_rows:
            normalized_date = normalize_date(
                values.get("date"), dayfirst=resolved_dayfirst
            )
            if not normalized_date:
                continue
            description = normalize_description(values.get("description"))
            if is_summary_or_balance_row(description):
                continue
            amount, transaction_type, row_currency = determine_amount_and_type(
                debit_val=values.get("debit"),
                credit_val=values.get("credit"),
                amount_val=values.get("amount"),
                type_hint=values.get("type") or description,
            )
            if amount is None:
                continue
            balance, _, _ = parse_numeric_amount(values.get("balance"))
            currency = (
                extract_currency(values.get("currency", ""))
                or row_currency
                or global_currency
            )
            category = values.get("category") or "Unknown"
            if category.casefold() in {"unknown", "uncategorized"}:
                category = categorize(description, transaction_type)
            transactions.append(
                Transaction(
                    transaction_id=generate_transaction_id(
                        filename,
                        row_index,
                        normalized_date,
                        description,
                        amount,
                        transaction_type,
                        currency,
                    ),
                    transaction_date=normalized_date,
                    description=description,
                    amount=amount,
                    currency=currency,
                    transaction_type=transaction_type,
                    category=category,
                    source_dataset=filename,
                    balance=round(balance, 2) if balance is not None else None,
                )
            )
            row_index += 1

        return transactions, warnings

    def _extract_from_text(
        self,
        pages_text: List[str],
        filename: str,
        global_currency: Optional[str],
        dayfirst_override: Optional[bool]
    ) -> Tuple[List[Transaction], List[str]]:
        """
        Line-based transaction parser for formatted text bank statements.
        """
        transactions: List[Transaction] = []
        warnings: List[str] = []

        all_lines: List[str] = []
        for p_text in pages_text:
            for line in p_text.splitlines():
                stripped = line.strip()
                if stripped:
                    all_lines.append(stripped)

        # First pass: collect raw candidate transaction lines
        raw_candidates: List[Tuple[str, str, List[str], Dict[str, int]]] = []
        date_samples: List[str] = []
        pending_line: Optional[str] = None
        incomplete_rows = 0
        text_header_mapping: Dict[str, int] = {}

        for line in all_lines:
            header_mapping, header_score = self._score_headers(line.split())
            if (
                header_score >= 2
                and "date" in header_mapping
                and any(
                    column in header_mapping
                    for column in ("amount", "debit", "credit")
                )
            ):
                text_header_mapping = header_mapping
                continue

            max_numeric_fields = (
                3
                if {"debit", "credit", "balance"}.issubset(text_header_mapping)
                else 2
            )
            parsed_line = self._parse_transaction_line(
                line, max_numeric_fields=max_numeric_fields
            )
            if parsed_line:
                if pending_line is not None:
                    incomplete_rows += 1
                    pending_line = None
                raw_dt, desc, amounts = parsed_line
                raw_candidates.append(
                    (raw_dt, desc, amounts, text_header_mapping.copy())
                )
                date_samples.append(raw_dt)
                continue

            if self._starts_with_transaction_date(line):
                if pending_line is not None:
                    incomplete_rows += 1
                pending_line = line
                continue

            if pending_line is not None:
                pending_line = f"{pending_line} {line}"
                parsed_line = self._parse_transaction_line(
                    pending_line, max_numeric_fields=max_numeric_fields
                )
                if parsed_line:
                    raw_dt, desc, amounts = parsed_line
                    raw_candidates.append(
                        (raw_dt, desc, amounts, text_header_mapping.copy())
                    )
                    date_samples.append(raw_dt)
                    pending_line = None

        if pending_line is not None:
            incomplete_rows += 1

        if not raw_candidates:
            if incomplete_rows:
                warnings.append(
                    f"Skipped {incomplete_rows} incomplete transaction row(s)."
                )
            return [], warnings

        # Infer date convention across collected lines
        resolved_dayfirst, date_warnings = infer_date_convention(
            raw_dates=date_samples,
            currency=global_currency,
            explicit_dayfirst=dayfirst_override if dayfirst_override is not None else self.dayfirst
        )
        warnings.extend(date_warnings)

        skipped_rows = incomplete_rows

        previous_balance: Optional[float] = None
        for row_idx, (raw_dt, desc, amounts, header_mapping) in enumerate(raw_candidates):
            normalized_dt = normalize_date(raw_dt, dayfirst=resolved_dayfirst)
            if not normalized_dt:
                skipped_rows += 1
                continue

            normalized_desc = normalize_description(desc)
            has_balance_column = "balance" in header_mapping
            row_balance = None
            if len(amounts) >= 2:
                row_balance, _, _ = parse_numeric_amount(amounts[-1])

            if is_summary_or_balance_row(normalized_desc):
                if re.search(
                    r"\b(?:opening|beginning|initial|starting)\s+bal(?:ance)?\b",
                    normalized_desc,
                    re.IGNORECASE,
                ):
                    if row_balance is None and amounts:
                        row_balance, _, _ = parse_numeric_amount(amounts[-1])
                    if row_balance is not None:
                        previous_balance = row_balance
                continue

            # Analyze amounts
            debit_val = None
            credit_val = None
            amount_val = None
            balance_val = None

            has_debit_column = "debit" in header_mapping
            has_credit_column = "credit" in header_mapping
            if has_debit_column and has_credit_column and len(amounts) >= 3:
                debit_val, credit_val = amounts[0], amounts[1]
                balance_val = amounts[-1] if has_balance_column else None
            elif has_debit_column != has_credit_column:
                if len(amounts) >= 2:
                    if has_debit_column:
                        debit_val = amounts[0]
                    else:
                        credit_val = amounts[0]
                    balance_val = amounts[-1] if has_balance_column else None
                elif amounts and not has_balance_column:
                    if has_debit_column:
                        debit_val = amounts[0]
                    else:
                        credit_val = amounts[0]
                else:
                    skipped_rows += 1
                    continue
            elif "amount" in header_mapping:
                if amounts and not (has_balance_column and len(amounts) == 1):
                    amount_val = amounts[0]
                    balance_val = amounts[-1] if len(amounts) >= 2 else None
                elif amounts:
                    skipped_rows += 1
                    continue
            elif len(amounts) == 1:
                amount_val = amounts[0]
            elif len(amounts) == 2:
                amount_val, balance_val = amounts
            elif len(amounts) >= 3:
                debit_val, credit_val = amounts[0], amounts[1]
                balance_val = amounts[-1]

            if (
                has_debit_column
                and has_credit_column
                and len(amounts) == 2
            ):
                transaction_value, _, _ = parse_numeric_amount(amounts[0])
                direction = None
                if previous_balance is not None and row_balance is not None:
                    balance_change = round(row_balance - previous_balance, 2)
                    if transaction_value is not None and math.isclose(
                        abs(balance_change),
                        abs(transaction_value),
                        abs_tol=0.02,
                    ):
                        direction = "expense" if balance_change < 0 else "income"
                if direction == "expense":
                    debit_val = amounts[0]
                elif direction == "income":
                    credit_val = amounts[0]
                elif transaction_value is not None and transaction_value < 0:
                    amount_val = amounts[0]
                else:
                    skipped_rows += 1
                    continue
            elif (
                not header_mapping
                and len(amounts) == 2
                and previous_balance is not None
                and row_balance is not None
            ):
                transaction_value, _, _ = parse_numeric_amount(amounts[0])
                balance_change = round(row_balance - previous_balance, 2)
                if (
                    transaction_value is not None
                    and transaction_value >= 0
                    and math.isclose(
                        abs(balance_change),
                        abs(transaction_value),
                        abs_tol=0.02,
                    )
                ):
                    if balance_change < 0:
                        debit_val = amounts[0]
                    elif balance_change > 0:
                        credit_val = amounts[0]
                    amount_val = None

            amount, txn_type, row_curr = determine_amount_and_type(
                debit_val=debit_val,
                credit_val=credit_val,
                amount_val=amount_val,
                type_hint=normalized_desc,
            )

            if amount is None:
                skipped_rows += 1
                continue

            resolved_currency = row_curr or extract_currency(desc) or global_currency

            resolved_balance = None
            if balance_val:
                bal_num, _, _ = parse_numeric_amount(balance_val)
                if bal_num is not None:
                    resolved_balance = round(bal_num, 2)
            elif has_balance_column and row_balance is not None:
                resolved_balance = round(row_balance, 2)
            if row_balance is not None:
                previous_balance = row_balance

            txn_id = generate_transaction_id(
                filename,
                row_idx,
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
                category=categorize(normalized_desc, txn_type),
                source_dataset=filename,
                balance=resolved_balance
            )
            transactions.append(txn)

        if skipped_rows > 0:
            warnings.append(f"Skipped {skipped_rows} non-transaction or malformed line(s).")

        return transactions, warnings

    def _parse_transaction_line(
        self, line: str, max_numeric_fields: int = 2
    ) -> Optional[Tuple[str, str, List[str]]]:
        """
        Parses a single text line that begins with a date followed by description and amounts.
        Returns (raw_date_str, description_str, list_of_amount_strings) or None.
        """
        date_pattern = (
            r"^(\d{4}[/.-]\d{1,2}[/.-]\d{1,2}|"
            r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|"
            r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|"
            r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})\b"
        )
        m_date = re.match(date_pattern, line.strip())
        if not m_date:
            return None

        raw_date = m_date.group(1)
        remainder = line.strip()[len(raw_date):].strip()

        tokens = remainder.split()
        amounts: List[str] = []
        desc_tokens = list(tokens)
        pending_suffix = ""

        # Extract numeric amounts from right to left
        for token in reversed(tokens):
            if len(amounts) >= max_numeric_fields:
                break
            clean_tok = token.strip(",;")
            if clean_tok.upper() in {"CR", "DR", "DB"}:
                pending_suffix = clean_tok.upper()
                desc_tokens.pop()
            elif re.search(r"\d", clean_tok) and re.fullmatch(
                r"[-+]?[$€£¥₹Rs.]*\(?[\d,]+(?:\.\d{1,2})?\)?(?:CR|DR|DB)?",
                clean_tok,
                re.IGNORECASE
            ):
                full_amt = clean_tok + (f" {pending_suffix}" if pending_suffix else "")
                pending_suffix = ""
                amounts.insert(0, full_amt)
                desc_tokens.pop()
            else:
                break

        if not amounts:
            return None

        desc = " ".join(desc_tokens).strip()
        return raw_date, desc, amounts

    @staticmethod
    def _starts_with_transaction_date(line: str) -> bool:
        return bool(
            re.match(
                r"^(?:\d{4}[/.-]\d{1,2}[/.-]\d{1,2}|"
                r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|"
                r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}|"
                r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4})\b",
                line.strip(),
            )
        )

    def _recognize_page(
        self, ocr_engine: Any, page: Any, filename: str, page_number: int
    ) -> Tuple[str, List[Dict[str, Any]]]:
        page_pixels_at_72_dpi = float(page.width) * float(page.height)
        if page_pixels_at_72_dpi <= 0:
            raise ValueError("PDF page has invalid dimensions.")
        if page_pixels_at_72_dpi > MAX_OCR_PAGE_PIXELS:
            raise ValueError("PDF page exceeds the OCR image size limit.")

        resolution = min(
            OCR_DPI,
            72 * math.sqrt(MAX_OCR_PAGE_PIXELS / page_pixels_at_72_dpi),
        )
        render_started = time.monotonic()
        _record_ocr_stage("pdf_page_render_started", page_number)
        logger.info(
            "Rendering scanned PDF page",
            extra={
                "source_filename": filename,
                "stage": "pdf_page_render_started",
                "page_number": page_number,
                "resolution": resolution,
            },
        )
        page_image = page.to_image(resolution=resolution).original
        _record_ocr_stage(
            "pdf_page_render_completed",
            page_number,
            time.monotonic() - render_started,
        )
        logger.info(
            "Rendered scanned PDF page",
            extra={
                "source_filename": filename,
                "stage": "pdf_page_render_completed",
                "page_number": page_number,
            },
        )
        inference_started = time.monotonic()
        _record_ocr_stage("ocr_inference_started", page_number)
        logger.info(
            "PaddleOCR inference started",
            extra={
                "source_filename": filename,
                "stage": "ocr_inference_started",
                "page_number": page_number,
            },
        )
        page_results = ocr_engine.predict(np.asarray(page_image))
        _record_ocr_stage(
            "ocr_inference_completed",
            page_number,
            time.monotonic() - inference_started,
        )
        logger.info(
            "PaddleOCR inference completed",
            extra={
                "source_filename": filename,
                "stage": "ocr_inference_completed",
                "page_number": page_number,
                "result_count": len(page_results),
            },
        )

        page_lines: List[Tuple[float, float, str, float, float]] = []
        unpositioned_text: List[str] = []
        for page_result in page_results:
            result_data = page_result.json.get("res", {})
            recognized_text = result_data.get("rec_texts", [])
            boxes = result_data.get("rec_boxes", [])
            if hasattr(boxes, "tolist"):
                boxes = boxes.tolist()
            polygons = result_data.get("rec_polys", [])
            if hasattr(polygons, "tolist"):
                polygons = polygons.tolist()

            for index, text in enumerate(recognized_text):
                if not str(text).strip():
                    continue
                box = boxes[index] if index < len(boxes) else None
                if (not box or len(box) < 4) and index < len(polygons):
                    polygon = polygons[index]
                    if len(polygon) >= 4 and all(len(point) >= 2 for point in polygon):
                        x_coords = [point[0] for point in polygon]
                        y_coords = [point[1] for point in polygon]
                        box = [
                            min(x_coords),
                            min(y_coords),
                            max(x_coords),
                            max(y_coords),
                        ]
                if box is not None and len(box) >= 4:
                    x_min, y_min, x_max, y_max = box[:4]
                    page_lines.append(
                        (
                            (float(y_min) + float(y_max)) / 2,
                            float(x_min),
                            str(text).strip(),
                            float(y_max) - float(y_min),
                            float(x_max),
                        )
                    )
                else:
                    unpositioned_text.append(str(text).strip())

        if not page_lines:
            return self._group_unpositioned_text(unpositioned_text), []
        page_lines.sort(key=lambda line: (line[0], line[1]))
        grouped_lines: List[List[Tuple[float, float, str, float, float]]] = []
        for line in page_lines:
            if grouped_lines:
                previous_group = grouped_lines[-1]
                group_center = sum(item[0] for item in previous_group) / len(previous_group)
                tolerance = max(8.0, min(line[3], previous_group[0][3]) * 0.6)
                if abs(line[0] - group_center) <= tolerance:
                    previous_group.append(line)
                    continue
            grouped_lines.append([line])

        positioned_words = [
            {
                "x0": x,
                "x1": x_max,
                "top": y - height / 2,
                "bottom": y + height / 2,
                "text": text,
            }
            for y, x, text, height, x_max in page_lines
        ]
        return "\n".join(
            " ".join(item[2] for item in sorted(group, key=lambda item: item[1]))
            for group in grouped_lines
        ), positioned_words

    def _group_unpositioned_text(self, text_fragments: List[str]) -> str:
        rows: List[List[str]] = []
        for fragment in text_fragments:
            if self._starts_with_transaction_date(fragment):
                rows.append([fragment])
            elif rows:
                rows[-1].append(fragment)
            elif fragment:
                rows.append([fragment])
        return "\n".join(" ".join(row) for row in rows)

    def _attempt_ocr_extraction(
        self,
        pdf: pdfplumber.pdf.PDF,
        filename: str,
        dayfirst: Optional[bool],
        pages: Optional[List[Tuple[int, Any]]] = None,
        allow_empty: bool = False,
    ) -> ExtractionResult:
        """Recognize selected scanned pages and parse their transaction rows."""
        selected_pages = pages or list(enumerate(pdf.pages, start=1))
        initialization_started = time.monotonic()
        try:
            _record_ocr_stage("ocr_model_initialization_started")
            logger.info(
                "PaddleOCR model initialization started",
                extra={
                    "source_filename": filename,
                    "stage": "ocr_model_initialization_started",
                },
            )
            ocr_engine = _get_ocr_engine()
            _record_ocr_stage(
                "ocr_model_initialization_completed",
                elapsed_seconds=time.monotonic() - initialization_started,
            )
            logger.info(
                "PaddleOCR model initialization completed",
                extra={
                    "source_filename": filename,
                    "stage": "ocr_model_initialization_completed",
                },
            )
        except ImportError:
            logger.warning("PaddleOCR is unavailable for scanned PDF %s", filename)
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="pdf",
                total_transactions=0,
                transactions=[],
                warnings=[],
                error=(
                    "Scanned PDF detected. Optical Character Recognition (OCR) "
                    "is not configured in this environment. Please upload a "
                    "digital/text-based statement or convert it to CSV/XLSX."
                ),
            )
        except Exception:
            logger.exception("Failed to initialize PaddleOCR for scanned PDF %s", filename)
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="pdf",
                total_transactions=0,
                transactions=[],
                warnings=[],
                error=(
                    "OCR engine could not be initialized. Please retry later or "
                    "upload a digital/text-based statement."
                ),
            )

        pages_text: List[str] = []
        pages_words: List[List[Dict[str, Any]]] = []
        warnings: List[str] = []
        _record_ocr_stage("pdf_rendering_started")
        logger.info(
            "Scanned PDF rendering started",
            extra={
                "source_filename": filename,
                "stage": "pdf_rendering_started",
                "page_count": len(selected_pages),
            },
        )
        for page_number, page in selected_pages:
            page_started = time.monotonic()
            try:
                page_text, page_words = self._recognize_page(
                    ocr_engine, page, filename, page_number
                )
                pages_text.append(page_text)
                pages_words.append(page_words)
            except Exception:
                warnings.append(f"OCR could not process PDF page {page_number}.")
                pages_text.append("")
                pages_words.append([])
                _record_ocr_stage(
                    "pdf_page_processing_failed",
                    page_number,
                    time.monotonic() - page_started,
                )
                logger.exception(
                    "OCR failed on scanned PDF page",
                    extra={
                        "source_filename": filename,
                        "stage": "ocr_page_failed",
                        "page_number": page_number,
                    },
                )
        _record_ocr_stage("pdf_rendering_completed")
        logger.info(
            "Scanned PDF rendering and OCR completed",
            extra={
                "source_filename": filename,
                "stage": "pdf_rendering_completed",
                "page_count": len(pages_text),
            },
        )

        recognized_text = "\n".join(pages_text).strip()
        if not recognized_text and not allow_empty:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="pdf",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error="Scanned PDF detected, but OCR could not recognize readable text.",
            )

        _record_ocr_stage("ocr_text_extraction_completed")
        logger.info(
            "OCR text extraction completed",
            extra={
                "source_filename": filename,
                "stage": "ocr_text_extraction_completed",
                "recognized_character_count": len(recognized_text),
            },
        )
        currency = extract_statement_currency(recognized_text)
        parsing_started = time.monotonic()
        _record_ocr_stage("transaction_parsing_started")
        logger.info(
            "OCR transaction parsing started",
            extra={
                "source_filename": filename,
                "stage": "transaction_parsing_started",
            },
        )
        line_transactions, parser_warnings = self._extract_from_text(
            pages_text, filename, currency, dayfirst
        )
        positioned_transactions, positioned_warnings = (
            self._extract_from_positioned_words(
                pages_words, filename, currency, dayfirst
            )
        )
        transactions = self._merge_transactions(
            positioned_transactions, line_transactions
        )
        warnings.extend(parser_warnings)
        warnings.extend(positioned_warnings)
        _record_ocr_stage(
            "transaction_parsing_completed",
            elapsed_seconds=time.monotonic() - parsing_started,
        )
        logger.info(
            "OCR transaction parsing completed",
            extra={
                "source_filename": filename,
                "stage": "transaction_parsing_completed",
                "transaction_count": len(transactions),
                "warning_count": len(warnings),
            },
        )
        if not transactions and not allow_empty:
            return ExtractionResult(
                success=False,
                filename=filename,
                file_type="pdf",
                total_transactions=0,
                transactions=[],
                warnings=warnings,
                error="OCR could not extract any valid transactions from the scanned PDF.",
            )

        return ExtractionResult(
            success=True,
            filename=filename,
            file_type="pdf",
            total_transactions=len(transactions),
            transactions=transactions,
            warnings=warnings,
            error=None,
        )


def parse_pdf(
    file_source: Union[str, Path, bytes, BinaryIO],
    filename: str = "statement.pdf",
    dayfirst: Optional[bool] = None
) -> ExtractionResult:
    """
    Convenience function to parse a PDF bank statement.
    """
    parser = PDFParser(dayfirst=dayfirst)
    return parser.parse(file_source, filename, dayfirst=dayfirst)


def is_scanned_pdf(pdf_bytes: bytes) -> bool:
    """Return whether any page has image content without sufficient extractable text."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                return False
            for page in pdf.pages:
                if _page_needs_ocr(page):
                    return True
            return False
    except Exception:
        return False
