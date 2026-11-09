"""
PDF Bank Statement Parser.
Extracts and normalizes transaction data from PDF bank statements using pdfplumber and pypdf.
Supports both table-structured layouts and formatted text layouts, multi-page statements,
scanned PDF detection, and safe handling of unreadable or encrypted files.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, BinaryIO
from pathlib import Path
import io
import re
import shutil
import pdfplumber
import pypdf

from models.transaction import Transaction, ExtractionResult
from services.transaction_normalizer import (
    normalize_date,
    normalize_description,
    determine_amount_and_type,
    parse_numeric_amount,
    generate_transaction_id,
    extract_currency,
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

            # Check for scanned PDF (empty text but images exist)
            all_text_chunks: List[str] = []
            total_images = 0
            for page in pdf.pages:
                text = page.extract_text() or ""
                all_text_chunks.append(text)
                total_images += len(page.images)

            full_text = "\n".join(all_text_chunks).strip()

            if not full_text:
                if total_images > 0:
                    # Scanned PDF detected
                    if self._is_ocr_available():
                        # OCR hook if configured in environment
                        return self._attempt_ocr_extraction(pdf, filename, dayfirst)
                    else:
                        return ExtractionResult(
                            success=False,
                            filename=filename,
                            file_type="pdf",
                            total_transactions=0,
                            transactions=[],
                            warnings=warnings,
                            error=(
                                "Scanned PDF detected. Optical Character Recognition (OCR) "
                                "is not configured in this environment. Please upload a "
                                "digital/text-based statement or convert it to CSV/XLSX."
                            )
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
            global_currency = extract_currency(full_text[:2000])

            # Attempt extraction:
            # 1. Try table-based extraction across pages
            transactions, table_warnings = self._extract_from_tables(
                pdf, filename, global_currency, dayfirst
            )

            # 2. If table extraction yielded no transactions, fall back to line-based text parsing
            if not transactions:
                transactions, line_warnings = self._extract_from_text(
                    all_text_chunks, filename, global_currency, dayfirst
                )
                warnings.extend(line_warnings)
            else:
                warnings.extend(table_warnings)

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
    ) -> Tuple[List[Transaction], List[str]]:
        """
        Extracts transactions from table structures detected by pdfplumber.
        """
        transactions: List[Transaction] = []
        warnings: List[str] = []

        active_col_mapping: Optional[Dict[str, int]] = None
        all_table_data_rows: List[List[Any]] = []

        for page in pdf.pages:
            tables = page.extract_tables()
            if not tables:
                continue

            for table in tables:
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
                        all_table_data_rows.append(row)

        if not active_col_mapping or not all_table_data_rows:
            return [], warnings

        # Infer date convention across table rows
        date_col = active_col_mapping["date"]
        raw_dates = [r[date_col] for r in all_table_data_rows if len(r) > date_col and r[date_col]]

        resolved_dayfirst, date_warnings = infer_date_convention(
            raw_dates=raw_dates,
            currency=global_currency,
            explicit_dayfirst=dayfirst_override if dayfirst_override is not None else self.dayfirst
        )
        warnings.extend(date_warnings)

        desc_col = active_col_mapping.get("description")
        debit_col = active_col_mapping.get("debit")
        credit_col = active_col_mapping.get("credit")
        amount_col = active_col_mapping.get("amount")
        balance_col = active_col_mapping.get("balance")
        type_col = active_col_mapping.get("type")
        cat_col = active_col_mapping.get("category")
        curr_col = active_col_mapping.get("currency")

        skipped_rows = 0

        for row_idx, row in enumerate(all_table_data_rows):
            def get_cell(idx: Optional[int]) -> Any:
                if idx is not None and 0 <= idx < len(row):
                    return row[idx]
                return None

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
                type_hint=str(raw_type) if raw_type is not None else None
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

            txn_id = generate_transaction_id(
                filename, row_idx, normalized_dt, normalized_desc, amount
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
            warnings.append(f"Skipped {skipped_rows} non-transaction or malformed table row(s).")

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
        raw_candidates: List[Tuple[str, str, List[str]]] = []
        date_samples: List[str] = []

        for line in all_lines:
            parsed_line = self._parse_transaction_line(line)
            if parsed_line:
                raw_dt, desc, amounts = parsed_line
                raw_candidates.append((raw_dt, desc, amounts))
                date_samples.append(raw_dt)

        if not raw_candidates:
            return [], warnings

        # Infer date convention across collected lines
        resolved_dayfirst, date_warnings = infer_date_convention(
            raw_dates=date_samples,
            currency=global_currency,
            explicit_dayfirst=dayfirst_override if dayfirst_override is not None else self.dayfirst
        )
        warnings.extend(date_warnings)

        skipped_rows = 0

        for row_idx, (raw_dt, desc, amounts) in enumerate(raw_candidates):
            normalized_dt = normalize_date(raw_dt, dayfirst=resolved_dayfirst)
            if not normalized_dt:
                skipped_rows += 1
                continue

            normalized_desc = normalize_description(desc)
            if is_summary_or_balance_row(normalized_desc):
                skipped_rows += 1
                continue

            # Analyze amounts
            # If 2 numbers: first is amount, second is balance
            # If 3 numbers: first is debit, second is credit, third is balance
            # If 1 number: first is amount
            debit_val = None
            credit_val = None
            amount_val = None
            balance_val = None

            if len(amounts) == 1:
                amount_val = amounts[0]
            elif len(amounts) == 2:
                amount_val = amounts[0]
                balance_val = amounts[1]
            elif len(amounts) >= 3:
                debit_val = amounts[0]
                credit_val = amounts[1]
                balance_val = amounts[2]

            amount, txn_type, row_curr = determine_amount_and_type(
                debit_val=debit_val,
                credit_val=credit_val,
                amount_val=amount_val
            )

            if amount is None:
                skipped_rows += 1
                continue

            resolved_currency = row_curr or global_currency

            resolved_balance = None
            if balance_val:
                bal_num, _, _ = parse_numeric_amount(balance_val)
                if bal_num is not None:
                    resolved_balance = round(bal_num, 2)

            txn_id = generate_transaction_id(
                filename, row_idx, normalized_dt, normalized_desc, amount
            )

            txn = Transaction(
                transaction_id=txn_id,
                transaction_date=normalized_dt,
                description=normalized_desc,
                amount=amount,
                currency=resolved_currency,
                transaction_type=txn_type,
                category="Unknown",
                source_dataset=filename,
                balance=resolved_balance
            )
            transactions.append(txn)

        if skipped_rows > 0:
            warnings.append(f"Skipped {skipped_rows} non-transaction or malformed line(s).")

        return transactions, warnings

    def _parse_transaction_line(self, line: str) -> Optional[Tuple[str, str, List[str]]]:
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

    def _is_ocr_available(self) -> bool:
        """
        Checks whether pytesseract and the tesseract binary are installed and available.
        """
        try:
            import pytesseract  # type: ignore # noqa: F401
            return shutil.which("tesseract") is not None
        except ImportError:
            return False

    def _attempt_ocr_extraction(
        self,
        pdf: pdfplumber.pdf.PDF,
        filename: str,
        dayfirst: Optional[bool]
    ) -> ExtractionResult:
        """
        Fallback OCR extraction when OCR engine is available.
        """
        return ExtractionResult(
            success=False,
            filename=filename,
            file_type="pdf",
            total_transactions=0,
            transactions=[],
            warnings=[],
            error="OCR processing is not enabled in this environment."
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

