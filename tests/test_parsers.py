"""
Unit tests for Statement Parsers: CSV & Excel Parsers.
Uses synthetic bank statement data to verify extraction, column mapping,
encoding handling, delimiter detection, and schema standardization.
"""
import io
import datetime
import logging
from types import SimpleNamespace
import openpyxl
import pypdf
import pytest

from services.csv_parser import CSVParser, parse_csv
from services.excel_parser import ExcelParser, parse_excel
from services.pdf_parser import PDFParser, is_scanned_pdf, parse_pdf
from models.transaction import Transaction, ExtractionResult
from services.transaction_normalizer import (
    extract_currency,
    correct_dates_from_leading_descriptions,
    generate_transaction_id,
    infer_date_convention,
    normalize_date,
    parse_numeric_amount,
)


class TestTransactionDateNormalization:
    def test_ambiguous_date_is_not_guessed_from_currency(self):
        convention, warnings = infer_date_convention(
            ["05/06/2024"], currency="LKR"
        )

        assert convention is None
        assert normalize_date("05/06/2024", dayfirst=convention) is None
        assert any("will be skipped" in warning for warning in warnings)

    @pytest.mark.parametrize("partial_date", ["Jan 5", "January 2024", "2024-01"])
    def test_partial_date_does_not_get_missing_components_invented(self, partial_date):
        assert normalize_date(partial_date) is None

    @pytest.mark.parametrize(
        ("complete_date", "expected"),
        [
            ("5 Jan 2024", "2024-01-05"),
            ("2024-01-05 13:15:00", "2024-01-05"),
            ("13/04/2024", "2024-04-13"),
        ],
    )
    def test_complete_unambiguous_date_formats_are_preserved(
        self, complete_date, expected
    ):
        assert normalize_date(complete_date) == expected

    def test_leading_description_dates_use_statement_day_first_evidence(self):
        transactions = [
            Transaction(
                transaction_id="dmy-anchor",
                transaction_date="2021-05-31",
                description="31-05-2021 Anchor",
                amount=10,
                currency=None,
                transaction_type="expense",
            ),
            Transaction(
                transaction_id="dmy-ambiguous",
                transaction_date="2021-06-02",
                description="06-02-2021 Purchase",
                amount=10,
                currency=None,
                transaction_type="expense",
            ),
        ]

        correct_dates_from_leading_descriptions(transactions)

        assert transactions[1].transaction_date == "2021-02-06"

    def test_leading_ambiguous_description_date_is_not_guessed(self):
        transaction = Transaction(
            transaction_id="ambiguous",
            transaction_date="2021-01-06",
            description="06-01-2021 Purchase",
            amount=10,
            currency=None,
            transaction_type="expense",
        )

        correct_dates_from_leading_descriptions([transaction])

        assert transaction.transaction_date == "2021-01-06"


class TestCSVParser:
    """Test suite for CSV bank statement parsing."""

    @pytest.mark.parametrize(
        ("currency_text", "expected"),
        [
            ("USD 1,234.56", "USD"),
            ("EUR 1.234,56", "EUR"),
            ("GBP 1,234.56", "GBP"),
            ("INR 1,234.56", "INR"),
            ("AED 1,234.56", "AED"),
            ("AUD 1,234.56", "AUD"),
            ("CAD 1,234.56", "CAD"),
            ("JPY 1,234", "JPY"),
            ("LKR 1,234.56", "LKR"),
            ("CHF 1'234.56", "CHF"),
            ("€ 1.234,56", "EUR"),
            ("£1,234.56", "GBP"),
            ("₹1,234.56", "INR"),
            ("US$1,234.56", "USD"),
            ("A$1,234.56", "AUD"),
            ("C$1,234.56", "CAD"),
            ("د.إ 1,234.56", "AED"),
            ("United Arab Emirates Dirham", "AED"),
        ],
    )
    def test_currency_detection_normalizes_explicit_formats(
        self, currency_text, expected
    ):
        assert extract_currency(currency_text) == expected

    @pytest.mark.parametrize(
        "ambiguous_currency", ["$1,234.56", "¥1,234", "Rs. 1,234.56"]
    )
    def test_ambiguous_currency_markers_are_not_guessed(self, ambiguous_currency):
        assert extract_currency(ambiguous_currency) is None

    def test_currency_detection_does_not_treat_common_words_as_currency_codes(self):
        assert extract_currency("All accounts are shown below") is None

    @pytest.mark.parametrize(
        ("raw_amount", "expected"),
        [
            ("USD 1,234.56", 1234.56),
            ("EUR 1.234,56", 1234.56),
            ("LKR 1,234.56", 1234.56),
            ("AED 1,234", 1234.0),
            ("US$1,234.56", 1234.56),
            ("CHF 1'234.56", 1234.56),
        ],
    )
    def test_currency_number_formats_parse_without_separator_confusion(
        self, raw_amount, expected
    ):
        amount, _, _ = parse_numeric_amount(raw_amount)
        assert amount == expected

    def test_transaction_fingerprint_uses_financial_identity_fields(self):
        first_id = generate_transaction_id(
            "first.csv", 0, "2024-01-01", "  Market   purchase ", 10.0,
            "expense", "LKR",
        )

        assert first_id == generate_transaction_id(
            "first.csv", 0, "2024-01-01", "market purchase", 10.0,
            "expense", "lkr",
        )
        assert first_id != generate_transaction_id(
            "first.csv", 0, "2024-01-01", "market purchase", 10.0,
            "income", "LKR",
        )
        assert first_id != generate_transaction_id(
            "first.csv", 0, "2024-01-01", "market purchase", 10.0,
            "expense", "USD",
        )
        assert first_id != generate_transaction_id(
            "first.csv", 1, "2024-01-01", "market purchase", 10.0,
            "expense", "LKR",
        )

    def test_standard_csv_with_signed_amounts(self):
        """Test standard comma-delimited CSV with single Amount column."""
        csv_data = """Date,Description,Amount,Balance
2024-01-05,Grocery Store Supermarket,-85.50,1414.50
2024-01-10,Monthly Salary Direct Deposit,3500.00,4914.50
2024-01-15,Electric Utility Bill,-120.00,4794.50
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="test_standard.csv")

        assert result.success is True
        assert result.file_type == "csv"
        assert result.total_transactions == 3
        assert len(result.transactions) == 3

        # Check expense row
        t1 = result.transactions[0]
        assert t1.transaction_date == "2024-01-05"
        assert t1.description == "Grocery Store Supermarket"
        assert t1.amount == 85.50
        assert t1.transaction_type == "expense"
        assert t1.balance == 1414.50
        assert t1.source_dataset == "test_standard.csv"
        assert t1.transaction_id.startswith("txn_")

        # Check income row
        t2 = result.transactions[1]
        assert t2.transaction_date == "2024-01-10"
        assert t2.description == "Monthly Salary Direct Deposit"
        assert t2.amount == 3500.00
        assert t2.transaction_type == "income"
        assert t2.balance == 4914.50

    def test_csv_without_explicit_currency_keeps_currency_null(self):
        csv_data = b"Date,Description,Amount\n2026-03-01,Local Store,-12.50\n"
        result = parse_csv(csv_data, filename="no_currency.csv")

        assert result.success is True
        assert len(result.transactions) == 1
        assert result.transactions[0].currency is None

    def test_csv_with_separate_debit_credit_columns(self):
        """Test CSV with separate Debit and Credit columns and currency symbol."""
        csv_data = """Currency: USD
Transaction Date,Details,Debit,Credit,Balance
15/01/2024,Coffee Shop,$4.50,,1000.00
16/01/2024,Freelance Payment,,$500.00,1500.00
17/01/2024,Bookstore Purchase,$29.99,,1470.01
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="debit_credit.csv")

        assert result.success is True
        assert result.total_transactions == 3

        # Debit should be expense
        t1 = result.transactions[0]
        assert t1.transaction_date == "2024-01-15"
        assert t1.amount == 4.50
        assert t1.transaction_type == "expense"
        assert t1.currency == "USD"

        # Credit should be income
        t2 = result.transactions[1]
        assert t2.transaction_date == "2024-01-16"
        assert t2.amount == 500.00
        assert t2.transaction_type == "income"
        assert t2.currency == "USD"

    def test_csv_with_semicolon_delimiter_and_european_numbers(self):
        """Test semicolon delimiter with European number formatting (comma decimal)."""
        csv_data = """Date;Narration;Debit;Credit;Balance
01.02.2024;Internet Provider;45,50;;954,50
05.02.2024;Tax Refund;;250,00;1204,50
"""
        result = parse_csv(
            csv_data.encode("utf-8"), filename="european.csv", dayfirst=True
        )

        assert result.success is True
        assert result.total_transactions == 2
        assert result.transactions[0].amount == 45.50
        assert result.transactions[0].transaction_type == "expense"
        assert result.transactions[1].amount == 250.00
        assert result.transactions[1].transaction_type == "income"

    def test_csv_skips_ambiguous_dates_without_convention(self):
        csv_data = (
            "Date,Description,Amount\n"
            "05/06/2024,Ambiguous purchase,-10.00\n"
        )

        result = parse_csv(csv_data.encode("utf-8"), filename="ambiguous.csv")

        assert result.success is True
        assert result.transactions == []
        assert any("will be skipped" in warning for warning in result.warnings)

    def test_csv_with_metadata_headers(self):
        """Test CSV where transaction table starts after metadata/summary lines."""
        csv_data = """Global Bank Statement
Account Number: XXXX-XXXX-1234
Period: 01/01/2024 to 31/01/2024
Currency: USD

Date,Particulars,Amount,Type
2024-01-02,Online Shopping,49.99,DEBIT
2024-01-08,Client Wire,1200.00,CREDIT
2024-01-12,ATM Withdrawal,100.00,EXPENSE
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="bank_statement.csv")

        assert result.success is True
        assert result.total_transactions == 3
        assert result.transactions[0].description == "Online Shopping"
        assert result.transactions[0].transaction_type == "expense"
        assert result.transactions[0].currency == "USD"
        assert result.transactions[1].transaction_type == "income"
        assert result.transactions[2].transaction_type == "expense"

    def test_csv_with_latin1_encoding(self):
        """Test CSV with Latin-1 (ISO-8859-1) encoding containing accents."""
        csv_data = "Date,Description,Amount\n2024-01-03,Café & Crêperie,-15.50\n"
        encoded_data = csv_data.encode("latin-1")

        result = parse_csv(encoded_data, filename="french.csv")
        assert result.success is True
        assert result.total_transactions == 1
        assert "Café & Crêperie" in result.transactions[0].description
        assert result.transactions[0].amount == 15.50

    def test_empty_csv_file(self):
        """Test rejection of empty CSV file."""
        result = parse_csv(b"", filename="empty.csv")
        assert result.success is False
        assert "empty" in (result.error or "").lower()
        assert result.total_transactions == 0

    def test_csv_with_only_headers(self):
        """Test CSV with header row but zero transaction rows."""
        csv_data = "Date,Description,Amount,Balance\n"
        result = parse_csv(csv_data.encode("utf-8"), filename="no_data.csv")
        assert result.success is True
        assert result.total_transactions == 0
        assert len(result.transactions) == 0

    def test_csv_missing_date_column(self):
        """Test CSV missing Date column."""
        csv_data = "Description,Amount,Balance\nItem 1,50.00,100.00\n"
        result = parse_csv(csv_data.encode("utf-8"), filename="missing_date.csv")
        assert result.success is False
        assert "required" in (result.error or "").lower()

    def test_csv_missing_amount_column(self):
        """Test CSV missing Amount/Debit/Credit columns."""
        csv_data = "Date,Description,Category\n2024-01-01,Groceries,Food\n"
        result = parse_csv(csv_data.encode("utf-8"), filename="missing_amount.csv")
        assert result.success is False
        assert "required" in (result.error or "").lower()

    def test_malformed_and_summary_rows_skipped_safely(self):
        """Test that subtotal, footer, and malformed rows are skipped safely with warnings."""
        csv_data = """Date,Description,Amount
2024-01-01,Gas Station,-45.00
InvalidDate,Corrupted Row,not_a_number
TOTAL,, -45.00
2024-01-02,Supermarket,-75.20
Thank you for banking with us,,
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="with_footers.csv")
        assert result.success is True
        assert result.total_transactions == 2
        assert len(result.warnings) > 0
        assert result.transactions[0].description == "Gas Station"
        assert result.transactions[1].description == "Supermarket"

    def test_description_based_categories_and_transfer_type(self):
        csv_data = """Date,Description,Amount
2026-03-01,Monthly Payroll,150000.00
2026-03-02,Keells Supermarket,-2500.00
2026-03-03,PickMe ride,-850.00
2026-03-04,Dialog utility bill,-1800.00
2026-03-05,Daraz shopping,-3200.00
2026-03-06,City pharmacy,-450.00
2026-03-07,Bank service fee,-50.00
2026-03-08,Internal transfer,-1000.00
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="categories.csv")

        assert result.success is True
        assert [txn.category for txn in result.transactions] == [
            "Salary",
            "Food",
            "Transport",
            "Utilities",
            "Shopping",
            "Healthcare",
            "Fees",
            "Transfer",
        ]
        assert result.transactions[0].transaction_type == "income"
        assert result.transactions[-1].transaction_type == "transfer"

    def test_rows_with_both_debit_and_credit_values_are_rejected(self):
        csv_data = """Date,Description,Debit,Credit,Balance
2026-03-01,Malformed row,100.00,200.00,1100.00
2026-03-02,Valid debit,25.00,,1075.00
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="bad_columns.csv")

        assert result.success is True
        assert result.total_transactions == 1
        assert result.transactions[0].description == "Valid debit"
        assert result.transactions[0].transaction_type == "expense"
        assert any("skipped" in warning.lower() for warning in result.warnings)

    def test_tsv_and_pipe_delimited(self):
        """Test tab-separated (TSV) and pipe-separated bank files."""
        tsv_data = "Posting Date\tPayee\tAmount\n2024-03-01\tCoffee Roasters\t-4.75\n"
        result_tsv = parse_csv(tsv_data.encode("utf-8"), filename="statement.tsv")
        assert result_tsv.success is True
        assert result_tsv.total_transactions == 1
        assert result_tsv.transactions[0].description == "Coffee Roasters"
        assert result_tsv.transactions[0].amount == 4.75
        assert result_tsv.transactions[0].transaction_type == "expense"

        pipe_data = "Txn Date|Narration|Amount|Balance\n2024-03-02|Gym Membership|-45.00|850.00\n"
        result_pipe = parse_csv(pipe_data.encode("utf-8"), filename="statement.txt")
        assert result_pipe.success is True
        assert result_pipe.total_transactions == 1
        assert result_pipe.transactions[0].description == "Gym Membership"

    def test_utf8_bom_encoding(self):
        """Test CSV with UTF-8 BOM encoding (standard Excel export)."""
        csv_data = "\ufeffDate,Description,Amount\n2024-01-10,Internet Subscription,-60.00\n"
        result = parse_csv(csv_data.encode("utf-8-sig"), filename="excel_export.csv")
        assert result.success is True
        assert result.total_transactions == 1
        assert result.transactions[0].transaction_date == "2024-01-10"
        assert result.transactions[0].amount == 60.00
        assert result.transactions[0].transaction_type == "expense"

    def test_parentheses_and_dr_cr_in_amounts(self):
        """Test parentheses (accounting negative) and DR/CR suffixes in amounts."""
        csv_data = """Date,Details,Amount
15 Jan 2024,Medical Clinic,(150.00)
16 Jan 2024,Consulting Fee,1200.00 CR
17 Jan 2024,Utility Charge,85.20 DR
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="accounting.csv")
        assert result.success is True
        assert result.total_transactions == 3

        # (150.00) -> expense
        assert result.transactions[0].amount == 150.00
        assert result.transactions[0].transaction_type == "expense"
        assert result.transactions[0].transaction_date == "2024-01-15"

        # 1200.00 CR -> income
        assert result.transactions[1].amount == 1200.00
        assert result.transactions[1].transaction_type == "income"

        # 85.20 DR -> expense
        assert result.transactions[2].amount == 85.20
        assert result.transactions[2].transaction_type == "expense"

    def test_explicit_amount_direction_overrides_description_keywords(self):
        csv_data = """Date,Description,Amount
2024-01-16,Consulting Fee,1200.00 CR
2024-01-17,International transfer,-250.00
2024-01-18,Service Fee,+12.00
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="directions.csv")

        assert result.success is True
        assert [txn.transaction_type for txn in result.transactions] == [
            "income",
            "expense",
            "income",
        ]

    def test_zero_amount_is_rejected_with_warning(self):
        csv_data = """Date,Description,Amount
2024-01-01,Zero amount,0.00
2024-01-02,Valid purchase,-5.00
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="zero.csv")

        assert result.success is True
        assert result.total_transactions == 1
        assert result.transactions[0].description == "Valid purchase"
        assert any("skipped" in warning.lower() for warning in result.warnings)

    def test_invalid_amount_text_is_rejected_instead_of_sanitized(self):
        csv_data = """Date,Description,Amount
2024-01-01,Corrupted amount,USD 1O0
2024-01-02,Combined columns,100 200
2024-01-03,Misgrouped amount,12.34.56
2024-01-04,Valid amount,"LKR 1,000.00"
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="invalid-amount.csv")

        assert result.success is True
        assert result.total_transactions == 1
        assert result.transactions[0].description == "Valid amount"
        assert result.transactions[0].amount == 1000.0
        assert result.transactions[0].currency == "LKR"
        assert any("skipped" in warning.lower() for warning in result.warnings)

    def test_category_and_currency_preservation(self):
        """Test that explicit category and currency columns are preserved."""
        csv_data = """Date,Description,Amount,Category,Currency
2024-02-15,Delta Airlines,-340.50,Travel,USD
2024-02-16,Dividend Payout,55.00,Investments,USD
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="categorized.csv")
        assert result.success is True
        assert result.total_transactions == 2
        assert result.transactions[0].category == "Travel"
        assert result.transactions[0].currency == "USD"
        assert result.transactions[1].category == "Investments"
        assert result.transactions[1].currency == "USD"

    def test_ambiguous_date_with_statement_level_resolution(self):
        """
        Verify that ambiguous dates like 03/04/2026 are not blindly guessed.
        When another row has 15/04/2026 (15 > 12), the entire file is proven
        to be DD/MM/YYYY, so 03/04/2026 parses as 2026-04-03 (April 3).
        """
        csv_data = """Date,Description,Amount
03/04/2026,Pharmacy,-15.00
15/04/2026,Supermarket,-50.00
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="dates.csv")
        assert result.success is True
        assert result.total_transactions == 2
        assert result.transactions[0].transaction_date == "2026-04-03"
        assert result.transactions[1].transaction_date == "2026-04-15"

    def test_ambiguous_date_generates_warning_and_supports_override(self):
        """
        When all dates are ambiguous (e.g. only 03/04/2026), ensure an explicit
        warning is generated in the result rather than silently guessing,
        and verify caller override via dayfirst parameter.
        """
        csv_data = """Date,Description,Amount
03/04/2026,One-off Payment,-25.00
"""
        # Without override: generates warning
        res_default = parse_csv(csv_data.encode("utf-8"), filename="ambiguous.csv")
        assert res_default.success is True
        assert any("Ambiguous date" in w for w in res_default.warnings)

        # With explicit dayfirst=True override -> April 3, 2026
        res_dmy = parse_csv(csv_data.encode("utf-8"), filename="ambiguous.csv", dayfirst=True)
        assert res_dmy.transactions[0].transaction_date == "2026-04-03"

        # With explicit dayfirst=False override -> March 4, 2026
        res_mdy = parse_csv(csv_data.encode("utf-8"), filename="ambiguous.csv", dayfirst=False)
        assert res_mdy.transactions[0].transaction_date == "2026-03-04"

    def test_canonical_schema_conformity(self):
        """
        Verify that to_canonical_dict() strictly produces the 8 fields
        documented in the repository README.md.
        """
        csv_data = "Date,Description,Amount,Balance\n2024-01-01,Groceries,-50.00,1000.00\n"
        result = parse_csv(csv_data.encode("utf-8"), filename="schema_test.csv")
        assert result.success is True
        txn = result.transactions[0]

        canonical = txn.to_canonical_dict()
        expected_fields = {
            "transaction_id",
            "transaction_date",
            "description",
            "amount",
            "currency",
            "transaction_type",
            "category",
            "source_dataset"
        }
        assert set(canonical.keys()) == expected_fields
        assert "balance" not in canonical
        assert canonical["amount"] == 50.00
        assert canonical["transaction_type"] == "expense"
        assert canonical["transaction_date"] == "2024-01-01"

    def test_extraction_result_serialization_strictly_has_eight_canonical_fields(self):
        """
        Verify that serializing ExtractionResult via model_dump() (used by FastAPI JSON response)
        strictly produces only the 8 canonical fields per transaction, omitting balance.
        """
        csv_data = "Date,Description,Amount,Balance\n2024-01-01,Groceries,-50.00,1000.00\n"
        result = parse_csv(csv_data.encode("utf-8"), filename="schema_test.csv")
        dumped = result.model_dump()

        assert "transactions" in dumped
        first_txn = dumped["transactions"][0]
        expected_fields = {
            "transaction_id",
            "transaction_date",
            "description",
            "amount",
            "currency",
            "transaction_type",
            "category",
            "source_dataset"
        }
        assert set(first_txn.keys()) == expected_fields
        assert "balance" not in first_txn

    def test_explicit_dayfirst_priority_and_contradiction_warning(self):
        """
        Verify that an explicitly supplied dayfirst value strictly takes priority
        over automatic inference, and generates a warning when statement data contradicts it,
        strictly rejecting contradictory rows.
        """
        # File has 15/04/2026 (which would auto-infer dayfirst=True)
        # But caller explicitly forces dayfirst=False (MM/DD)
        csv_data = """Date,Description,Amount
03/04/2026,Medical,-100.00
15/04/2026,Supermarket,-50.00
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="forced_format.csv", dayfirst=False)

        # 1. Total transactions is 1 because 15/04/2026 (month 15) is invalid under MM/DD and is rejected
        assert result.total_transactions == 1
        assert len(result.transactions) == 1

        # 2. 03/04/2026 is parsed as March 4 (2026-03-04) based on dayfirst=False priority
        assert result.transactions[0].transaction_date == "2026-03-04"
        assert result.transactions[0].description == "Medical"

        # 3. Contradiction with 15/04/2026 is explicitly warned
        assert any("contradictory" in w.lower() for w in result.warnings)

    def test_inconsistent_date_conventions_flagged(self):
        """
        Verify that inconsistent date conventions across rows (found both DD > 12 and MM > 12)
        are flagged with an explicit warning and the conflicting rows are rejected.
        """
        csv_data = """Date,Description,Amount
15/01/2024,First Row,-20.00
01/25/2024,Second Row,-30.00
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="mixed_dates.csv")

        # 1. Mixed convention warning is generated
        assert any("mixed" in w.lower() or "inconsistent" in w.lower() for w in result.warnings)

        # 2. Statement convention inferred as DD/MM/YYYY; 01/25/2024 has month=25 and is rejected
        assert result.total_transactions == 1
        assert len(result.transactions) == 1
        assert result.transactions[0].description == "First Row"
        assert result.transactions[0].transaction_date == "2024-01-15"

    def test_invalid_calendar_dates_rejected(self):
        """
        Verify that invalid calendar dates (e.g., February 31 or Feb 29 in non-leap year)
        are strictly rejected without fabricating or silently adjusting dates.
        """
        csv_data = """Date,Description,Amount
31/02/2026,Invalid Feb 31,-50.00
29/02/2025,Invalid Leap Day 2025,-75.00
29/02/2024,Valid Leap Day 2024,-100.00
31/04/2026,Invalid April 31,-40.00
"""
        result = parse_csv(csv_data.encode("utf-8"), filename="invalid_dates.csv", dayfirst=True)

        assert result.success is True
        # Only 29/02/2024 is a real calendar date (2024 is a leap year; April and Feb do not have 31 days)
        assert result.total_transactions == 1
        assert len(result.transactions) == 1
        assert result.transactions[0].description == "Valid Leap Day 2024"
        assert result.transactions[0].transaction_date == "2024-02-29"
        assert any("skipped" in w.lower() for w in result.warnings)


def create_test_xlsx(rows, sheet_name="Statement", extra_sheets=None) -> bytes:
    """Helper to generate an in-memory .xlsx file bytes."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    for r in rows:
        ws.append(r)
    if extra_sheets:
        for name, sheet_rows in extra_sheets.items():
            new_ws = wb.create_sheet(title=name)
            for r in sheet_rows:
                new_ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestExcelParser:
    """Test suite for Excel bank statement parsing (.xlsx and .xls)."""

    def test_standard_xlsx_with_signed_amounts(self):
        """Test standard .xlsx file with single Amount column."""
        rows = [
            ["Date", "Description", "Amount", "Balance"],
            [datetime.date(2024, 1, 5), "Grocery Store Supermarket", -85.50, 1414.50],
            [datetime.date(2024, 1, 10), "Monthly Salary Direct Deposit", 3500.00, 4914.50],
            [datetime.date(2024, 1, 15), "Electric Utility Bill", -120.00, 4794.50],
        ]
        xlsx_bytes = create_test_xlsx(rows)
        result = parse_excel(xlsx_bytes, filename="standard.xlsx")

        assert result.success is True
        assert result.file_type == "excel"
        assert result.total_transactions == 3
        assert len(result.transactions) == 3

        # Expense row
        t1 = result.transactions[0]
        assert t1.transaction_date == "2024-01-05"
        assert t1.description == "Grocery Store Supermarket"
        assert t1.amount == 85.50
        assert t1.transaction_type == "expense"
        assert t1.balance == 1414.50
        assert t1.source_dataset == "standard.xlsx"
        assert t1.transaction_id.startswith("txn_")

        # Income row
        t2 = result.transactions[1]
        assert t2.transaction_date == "2024-01-10"
        assert t2.description == "Monthly Salary Direct Deposit"
        assert t2.amount == 3500.00
        assert t2.transaction_type == "income"
        assert t2.balance == 4914.50

    def test_xlsx_with_separate_debit_credit_and_balance(self):
        """Test .xlsx file with separate Debit and Credit columns and Balance."""
        rows = [
            ["Transaction Date", "Details", "Debit", "Credit", "Balance"],
            [datetime.date(2024, 1, 15), "Coffee Shop", 4.50, None, 1000.00],
            [datetime.date(2024, 1, 16), "Freelance Payment", None, 500.00, 1500.00],
            [datetime.date(2024, 1, 17), "Bookstore Purchase", 29.99, None, 1470.01],
        ]
        xlsx_bytes = create_test_xlsx(rows)
        result = parse_excel(xlsx_bytes, filename="debit_credit.xlsx")

        assert result.success is True
        assert result.total_transactions == 3

        # Debit should be expense
        t1 = result.transactions[0]
        assert t1.transaction_date == "2024-01-15"
        assert t1.amount == 4.50
        assert t1.transaction_type == "expense"
        assert t1.balance == 1000.00

        # Credit should be income
        t2 = result.transactions[1]
        assert t2.transaction_date == "2024-01-16"
        assert t2.amount == 500.00
        assert t2.transaction_type == "income"
        assert t2.balance == 1500.00

    def test_xlsx_with_metadata_headers(self):
        """Test .xlsx with leading metadata rows before the actual transaction table."""
        rows = [
            ["Global Bank Statement"],
            ["Account Number: XXXX-XXXX-1234"],
            ["Period: 01/01/2024 to 31/01/2024"],
            ["Currency: USD"],
            [],  # Empty row
            ["Date", "Particulars", "Amount", "Type"],
            [datetime.date(2024, 1, 2), "Online Shopping", 49.99, "DEBIT"],
            [datetime.date(2024, 1, 8), "Client Wire", 1200.00, "CREDIT"],
            [datetime.date(2024, 1, 12), "ATM Withdrawal", 100.00, "EXPENSE"],
        ]
        xlsx_bytes = create_test_xlsx(rows)
        result = parse_excel(xlsx_bytes, filename="statement_metadata.xlsx")

        assert result.success is True
        assert result.total_transactions == 3
        assert result.transactions[0].description == "Online Shopping"
        assert result.transactions[0].transaction_type == "expense"
        assert result.transactions[0].currency == "USD"
        assert result.transactions[1].transaction_type == "income"
        assert result.transactions[2].transaction_type == "expense"

    def test_xlsx_multi_sheet_auto_detection(self):
        """Test multi-sheet workbook where the transaction table is on a non-default sheet."""
        overview_rows = [
            ["Welcome to your statement report"],
            ["Please find your transactions on the next sheet"],
        ]
        tx_rows = [
            ["Date", "Description", "Amount"],
            [datetime.date(2024, 2, 1), "Internet Bill", -60.00],
            [datetime.date(2024, 2, 5), "Consulting", 800.00],
        ]
        xlsx_bytes = create_test_xlsx(overview_rows, sheet_name="Overview", extra_sheets={"Transactions": tx_rows})
        result = parse_excel(xlsx_bytes, filename="multi_sheet.xlsx")

        assert result.success is True
        assert result.total_transactions == 2
        assert result.transactions[0].description == "Internet Bill"
        assert result.transactions[1].description == "Consulting"

    def test_xlsx_explicit_sheet_selection(self):
        """Test explicit sheet selection by name and index."""
        tx1 = [
            ["Date", "Description", "Amount"],
            [datetime.date(2024, 1, 1), "Sheet 1 Item", -10.00],
        ]
        tx2 = [
            ["Date", "Description", "Amount"],
            [datetime.date(2024, 1, 2), "Sheet 2 Item", -20.00],
        ]
        xlsx_bytes = create_test_xlsx(tx1, sheet_name="Checking", extra_sheets={"Savings": tx2})

        # By name
        res_savings = parse_excel(xlsx_bytes, filename="accounts.xlsx", sheet_name="Savings")
        assert res_savings.success is True
        assert res_savings.transactions[0].description == "Sheet 2 Item"

        # By index
        res_checking = parse_excel(xlsx_bytes, filename="accounts.xlsx", sheet_name=0)
        assert res_checking.success is True
        assert res_checking.transactions[0].description == "Sheet 1 Item"

        # Non-existent sheet returns error
        res_err = parse_excel(xlsx_bytes, filename="accounts.xlsx", sheet_name="CreditCard")
        assert res_err.success is False
        assert "not found" in (res_err.error or "").lower()

    def test_xlsx_with_numeric_serial_dates(self):
        """Test that Excel numeric serial date numbers (e.g. 45300) are converted correctly."""
        rows = [
            ["Date", "Description", "Amount"],
            [45300, "Serial Date Transaction", -75.00],
        ]
        xlsx_bytes = create_test_xlsx(rows)
        result = parse_excel(xlsx_bytes, filename="serials.xlsx")

        assert result.success is True
        assert result.total_transactions == 1
        assert result.transactions[0].transaction_date == "2024-01-09"
        assert result.transactions[0].amount == 75.00

    def test_empty_xlsx_file(self):
        """Test rejection of 0-byte Excel file."""
        result = parse_excel(b"", filename="empty.xlsx")
        assert result.success is False
        assert "empty" in (result.error or "").lower()

    def test_corrupted_xlsx_file(self):
        """Test corrupted binary file with .xlsx extension."""
        corrupted_bytes = b"PK\x03\x04CorruptedNonZipData"
        result = parse_excel(corrupted_bytes, filename="corrupted.xlsx")
        assert result.success is False
        assert "invalid" in (result.error or "").lower() or "corrupted" in (result.error or "").lower()

    def test_empty_worksheet(self):
        """Test worksheet with 0 rows."""
        xlsx_bytes = create_test_xlsx([])
        result = parse_excel(xlsx_bytes, filename="empty_sheet.xlsx")
        assert result.success is False
        assert "could not identify" in (result.error or "").lower() or "empty" in (result.error or "").lower()

    def test_xlsx_with_only_headers(self):
        """Test worksheet with headers but no transactions."""
        rows = [["Date", "Description", "Amount", "Balance"]]
        xlsx_bytes = create_test_xlsx(rows)
        result = parse_excel(xlsx_bytes, filename="only_headers.xlsx")

        assert result.success is True
        assert result.total_transactions == 0
        assert len(result.transactions) == 0

    def test_xlsx_missing_required_columns(self):
        """Test worksheet missing required Date or Amount columns."""
        rows_no_date = [
            ["Description", "Amount", "Balance"],
            ["Item 1", -50.00, 100.00],
        ]
        result1 = parse_excel(create_test_xlsx(rows_no_date), filename="no_date.xlsx")
        assert result1.success is False
        assert "required" in (result1.error or "").lower()

        rows_no_amt = [
            ["Date", "Description", "Category"],
            [datetime.date(2024, 1, 1), "Item 1", "Food"],
        ]
        result2 = parse_excel(create_test_xlsx(rows_no_amt), filename="no_amt.xlsx")
        assert result2.success is False
        assert "required" in (result2.error or "").lower()

    def test_corrupted_legacy_xls_error(self):
        """Test malformed legacy .xls content is rejected without transactions."""
        result = parse_excel(b"FakeLegacyXlsContent", filename="statement.xls")
        assert result.success is False
        assert result.total_transactions == 0
        assert len(result.transactions) == 0
        assert "legacy excel" in (result.error or "").lower()

    def test_xlsx_canonical_schema_serialization(self):
        """Test that model_dump() strictly outputs the 8 canonical fields for Excel extractions."""
        rows = [
            ["Date", "Description", "Amount", "Balance"],
            [datetime.date(2024, 1, 1), "Groceries", -50.00, 1000.00],
        ]
        xlsx_bytes = create_test_xlsx(rows)
        result = parse_excel(xlsx_bytes, filename="schema_test.xlsx")
        assert result.success is True

        dumped = result.model_dump()
        first_txn = dumped["transactions"][0]
        expected_fields = {
            "transaction_id",
            "transaction_date",
            "description",
            "amount",
            "currency",
            "transaction_type",
            "category",
            "source_dataset"
        }
        assert set(first_txn.keys()) == expected_fields
        assert "balance" not in first_txn

    def test_xlsx_summary_and_invalid_rows_skipped_safely(self):
        """Test that invalid calendar dates and subtotal rows are skipped safely with warnings."""
        rows = [
            ["Date", "Description", "Amount"],
            [datetime.date(2024, 1, 1), "Gas Station", -45.00],
            ["31/02/2026", "Invalid Date Feb 31", -10.00],
            ["TOTAL", "Summary row", -55.00],
            [datetime.date(2024, 1, 2), "Supermarket", -75.20],
        ]
        xlsx_bytes = create_test_xlsx(rows)
        result = parse_excel(xlsx_bytes, filename="with_invalid.xlsx")

        assert result.success is True
        assert result.total_transactions == 2
        assert result.transactions[0].description == "Gas Station"
        assert result.transactions[1].description == "Supermarket"
        assert len(result.warnings) > 0


def create_test_text_pdf(lines: list[str]) -> bytes:
    """Helper to synthesize a standard single-page PDF containing text lines."""
    stream_lines = []
    y = 720
    for line in lines:
        safe_line = str(line).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_lines.append(f"BT /F1 10 Tf 50 {y} Td ({safe_line}) Tj ET")
        y -= 20
    content_bytes = "\n".join(stream_lines).encode("latin-1")

    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj4 = b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    obj5 = b"5 0 obj\n<< /Length " + str(len(content_bytes)).encode("ascii") + b" >>\nstream\n" + content_bytes + b"\nendstream\nendobj\n"
    obj3 = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"

    header = b"%PDF-1.4\n"
    offsets = [0]
    pos = len(header)
    for o in [obj1, obj2, obj3, obj4, obj5]:
        offsets.append(pos)
        pos += len(o)

    xref = f"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n"
    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{pos}\n%%EOF\n"
    return header + obj1 + obj2 + obj3 + obj4 + obj5 + xref.encode("ascii") + trailer.encode("ascii")


def create_test_table_pdf(
    headers: list[str],
    rows: list[list[str]],
    col_widths: list[float] | None = None,
    page_size: tuple[float, float] = (612.0, 792.0),
) -> bytes:
    """Helper to synthesize a PDF with a bordered grid table and cell contents."""
    if col_widths is None:
        if len(headers) == 5:
            col_widths = [75.0, 175.0, 80.0, 80.0, 90.0]
        elif len(headers) == 4:
            col_widths = [80.0, 220.0, 100.0, 100.0]
        else:
            col_widths = [500.0 / len(headers)] * len(headers)

    total_width = sum(col_widths)
    x_start = 50.0
    y_start = page_size[1] - 92.0
    row_height = 20.0

    col_x = [x_start]
    for w in col_widths:
        col_x.append(col_x[-1] + w)

    stream_lines = []
    # Write headers
    for c_idx, h in enumerate(headers):
        x = col_x[c_idx] + 4.0
        safe = str(h).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_lines.append(f"BT /F1 8 Tf {x} {y_start - 14} Td ({safe}) Tj ET")

    # Write data rows
    y = y_start - row_height
    for row in rows:
        for c_idx, val in enumerate(row):
            x = col_x[c_idx] + 4.0
            safe = str(val).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_lines.append(f"BT /F1 8 Tf {x} {y - 14} Td ({safe}) Tj ET")
        y -= row_height

    total_rows = len(rows) + 1
    y_bottom = y_start - total_rows * row_height

    # Horizontal lines
    for i in range(total_rows + 1):
        line_y = y_start - i * row_height
        stream_lines.append(f"{x_start} {line_y} m {x_start + total_width} {line_y} l S")

    # Vertical lines
    for x in col_x:
        stream_lines.append(f"{x} {y_bottom} m {x} {y_start} l S")

    content_bytes = "\n".join(stream_lines).encode("latin-1")

    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj4 = b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    obj5 = b"5 0 obj\n<< /Length " + str(len(content_bytes)).encode("ascii") + b" >>\nstream\n" + content_bytes + b"\nendstream\nendobj\n"
    obj3 = (
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 "
        f"{page_size[0]} {page_size[1]}] /Resources << /Font << /F1 4 0 R >> "
        ">> /Contents 5 0 R >>\nendobj\n"
    ).encode("ascii")

    header = b"%PDF-1.4\n"
    offsets = [0]
    pos = len(header)
    for o in [obj1, obj2, obj3, obj4, obj5]:
        offsets.append(pos)
        pos += len(o)

    xref = f"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n"
    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{pos}\n%%EOF\n"
    return header + obj1 + obj2 + obj3 + obj4 + obj5 + xref.encode("ascii") + trailer.encode("ascii")


def create_test_multipage_pdf(pages_lines: list[list[str]]) -> bytes:
    """Helper to synthesize a multi-page PDF document."""
    writer = pypdf.PdfWriter()
    for lines in pages_lines:
        single_page_bytes = create_test_text_pdf(lines)
        reader = pypdf.PdfReader(io.BytesIO(single_page_bytes))
        writer.add_page(reader.pages[0])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def create_test_encrypted_pdf(password: str = "secret123") -> bytes:
    """Helper to synthesize an encrypted, password-protected PDF document."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt(password)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def create_test_scanned_pdf() -> bytes:
    """Helper to synthesize a scanned-style PDF with an image XObject and no text characters."""
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<<
  /Type /Page
  /Parent 2 0 R
  /MediaBox [0 0 100 100]
  /Resources <<
    /XObject << /Im1 4 0 R >>
  >>
  /Contents 5 0 R
>>
endobj
4 0 obj
<<
  /Type /XObject
  /Subtype /Image
  /Width 1
  /Height 1
  /ColorSpace /DeviceRGB
  /BitsPerComponent 8
  /Length 3
>>
stream
\xff\x00\x00
endstream
endobj
5 0 obj
<< /Length 17 >>
stream
q 10 0 0 10 0 0 cm /Im1 Do Q
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000373 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
440
%%EOF
"""


class TestPDFParser:
    """Test suite for PDF bank statement parsing."""

    def test_pdf_table_with_debit_credit_and_balance(self):
        """Test structured table PDF with separate Debit, Credit, and Balance columns."""
        headers = ["Date", "Description", "Debit", "Credit", "Balance"]
        rows = [
            ["2024-01-05", "Grocery Store Supermarket", "85.50", "", "1414.50"],
            ["2024-01-10", "Monthly Salary Direct Deposit", "", "3500.00", "4914.50"],
            ["2024-01-15", "Electric Utility Bill", "120.00", "", "4794.50"],
        ]
        pdf_bytes = create_test_table_pdf(headers, rows)
        result = parse_pdf(pdf_bytes, filename="statement_table.pdf")

        assert result.success is True
        assert result.file_type == "pdf"
        assert result.total_transactions == 3
        assert len(result.transactions) == 3

        # Expense verification
        t1 = result.transactions[0]
        assert t1.transaction_date == "2024-01-05"
        assert t1.description == "Grocery Store Supermarket"
        assert t1.amount == 85.50
        assert t1.transaction_type == "expense"
        assert t1.balance == 1414.50
        assert t1.source_dataset == "statement_table.pdf"

        # Income verification
        t2 = result.transactions[1]
        assert t2.transaction_date == "2024-01-10"
        assert t2.description == "Monthly Salary Direct Deposit"
        assert t2.amount == 3500.00
        assert t2.transaction_type == "income"
        assert t2.balance == 4914.50

        # Expense verification
        t3 = result.transactions[2]
        assert t3.transaction_date == "2024-01-15"
        assert t3.amount == 120.00
        assert t3.transaction_type == "expense"
        assert t3.balance == 4794.50

    def test_pdf_reference_table_returns_only_fourteen_transactions(self, monkeypatch):
        headers = [
            "Date", "Description", "Currency", "Reference",
            "Debit", "Credit", "Balance",
        ]
        entries = [
            ("Grocery Store", "expense", 1250.00),
            ("Mobile Reload", "expense", 500.00),
            ("Online Shopping", "expense", 7500.00),
            ("Monthly Salary", "income", 125000.00),
            ("Fuel Station", "expense", 3200.00),
            ("Water Bill", "expense", 1800.00),
            ("Coffee Shop", "expense", 650.00),
            ("Pharmacy", "expense", 2100.00),
            ("Client Payment", "income", 24000.00),
            ("Restaurant", "expense", 3400.00),
            ("Internet Bill", "expense", 4500.00),
            ("Book Shop", "expense", 2750.00),
            ("Taxi Fare", "expense", 1200.00),
            ("Electricity Bill", "expense", 6200.00),
        ]
        running_balance = 100000.00
        rows = []
        expected = []
        for day, (description, transaction_type, amount) in enumerate(entries, start=1):
            debit = f"{amount:.2f}" if transaction_type == "expense" else ""
            credit = f"{amount:.2f}" if transaction_type == "income" else ""
            running_balance += amount if transaction_type == "income" else -amount
            rows.append([
                f"2026-03-{day:02d}",
                description,
                "LKR",
                f"260300{day:02d}",
                debit,
                credit,
                f"{running_balance:.2f}",
            ])
            expected.append((description, amount, transaction_type, running_balance))

        fallback_inputs = {}
        original_positioned = PDFParser._extract_from_positioned_words
        original_text = PDFParser._extract_from_text

        def capture_positioned(self, pages_words, *args, **kwargs):
            fallback_inputs["positioned"] = pages_words
            return original_positioned(self, pages_words, *args, **kwargs)

        def capture_text(self, pages_text, *args, **kwargs):
            fallback_inputs["text"] = pages_text
            return original_text(self, pages_text, *args, **kwargs)

        monkeypatch.setattr(PDFParser, "_extract_from_positioned_words", capture_positioned)
        monkeypatch.setattr(PDFParser, "_extract_from_text", capture_text)
        pdf_bytes = create_test_table_pdf(
            headers,
            rows,
            col_widths=[60.0, 130.0, 55.0, 65.0, 60.0, 60.0, 70.0],
        )

        result = parse_pdf(pdf_bytes, filename="fourteen_transactions.pdf")

        assert result.success is True
        assert result.total_transactions == 14
        assert fallback_inputs == {"positioned": [], "text": []}
        assert [
            (
                transaction.description,
                transaction.amount,
                transaction.transaction_type,
                transaction.balance,
            )
            for transaction in result.transactions
        ] == expected
        assert all(transaction.currency == "LKR" for transaction in result.transactions)
        assert result.transactions[1].category == "Utilities"
        assert result.transactions[2].category == "Shopping"

    def test_pdf_table_with_signed_amounts(self):
        """Test structured table PDF with a single signed Amount column."""
        headers = ["Date", "Description", "Amount", "Balance"]
        rows = [
            ["2024-02-01", "Coffee Shop Cafe", "-4.50", "995.50"],
            ["2024-02-05", "Client Invoice Payment", "1200.00", "2195.50"],
        ]
        pdf_bytes = create_test_table_pdf(headers, rows)
        result = parse_pdf(pdf_bytes, filename="signed_table.pdf")

        assert result.success is True
        assert result.total_transactions == 2
        assert result.transactions[0].amount == 4.50
        assert result.transactions[0].transaction_type == "expense"
        assert result.transactions[1].amount == 1200.00
        assert result.transactions[1].transaction_type == "income"

    def test_pdf_ambiguous_currency_symbol_keeps_currency_null(self):
        pdf_bytes = create_test_table_pdf(
            ["Date", "Description", "Debit", "Credit", "Balance"],
            [["2026-03-01", "Local Store", "$12.50", "", "$987.50"]],
        )

        result = parse_pdf(pdf_bytes, filename="ambiguous_currency.pdf")

        assert result.success is True
        assert result.total_transactions == 1
        assert result.transactions[0].amount == 12.50
        assert result.transactions[0].currency is None

    def test_pdf_transaction_currency_does_not_leak_to_other_rows(self):
        pdf_bytes = create_test_table_pdf(
            ["Date", "Description", "Debit", "Credit", "Balance", "Currency"],
            [
                ["2026-03-01", "Unspecified Store", "12.50", "", "987.50", ""],
                ["2026-03-02", "Euro Store", "10.00", "", "977.50", "EUR"],
            ],
        )

        result = parse_pdf(pdf_bytes, filename="mixed_currency.pdf")

        assert result.success is True
        assert result.transactions[0].currency is None
        assert result.transactions[1].currency == "EUR"

    def test_pdf_text_columns_exclude_balance_and_infer_debit_credit(self):
        lines = [
            "Date Description Debit Credit Balance",
            "2026-03-01 Opening Balance 1000.00 1000.00",
            "2026-03-02 Keells Grocery 25.00 975.00",
            "2026-03-03 Monthly Salary 500.00 1475.00",
        ]
        transactions, warnings = PDFParser()._extract_from_text(
            ["\n".join(lines)], "text-columns.pdf", "LKR", None
        )

        assert warnings == []
        assert len(transactions) == 2
        assert transactions[0].amount == 25.00
        assert transactions[0].balance == 975.00
        assert transactions[0].transaction_type == "expense"
        assert transactions[0].category == "Food"
        assert transactions[1].amount == 500.00
        assert transactions[1].balance == 1475.00
        assert transactions[1].transaction_type == "income"
        assert transactions[1].category == "Salary"

    def test_pdf_text_parser_preserves_numeric_description_suffix(self):
        transactions, warnings = PDFParser()._extract_from_text(
            ["2024-01-01 Transaction 1 -1.00 9.00"],
            "numeric-description.pdf",
            None,
            None,
        )

        assert warnings == []
        assert len(transactions) == 1
        assert transactions[0].description == "Transaction 1"
        assert transactions[0].amount == 1.0
        assert transactions[0].balance == 9.0
        assert transactions[0].transaction_type == "expense"

    def test_pdf_prefers_structured_row_over_conflicting_fallback(self):
        structured = Transaction(
            transaction_id="structured",
            transaction_date="2024-02-01",
            description="Card payment",
            amount=10.0,
            currency="USD",
            transaction_type="expense",
            category="Unknown",
            source_dataset="synthetic.pdf",
            balance=90.0,
        )
        fallback = structured.model_copy(
            update={
                "transaction_id": "fallback",
                "balance": 91.0,
            }
        )

        result = PDFParser._merge_transactions([structured], [fallback])

        assert result == [structured]

    def test_pdf_merge_prefers_table_direction_over_conflicting_fallback(self):
        table_row = Transaction(
            transaction_id="table-row",
            transaction_date="2024-02-01",
            description="Card payment",
            amount=10.0,
            currency="USD",
            transaction_type="expense",
            category="Unknown",
            source_dataset="synthetic.pdf",
        )
        fallback_row = table_row.model_copy(
            update={
                "transaction_id": "fallback-row",
                "transaction_type": "income",
            }
        )

        result = PDFParser._merge_transactions([table_row], [fallback_row])

        assert result == [table_row]

    def test_pdf_merge_preserves_repeated_same_day_transactions(self):
        first = Transaction(
            transaction_id="first",
            transaction_date="2024-02-01",
            description="Market purchase",
            amount=10.0,
            currency="USD",
            transaction_type="expense",
            category="Unknown",
            source_dataset="synthetic.pdf",
        )
        repeated = first.model_copy(update={"transaction_id": "repeated"})
        fallback_matches = [
            first.model_copy(update={"transaction_id": "fallback-1"}),
            first.model_copy(update={"transaction_id": "fallback-2"}),
        ]

        result = PDFParser._merge_transactions(
            [first, repeated], fallback_matches
        )

        assert [transaction.transaction_id for transaction in result] == [
            "first",
            "repeated",
        ]

    def test_pdf_merge_preserves_debit_credit_collisions(self):
        debit = Transaction(
            transaction_id="debit",
            transaction_date="2024-02-01",
            description="Transfer",
            amount=10.0,
            currency="USD",
            transaction_type="expense",
            category="Unknown",
            source_dataset="synthetic.pdf",
        )
        credit = debit.model_copy(
            update={"transaction_id": "credit", "transaction_type": "income"}
        )

        result = PDFParser._merge_transactions([debit, credit])

        assert [transaction.transaction_type for transaction in result] == [
            "expense",
            "income",
        ]

    @pytest.mark.parametrize("rotation", [90, 180, 270])
    def test_pdf_rotated_page_is_normalized_before_extraction(self, rotation):
        source = pypdf.PdfReader(
            io.BytesIO(
                create_test_table_pdf(
                    ["Date", "Description", "Amount", "Balance"],
                    [["2024-02-15", "Rotation test", "-10.00", "90.00"]],
                )
            )
        )
        source.pages[0].rotate(rotation)
        writer = pypdf.PdfWriter(clone_from=source)
        output = io.BytesIO()
        writer.write(output)

        result = parse_pdf(output.getvalue(), filename="rotated.pdf")

        assert result.success is True
        assert result.total_transactions == 1
        assert result.transactions[0].transaction_date == "2024-02-15"
        assert result.transactions[0].description == "Rotation test"
        assert result.transactions[0].transaction_type == "expense"

    def test_pdf_text_fallback_extraction(self):
        """Test fallback to line-based text parsing for non-grid formatted statements."""
        lines = [
            "2024-01-05 Grocery Store Supermarket -85.50 1414.50",
            "2024-01-10 Monthly Salary Direct Deposit 3500.00 4914.50",
            "2024-01-15 Electric Utility Bill -120.00 4794.50"
        ]
        pdf_bytes = create_test_text_pdf(lines)
        result = parse_pdf(pdf_bytes, filename="text_statement.pdf")

        assert result.success is True
        assert result.file_type == "pdf"
        assert result.total_transactions == 3
        assert result.transactions[0].description == "Grocery Store Supermarket"
        assert result.transactions[0].amount == 85.50
        assert result.transactions[0].transaction_type == "expense"
        assert result.transactions[0].balance == 1414.50
        assert result.transactions[1].amount == 3500.00
        assert result.transactions[1].transaction_type == "income"

    def test_pdf_text_with_dr_cr_suffixes(self):
        """Test formatted text statements using CR/DR indicators."""
        lines = [
            "2024-03-01 Pharmacy Medicines 45.00 DR 1000.00",
            "2024-03-02 Consulting Dividend 300.00 CR 1300.00",
        ]
        pdf_bytes = create_test_text_pdf(lines)
        result = parse_pdf(pdf_bytes, filename="drcr_statement.pdf")

        assert result.success is True
        assert result.total_transactions == 2
        assert result.transactions[0].amount == 45.00
        assert result.transactions[0].transaction_type == "expense"
        assert result.transactions[1].amount == 300.00
        assert result.transactions[1].transaction_type == "income"

    def test_pdf_multipage_statement(self):
        """Test multi-page statement parsing aggregating transactions across pages."""
        page1 = [
            "2024-01-01 Page One Transaction -25.00 1000.00",
            "2024-01-02 Page One Paycheck 2000.00 3000.00",
        ]
        page2 = [
            "2024-01-03 Page Two Transaction -75.00 2925.00",
            "2024-01-04 Page Two Utility -110.00 2815.00",
        ]
        pdf_bytes = create_test_multipage_pdf([page1, page2])
        result = parse_pdf(pdf_bytes, filename="multipage.pdf")

        assert result.success is True
        assert result.total_transactions == 4
        assert [t.transaction_date for t in result.transactions] == [
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"
        ]

    def test_pdf_multipage_tables_with_different_column_layouts(self):
        """Preserve each page's own headers when table layouts change."""
        page_tables = [
            (
                ["Date", "Description", "Debit", "Credit", "Balance"],
                [["2024-11-06", "Card purchase", "USD $12.50", "", "$987.50"]],
                {},
            ),
            (
                [
                    "Posting Date",
                    "Details",
                    "Net Amount",
                    "D/C",
                    "Available Balance",
                    "Currency",
                ],
                [["17.11.2024", "Refund", "18.25", "CR", "1,005.75", "EUR"]],
                {
                    "col_widths": [115.0, 155.0, 100.0, 45.0, 150.0, 80.0],
                    "page_size": (842.0, 595.0),
                },
            ),
        ]
        writer = pypdf.PdfWriter()
        for headers, rows, layout in page_tables:
            page_pdf = pypdf.PdfReader(
                io.BytesIO(create_test_table_pdf(headers, rows, **layout))
            )
            writer.add_page(page_pdf.pages[0])
        output = io.BytesIO()
        writer.write(output)

        result = parse_pdf(output.getvalue(), filename="varied-layouts.pdf")

        assert result.success is True
        assert result.total_transactions == 2
        assert [
            (txn.transaction_date, txn.description, txn.amount, txn.transaction_type)
            for txn in result.transactions
        ] == [
            ("2024-11-06", "Card purchase", 12.5, "expense"),
            ("2024-11-17", "Refund", 18.25, "income"),
        ]
        assert result.transactions[0].currency == "USD"
        assert result.transactions[0].balance == 987.5
        assert result.transactions[1].currency == "EUR"
        assert result.transactions[1].balance == 1005.75

    def test_pdf_transaction_row_can_continue_across_pages(self):
        """Join a wrapped transaction line split by a page boundary."""
        pdf_bytes = create_test_multipage_pdf(
            [
                ["2024-12-01 International transfer to"],
                ["savings account -250.00 750.00"],
            ]
        )

        result = parse_pdf(pdf_bytes, filename="wrapped-row.pdf")

        assert result.success is True
        assert result.total_transactions == 1
        assert result.transactions[0].description == (
            "International transfer to savings account"
        )
        assert result.transactions[0].amount == 250.0
        assert result.transactions[0].transaction_type == "expense"
        assert result.transactions[0].balance == 750.0

    def test_pdf_password_protected_encrypted_rejection(self):
        """Test clean rejection of encrypted/password-protected PDF files."""
        enc_bytes = create_test_encrypted_pdf(password="secret123")
        result = parse_pdf(enc_bytes, filename="encrypted.pdf")

        assert result.success is False
        assert result.total_transactions == 0
        assert len(result.transactions) == 0
        assert "password" in (result.error or "").lower()

    def test_pdf_scanned_ocr_unavailable(self, monkeypatch):
        """Test scanned PDFs retain a clear fallback when PaddleOCR is unavailable."""
        def unavailable_ocr_engine():
            raise ModuleNotFoundError("No module named 'paddleocr'")

        monkeypatch.setattr("services.pdf_parser._get_ocr_engine", unavailable_ocr_engine)
        scanned_bytes = create_test_scanned_pdf()
        result = parse_pdf(scanned_bytes, filename="scanned.pdf")

        assert result.success is False
        assert result.total_transactions == 0
        assert len(result.transactions) == 0
        assert "scanned" in (result.error or "").lower()
        assert "not configured" in (result.error or "").lower()

    def test_pdf_scanned_uses_paddle_ocr_and_extracts_transactions(
        self, monkeypatch, caplog
    ):
        """Test PaddleOCR output is grouped into transaction rows and parsed."""
        caplog.set_level(logging.INFO, logger="services.pdf_parser")

        class FakeOCRResult:
            json = {
                "res": {
                    "rec_texts": [
                        "2024-01-05",
                        "Grocery Store",
                        "-85.50",
                        "1414.50",
                    ],
                    "rec_boxes": [
                        [10, 20, 90, 40],
                        [100, 20, 220, 40],
                        [400, 20, 460, 40],
                        [470, 20, 530, 40],
                    ],
                }
            }

        class FakePaddleOCR:
            def predict(self, image):
                assert image.ndim == 3
                return [FakeOCRResult()]

        monkeypatch.setattr(
            "services.pdf_parser._get_ocr_engine", lambda: FakePaddleOCR()
        )
        result = parse_pdf(create_test_scanned_pdf(), filename="scanned.pdf")

        assert result.success is True
        assert result.total_transactions == 1
        assert result.transactions[0].transaction_date == "2024-01-05"
        assert result.transactions[0].description == "Grocery Store"
        assert result.transactions[0].amount == 85.50
        assert result.transactions[0].transaction_type == "expense"
        assert any(
            getattr(record, "stage", None) == "ocr_inference_started"
            for record in caplog.records
        )
        assert any(
            getattr(record, "stage", None) == "transaction_parsing_completed"
            for record in caplog.records
        )

    def test_pdf_mixed_digital_and_scanned_pages(self, monkeypatch):
        class FakeOCRResult:
            json = {
                "res": {
                    "rec_texts": [
                        "2024-01-05 Cash deposit 35.00 CR 125.00"
                    ],
                    "rec_boxes": [[10, 20, 500, 60]],
                }
            }

        class FakePaddleOCR:
            def predict(self, image):
                return [FakeOCRResult()]

        writer = pypdf.PdfWriter()
        digital_page = pypdf.PdfReader(
            io.BytesIO(
                create_test_text_pdf(
                    ["2024-01-03 Online payment -10.00 90.00"]
                )
            )
        ).pages[0]
        scanned_page = pypdf.PdfReader(
            io.BytesIO(create_test_scanned_pdf())
        ).pages[0]
        writer.add_page(digital_page)
        writer.add_page(scanned_page)
        output = io.BytesIO()
        writer.write(output)
        pdf_bytes = output.getvalue()

        monkeypatch.setattr(
            "services.pdf_parser._get_ocr_engine", lambda: FakePaddleOCR()
        )

        assert is_scanned_pdf(pdf_bytes) is True
        result = parse_pdf(pdf_bytes, filename="mixed-pages.pdf")

        assert result.success is True
        assert {
            (txn.transaction_date, txn.description, txn.transaction_type)
            for txn in result.transactions
        } == {
            ("2024-01-03", "Online payment", "expense"),
            ("2024-01-05", "Cash deposit", "income"),
        }

    def test_pdf_scanned_uses_recognition_polygons_when_boxes_are_missing(self, monkeypatch):
        class FakeOCRResult:
            json = {
                "res": {
                    "rec_texts": [
                        "2024-01-05",
                        "Grocery Store",
                        "-85.50",
                        "1414.50",
                    ],
                    "rec_polys": [
                        [[10, 20], [90, 20], [90, 40], [10, 40]],
                        [[100, 20], [220, 20], [220, 40], [100, 40]],
                        [[400, 20], [460, 20], [460, 40], [400, 40]],
                        [[470, 20], [530, 20], [530, 40], [470, 40]],
                    ],
                }
            }

        class FakePaddleOCR:
            def predict(self, image):
                return [FakeOCRResult()]

        monkeypatch.setattr(
            "services.pdf_parser._get_ocr_engine", lambda: FakePaddleOCR()
        )
        result = parse_pdf(create_test_scanned_pdf(), filename="scanned.pdf")

        assert result.success is True
        assert result.total_transactions == 1
        assert result.transactions[0].transaction_date == "2024-01-05"
        assert result.transactions[0].description == "Grocery Store"
        assert result.transactions[0].amount == 85.50
        assert result.transactions[0].transaction_type == "expense"

    def test_scanned_pdf_ocr_caps_rendering_resolution(self, monkeypatch):
        class FakeOCRResult:
            json = {
                "res": {
                    "rec_texts": ["2024-01-05 Grocery Store -85.50 1414.50"],
                    "rec_boxes": [[10, 20, 500, 60]],
                }
            }

        class FakePage:
            width = 612
            height = 792
            resolution = None

            def to_image(self, resolution):
                self.resolution = resolution
                return SimpleNamespace(original=[[[0, 0, 0]]])

        class FakeOCR:
            def predict(self, image):
                return [FakeOCRResult()]

        page = FakePage()
        monkeypatch.setattr("services.pdf_parser._get_ocr_engine", lambda: FakeOCR())
        result = PDFParser()._attempt_ocr_extraction(
            SimpleNamespace(pages=[page]), "scanned.pdf", None
        )

        assert result.success is True
        assert page.resolution <= 150

    def test_scanned_pdf_ocr_accepts_more_than_twenty_pages(self, monkeypatch):
        class FakeOCR:
            pass

        monkeypatch.setattr("services.pdf_parser._get_ocr_engine", lambda: FakeOCR())
        monkeypatch.setattr(
            PDFParser,
            "_recognize_page",
            lambda self, engine, page, filename, page_number: (
                f"2024-01-{page_number:02d} Transaction {page_number} "
                "-1.00 9.00",
                [],
            ),
        )
        pages = [(number, object()) for number in range(1, 22)]

        result = PDFParser()._attempt_ocr_extraction(
            SimpleNamespace(pages=[page for _, page in pages]),
            "multipage-scan.pdf",
            None,
            pages=pages,
        )

        assert result.success is True
        assert result.total_transactions == 21

    def test_pdf_empty_file_rejection(self):
        """Test that zero-byte PDF upload is cleanly rejected."""
        result = parse_pdf(b"", filename="empty.pdf")
        assert result.success is False
        assert result.total_transactions == 0
        assert "empty" in (result.error or "").lower()

    def test_pdf_corrupted_header_rejection(self):
        """Test rejection of file with invalid/missing %PDF header signature."""
        result = parse_pdf(b"This is not a real PDF file header", filename="corrupted.pdf")
        assert result.success is False
        assert result.total_transactions == 0
        assert "%pdf" in (result.error or "").lower()

    def test_pdf_no_readable_content(self):
        """Test rejection of blank PDF containing neither text nor images."""
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        buf = io.BytesIO()
        writer.write(buf)
        blank_bytes = buf.getvalue()

        result = parse_pdf(blank_bytes, filename="blank.pdf")
        assert result.success is False
        assert result.total_transactions == 0
        assert "no readable text" in (result.error or "").lower()

    def test_pdf_canonical_schema_serialization(self):
        """Test that model_dump() strictly outputs the 8 canonical fields for PDF extractions."""
        lines = ["2024-01-01 Supermarket Groceries -50.00 1000.00"]
        pdf_bytes = create_test_text_pdf(lines)
        result = parse_pdf(pdf_bytes, filename="canonical_test.pdf")

        assert result.success is True
        dumped = result.model_dump()
        first_txn = dumped["transactions"][0]
        expected_fields = {
            "transaction_id",
            "transaction_date",
            "description",
            "amount",
            "currency",
            "transaction_type",
            "category",
            "source_dataset"
        }
        assert set(first_txn.keys()) == expected_fields
        assert "balance" not in first_txn

    def test_pdf_ambiguous_date_with_dayfirst_override(self):
        """Test explicit dayfirst override on ambiguous dates in PDF statements."""
        # 05/06/2024: May 6 if dayfirst=False, June 5 if dayfirst=True
        lines = ["05/06/2024 Ambiguous Date Item -20.00 500.00"]
        pdf_bytes = create_test_text_pdf(lines)

        result_dayfirst = parse_pdf(pdf_bytes, filename="ambig.pdf", dayfirst=True)
        assert result_dayfirst.success is True
        assert result_dayfirst.transactions[0].transaction_date == "2024-06-05"

        result_monthfirst = parse_pdf(pdf_bytes, filename="ambig.pdf", dayfirst=False)
        assert result_monthfirst.success is True
        assert result_monthfirst.transactions[0].transaction_date == "2024-05-06"

    def test_pdf_corrects_inconsistent_date_from_leading_description(self):
        lines = [
            "2021-01-06 05-31-2021 Grocery purchase -10.00 90.00",
            "2021-01-06 06-01-2021 Transit purchase -5.00 85.00",
        ]
        result = parse_pdf(
            create_test_text_pdf(lines), filename="leading-description-date.pdf"
        )

        assert result.success is True
        assert [
            transaction.transaction_date for transaction in result.transactions
        ] == ["2021-05-31", "2021-06-01"]
        assert [
            transaction.description for transaction in result.transactions
        ] == ["05-31-2021 Grocery purchase", "06-01-2021 Transit purchase"]

    def test_pdf_summary_and_header_lines_skipped_safely(self):
        """Test that bank header and summary lines are safely ignored with warnings."""
        lines = [
            "FIRST NATIONAL BANK STATEMENT",
            "Account Number: 123456789",
            "Statement Period: Jan 01 2024 to Jan 31 2024",
            "2024-01-10 Hardware Store -45.00 1200.00",
            "Total Withdrawals: 45.00",
            "Page 1 of 1"
        ]
        pdf_bytes = create_test_text_pdf(lines)
        result = parse_pdf(pdf_bytes, filename="with_headers.pdf")

        assert result.success is True
        assert result.total_transactions == 1
        assert result.transactions[0].description == "Hardware Store"
        assert result.transactions[0].amount == 45.00

    def test_pdf_currency_symbol_detection(self):
        """Test extraction of currency symbols in statement text."""
        lines = ["2024-01-15 British Merchant -\xa375.00 \xa3925.00"]
        pdf_bytes = create_test_text_pdf(lines)
        result = parse_pdf(pdf_bytes, filename="currency_test.pdf")

        assert result.success is True
        assert result.total_transactions == 1
        assert result.transactions[0].currency == "GBP"
        assert result.transactions[0].amount == 75.00

    def test_pdf_stream_and_path_inputs(self, tmp_path):
        """Test that PDFParser handles bytes, BytesIO stream, and file path inputs."""
        lines = ["2024-01-01 Store Transaction -10.00 990.00"]
        pdf_bytes = create_test_text_pdf(lines)

        # 1. Bytes input
        r1 = parse_pdf(pdf_bytes)
        assert r1.success is True and r1.total_transactions == 1

        # 2. BytesIO stream input
        r2 = parse_pdf(io.BytesIO(pdf_bytes))
        assert r2.success is True and r2.total_transactions == 1

        # 3. File path input
        pdf_file = tmp_path / "temp_statement.pdf"
        pdf_file.write_bytes(pdf_bytes)
        r3 = parse_pdf(str(pdf_file))
        assert r3.success is True and r3.total_transactions == 1

    def test_pdf_opening_closing_and_summary_rows_excluded(self):
        """Test that opening balance, closing balance, and summary totals are not extracted as txns."""
        # 1. Text PDF test
        text_lines = [
            "2024-01-01 Opening Balance 5000.00 5000.00",
            "2024-01-05 Grocery Store Supermarket -50.00 4950.00",
            "2024-01-10 Balance Transfer Credit Card -200.00 4750.00",
            "2024-01-15 Total Energies Gas Station -45.00 4705.00",
            "2024-01-31 Closing Balance 4705.00 4705.00",
            "2024-01-31 Total Withdrawals 295.00"
        ]
        pdf_bytes = create_test_text_pdf(text_lines)
        res = parse_pdf(pdf_bytes, filename="balance_test.pdf")

        assert res.success is True
        assert res.total_transactions == 3
        descriptions = [t.description for t in res.transactions]
        assert "Opening Balance" not in descriptions
        assert "Closing Balance" not in descriptions
        assert "Total Withdrawals" not in descriptions
        assert "Grocery Store Supermarket" in descriptions
        assert "Balance Transfer Credit Card" in descriptions
        assert "Total Energies Gas Station" in descriptions

        # 2. Table PDF test
        headers = ["Date", "Description", "Debit", "Credit", "Balance"]
        rows = [
            ["2024-01-01", "Beginning Balance", "", "", "5000.00"],
            ["2024-01-05", "Grocery Store", "85.50", "", "4914.50"],
            ["2024-01-10", "Balance Transfer Payment", "200.00", "", "4714.50"],
            ["2024-01-31", "Ending Balance", "", "", "4714.50"],
            ["2024-01-31", "Total Debits", "285.50", "", "4714.50"],
        ]
        table_pdf = create_test_table_pdf(headers, rows)
        res_table = parse_pdf(table_pdf, filename="table_bal.pdf")

        assert res_table.success is True
        assert res_table.total_transactions == 2
        table_desc = [t.description for t in res_table.transactions]
        assert "Beginning Balance" not in table_desc
        assert "Ending Balance" not in table_desc
        assert "Total Debits" not in table_desc
        assert "Grocery Store" in table_desc
        assert "Balance Transfer Payment" in table_desc

    def test_pdf_stream_seek_reset_and_eof_handling(self):
        """Test stream position safety: advances stream to EOF, verifies parser rewinds, reads, and resets."""
        lines = ["2024-01-01 Store Transaction -10.00 990.00"]
        pdf_bytes = create_test_text_pdf(lines)
        stream = io.BytesIO(pdf_bytes)

        # Move pointer to end of stream (simulating a previously read stream)
        stream.read()
        assert stream.tell() == len(pdf_bytes)

        res = parse_pdf(stream, filename="stream_eof.pdf")
        assert res.success is True
        assert res.total_transactions == 1
        # Verify stream pointer was safely reset to 0 for downstream callers
        assert stream.tell() == 0

    def test_pdf_corrupted_body_syntax_error(self):
        """Test that PDF starting with %PDF but containing corrupted body returns a clean error."""
        corrupted_bytes = b"%PDF-1.4\n1 0 obj\n<< Corrupted junk content without valid PDF structure"
        res = parse_pdf(corrupted_bytes, filename="corrupt_body.pdf")

        assert res.success is False
        assert res.total_transactions == 0
        assert len(res.transactions) == 0
        assert res.error is not None
        assert "corrupted" in res.error.lower() or "invalid" in res.error.lower()
