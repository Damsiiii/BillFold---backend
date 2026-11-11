"""
API Unit and Integration Tests for Statement Upload Endpoint.
Tests POST /api/upload for CSV, XLSX, and PDF bank statements,
including format dispatching, size limits, signature verification, error codes,
schema consistency, and file closure.
"""
import io
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch
import pytest
from fastapi import UploadFile
import numpy as np
from PIL import Image

from api.upload import upload_statement
from models.transaction import ExtractionResult
from services.image_parser import parse_image
from tests.test_parsers import (
    create_test_xlsx,
    create_test_table_pdf,
    create_test_text_pdf,
    create_test_encrypted_pdf,
    create_test_scanned_pdf,
)


def make_upload_file(data: bytes, filename: str) -> UploadFile:
    """Helper to construct a FastAPI UploadFile from bytes and filename."""
    return UploadFile(filename=filename, file=io.BytesIO(data))


def create_test_image(format_name: str) -> bytes:
    image = Image.new("RGB", (500, 100), color="white")
    output = io.BytesIO()
    image.save(output, format=format_name)
    return output.getvalue()


class TestUploadAPI:
    """Test suite for POST /api/upload endpoint."""

    # 1. Successful uploads
    def test_successful_csv_upload(self):
        """Test successful CSV statement upload returns 200 OK and canonical transactions."""
        csv_data = (
            b"Date,Description,Amount,Balance\n"
            b"2024-01-05,Supermarket Groceries,-85.50,1414.50\n"
            b"2024-01-10,Salary Direct Deposit,3500.00,4914.50\n"
        )
        file = make_upload_file(csv_data, "january_statement.csv")
        response = upload_statement(file)

        assert response.status_code == 200
        body = json.loads(response.body.decode("utf-8"))

        assert body["success"] is True
        assert body["filename"] == "january_statement.csv"
        assert body["file_type"] == "csv"
        assert body["total_transactions"] == 2
        assert len(body["transactions"]) == 2
        assert body["error"] is None

        # Check expense transaction
        t1 = body["transactions"][0]
        assert t1["transaction_date"] == "2024-01-05"
        assert t1["description"] == "Supermarket Groceries"
        assert t1["amount"] == 85.50
        assert t1["transaction_type"] == "expense"
        assert t1["source_dataset"] == "january_statement.csv"

        # Check income transaction
        t2 = body["transactions"][1]
        assert t2["transaction_date"] == "2024-01-10"
        assert t2["amount"] == 3500.00
        assert t2["transaction_type"] == "income"

    def test_successful_xlsx_upload(self):
        """Test successful XLSX statement upload returns 200 OK and canonical transactions."""
        rows = [
            ["Date", "Description", "Debit", "Credit", "Balance"],
            ["2024-02-01", "Coffee Shop Cafe", "4.50", "", "995.50"],
            ["2024-02-05", "Client Invoice Pmt", "", "1200.00", "2195.50"],
        ]
        xlsx_bytes = create_test_xlsx(rows)
        file = make_upload_file(xlsx_bytes, "q1_statement.xlsx")
        response = upload_statement(file)

        assert response.status_code == 200
        body = json.loads(response.body.decode("utf-8"))

        assert body["success"] is True
        assert body["filename"] == "q1_statement.xlsx"
        assert body["file_type"] in {"excel", "xlsx"}
        assert body["total_transactions"] == 2
        assert body["transactions"][0]["transaction_type"] == "expense"
        assert body["transactions"][1]["transaction_type"] == "income"

    def test_successful_legacy_xls_upload(self, monkeypatch):
        class FakeSheet:
            name = "Transactions"
            rows = [
                ["Date", "Description", "Debit", "Credit", "Balance"],
                ["2024-02-01", "Coffee Shop", 4.5, None, 995.5],
            ]
            nrows = len(rows)

            def row(self, index):
                return [
                    SimpleNamespace(
                        ctype=0 if value is None else (2 if isinstance(value, float) else 1),
                        value=value,
                    )
                    for value in self.rows[index]
                ]

        class FakeWorkbook:
            datemode = 0
            nsheets = 1

            def __init__(self):
                self.sheet = FakeSheet()

            def release_resources(self):
                pass

            def sheet_names(self):
                return [self.sheet.name]

            def sheet_by_name(self, name):
                assert name == self.sheet.name
                return self.sheet

        fake_xlrd = SimpleNamespace(
            XL_CELL_EMPTY=0,
            XL_CELL_BLANK=6,
            XL_CELL_ERROR=5,
            XL_CELL_DATE=3,
            xldate=SimpleNamespace(xldate_as_datetime=lambda value, mode: value),
            open_workbook=lambda **kwargs: FakeWorkbook(),
        )
        monkeypatch.setitem(sys.modules, "xlrd", fake_xlrd)

        file = make_upload_file(
            b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1synthetic-xls",
            "legacy_statement.xls",
        )
        response = upload_statement(file)
        body = json.loads(response.body.decode("utf-8"))

        assert response.status_code == 200
        assert body["total_transactions"] == 1
        assert body["transactions"][0]["description"] == "Coffee Shop"
        assert body["transactions"][0]["amount"] == 4.5
        assert body["transactions"][0]["transaction_type"] == "expense"

    def test_successful_pdf_upload(self):
        """Test successful PDF statement upload returns 200 OK and canonical transactions."""
        headers = ["Date", "Description", "Debit", "Credit", "Balance"]
        rows = [
            ["2024-03-01", "Office Supplies Store", "45.00", "", "1500.00"],
            ["2024-03-15", "Consulting Fee Deposit", "", "2000.00", "3500.00"],
        ]
        pdf_bytes = create_test_table_pdf(headers, rows)
        file = make_upload_file(pdf_bytes, "march_statement.pdf")
        response = upload_statement(file)

        assert response.status_code == 200
        body = json.loads(response.body.decode("utf-8"))

        assert body["success"] is True
        assert body["filename"] == "march_statement.pdf"
        assert body["file_type"] == "pdf"
        assert body["total_transactions"] == 2
        assert body["transactions"][0]["amount"] == 45.00
        assert body["transactions"][1]["amount"] == 2000.00

    @pytest.mark.parametrize(
        ("extension", "image_format"),
        [("jpg", "JPEG"), ("jpeg", "JPEG"), ("png", "PNG")],
    )
    def test_statement_image_extensions_are_validated_and_dispatched(
        self, extension, image_format
    ):
        image_bytes = create_test_image(image_format)
        file = make_upload_file(image_bytes, f"statement.{extension}")
        parser_result = ExtractionResult(
            success=True,
            filename=f"statement.{extension}",
            file_type=extension,
            total_transactions=0,
            transactions=[],
            warnings=[],
            error=None,
        )

        with patch("api.upload.parse_image", return_value=parser_result) as image_parser:
            response = upload_statement(file)

        assert response.status_code == 200
        image_parser.assert_called_once()

    def test_image_ocr_extracts_transaction_and_explicit_currency(self, monkeypatch):
        image_bytes = create_test_image("PNG")
        recognized = [
            ("Date", 0, 40, 10, 20),
            ("Description", 70, 145, 10, 20),
            ("Debit", 180, 220, 10, 20),
            ("Credit", 250, 295, 10, 20),
            ("Balance", 320, 380, 10, 20),
            ("Currency", 410, 470, 10, 20),
            ("2026-03-01", 0, 55, 40, 50),
            ("Mobile", 75, 110, 40, 50),
            ("Reload", 115, 145, 40, 50),
            ("25.50", 180, 220, 40, 50),
            ("950.00", 320, 380, 40, 50),
            ("AED", 420, 450, 40, 50),
        ]
        result_data = {
            "rec_texts": [item[0] for item in recognized],
            "rec_boxes": np.array([
                [item[1], item[3], item[2], item[4]] for item in recognized
            ]),
            "rec_scores": np.array([0.99] * len(recognized)),
        }

        class FakeOCR:
            def predict(self, image):
                assert image.shape == (100, 500, 3)
                return [SimpleNamespace(json={"res": result_data})]

        monkeypatch.setattr("services.image_parser._get_ocr_engine", FakeOCR)

        result = parse_image(
            io.BytesIO(image_bytes),
            filename="statement.png",
            file_type="png",
        )

        assert result.success is True
        assert result.total_transactions == 1
        transaction = result.transactions[0]
        assert transaction.transaction_date == "2026-03-01"
        assert transaction.description == "Mobile Reload"
        assert transaction.amount == 25.50
        assert transaction.transaction_type == "expense"
        assert transaction.currency == "AED"

    def test_image_ocr_prefers_consistent_leading_description_dates(self, monkeypatch):
        image_bytes = io.BytesIO()
        Image.new("RGB", (700, 430), color="white").save(image_bytes, format="PNG")

        rows = [
            ("05-31-2021", "2021-01-06"),
            ("06-01-2021", "2021-01-06"),
            ("06-03-2021", "2021-02-06"),
            ("06-04-2021", "2021-07-06"),
            ("06-07-2021", "2021-08-06"),
            ("06-08-2021", "2021-10-06"),
            ("06-09-2021", "2021-11-06"),
        ]
        recognized = [
            ("Date", 0, 60, 10, 22),
            ("Description", 100, 300, 10, 22),
            ("Debit", 350, 400, 10, 22),
            ("Credit", 450, 500, 10, 22),
            ("Balance", 550, 630, 10, 22),
        ]
        for index, (description_date, extracted_date) in enumerate(rows):
            y = 40 + index * 35
            recognized.extend([
                (extracted_date, 0, 80, y, y + 14),
                (f"{description_date} Purchase {index}", 100, 310, y, y + 14),
                ("10.00", 350, 400, y, y + 14),
                (f"{990 - index * 10:.2f}", 550, 630, y, y + 14),
            ])

        recognized.extend([
            ("2021-06-12", 0, 80, 295, 309),
            ("Ordinary purchase", 100, 310, 295, 309),
            ("10.00", 350, 400, 295, 309),
            ("920.00", 550, 630, 295, 309),
            ("2021-05-31", 0, 80, 330, 344),
            ("05-31-2021 Consistent date", 100, 310, 330, 344),
            ("10.00", 350, 400, 330, 344),
            ("910.00", 550, 630, 330, 344),
        ])
        result_data = {
            "rec_texts": [item[0] for item in recognized],
            "rec_boxes": np.array([
                [item[1], item[3], item[2], item[4]] for item in recognized
            ]),
            "rec_scores": np.array([0.99] * len(recognized)),
        }

        class FakeOCR:
            def predict(self, image):
                return [SimpleNamespace(json={"res": result_data})]

        monkeypatch.setattr("services.image_parser._get_ocr_engine", FakeOCR)
        result = parse_image(
            image_bytes.getvalue(), "date-regression.png", "png"
        )

        assert result.success is True
        assert result.total_transactions == 9
        assert [
            transaction.transaction_date
            for transaction in result.transactions[:7]
        ] == [
            "2021-05-31",
            "2021-06-01",
            "2021-06-03",
            "2021-06-04",
            "2021-06-07",
            "2021-06-08",
            "2021-06-09",
        ]
        assert result.transactions[7].transaction_date == "2021-06-12"
        assert result.transactions[8].transaction_date == "2021-05-31"
        assert result.transactions[0].description.startswith("05-31-2021")
        assert all(transaction.currency is None for transaction in result.transactions)

        response = upload_statement(
            make_upload_file(image_bytes.getvalue(), "date-regression.png"),
            dayfirst=None,
        )
        body = json.loads(response.body.decode("utf-8"))
        assert response.status_code == 200
        assert [
            transaction["transaction_date"]
            for transaction in body["transactions"][:7]
        ] == [
            "2021-05-31",
            "2021-06-01",
            "2021-06-03",
            "2021-06-04",
            "2021-06-07",
            "2021-06-08",
            "2021-06-09",
        ]

    def test_image_with_low_confidence_ocr_does_not_create_transactions(
        self, monkeypatch
    ):
        image_bytes = create_test_image("PNG")
        result_data = {
            "rec_texts": ["2026-03-01", "Unknown purchase", "10.00"],
            "rec_boxes": [
                [0, 10, 70, 20],
                [80, 10, 180, 20],
                [200, 10, 240, 20],
            ],
            "rec_scores": [0.3, 0.4, 0.2],
        }

        class FakeOCR:
            def predict(self, image):
                return [SimpleNamespace(json={"res": result_data})]

        monkeypatch.setattr("services.image_parser._get_ocr_engine", FakeOCR)
        file = make_upload_file(image_bytes, "blurry.png")
        response = upload_statement(file)

        assert response.status_code == 422
        body = json.loads(response.body.decode("utf-8"))
        assert body["transactions"] == []
        assert any("low-confidence" in warning for warning in body["warnings"])

    def test_malformed_image_is_rejected_without_ocr(self):
        file = make_upload_file(b"\x89PNG\r\n\x1a\nnot-an-image", "broken.png")
        response = upload_statement(file)

        assert response.status_code == 400
        body = json.loads(response.body.decode("utf-8"))
        assert body["transactions"] == []
        assert "unreadable" in body["error"].lower()

    # 2. Unsupported file types
    def test_unsupported_extension_rejected_400(self):
        """Test upload with unsupported extension returns 400 Bad Request."""
        file = make_upload_file(b"some text data", "notes.txt")
        response = upload_statement(file)

        assert response.status_code == 400
        body = json.loads(response.body.decode("utf-8"))
        assert body["success"] is False
        assert body["total_transactions"] == 0
        assert "unsupported" in body["error"].lower()

    def test_corrupted_legacy_xls_rejected_400(self):
        """Test malformed legacy XLS content returns a useful validation error."""
        file = make_upload_file(b"FakeLegacyXlsData", "statement.xls")
        response = upload_statement(file)

        assert response.status_code == 400
        body = json.loads(response.body.decode("utf-8"))
        assert body["success"] is False
        assert body["total_transactions"] == 0
        assert "legacy excel" in body["error"].lower()

    # 3. Empty and oversized files
    def test_empty_file_rejected_400(self):
        """Test 0-byte file upload returns 400 Bad Request."""
        file = make_upload_file(b"", "empty_statement.csv")
        response = upload_statement(file)

        assert response.status_code == 400
        body = json.loads(response.body.decode("utf-8"))
        assert body["success"] is False
        assert body["total_transactions"] == 0
        assert "empty" in body["error"].lower()

    def test_oversized_file_rejected_413(self):
        """Test file exceeding 15 MB returns 413 Payload Too Large."""
        large_bytes = b"%PDF-1.4\n" + b"X" * (15 * 1024 * 1024 + 1024)
        file = make_upload_file(large_bytes, "giant_statement.pdf")
        response = upload_statement(file)

        assert response.status_code == 413
        body = json.loads(response.body.decode("utf-8"))
        assert body["success"] is False
        assert body["total_transactions"] == 0
        assert "exceeds" in body["error"].lower()
        assert "15" in body["error"]

    # 4. Invalid signatures and disguised files
    def test_pdf_invalid_signature_rejected_400(self):
        """Test file named .pdf without %PDF signature returns 400 Bad Request."""
        file = make_upload_file(b"This is not a real PDF header", "fake.pdf")
        response = upload_statement(file)

        assert response.status_code == 400
        body = json.loads(response.body.decode("utf-8"))
        assert body["success"] is False
        assert "%pdf" in body["error"].lower()

    def test_xlsx_invalid_signature_rejected_400(self):
        """Test file named .xlsx without ZIP signature returns 400 Bad Request."""
        file = make_upload_file(b"Plain text file masquerading as xlsx", "fake.xlsx")
        response = upload_statement(file)

        assert response.status_code == 400
        body = json.loads(response.body.decode("utf-8"))
        assert body["success"] is False
        assert "zip" in body["error"].lower() or "signature" in body["error"].lower()

    def test_csv_disguised_binary_rejected_400(self):
        """Test binary file named .csv returns 400 Bad Request."""
        file = make_upload_file(b"%PDF-1.4\nDisguised PDF as CSV", "fake.csv")
        response = upload_statement(file)

        assert response.status_code == 400
        body = json.loads(response.body.decode("utf-8"))
        assert body["success"] is False
        assert "signature" in body["error"].lower() or "binary" in body["error"].lower()

    # 5. Parser extraction failures (422 Unprocessable Content)
    def test_scanned_pdf_returns_422(self):
        """Test scanned PDF (images with no text) returns 422 Unprocessable Content."""
        scanned_bytes = create_test_scanned_pdf()
        file = make_upload_file(scanned_bytes, "scanned_doc.pdf")
        response = upload_statement(file)

        assert response.status_code == 422
        body = json.loads(response.body.decode("utf-8"))
        assert body["success"] is False
        assert body["total_transactions"] == 0
        assert "scanned" in body["error"].lower()

    def test_encrypted_pdf_returns_422(self):
        """Test password-encrypted PDF returns 422 Unprocessable Content."""
        enc_bytes = create_test_encrypted_pdf("secretpass")
        file = make_upload_file(enc_bytes, "protected.pdf")
        response = upload_statement(file)

        assert response.status_code == 422
        body = json.loads(response.body.decode("utf-8"))
        assert body["success"] is False
        assert body["total_transactions"] == 0
        assert "password" in body["error"].lower()

    def test_corrupted_pdf_structure_returns_422(self):
        """Test PDF starting with %PDF but having invalid structure returns 422."""
        corrupt_bytes = b"%PDF-1.4\nCorrupted structure without valid objects"
        file = make_upload_file(corrupt_bytes, "corrupt.pdf")
        response = upload_statement(file)

        assert response.status_code == 422
        body = json.loads(response.body.decode("utf-8"))
        assert body["success"] is False
        assert body["total_transactions"] == 0

    # 6. Response schema consistency
    def test_response_schema_strictly_contains_eight_canonical_fields(self):
        """Test that every extracted transaction contains strictly the 8 canonical fields."""
        csv_data = b"Date,Description,Amount\n2024-01-01,Supermarket,-50.00\n"
        file = make_upload_file(csv_data, "schema_test.csv")
        response = upload_statement(file)

        assert response.status_code == 200
        body = json.loads(response.body.decode("utf-8"))

        first_txn = body["transactions"][0]
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

    def test_error_response_schema_consistency(self):
        """Test that error responses conform strictly to the ExtractionResult structure."""
        file = make_upload_file(b"", "empty.csv")
        response = upload_statement(file)

        body = json.loads(response.body.decode("utf-8"))
        expected_top_level = {
            "success",
            "filename",
            "file_type",
            "total_transactions",
            "transactions",
            "warnings",
            "error"
        }
        assert set(body.keys()) == expected_top_level
        assert body["success"] is False
        assert body["transactions"] == []

    # 7. Path traversal sanitization & stream closure
    def test_path_traversal_filename_sanitized_in_response(self):
        """Test that filenames with path traversal markers are sanitized in the API response."""
        csv_data = b"Date,Description,Amount\n2024-01-01,Groceries,-15.00\n"
        file = make_upload_file(csv_data, "../../etc/shadow.csv")
        response = upload_statement(file)

        assert response.status_code == 200
        body = json.loads(response.body.decode("utf-8"))
        assert body["filename"] == "shadow.csv"
        assert "/" not in body["filename"]
        assert "\\" not in body["filename"]

    def test_file_stream_closed_after_upload(self):
        """Test that the uploaded file stream is properly closed in the finally block."""
        csv_data = b"Date,Description,Amount\n2024-01-01,Groceries,-10.00\n"
        file = make_upload_file(csv_data, "test_close.csv")
        assert not file.file.closed

        upload_statement(file)
        assert file.file.closed

    def test_unexpected_parser_exception_returns_500_without_stack_trace(self):
        """Test that an unexpected parser exception returns HTTP 500 without leaking stack traces."""
        csv_data = b"Date,Description,Amount\n2024-01-01,Store,-10.00\n"
        file = make_upload_file(csv_data, "crash.csv")

        with patch("api.upload.parse_csv", side_effect=RuntimeError("Secret database path: /etc/secrets/db")):
            response = upload_statement(file)

        assert response.status_code == 500
        body = json.loads(response.body.decode("utf-8"))
        assert body["success"] is False
        assert body["total_transactions"] == 0
        assert body["transactions"] == []
        assert body["error"] == "An internal server error occurred while processing the statement."
        # Confirm sensitive stack traces or internal paths are never exposed
        assert "/etc/secrets" not in body["error"]
        assert "RuntimeError" not in body["error"]
        assert "traceback" not in body["error"].lower()
        # Confirm file is closed even after 500
        assert file.file.closed

    def test_file_closed_in_all_error_scenarios(self):
        """Verify uploaded file stream is properly closed in all error scenarios (400, 413, 422)."""
        # 400 - Unsupported extension
        f1 = make_upload_file(b"content", "doc.txt")
        upload_statement(f1)
        assert f1.file.closed

        # 400 - Bad signature
        f2 = make_upload_file(b"bad_signature", "doc.pdf")
        upload_statement(f2)
        assert f2.file.closed

        # 413 - Oversized
        f3 = make_upload_file(b"%PDF" + b"X" * (15 * 1024 * 1024 + 100), "huge.pdf")
        upload_statement(f3)
        assert f3.file.closed

        # 422 - Scanned PDF
        f4 = make_upload_file(create_test_scanned_pdf(), "scan.pdf")
        upload_statement(f4)
        assert f4.file.closed

    def test_safe_filename_used_as_transaction_source_dataset(self):
        """Verify that sanitized safe filename is used as the source_dataset on all transactions."""
        csv_data = b"Date,Description,Amount\n2024-01-01,Groceries,-15.00\n"
        dirty_filename = "../../../var/data/monthly_report.csv"
        file = make_upload_file(csv_data, dirty_filename)

        response = upload_statement(file)
        assert response.status_code == 200
        body = json.loads(response.body.decode("utf-8"))

        assert body["filename"] == "monthly_report.csv"
        assert body["transactions"][0]["source_dataset"] == "monthly_report.csv"
        assert "../" not in body["transactions"][0]["source_dataset"]


class TestEndToEndWorkflow:
    """Comprehensive End-to-End workflow verification across CSV, XLSX, and PDF statements."""

    def test_e2e_csv_upload_workflow(self):
        """Verify full CSV upload -> validation -> normalization -> API JSON response."""
        csv_content = (
            b"Date,Description,Debit,Credit,Balance\n"
            b"2024-01-01,Opening Balance,,,5000.00\n"
            b"2024-01-05,Office Supplies,75.25,,4924.75\n"
            b"2024-01-10,Client Consultation Deposit,,1250.00,6174.75\n"
            b"2024-01-15,Utility Electric Company,120.00,,6054.75\n"
            b"31/02/2024,Invalid Date February 31,50.00,,6004.75\n"
            b"2024-01-31,Closing Balance,,,6054.75\n"
            b"TOTAL,Summary Debits,195.25,,\n"
        )
        file = make_upload_file(csv_content, "../../danger/january_statement.csv")
        response = upload_statement(file)

        assert response.status_code == 200
        body = json.loads(response.body.decode("utf-8"))

        assert body["success"] is True
        assert body["filename"] == "january_statement.csv"
        assert body["file_type"] == "csv"
        # 3 transactions: Office Supplies (expense), Consultation (income), Utility (expense)
        # Opening Balance, Closing Balance, Summary Debits, and Feb 31 excluded
        assert body["total_transactions"] == 3
        assert len(body["warnings"]) > 0

        for txn in body["transactions"]:
            assert txn["amount"] > 0
            assert set(txn.keys()) == {
                "transaction_id",
                "transaction_date",
                "description",
                "amount",
                "currency",
                "transaction_type",
                "category",
                "source_dataset"
            }
            assert txn["source_dataset"] == "january_statement.csv"

        assert body["transactions"][0]["transaction_type"] == "expense"
        assert body["transactions"][1]["transaction_type"] == "income"
        assert body["transactions"][2]["transaction_type"] == "expense"

    def test_e2e_xlsx_upload_workflow(self):
        """Verify full XLSX upload -> validation -> normalization -> API JSON response."""
        rows = [
            ["Date", "Description", "Amount", "Balance"],
            ["2024-02-01", "Opening Balance", 1000.0, 1000.0],
            ["2024-02-03", "Hardware Store", -145.50, 854.50],
            ["2024-02-10", "Consulting Invoice", 800.00, 1654.50],
            ["2024-02-28", "Closing Balance", 1654.50, 1654.50],
        ]
        xlsx_bytes = create_test_xlsx(rows)
        file = make_upload_file(xlsx_bytes, "C:\\temp\\q1_bank.xlsx")
        response = upload_statement(file)

        assert response.status_code == 200
        body = json.loads(response.body.decode("utf-8"))

        assert body["success"] is True
        assert body["filename"] == "q1_bank.xlsx"
        assert body["total_transactions"] == 2
        assert body["transactions"][0]["description"] == "Hardware Store"
        assert body["transactions"][0]["amount"] == 145.50
        assert body["transactions"][0]["transaction_type"] == "expense"
        assert body["transactions"][1]["description"] == "Consulting Invoice"
        assert body["transactions"][1]["amount"] == 800.00
        assert body["transactions"][1]["transaction_type"] == "income"

    def test_e2e_pdf_upload_workflow(self):
        """Verify full PDF upload -> validation -> normalization -> API JSON response."""
        headers = ["Date", "Description", "Debit", "Credit", "Balance"]
        pdf_rows = [
            ["2024-03-01", "Beginning Balance", "", "", "2000.00"],
            ["2024-03-05", "European Merchant", "150.00", "", "1850.00"],
            ["2024-03-15", "Direct Deposit Salary", "", "3200.00", "5050.00"],
            ["2024-03-31", "Ending Balance", "", "", "5050.00"],
        ]
        pdf_bytes = create_test_table_pdf(headers, pdf_rows)
        file = make_upload_file(pdf_bytes, "..\\..\\pdf\\statement_march.pdf")
        response = upload_statement(file)

        assert response.status_code == 200
        body = json.loads(response.body.decode("utf-8"))

        assert body["success"] is True
        assert body["filename"] == "statement_march.pdf"
        assert body["total_transactions"] == 2
        assert body["transactions"][0]["amount"] == 150.00
        assert body["transactions"][0]["transaction_type"] == "expense"
        assert body["transactions"][1]["amount"] == 3200.00
        assert body["transactions"][1]["transaction_type"] == "income"
