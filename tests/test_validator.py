"""
Unit tests for File Validation Service.
Tests format support, signature verification, 15 MB size limit, path traversal defense,
empty file handling, and stream safety.
"""
import io
import zipfile
from types import SimpleNamespace
import pytest

from services.file_validator import (
    FileValidator,
    FileValidationResult,
    validate_file,
    sanitize_filename,
    MAX_FILE_SIZE_BYTES
)


def create_valid_test_xlsx() -> bytes:
    """Helper to synthesize a minimal valid XLSX ZIP archive containing workbook parts."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"></Types>')
        zf.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"></workbook>')
        zf.writestr("xl/worksheets/sheet1.xml", "<worksheet></worksheet>")
    return buf.getvalue()


def create_non_excel_zip() -> bytes:
    """Helper to synthesize a ZIP archive that is NOT an Excel spreadsheet (e.g. a document zip)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", "<w:document></w:document>")
        zf.writestr("docProps/app.xml", "<Properties></Properties>")
    return buf.getvalue()


class TestFileValidator:
    """Test suite for FileValidator service."""

    # 1. Valid file validation
    def test_valid_pdf_file(self):
        """Test validation of a valid PDF statement."""
        pdf_bytes = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<< /Root 1 0 R >>\n%%EOF"
        result = validate_file(pdf_bytes, filename="statement_january.pdf")

        assert result.is_valid is True
        assert result.file_type == "pdf"
        assert result.safe_filename == "statement_january.pdf"
        assert result.file_size == len(pdf_bytes)
        assert result.error is None

    def test_valid_csv_file(self):
        """Test validation of a standard text CSV statement."""
        csv_bytes = b"Date,Description,Amount,Balance\n2024-01-01,Groceries,-50.00,1000.00\n"
        result = validate_file(csv_bytes, filename="bank_export.csv")

        assert result.is_valid is True
        assert result.file_type == "csv"
        assert result.safe_filename == "bank_export.csv"
        assert result.file_size == len(csv_bytes)
        assert result.error is None

    def test_valid_xlsx_file(self):
        """Test validation of a valid modern Excel (.xlsx) file."""
        xlsx_bytes = create_valid_test_xlsx()
        result = validate_file(xlsx_bytes, filename="financials_2024.xlsx")

        assert result.is_valid is True
        assert result.file_type == "xlsx"
        assert result.safe_filename == "financials_2024.xlsx"
        assert result.file_size == len(xlsx_bytes)
        assert result.error is None

    # 2. Unsupported extensions
    def test_unsupported_extensions_rejected(self):
        """Test rejection of unsupported file extensions."""
        unsupported = [
            ("data.txt", b"plain text data"),
            ("export.json", b'{"data": []}'),
            ("archive.zip", b"PK\x03\x04..."),
            ("document.docx", b"PK\x03\x04..."),
            ("script.py", b"print('hello')"),
            ("image.gif", b"GIF89a"),
            ("malware.exe", b"MZ\x90\x00"),
        ]
        for name, content in unsupported:
            res = validate_file(content, filename=name)
            assert res.is_valid is False
            assert "unsupported" in (res.error or "").lower()

    def test_legacy_xls_is_validated_with_the_xls_reader(self, monkeypatch):
        """Test that legacy XLS content is inspected rather than rejected by extension."""
        fake_workbook = SimpleNamespace(
            nsheets=1,
            release_resources=lambda: None,
        )
        fake_xlrd = SimpleNamespace(
            open_workbook=lambda **kwargs: fake_workbook,
        )
        monkeypatch.setitem(__import__("sys").modules, "xlrd", fake_xlrd)
        result = validate_file(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1dummy", filename="old_statement.xls")
        assert result.is_valid is True
        assert result.file_type == "xls"

    def test_mismatched_mime_type_is_rejected(self):
        result = validate_file(
            b"%PDF-1.4\nstatement",
            filename="statement.pdf",
            content_type="image/jpeg",
        )
        assert result.is_valid is False
        assert "content type" in (result.error or "").lower()

    # 3. File size enforcement
    def test_empty_file_rejected(self):
        """Test rejection of empty 0-byte file."""
        result = validate_file(b"", filename="empty.pdf")
        assert result.is_valid is False
        assert result.file_size == 0
        assert "empty" in (result.error or "").lower()

    def test_file_exceeding_15mb_rejected(self):
        """Test rejection of files exceeding the 15 MB limit."""
        # Test with custom validator where limit is 1000 bytes
        validator = FileValidator(max_size_bytes=1000)
        large_content = b"%PDF" + b"X" * 1500
        result = validator.validate(large_content, filename="large.pdf")

        assert result.is_valid is False
        assert result.file_size == len(large_content)
        assert "exceeds" in (result.error or "").lower()
        assert "15" in (result.error or "") or "mb" in (result.error or "").lower()

    def test_standard_15mb_boundary(self):
        """Verify the exact MAX_FILE_SIZE_BYTES constant value."""
        assert MAX_FILE_SIZE_BYTES == 15 * 1024 * 1024
        validator = FileValidator()
        assert validator.max_size_bytes == 15728640

    # 4. Signature & magic bytes validation
    def test_pdf_missing_header_signature_rejected(self):
        """Test that a file named .pdf without %PDF signature is rejected."""
        corrupted_pdf = b"NOT_A_PDF_HEADER\nSome random content"
        result = validate_file(corrupted_pdf, filename="fake.pdf")

        assert result.is_valid is False
        assert result.file_type == "pdf"
        assert "%pdf" in (result.error or "").lower()

    def test_xlsx_missing_zip_signature_rejected(self):
        """Test that an .xlsx file not starting with PK\\x03\\x04 is rejected."""
        fake_xlsx = b"Plain text disguised as excel sheet"
        result = validate_file(fake_xlsx, filename="fake.xlsx")

        assert result.is_valid is False
        assert result.file_type == "xlsx"
        assert "zip" in (result.error or "").lower() or "signature" in (result.error or "").lower()

    def test_xlsx_non_excel_zip_archive_rejected(self):
        """Test that a valid ZIP archive that does not contain Excel parts is rejected."""
        non_excel_zip = create_non_excel_zip()
        result = validate_file(non_excel_zip, filename="fake_calc.xlsx")

        assert result.is_valid is False
        assert result.file_type == "xlsx"
        assert "excel" in (result.error or "").lower()

    def test_xlsx_corrupted_zip_archive_rejected(self):
        """Test that a truncated/corrupted ZIP archive named .xlsx is rejected."""
        corrupt_zip = b"PK\x03\x04\x14\x00\x00\x00truncated junk"
        result = validate_file(corrupt_zip, filename="corrupt.xlsx")

        assert result.is_valid is False
        assert "corrupted" in (result.error or "").lower() or "archive" in (result.error or "").lower()

    def test_csv_with_binary_signatures_rejected(self):
        """Test that binary files renamed to .csv are rejected by signature checks."""
        binary_files = [
            (b"%PDF-1.4\n...", "PDF renamed to CSV"),
            (b"PK\x03\x04...", "ZIP renamed to CSV"),
            (b"MZ\x90\x00\x03\x00...", "EXE renamed to CSV"),
            (b"\x7fELF\x02\x01...", "ELF binary renamed to CSV"),
            (b"\x89PNG\r\n\x1a\n...", "PNG renamed to CSV"),
            (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1...", "Legacy XLS renamed to CSV"),
        ]
        for data, desc in binary_files:
            result = validate_file(data, filename="suspicious.csv")
            assert result.is_valid is False, f"Failed to reject {desc}"
            assert "signature" in (result.error or "").lower() or "binary" in (result.error or "").lower()

    def test_csv_with_null_bytes_rejected(self):
        """Test that a file containing null bytes is rejected as binary data."""
        data_with_nulls = b"Date,Description,Amount\n2024-01-01,Payment\x00Injected,100.00\n"
        result = validate_file(data_with_nulls, filename="binary_data.csv")

        assert result.is_valid is False
        assert "null bytes" in (result.error or "").lower() or "binary" in (result.error or "").lower()

    def test_csv_with_excessive_non_printable_bytes_rejected(self):
        """Test that random binary bytes decoded as Latin-1 are caught by non-printable threshold."""
        random_binary = bytes(range(1, 31)) * 10
        result = validate_file(random_binary, filename="garbage.csv")

        assert result.is_valid is False
        assert "non-printable" in (result.error or "").lower() or "binary" in (result.error or "").lower()

    # 5. Filename sanitization & path traversal defense
    def test_path_traversal_sanitization(self):
        """Test that path traversal sequences are cleanly stripped."""
        cases = [
            ("../../etc/passwd.pdf", "passwd.pdf"),
            ("..\\..\\windows\\system32\\secret.xlsx", "secret.xlsx"),
            ("/var/log/statements/bank.csv", "bank.csv"),
            ("C:\\Users\\Administrator\\Desktop\\report.pdf", "report.pdf"),
            ("folder/subfolder/statement.csv", "statement.csv"),
            ("   spaces_trimmed.pdf   ", "spaces_trimmed.pdf"),
        ]
        for dirty, expected_clean in cases:
            cleaned = sanitize_filename(dirty)
            assert cleaned == expected_clean

    def test_malicious_dot_filenames_rejected(self):
        """Test that filenames consisting only of dots or traversal markers are rejected."""
        malicious = ["..", "...", "../", "..\\", " . ", ""]
        for m in malicious:
            res = validate_file(b"%PDF-1.4", filename=m)
            assert res.is_valid is False
            assert "filename" in (res.error or "").lower() or "traversal" in (res.error or "").lower()

    def test_null_byte_in_filename_sanitized(self):
        """Test that null byte injection in filenames is stripped safely."""
        dirty = "statement.csv\x00.exe"
        cleaned = sanitize_filename(dirty)
        assert "\x00" not in cleaned
        assert cleaned == "statement.csv.exe"

    def test_reserved_characters_in_filename_sanitized(self):
        """Test that filesystem reserved characters (<>:\"/\\|?*) are replaced."""
        dirty = 'statement<2024>:final*?.pdf'
        cleaned = sanitize_filename(dirty)
        for char in '<>:"/\\|?*':
            assert char not in cleaned
        assert cleaned.endswith(".pdf")

    # 6. Stream safety and preservation
    def test_stream_position_preserved_and_rewound(self):
        """Test that validating a stream does not advance or consume the caller's stream."""
        pdf_bytes = b"%PDF-1.4\nTest statement content"
        stream = io.BytesIO(pdf_bytes)

        # 1. Test when stream is passed at position 0
        res1 = validate_file(stream, filename="stream.pdf")
        assert res1.is_valid is True
        assert stream.tell() == 0  # Still at start

        # 2. Test when stream was previously read to EOF
        stream.read()
        assert stream.tell() == len(pdf_bytes)
        res2 = validate_file(stream, filename="stream_eof.pdf")
        assert res2.is_valid is True
        assert stream.tell() == 0  # Safely rewound to start

    def test_filepath_input_handling(self, tmp_path):
        """Test validation when given a filesystem Path or path string."""
        test_file = tmp_path / "valid_statement.pdf"
        test_file.write_bytes(b"%PDF-1.4\nValid content")

        res_path = validate_file(test_file)
        assert res_path.is_valid is True
        assert res_path.file_type == "pdf"
        assert res_path.safe_filename == "valid_statement.pdf"

        res_str = validate_file(str(test_file))
        assert res_str.is_valid is True
        assert res_str.safe_filename == "valid_statement.pdf"
