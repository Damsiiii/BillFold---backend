"""
File Validation Service for Statement Uploads.
Validates uploaded bank statements against security, size, format, extension,
magic bytes, and content integrity requirements before processing.
"""
from typing import Any, Dict, List, Optional, Tuple, Union, BinaryIO
from pathlib import Path
import os
import re
import io
import zipfile
from pydantic import BaseModel, Field


# Constants
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB
SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx"}
SUPPORTED_FILE_TYPES = {"pdf", "csv", "xlsx"}

# Known binary signatures that must NEVER be accepted as CSV
BINARY_SIGNATURES = [
    (b"%PDF", "PDF document"),
    (b"PK\x03\x04", "ZIP / Office archive"),
    (b"PK\x05\x06", "Empty ZIP archive"),
    (b"PK\x07\x08", "Spanned ZIP archive"),
    (b"MZ", "Windows executable (EXE/DLL)"),
    (b"\x7fELF", "Linux executable (ELF)"),
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"GIF87a", "GIF image"),
    (b"GIF89a", "GIF image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"\x1f\x8b", "GZIP compressed archive"),
    (b"\x42\x5a\x68", "BZIP2 compressed archive"),
    (b"\x52\x61\x72\x21\x1a\x07", "RAR archive"),
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "Legacy Microsoft Compound Document (.xls / .doc)"),
]


class FileValidationResult(BaseModel):
    """
    Structured outcome of file validation.
    """
    is_valid: bool = Field(description="Whether the file passed all validation checks")
    file_type: Optional[str] = Field(default=None, description="Normalized file type: 'pdf', 'csv', or 'xlsx'")
    safe_filename: str = Field(description="Sanitized, safe filename with traversal components stripped")
    file_size: int = Field(default=0, description="Size of the file in bytes")
    error: Optional[str] = Field(default=None, description="Detailed explanation of failure if invalid")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings encountered during validation")


def sanitize_filename(raw_filename: Optional[str]) -> str:
    """
    Sanitizes an uploaded filename to prevent directory traversal and injection attacks.
    Strips directory separators, relative path markers (../, ..\\), null bytes, Windows drive
    letters, and control characters.
    """
    if not raw_filename or not isinstance(raw_filename, str):
        return ""

    # Remove null bytes and control characters
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", raw_filename).strip()
    if not cleaned:
        return ""

    # Normalize backslashes to forward slashes for cross-platform processing
    normalized = cleaned.replace("\\", "/")

    # Extract the base component
    base = normalized.rstrip("/").rsplit("/", 1)[-1].strip()

    # Strip Windows drive letters (e.g., C: or D:)
    base = re.sub(r"^[a-zA-Z]:", "", base).strip()

    # Reject if the filename consists exclusively of dots, dashes, underscores, or spaces
    if set(base).issubset({".", " ", "_", "-"}):
        return ""

    # Replace filesystem reserved characters: <>:"/\|?* with underscore
    base = re.sub(r'[<>:"/\\|?*]', "_", base)

    # Restrict maximum filename length to 255 characters while preserving extension
    if len(base) > 255:
        ext = Path(base).suffix
        stem = Path(base).stem[: 255 - len(ext)]
        base = stem + ext

    return base


class FileValidator:
    """
    Validates uploaded bank statement files for size, extension, signatures, and format integrity.
    """

    def __init__(self, max_size_bytes: int = MAX_FILE_SIZE_BYTES):
        self.max_size_bytes = max_size_bytes

    def validate(
        self,
        file_source: Union[str, Path, bytes, BinaryIO],
        filename: Optional[str] = None
    ) -> FileValidationResult:
        """
        Validates an uploaded file.
        Accepts a filesystem path, raw bytes, or a seekable/readable binary stream.
        Preserves stream position and does not mutate or consume the uploaded source.
        """
        warnings: List[str] = []

        # 1. Resolve raw filename from source if not provided
        if filename is not None:
            resolved_filename = filename.strip()
            if not resolved_filename or not sanitize_filename(resolved_filename):
                return FileValidationResult(
                    is_valid=False,
                    file_type=None,
                    safe_filename="unknown",
                    file_size=0,
                    error="Invalid or malicious filename: Contains unsafe traversal patterns or empty name.",
                    warnings=warnings
                )
        else:
            if isinstance(file_source, (str, Path)):
                resolved_filename = Path(file_source).name
            elif hasattr(file_source, "name") and isinstance(file_source.name, str):
                resolved_filename = file_source.name
            else:
                resolved_filename = "statement"

        # 2. Sanitize filename & prevent path traversal
        safe_name = sanitize_filename(resolved_filename)
        if not safe_name:
            return FileValidationResult(
                is_valid=False,
                file_type=None,
                safe_filename="unknown",
                file_size=0,
                error="Invalid or malicious filename: Contains unsafe traversal patterns or empty name.",
                warnings=warnings
            )

        # 3. Read binary content without mutating or permanently consuming stream
        raw_bytes, file_size, read_error = self._read_bytes_safely(file_source)
        if read_error:
            return FileValidationResult(
                is_valid=False,
                file_type=None,
                safe_filename=safe_name,
                file_size=0,
                error=read_error,
                warnings=warnings
            )

        # 4. Enforce maximum file size check (15 MB)
        if file_size > self.max_size_bytes:
            mb_size = file_size / (1024 * 1024)
            max_mb = self.max_size_bytes / (1024 * 1024)
            return FileValidationResult(
                is_valid=False,
                file_type=None,
                safe_filename=safe_name,
                file_size=file_size,
                error=f"File size ({mb_size:.2f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB.",
                warnings=warnings
            )

        # 5. Enforce non-empty file check
        if file_size == 0 or not raw_bytes:
            return FileValidationResult(
                is_valid=False,
                file_type=None,
                safe_filename=safe_name,
                file_size=0,
                error="Uploaded file is empty (0 bytes).",
                warnings=warnings
            )

        # 6. Validate file extension
        ext = Path(safe_name).suffix.lower()
        if not ext:
            return FileValidationResult(
                is_valid=False,
                file_type=None,
                safe_filename=safe_name,
                file_size=file_size,
                error="File has no extension. Supported formats are: .pdf, .csv, .xlsx.",
                warnings=warnings
            )

        # Explicit rejection for legacy .xls with conversion guidance
        if ext == ".xls":
            return FileValidationResult(
                is_valid=False,
                file_type=None,
                safe_filename=safe_name,
                file_size=file_size,
                error="Legacy Excel format (.xls) is not supported. Please convert the statement to modern Excel format (.xlsx) or export as CSV/PDF.",
                warnings=warnings
            )

        if ext not in SUPPORTED_EXTENSIONS:
            allowed_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            return FileValidationResult(
                is_valid=False,
                file_type=None,
                safe_filename=safe_name,
                file_size=file_size,
                error=f"Unsupported file format '{ext}'. Allowed formats are: {allowed_str}.",
                warnings=warnings
            )

        file_type = ext.lstrip(".")

        # 7. Validate magic bytes / file signature & internal structure
        if file_type == "pdf":
            sig_valid, sig_err = self._validate_pdf_signature(raw_bytes)
        elif file_type == "xlsx":
            sig_valid, sig_err = self._validate_xlsx_signature(raw_bytes)
        elif file_type == "csv":
            sig_valid, sig_err = self._validate_csv_content(raw_bytes)
        else:
            sig_valid = False
            sig_err = f"Unknown file type: {file_type}"

        if not sig_valid:
            return FileValidationResult(
                is_valid=False,
                file_type=file_type,
                safe_filename=safe_name,
                file_size=file_size,
                error=sig_err,
                warnings=warnings
            )

        return FileValidationResult(
            is_valid=True,
            file_type=file_type,
            safe_filename=safe_name,
            file_size=file_size,
            error=None,
            warnings=warnings
        )

    def _read_bytes_safely(
        self,
        file_source: Union[str, Path, bytes, BinaryIO]
    ) -> Tuple[Optional[bytes], int, Optional[str]]:
        """
        Safely reads binary data while preserving stream position for downstream callers.
        """
        if isinstance(file_source, (str, Path)):
            path = Path(file_source)
            if not path.exists():
                return None, 0, f"File not found: {path.name}"
            try:
                file_size = path.stat().st_size
                if file_size > self.max_size_bytes:
                    return None, file_size, None
                data = path.read_bytes()
                return data, file_size, None
            except Exception as e:
                return None, 0, f"Error reading file: {str(e)}"
        elif isinstance(file_source, bytes):
            return file_source, len(file_source), None
        elif hasattr(file_source, "read"):
            try:
                # If stream is seekable, probe size without loading large payloads into memory
                if hasattr(file_source, "seek") and hasattr(file_source, "tell"):
                    file_source.seek(0, os.SEEK_END)
                    stream_size = file_source.tell()
                    file_source.seek(0)
                    if stream_size > self.max_size_bytes:
                        return None, stream_size, None

                # Otherwise or if within size limits, read safely
                if hasattr(file_source, "seek"):
                    file_source.seek(0)
                data = file_source.read()
                # Rewind stream pointer back to beginning for downstream safety
                if hasattr(file_source, "seek"):
                    file_source.seek(0)

                if isinstance(data, str):
                    encoded = data.encode("utf-8")
                    return encoded, len(encoded), None
                return data, len(data), None
            except Exception as e:
                return None, 0, f"Error reading stream: {str(e)}"
        return None, 0, "Unsupported file source type."

    def _validate_pdf_signature(self, raw_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """
        Validates %PDF header signature.
        """
        if not raw_bytes.startswith(b"%PDF"):
            return False, "File content does not match PDF format: Missing %PDF header signature."
        return True, None

    def _validate_xlsx_signature(self, raw_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """
        Validates modern .xlsx ZIP archive signature and confirms it is an Excel workbook.
        """
        if not raw_bytes.startswith(b"PK\x03\x04"):
            return False, "File content does not match Excel (.xlsx) format: Missing ZIP container signature."

        # Verify ZIP archive structure and workbook components
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                namelist = zf.namelist()
                has_workbook = any(
                    name.startswith("xl/") or name == "xl/workbook.xml"
                    for name in namelist
                )
                has_content_types = "[Content_Types].xml" in namelist
                if not (has_workbook or has_content_types):
                    return False, "File is a valid ZIP archive but does not contain Excel spreadsheet parts."
        except zipfile.BadZipFile:
            return False, "Corrupted or invalid Excel file: Bad ZIP archive."
        except Exception as e:
            return False, f"Invalid Excel archive structure: {str(e)}"

        return True, None

    def _validate_csv_content(self, raw_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """
        Validates that CSV content is plain text and not a renamed binary file.
        """
        # Check against known binary signatures
        for sig, sig_name in BINARY_SIGNATURES:
            if raw_bytes.startswith(sig):
                return False, f"File signature matches {sig_name}, not a plain text CSV statement."

        # Check for null bytes in sample (definitive binary indicator)
        sample = raw_bytes[:8192]
        if b"\x00" in sample:
            return False, "File contains binary data (null bytes detected) and is not a valid text-based CSV."

        # Verify readability with supported encodings
        encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
        decoded_text: Optional[str] = None
        for enc in encodings:
            try:
                decoded_text = sample.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if decoded_text is None:
            return False, "Unable to decode file content with supported text encodings (UTF-8, Latin-1, CP1252)."

        # Check non-printable ASCII density in sample
        non_printable = 0
        total_chars = len(decoded_text)
        if total_chars > 0:
            for ch in decoded_text:
                if ord(ch) < 32 and ch not in "\r\n\t":
                    non_printable += 1
            if (non_printable / total_chars) > 0.15:
                return False, "File contains excessive non-printable characters and appears to be a binary format."

        return True, None


def validate_file(
    file_source: Union[str, Path, bytes, BinaryIO],
    filename: Optional[str] = None,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES
) -> FileValidationResult:
    """
    Convenience function to validate an uploaded statement file.
    """
    validator = FileValidator(max_size_bytes=max_size_bytes)
    return validator.validate(file_source, filename=filename)
