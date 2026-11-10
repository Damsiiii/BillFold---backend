"""
API endpoint for bank statement upload and transaction extraction.
Tharanya's Module: Statement Upload & Data Extraction.
Provides POST /api/upload for uploading PDF, CSV, and XLSX statements,
validating file security, format, size, and dispatching to canonical parsers.
"""
import logging
from typing import Optional
from fastapi import FastAPI, APIRouter, UploadFile, File, Query, status
from fastapi.responses import JSONResponse

from models.transaction import ExtractionResult
from services.file_validator import validate_file, sanitize_filename
from services.pdf_parser import parse_pdf
from services.csv_parser import parse_csv
from services.excel_parser import parse_excel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Statement Upload"])


@router.post(
    "/upload",
    response_model=ExtractionResult,
    responses={
        200: {"description": "Successful transaction extraction", "model": ExtractionResult},
        400: {"description": "Invalid file, unsupported format, or validation failure", "model": ExtractionResult},
        413: {"description": "File size exceeds 15 MB limit", "model": ExtractionResult},
        422: {"description": "Unprocessable statement content", "model": ExtractionResult},
        500: {"description": "Internal server error during extraction", "model": ExtractionResult},
    },
    summary="Upload and extract bank statement transactions",
    description="Accepts a PDF, CSV, or XLSX bank statement file, validates it, and extracts normalized transactions into the project canonical schema."
)
def upload_statement(
    file: UploadFile = File(..., description="Bank statement file (.pdf, .csv, .xlsx)"),
    dayfirst: Optional[bool] = Query(
        default=None,
        description="Explicit date convention override: True for DD/MM/YYYY, False for MM/DD/YYYY"
    )
) -> JSONResponse:
    """
    Handles upload, validation, and parsing of bank statements.
    """
    raw_filename = file.filename or "statement"
    safe_name = sanitize_filename(raw_filename) or "statement"

    try:
        # 1. Validate uploaded file using services/file_validator.py
        validation = validate_file(file.file, filename=file.filename)

        if not validation.is_valid:
            # Distinguish between file size limit (413) and general invalid request (400)
            if "exceeds" in (validation.error or "").lower() or (validation.file_size > 15 * 1024 * 1024):
                status_code = 413
            else:
                status_code = 400

            err_result = ExtractionResult(
                success=False,
                filename=validation.safe_filename,
                file_type=validation.file_type or "unknown",
                total_transactions=0,
                transactions=[],
                warnings=validation.warnings,
                error=validation.error,
            )
            return JSONResponse(status_code=status_code, content=err_result.model_dump())

        validated_name = validation.safe_filename
        file_type = validation.file_type

        # 2. Dispatch to the appropriate statement parser
        if file_type == "pdf":
            result = parse_pdf(file.file, filename=validated_name, dayfirst=dayfirst)
        elif file_type == "csv":
            result = parse_csv(file.file, filename=validated_name, dayfirst=dayfirst)
        elif file_type == "xlsx":
            result = parse_excel(file.file, filename=validated_name, dayfirst=dayfirst)
        else:
            # Defensive guard
            err_result = ExtractionResult(
                success=False,
                filename=validated_name,
                file_type=file_type or "unknown",
                total_transactions=0,
                transactions=[],
                warnings=[],
                error=f"Unsupported file type: '{file_type}'. Supported formats: .pdf, .csv, .xlsx",
            )
            return JSONResponse(status_code=400, content=err_result.model_dump())

        # 3. Check parser outcome
        if not result.success:
            return JSONResponse(
                status_code=422,
                content=result.model_dump(),
            )

        # 4. Successful extraction (200 OK)
        return JSONResponse(
            status_code=200,
            content=result.model_dump(),
        )

    except Exception:
        logger.exception(
            "Statement extraction failed unexpectedly",
            extra={"source_filename": safe_name, "stage": "statement_extraction_failed"},
        )
        err_result = ExtractionResult(
            success=False,
            filename=safe_name,
            file_type="unknown",
            total_transactions=0,
            transactions=[],
            warnings=[],
            error="An internal server error occurred while processing the statement.",
        )
        return JSONResponse(
            status_code=500,
            content=err_result.model_dump(),
        )
    finally:
        # 5. Ensure uploaded file is closed properly
        try:
            file.file.close()
        except Exception:
            pass


# Create standalone FastAPI application instance
app = FastAPI(
    title="BillFold Statement Upload API",
    description="API for bank statement uploads and canonical transaction data extraction.",
    version="1.0.0"
)
app.include_router(router)
