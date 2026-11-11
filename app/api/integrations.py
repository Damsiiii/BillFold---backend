"""Integration endpoints for Members 1, 2, and 3.

Provides structured route registration and integration points.
Business logic is deferred to the respective member modules via pluggable dependency handlers.
"""

import io
import hashlib
import json
import logging
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.responses import Response
from starlette.datastructures import UploadFile as StarletteUploadFile
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from api.upload import upload_statement as extract_statement
from app.database import crud
from app.database.database import SessionLocal
from app.models.upload_claim import StatementUploadClaim
from app.models.upload_job import UploadJob
from app.services.analytics_engine import (
    compute_category_spending,
    compute_financial_summary,
)
from app.services.anomaly_detection import detect_anomalies as run_anomaly_detection
from app.services.budget_engine import (
    detect_overspending as run_overspending_detection,
    generate_suggested_budgets,
)
from app.services.savings_engine import calculate_savings_recommendations
from services.file_validator import MAX_FILE_SIZE_BYTES, validate_file
from services.pdf_parser import is_scanned_pdf, ocr_stage_callback
from services.transaction_normalizer import correct_dates_from_leading_descriptions

router = APIRouter(tags=["Team Integrations"])
logger = logging.getLogger(__name__)
OCR_MAX_QUEUED_JOBS = 2
_ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="statement-ocr")
_ocr_job_slots = threading.BoundedSemaphore(OCR_MAX_QUEUED_JOBS)
OCR_JOB_TIMEOUT = timedelta(minutes=5)
OCR_JOB_STALE_AFTER = OCR_JOB_TIMEOUT + timedelta(minutes=1)
OCR_SCAN_DETECTION_TIMEOUT = 60.0
OCR_PERSISTENCE_TIMEOUT = 60.0
OCR_STATUS_UPDATE_TIMEOUT = 15.0
SQLITE_BUSY_TIMEOUT_MS = 5_000
_isolated_process_lock = threading.Lock()
_unreaped_process: Optional[multiprocessing.Process] = None


class OCRProcessTerminationError(RuntimeError):
    """Raised when an isolated operation may still be running."""


class ApiResponseEnvelope(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None


# Pluggable handler registries for Members 1, 2, and 3
# Other members plug in their business logic functions here without modifying route code.
class ModuleRegistry:
    member1_upload_handler: Optional[Callable] = None
    member2_transactions_handler: Optional[Callable] = None
    member3_summary_handler: Optional[Callable] = None
    member3_categories_handler: Optional[Callable] = None
    member3_budget_handler: Optional[Callable] = None
    member3_savings_handler: Optional[Callable] = None
    member3_overspending_handler: Optional[Callable] = None
    member3_anomalies_handler: Optional[Callable] = None


registry = ModuleRegistry()


async def _upload_statement(file: UploadFile, user_id: str) -> dict[str, Any] | Response:
    filename = file.filename or "statement.pdf"
    upload_bytes = await file.read(MAX_FILE_SIZE_BYTES + 1)
    await file.seek(0)
    job_id: Optional[str] = None
    validation = validate_file(
        upload_bytes,
        filename=filename,
        content_type=file.content_type,
    )
    if not validation.is_valid:
        status_code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if validation.file_size > MAX_FILE_SIZE_BYTES
            else status.HTTP_400_BAD_REQUEST
        )
        await file.close()
        return _statement_processing_error(
            validation.safe_filename,
            status_code,
            validation.error or "Uploaded statement failed validation.",
            file_type=validation.file_type or "unknown",
            warnings=validation.warnings,
        )
    filename = validation.safe_filename
    file_type = validation.file_type or "unknown"
    if len(upload_bytes) <= MAX_FILE_SIZE_BYTES:
        content_hash = hashlib.sha256(upload_bytes).hexdigest()
        job, claimed = _claim_statement_upload(user_id, filename, content_hash)
        if not claimed:
            await file.close()
            return _existing_statement_upload_response(job)
        job_id = job.id

    try:
        if (
            file_type == "pdf"
            and len(upload_bytes) <= MAX_FILE_SIZE_BYTES
            and job_id is not None
        ):
            try:
                scanned = await run_in_threadpool(
                    _is_scanned_pdf_isolated, upload_bytes
                )
            except TimeoutError:
                _update_ocr_job(
                    job_id,
                    status_value="failed",
                    error="PDF scan detection exceeded its processing timeout.",
                    expected_status=("processing",),
                )
                logger.error(
                    "PDF scan detection timed out",
                    extra={
                        "source_filename": filename,
                        "stage": "pdf_scan_detection_timeout",
                        "timeout_seconds": OCR_SCAN_DETECTION_TIMEOUT,
                    },
                )
                return _statement_processing_error(
                    filename,
                    status.HTTP_504_GATEWAY_TIMEOUT,
                    "PDF scan detection exceeded its processing timeout.",
                )
            except Exception:
                _update_ocr_job(
                    job_id,
                    status_value="failed",
                    error="An internal server error occurred while processing the statement.",
                    expected_status=("processing",),
                )
                logger.exception(
                    "Could not safely inspect uploaded PDF",
                    extra={
                        "source_filename": filename,
                        "stage": "pdf_scan_detection_failed",
                    },
                )
                return _statement_processing_error(
                    filename,
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "An internal server error occurred while processing the statement.",
                )
            if scanned:
                await file.close()
                return _enqueue_ocr_job(
                    upload_bytes, filename, user_id, job_id, file_type
                )

        if file_type in {"jpg", "jpeg", "png"} and job_id is not None:
            await file.close()
            return _enqueue_ocr_job(
                upload_bytes, filename, user_id, job_id, file_type
            )

        extraction_response = await run_in_threadpool(
            _extract_and_save, file, user_id, job_id
        )
        if extraction_response.status_code != status.HTTP_200_OK:
            if job_id is not None:
                _update_ocr_job(
                    job_id,
                    status_value="failed",
                    error="Statement extraction failed.",
                    expected_status=("processing",),
                )
            return extraction_response
        return json.loads(extraction_response.body)
    except Exception:
        if job_id is not None:
            try:
                _update_ocr_job(
                    job_id,
                    status_value="failed",
                    error="An internal server error occurred while processing the statement.",
                    expected_status=("processing",),
                )
            except Exception:
                logger.exception(
                    "Could not mark statement upload as failed",
                    extra={"job_id": job_id, "stage": "upload_failure_status_failed"},
                )
        raise


def _claim_statement_upload(
    user_id: str, filename: str, content_hash: str
) -> tuple[UploadJob, bool]:
    db = SessionLocal()
    try:
        claim = db.get(StatementUploadClaim, (user_id, content_hash))
        if claim is None:
            job = UploadJob(
                id=str(uuid4()),
                user_id=user_id,
                filename=filename,
                content_hash=content_hash,
                status="processing",
            )
            claim = StatementUploadClaim(
                user_id=user_id,
                content_hash=content_hash,
                job_id=job.id,
            )
            db.add_all((job, claim))
            try:
                db.commit()
                db.refresh(job)
                return job, True
            except IntegrityError:
                db.rollback()
                claim = db.get(StatementUploadClaim, (user_id, content_hash))
                if claim is None:
                    raise RuntimeError(
                        "Could not claim this statement upload safely."
                    )

        job = db.query(UploadJob).filter(UploadJob.id == claim.job_id).first()
        if job is None:
            raise RuntimeError(
                "Statement upload claim refers to a missing upload job."
            )
        if job.status != "failed":
            return job, False

        updated = (
            db.query(UploadJob)
            .filter(UploadJob.id == job.id, UploadJob.status == "failed")
            .update(
                {
                    UploadJob.filename: filename,
                    UploadJob.status: "processing",
                    UploadJob.result_json: None,
                    UploadJob.error: None,
                    UploadJob.is_async: False,
                },
                synchronize_session=False,
            )
        )
        db.commit()
        job = db.query(UploadJob).filter(UploadJob.id == job.id).one()
        return job, bool(updated)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _existing_statement_upload_response(job: UploadJob) -> dict[str, Any] | Response:
    if job.status == "completed" and job.result_json:
        result = json.loads(job.result_json)
        if result.get("file_type") in {"jpg", "jpeg", "png", "pdf"}:
            correct_dates_from_leading_descriptions(result.get("transactions", []))
        return result
    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": job.status,
            "message": "Statement upload has already been accepted.",
            "data": {"job_id": job.id, "status": job.status},
        },
    )


def _statement_processing_error(
    filename: str,
    status_code: int,
    error: str,
    file_type: str = "pdf",
    warnings: Optional[list[str]] = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "filename": filename,
            "file_type": file_type,
            "total_transactions": 0,
            "transactions": [],
            "warnings": warnings or [],
            "error": error,
        },
    )


def _transactions_for_persistence(
    extraction_result: dict[str, Any],
) -> list[dict[str, Any]]:
    transactions = []
    for transaction in extraction_result["transactions"]:
        transaction_type = transaction["transaction_type"].strip().lower()
        if transaction_type == "expense":
            transaction_type = "debit"
        elif transaction_type == "income":
            transaction_type = "credit"
        transactions.append(
            {
                "date": date.fromisoformat(transaction["transaction_date"]),
                "description": transaction["description"],
                "amount": transaction["amount"],
                "type": transaction_type,
                "category": transaction["category"],
                "currency": transaction["currency"],
            }
        )
    return transactions


def _persist_extraction(
    extraction_response: Response, user_id: str, job_id: Optional[str] = None
) -> Response:
    if extraction_response.status_code != status.HTTP_200_OK:
        return extraction_response
    extraction_result = json.loads(extraction_response.body)
    transactions = _transactions_for_persistence(extraction_result)

    db = SessionLocal()
    try:
        logger.info(
            "Persisting extracted statement transactions",
            extra={
                "job_id": job_id,
                "stage": "database_persistence_started",
                "user_id": user_id,
            },
        )
        crud.save_transactions(
            db,
            user_id=user_id,
            transactions=transactions,
            source_file=extraction_result["filename"],
            commit=False,
        )
        if job_id is not None:
            job = db.query(UploadJob).filter(UploadJob.id == job_id).first()
            if job is None or job.status != "processing":
                raise LookupError("Statement upload is no longer processing.")
            job.status = "completed"
            job.result_json = json.dumps(extraction_result)
            job.error = None
        db.commit()
        logger.info(
            "Persisted extracted statement transactions",
            extra={
                "job_id": job_id,
                "stage": "database_persistence_completed",
                "user_id": user_id,
            },
        )
    except Exception as exc:
        logger.exception("Failed to persist extracted statement transactions")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred while processing the statement.",
        ) from exc
    finally:
        db.close()

    return extraction_response


def _extract_and_save(
    file: UploadFile, user_id: str, job_id: Optional[str] = None
) -> Response:
    return _persist_extraction(
        extract_statement(file, dayfirst=None), user_id, job_id
    )


def _extract_statement_in_child(
    pdf_bytes: bytes, filename: str, connection
) -> None:
    def report_stage(
        stage: str,
        page_number: Optional[int] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> None:
        try:
            connection.send(
                {
                    "type": "stage",
                    "stage": stage,
                    "page_number": page_number,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
        except (BrokenPipeError, EOFError, OSError):
            logger.warning(
                "Could not report OCR progress to the supervising worker",
                extra={"source_filename": filename, "stage": "progress_channel_closed"},
            )

    callback_token = ocr_stage_callback.set(report_stage)
    try:
        upload = StarletteUploadFile(filename=filename, file=io.BytesIO(pdf_bytes))
        response = extract_statement(upload, dayfirst=None)
        connection.send(
            {
                "type": "result",
                "status_code": response.status_code,
                "body": response.body,
            }
        )
    except Exception:
        logger.exception(
            "Isolated statement extraction process failed",
            extra={"source_filename": filename, "stage": "isolated_extraction_failed"},
        )
        try:
            connection.send({"type": "error"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        ocr_stage_callback.reset(callback_token)
        connection.close()


def _detect_scanned_pdf_in_child(pdf_bytes: bytes, connection) -> None:
    try:
        connection.send(
            {"type": "scan_detection", "is_scanned": is_scanned_pdf(pdf_bytes)}
        )
    except (BrokenPipeError, EOFError, OSError):
        logger.warning(
            "Could not return PDF scan-detection result",
            extra={"stage": "scan_detection_channel_closed"},
        )
    finally:
        connection.close()


def _is_scanned_pdf_isolated(pdf_bytes: bytes) -> bool:
    message = _run_isolated_process(
        _detect_scanned_pdf_in_child,
        (pdf_bytes,),
        OCR_SCAN_DETECTION_TIMEOUT,
        lambda _stage, _page_number, _elapsed_seconds: None,
    )
    if message is None:
        raise TimeoutError("PDF scan detection exceeded its timeout.")
    if message.get("type") != "scan_detection":
        raise RuntimeError("PDF scan detection failed in the isolated process.")
    return bool(message["is_scanned"])


def _persist_ocr_job_in_child(
    extraction_result: dict[str, Any],
    user_id: str,
    job_id: str,
    database_url: str,
    connection,
) -> None:
    engine = None
    db = None
    sqlite_connection = None
    try:
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False, "timeout": 5.0},
        )
        db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
        if db.get_bind().dialect.name != "sqlite":
            raise RuntimeError("Bounded OCR persistence requires SQLite.")

        deadline = time.monotonic() + OCR_PERSISTENCE_TIMEOUT
        sqlite_connection = db.connection().connection.driver_connection
        sqlite_connection.execute(
            f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}"
        )
        sqlite_connection.set_progress_handler(
            lambda: int(time.monotonic() >= deadline),
            1000,
        )
        saved_count = crud.save_transactions(
            db,
            user_id=user_id,
            transactions=_transactions_for_persistence(extraction_result),
            source_file=extraction_result["filename"],
            deadline=deadline,
            commit=False,
        )
        job = db.query(UploadJob).filter(UploadJob.id == job_id).first()
        if job is None or job.status != "processing":
            raise LookupError("Scanned statement job is no longer processing.")
        job.status = "completed"
        job.result_json = json.dumps(extraction_result)
        job.error = None
        db.commit()
        connection.send(
            {"type": "persistence_completed", "saved_count": saved_count}
        )
    except Exception:
        if db is not None:
            db.rollback()
        logger.exception(
            "Isolated OCR transaction persistence failed",
            extra={"job_id": job_id, "stage": "database_persistence_failed"},
        )
        try:
            connection.send({"type": "error"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if sqlite_connection is not None:
            try:
                sqlite_connection.set_progress_handler(None, 0)
            except Exception:
                logger.exception(
                    "Could not clear SQLite persistence deadline handler",
                    extra={
                        "job_id": job_id,
                        "stage": "persistence_deadline_cleanup_failed",
                    },
                )
        if db is not None:
            db.close()
        if engine is not None:
            engine.dispose()
        connection.close()


def _persist_ocr_job_isolated(
    extraction_result: dict[str, Any],
    user_id: str,
    job_id: str,
) -> Response:
    bound_engine = SessionLocal.kw.get("bind")
    if bound_engine is None or bound_engine.dialect.name != "sqlite":
        return _statement_processing_error(
            extraction_result["filename"],
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Bounded OCR persistence requires SQLite.",
        )

    try:
        message = _run_isolated_process(
            _persist_ocr_job_in_child,
            (
                extraction_result,
                user_id,
                job_id,
                str(bound_engine.url),
            ),
            OCR_PERSISTENCE_TIMEOUT,
            lambda _stage, _page_number, _elapsed_seconds: None,
        )
    except Exception:
        logger.exception(
            "Could not safely persist OCR transactions",
            extra={
                "job_id": job_id,
                "source_filename": extraction_result["filename"],
                "stage": "database_persistence_process_failed",
            },
        )
        return _statement_processing_error(
            extraction_result["filename"],
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "An internal server error occurred while processing the statement.",
        )

    if message is None:
        return _statement_processing_error(
            extraction_result["filename"],
            status.HTTP_504_GATEWAY_TIMEOUT,
            "Statement transaction persistence exceeded its processing timeout.",
        )
    if message.get("type") != "persistence_completed":
        return _statement_processing_error(
            extraction_result["filename"],
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "An internal server error occurred while processing the statement.",
        )
    return Response(content=b"{}", status_code=status.HTTP_200_OK)


def _update_ocr_job_in_child(
    job_id: str,
    status_value: str,
    result: Optional[dict[str, Any]],
    error: Optional[str],
    expected_status: Optional[tuple[str, ...]],
    database_url: str,
    connection,
) -> None:
    engine = None
    db = None
    try:
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False, "timeout": 5.0},
        )
        db = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
        job = db.query(UploadJob).filter(UploadJob.id == job_id).first()
        if job is None:
            raise LookupError(f"Scanned statement job {job_id} was not found.")
        if expected_status is not None and job.status not in expected_status:
            connection.send({"type": "status_updated", "updated": False})
            return
        job.status = status_value
        job.result_json = json.dumps(result) if result is not None else None
        job.error = error
        db.commit()
        connection.send({"type": "status_updated", "updated": True})
    except Exception:
        if db is not None:
            db.rollback()
        logger.exception(
            "Isolated OCR job-status update failed",
            extra={"job_id": job_id, "stage": "job_status_update_failed"},
        )
        try:
            connection.send({"type": "error"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if db is not None:
            db.close()
        if engine is not None:
            engine.dispose()
        connection.close()


def _update_ocr_job_isolated(
    job_id: str,
    status_value: str,
    result: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
    expected_status: Optional[tuple[str, ...]] = None,
) -> Optional[bool]:
    bound_engine = SessionLocal.kw.get("bind")
    if bound_engine is None or bound_engine.dialect.name != "sqlite":
        logger.error(
            "Could not safely update OCR job status without SQLite",
            extra={"job_id": job_id, "stage": "job_status_update_unsupported"},
        )
        return None
    try:
        message = _run_isolated_process(
            _update_ocr_job_in_child,
            (
                job_id,
                status_value,
                result,
                error,
                expected_status,
                str(bound_engine.url),
            ),
            OCR_STATUS_UPDATE_TIMEOUT,
            lambda _stage, _page_number, _elapsed_seconds: None,
        )
    except Exception:
        logger.exception(
            "Could not safely update OCR job status",
            extra={"job_id": job_id, "stage": "job_status_update_process_failed"},
        )
        return None
    if message is None or message.get("type") != "status_updated":
        return None
    return bool(message["updated"])


def _terminate_process(process: multiprocessing.Process) -> bool:
    for action in (process.terminate, process.kill):
        try:
            if not process.is_alive():
                return True
            action()
            process.join(timeout=1)
        except Exception:
            logger.exception(
                "Could not terminate isolated statement process",
                extra={"stage": "isolated_process_termination_attempt_failed"},
            )
    try:
        if not process.is_alive():
            return True
    except Exception:
        logger.exception(
            "Could not determine whether isolated statement process is alive",
            extra={"stage": "isolated_process_state_unknown"},
        )
    logger.critical(
        "Isolated statement process remains alive after terminate and kill",
        extra={"stage": "isolated_process_termination_failed"},
    )
    return False


def _run_isolated_process(
    target: Callable,
    args: tuple[Any, ...],
    timeout_seconds: float,
    on_stage: Callable[[str, Optional[int], Optional[float]], None],
) -> Optional[dict[str, Any]]:
    global _unreaped_process
    deadline = time.monotonic() + timeout_seconds
    if not _isolated_process_lock.acquire(timeout=timeout_seconds):
        return None
    receive_connection = None
    send_connection = None
    process = None
    try:
        if _unreaped_process is not None:
            if _unreaped_process.is_alive():
                raise OCRProcessTerminationError(
                    "A previous isolated statement process is still alive."
                )
            _unreaped_process.close()
            _unreaped_process = None

        if time.monotonic() >= deadline:
            return None
        context = multiprocessing.get_context("spawn")
        receive_connection, send_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=target,
            args=(*args, send_connection),
            daemon=True,
        )
        try:
            process.start()
        except Exception:
            logger.exception("Could not start isolated statement extraction process")
            raise
        send_connection.close()
        _unreaped_process = process

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if not _terminate_process(process):
                    raise OCRProcessTerminationError(
                        "Timed-out statement process could not be terminated."
                    )
                return None
            if receive_connection.poll(min(0.1, remaining)):
                message = receive_connection.recv()
                if message["type"] == "stage":
                    on_stage(
                        message["stage"],
                        message.get("page_number"),
                        message.get("elapsed_seconds"),
                    )
                    continue
                process.join(timeout=1)
                if process.is_alive() and not _terminate_process(process):
                    raise OCRProcessTerminationError(
                        "Statement process remained alive after returning a result."
                    )
                return message
            if not process.is_alive():
                if receive_connection.poll():
                    continue
                return {"type": "error"}
    finally:
        if receive_connection is not None:
            receive_connection.close()
        if send_connection is not None:
            send_connection.close()
        if process is not None and process.pid is not None:
            try:
                if process.is_alive() and not _terminate_process(process):
                    _unreaped_process = process
                if not process.is_alive():
                    process.close()
                    if _unreaped_process is process:
                        _unreaped_process = None
            finally:
                _isolated_process_lock.release()
        else:
            _isolated_process_lock.release()


def _run_isolated_extraction(
    pdf_bytes: bytes,
    filename: str,
    progress: dict[str, Any],
    progress_lock: threading.Lock,
    file_type: str = "pdf",
) -> Response:
    extraction_started = time.monotonic()

    def record_stage(
        stage: str,
        page_number: Optional[int] = None,
        operation_elapsed: Optional[float] = None,
    ) -> None:
        _record_job_stage(
            progress,
            progress_lock,
            stage,
            page_number=page_number,
            operation_elapsed=operation_elapsed,
        )
        with progress_lock:
            job_id = progress["job_id"]
        logger.info(
            "Scanned statement extraction stage",
            extra={
                "job_id": job_id,
                "source_filename": filename,
                "stage": stage,
                "page_number": page_number,
                "elapsed_seconds": round(time.monotonic() - extraction_started, 3),
                "operation_elapsed_seconds": (
                    round(operation_elapsed, 3)
                    if operation_elapsed is not None
                    else None
                ),
            },
        )

    try:
        message = _run_isolated_process(
            _extract_statement_in_child,
            (pdf_bytes, filename),
            OCR_JOB_TIMEOUT.total_seconds(),
            record_stage,
        )
    except Exception:
        logger.exception(
            "Could not run isolated statement extraction",
            extra={"source_filename": filename, "stage": "isolated_process_start_failed"},
        )
        message = {"type": "error"}

    if message is None:
        with progress_lock:
            stalled_stage = progress["stage"]
            stalled_page = progress.get("page_number")
        elapsed_seconds = time.monotonic() - extraction_started
        error = (
            f"Statement processing exceeded the {OCR_JOB_TIMEOUT} timeout "
            f"during stage '{stalled_stage}'. Please retry."
        )
        logger.error(
            "Isolated statement extraction timed out",
            extra={
                "source_filename": filename,
                "job_id": progress.get("job_id"),
                "stage": "job_timeout",
                "stalled_stage": stalled_stage,
                "page_number": stalled_page,
                "elapsed_seconds": round(elapsed_seconds, 3),
            },
        )
        status_code = status.HTTP_504_GATEWAY_TIMEOUT
        payload = {
            "success": False,
            "filename": filename,
            "file_type": file_type,
            "total_transactions": 0,
            "transactions": [],
            "warnings": [],
            "error": error,
        }
        return JSONResponse(status_code=status_code, content=payload)

    if message.get("type") != "result":
        payload = {
            "success": False,
            "filename": filename,
            "file_type": file_type,
            "total_transactions": 0,
            "transactions": [],
            "warnings": [],
            "error": "An internal server error occurred while processing the statement.",
        }
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload,
        )
    return Response(content=message["body"], status_code=message["status_code"])


def _enqueue_ocr_job(
    pdf_bytes: bytes,
    filename: str,
    user_id: str,
    job_id: str,
    file_type: str = "pdf",
) -> JSONResponse:
    if not _ocr_job_slots.acquire(blocking=False):
        _update_ocr_job(
            job_id,
            status_value="failed",
            error="Statement processing queue is full. Please retry.",
            expected_status=("processing",),
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "message": "Statement processing is busy. Please retry shortly.",
            },
            headers={"Retry-After": "30"},
        )

    db = SessionLocal()
    try:
        job = db.query(UploadJob).filter(UploadJob.id == job_id).one()
        job.filename = filename
        job.status = "queued"
        job.is_async = True
        db.commit()
    except Exception as exc:
        db.rollback()
        _ocr_job_slots.release()
        logger.exception("Failed to create statement upload job")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not queue statement for processing.",
        ) from exc
    finally:
        db.close()

    progress_lock = threading.Lock()
    progress: dict[str, Any] = {
        "stage": "job_queued",
        "job_id": job_id,
        "source_filename": filename,
        "page_number": None,
    }
    try:
        logger.info(
            "Queued statement processing job",
            extra={
                "job_id": job_id,
                "source_filename": filename,
                "stage": "job_queued",
            },
        )
        _ocr_executor.submit(
            _process_ocr_job,
            job_id,
            pdf_bytes,
            filename,
            user_id,
            progress,
            progress_lock,
            file_type,
        )
    except Exception as exc:
        _ocr_job_slots.release()
        _update_ocr_job(
            job_id,
            status_value="failed",
            error="Scanned statement could not be queued for processing.",
        )
        logger.exception("Failed to submit statement job %s", job_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Statement processing is temporarily unavailable.",
        ) from exc

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "processing",
            "message": "Statement accepted for background processing.",
            "data": {"job_id": job_id, "status": "queued"},
        },
    )


def _update_ocr_job(
    job_id: str,
    status_value: str,
    result: Optional[dict[str, Any]] = None,
    error: Optional[str] = None,
    expected_status: Optional[tuple[str, ...]] = None,
) -> bool:
    db = SessionLocal()
    try:
        job = db.query(UploadJob).filter(UploadJob.id == job_id).first()
        if job is None:
            raise LookupError(f"Scanned statement job {job_id} was not found.")
        if expected_status is not None and job.status not in expected_status:
            logger.info(
                "Ignored stale scanned statement status transition",
                extra={
                    "job_id": job_id,
                    "stage": "status_transition_ignored",
                    "current_status": job.status,
                    "requested_status": status_value,
                },
            )
            return False
        job.status = status_value
        job.result_json = json.dumps(result) if result is not None else None
        job.error = error
        db.commit()
        logger.info(
            "Persisted scanned statement job status",
            extra={
                "job_id": job_id,
                "stage": "job_status_persisted",
                "job_status": status_value,
            },
        )
        return True
    except Exception:
        db.rollback()
        logger.exception("Failed to update scanned statement job %s", job_id)
        raise
    finally:
        db.close()


def expire_stale_ocr_jobs() -> int:
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - OCR_JOB_STALE_AFTER
    db = SessionLocal()
    try:
        stale_jobs = (
            db.query(UploadJob)
            .filter(
                UploadJob.status.in_(("queued", "processing")),
                UploadJob.updated_at < cutoff,
            )
            .all()
        )
        for job in stale_jobs:
            job.status = "failed"
            job.error = (
                "Statement processing stopped before completion. Please upload it again."
            )
        if stale_jobs:
            db.commit()
            logger.warning(
                "Marked stale statement upload jobs as failed",
                extra={
                    "stage": "stale_jobs_failed",
                    "job_count": len(stale_jobs),
                },
            )
        return len(stale_jobs)
    except Exception:
        db.rollback()
        logger.exception("Could not expire stale statement upload jobs")
        raise
    finally:
        db.close()


def _record_job_stage(
    progress: dict[str, Any],
    progress_lock: threading.Lock,
    stage: str,
    page_number: Optional[int] = None,
    operation_elapsed: Optional[float] = None,
) -> None:
    with progress_lock:
        progress["stage"] = stage
        progress["page_number"] = page_number
        progress["operation_elapsed_seconds"] = operation_elapsed


def _process_ocr_job(
    job_id: str,
    pdf_bytes: bytes,
    filename: str,
    user_id: str,
    progress: dict[str, Any],
    progress_lock: threading.Lock,
    file_type: str = "pdf",
) -> None:
    job_started = time.monotonic()
    try:
        _record_job_stage(progress, progress_lock, "job_dequeued")
        logger.info(
            "Statement worker dequeued job",
            extra={
                "job_id": job_id,
                "source_filename": filename,
                "stage": "job_dequeued",
                "elapsed_seconds": 0.0,
            },
        )
        started = _update_ocr_job_isolated(
            job_id,
            status_value="processing",
            expected_status=("queued",),
        )
        if not started:
            return

        _record_job_stage(progress, progress_lock, "extraction_started")
        logger.info(
            "Started statement extraction",
            extra={
                "job_id": job_id,
                "source_filename": filename,
                "stage": "extraction_started",
                "elapsed_seconds": round(time.monotonic() - job_started, 3),
            },
        )
        if file_type == "pdf":
            extraction_response = _run_isolated_extraction(
                pdf_bytes, filename, progress, progress_lock
            )
        else:
            extraction_response = _run_isolated_extraction(
                pdf_bytes, filename, progress, progress_lock, file_type
            )
        result = json.loads(extraction_response.body)
        _record_job_stage(progress, progress_lock, "extraction_completed")
        logger.info(
            "Statement extraction returned",
            extra={
                "job_id": job_id,
                "source_filename": filename,
                "stage": "extraction_completed",
                "http_status": extraction_response.status_code,
                "elapsed_seconds": round(time.monotonic() - job_started, 3),
            },
        )

        still_processing = _update_ocr_job_isolated(
            job_id,
            status_value="processing",
            expected_status=("processing",),
        )
        if not still_processing:
            logger.warning(
                "Discarding extraction that returned after its job timed out",
                extra={
                    "job_id": job_id,
                    "source_filename": filename,
                    "stage": "late_extraction_discarded",
                },
            )
            return
        if extraction_response.status_code == status.HTTP_200_OK:
            _record_job_stage(progress, progress_lock, "database_persistence_started")
            persistence_started = time.monotonic()
            logger.info(
                "Statement transaction persistence started",
                extra={
                    "job_id": job_id,
                    "source_filename": filename,
                    "stage": "database_persistence_started",
                },
            )
            persistence_response = _persist_ocr_job_isolated(
                result, user_id, job_id
            )
            _record_job_stage(progress, progress_lock, "database_persistence_completed")
            logger.info(
                "Statement transaction persistence completed",
                extra={
                    "job_id": job_id,
                    "source_filename": filename,
                    "stage": "database_persistence_completed",
                    "elapsed_seconds": round(time.monotonic() - persistence_started, 3),
                    "total_elapsed_seconds": round(
                        time.monotonic() - job_started, 3
                    ),
                },
            )
            if persistence_response.status_code == status.HTTP_200_OK:
                _record_job_stage(progress, progress_lock, "job_completion_started")
                logger.info(
                    "Scanned statement job completed",
                    extra={
                        "job_id": job_id,
                        "source_filename": filename,
                        "stage": "job_completed",
                        "elapsed_seconds": round(time.monotonic() - job_started, 3),
                    },
                )
            else:
                persistence_result = json.loads(persistence_response.body)
                failure_saved = _update_ocr_job_isolated(
                    job_id,
                    status_value="failed",
                    result=persistence_result,
                    error=persistence_result.get(
                        "error", "Statement transaction persistence failed."
                    ),
                    expected_status=("processing",),
                )
                if not failure_saved:
                    logger.error(
                        "Could not persist OCR job failure status",
                        extra={
                            "job_id": job_id,
                            "source_filename": filename,
                            "stage": "job_failure_status_persistence_failed",
                        },
                    )
                logger.error(
                    "Scanned statement job failed during transaction persistence",
                    extra={
                        "job_id": job_id,
                        "source_filename": filename,
                        "stage": "job_failed",
                        "http_status": persistence_response.status_code,
                        "elapsed_seconds": round(time.monotonic() - job_started, 3),
                    },
                )
        else:
            failure_saved = _update_ocr_job_isolated(
                job_id,
                status_value="failed",
                result=result,
                error=result.get("error", "Statement extraction failed."),
                expected_status=("processing",),
            )
            if not failure_saved:
                logger.error(
                    "Could not persist OCR job failure status",
                    extra={
                        "job_id": job_id,
                        "source_filename": filename,
                        "stage": "job_failure_status_persistence_failed",
                    },
                )
            logger.info(
                "Scanned statement job failed",
                extra={
                    "job_id": job_id,
                    "source_filename": filename,
                    "stage": "job_failed",
                    "elapsed_seconds": round(time.monotonic() - job_started, 3),
                },
            )
    except Exception as exc:
        _record_job_stage(progress, progress_lock, "worker_failed")
        logger.exception(
            "Background processing failed for scanned statement job %s",
            job_id,
            extra={
                "job_id": job_id,
                "source_filename": filename,
                "stage": "worker_failed",
                "elapsed_seconds": round(time.monotonic() - job_started, 3),
            },
        )
        failure_saved = _update_ocr_job_isolated(
            job_id,
            status_value="failed",
            error="An internal server error occurred while processing the statement.",
            expected_status=("processing",),
        )
        if not failure_saved:
            logger.error(
                "Could not save failure state for OCR job %s",
                job_id,
                extra={
                    "job_id": job_id,
                    "stage": "failure_state_persistence_failed",
                },
            )
    finally:
        _ocr_job_slots.release()


@router.get(
    "/upload-status/{job_id}",
    response_model=ApiResponseEnvelope,
    summary="Get statement upload status",
    description="Poll the status of a scanned PDF or statement image processed asynchronously.",
)
async def get_upload_status(
    job_id: str,
    user_id: str = Query(default="demo_user", description="User ID associated with the upload"),
):
    expire_stale_ocr_jobs()
    db = SessionLocal()
    try:
        job = (
            db.query(UploadJob)
            .filter(UploadJob.id == job_id, UploadJob.user_id == user_id)
            .first()
        )
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Statement upload job was not found.",
            )
        data: dict[str, Any] = {
            "job_id": job.id,
            "status": job.status,
            "filename": job.filename,
        }
        if job.result_json:
            result = json.loads(job.result_json)
            if result.get("file_type") in {"jpg", "jpeg", "png", "pdf"}:
                correct_dates_from_leading_descriptions(
                    result.get("transactions", [])
                )
            data["result"] = result
        if job.error:
            data["error"] = job.error
        return ApiResponseEnvelope(
            status=job.status,
            message=f"Statement upload is {job.status}.",
            data=data,
        )
    finally:
        db.close()


def _load_user_transactions(user_id: str) -> pd.DataFrame:
    db = SessionLocal()
    try:
        rows = crud.get_transactions(db, user_id=user_id, limit=None)
        transactions = []
        for row in rows:
            transaction = crud.to_dict(row)
            transaction_type = transaction["type"].strip().lower()
            if transaction_type == "debit":
                transaction_type = "expense"
            elif transaction_type == "credit":
                transaction_type = "income"
            transactions.append(
                {
                    "transaction_id": transaction["id"],
                    "transaction_date": transaction["date"],
                    "description": transaction["description"],
                    "amount": transaction["amount"],
                    "currency": transaction["currency"],
                    "transaction_type": transaction_type,
                    "category": transaction["category"],
                    "source_dataset": row.source_file,
                }
            )
        return pd.DataFrame(transactions)
    finally:
        db.close()


async def _get_transactions(user_id: str, limit: int) -> dict[str, Any]:
    db = SessionLocal()
    try:
        rows = crud.get_transactions(db, user_id=user_id, limit=limit)
        return {
            "transactions": [crud.to_dict(row) for row in rows],
            "count": len(rows),
        }
    finally:
        db.close()


async def _get_summary(user_id: str) -> dict[str, Any]:
    return compute_financial_summary(_load_user_transactions(user_id))


async def _get_categories(user_id: str) -> list[dict[str, Any]]:
    return compute_category_spending(_load_user_transactions(user_id))


async def _get_budgets(user_id: str) -> dict[str, Any]:
    return generate_suggested_budgets(_load_user_transactions(user_id))


async def _get_savings(user_id: str) -> dict[str, Any]:
    return calculate_savings_recommendations(_load_user_transactions(user_id))


async def _get_overspending(user_id: str) -> dict[str, Any]:
    return run_overspending_detection(_load_user_transactions(user_id))


async def _get_anomalies(user_id: str) -> dict[str, Any]:
    return run_anomaly_detection(_load_user_transactions(user_id))


registry.member1_upload_handler = _upload_statement
registry.member2_transactions_handler = _get_transactions
registry.member3_summary_handler = _get_summary
registry.member3_categories_handler = _get_categories
registry.member3_budget_handler = _get_budgets
registry.member3_savings_handler = _get_savings
registry.member3_overspending_handler = _get_overspending
registry.member3_anomalies_handler = _get_anomalies


def _not_implemented_response(module_name: str, responsible_member: str) -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=f"{module_name} is awaiting integration from {responsible_member}.",
    )


# Member 1: Statement Upload and Extraction
@router.post(
    "/upload-statement",
    response_model=ApiResponseEnvelope,
    responses={
        202: {"description": "Scanned PDF or statement image accepted for background processing."},
        503: {"description": "Statement processing queue is full or unavailable."},
    },
    summary="Upload Statement (Member 1)",
    description=(
        "Accepts PDF, CSV, XLS, XLSX, JPG, JPEG, or PNG bank statements for extraction."
    ),
)
async def upload_statement(
    file: UploadFile = File(
        ...,
        description="Bank statement file (.pdf, .csv, .xls, .xlsx, .jpg, .jpeg, .png)",
    ),
    user_id: str = Query(default="demo_user", description="User ID associated with the statement"),
):
    if registry.member1_upload_handler:
        result = await registry.member1_upload_handler(file=file, user_id=user_id)
        if isinstance(result, Response):
            return result
        return ApiResponseEnvelope(
            status="success",
            message="Statement processed successfully",
            data=result,
        )
    _not_implemented_response("Statement Upload & Extraction", "Member 1")


# Member 2: Transactions Retrieval
@router.get(
    "/transactions",
    response_model=ApiResponseEnvelope,
    summary="Get Transactions (Member 2)",
    description="Retrieves processed and categorized user transactions from SQLite database.",
)
async def get_transactions(
    user_id: str = Query(default="demo_user", description="User ID"),
    limit: int = Query(default=50, ge=1, le=500),
):
    if registry.member2_transactions_handler:
        result = await registry.member2_transactions_handler(user_id=user_id, limit=limit)
        return ApiResponseEnvelope(
            status="success",
            message="Transactions retrieved successfully",
            data=result,
        )
    _not_implemented_response("Transaction Storage & Retrieval", "Member 2")


# Member 3: Analytics and Recommendations
@router.get(
    "/analytics/summary",
    response_model=ApiResponseEnvelope,
    summary="Financial Summary (Member 3)",
    description="Provides total income, expenses, and net cash flow.",
)
async def get_financial_summary(
    user_id: str = Query(default="demo_user", description="User ID"),
):
    if registry.member3_summary_handler:
        result = await registry.member3_summary_handler(user_id=user_id)
        return ApiResponseEnvelope(
            status="success",
            message="Financial summary computed successfully",
            data=result,
        )
    _not_implemented_response("Financial Summary Analytics", "Member 3")


@router.get(
    "/analytics/categories",
    response_model=ApiResponseEnvelope,
    summary="Category Breakdown (Member 3)",
    description="Provides category-wise spending breakdown and percentages.",
)
async def get_category_analytics(
    user_id: str = Query(default="demo_user", description="User ID"),
):
    if registry.member3_categories_handler:
        result = await registry.member3_categories_handler(user_id=user_id)
        return ApiResponseEnvelope(
            status="success",
            message="Category breakdown computed successfully",
            data=result,
        )
    _not_implemented_response("Category Analytics", "Member 3")


@router.get(
    "/recommendations/budget",
    response_model=ApiResponseEnvelope,
    summary="Budget Recommendations (Member 3)",
    description="Provides personalized monthly budget recommendations.",
)
async def get_budget_recommendations(
    user_id: str = Query(default="demo_user", description="User ID"),
):
    if registry.member3_budget_handler:
        result = await registry.member3_budget_handler(user_id=user_id)
        return ApiResponseEnvelope(
            status="success",
            message="Budget recommendations generated successfully",
            data=result,
        )
    _not_implemented_response("Budget Recommendations", "Member 3")


@router.get(
    "/recommendations/savings",
    response_model=ApiResponseEnvelope,
    summary="Savings Recommendations (Member 3)",
    description="Provides actionable opportunities for monthly savings.",
)
async def get_savings_recommendations(
    user_id: str = Query(default="demo_user", description="User ID"),
):
    if registry.member3_savings_handler:
        result = await registry.member3_savings_handler(user_id=user_id)
        return ApiResponseEnvelope(
            status="success",
            message="Savings recommendations generated successfully",
            data=result,
        )
    _not_implemented_response("Savings Recommendations", "Member 3")


@router.get(
    "/detection/overspending",
    response_model=ApiResponseEnvelope,
    summary="Overspending Detection (Member 3)",
    description="Detects categories exceeding expected budget limits.",
)
async def detect_overspending(
    user_id: str = Query(default="demo_user", description="User ID"),
):
    if registry.member3_overspending_handler:
        result = await registry.member3_overspending_handler(user_id=user_id)
        return ApiResponseEnvelope(
            status="success",
            message="Overspending detection completed successfully",
            data=result,
        )
    _not_implemented_response("Overspending Detection", "Member 3")


@router.get(
    "/detection/anomalies",
    response_model=ApiResponseEnvelope,
    summary="Anomaly Detection (Member 3)",
    description="Identifies unusual or suspicious transactions.",
)
async def detect_anomalies(
    user_id: str = Query(default="demo_user", description="User ID"),
):
    if registry.member3_anomalies_handler:
        result = await registry.member3_anomalies_handler(user_id=user_id)
        return ApiResponseEnvelope(
            status="success",
            message="Anomaly detection completed successfully",
            data=result,
        )
    _not_implemented_response("Anomaly Detection", "Member 3")
