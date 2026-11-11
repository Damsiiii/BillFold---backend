import io
import hashlib
import json
import threading
import time
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app
from app.api import integrations
from app.api.integrations import registry
from app.database import database as database_module
from app.database.database import Base
from app.models.transaction import Transaction
from app.models.upload_claim import StatementUploadClaim
from app.models.upload_job import UploadJob
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.test_parsers import (
    create_test_scanned_pdf,
    create_test_table_pdf,
    create_test_xlsx,
)

client = TestClient(app)


def _blocked_child_process(connection):
    connection.send(
        {
            "type": "stage",
            "stage": "ocr_inference_started",
            "page_number": 2,
            "elapsed_seconds": None,
        }
    )
    time.sleep(30)


def _successful_child_process(connection):
    connection.send({"type": "result", "status_code": 200, "body": b"done"})


class _UnstoppableProcess:
    def __init__(self):
        self.terminate_calls = 0
        self.kill_calls = 0

    def is_alive(self):
        return True

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1

    def join(self, timeout=None):
        pass


HANDLER_NAMES = (
    "member1_upload_handler",
    "member2_transactions_handler",
    "member3_summary_handler",
    "member3_categories_handler",
    "member3_budget_handler",
    "member3_savings_handler",
    "member3_overspending_handler",
    "member3_anomalies_handler",
)


@pytest.fixture(autouse=True)
def reset_registry():
    original_handlers = {name: getattr(registry, name) for name in HANDLER_NAMES}
    yield
    for name, handler in original_handlers.items():
        setattr(registry, name, handler)


@pytest.fixture
def sqlite_database(monkeypatch, tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'integrations.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    test_session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(integrations, "SessionLocal", test_session)

    db = test_session()
    db.add_all(
        [
            Transaction(
                user_id="alice",
                date=date(2026, 1, 1),
                description="Monthly salary",
                amount=4000.0,
                type="credit",
                category="Income",
                currency="LKR",
            ),
            Transaction(
                user_id="alice",
                date=date(2026, 1, 2),
                description="Grocery store",
                amount=500.0,
                type="debit",
                category="Food",
                currency="LKR",
            ),
            Transaction(
                user_id="bob",
                date=date(2026, 1, 3),
                description="Other user's purchase",
                amount=1000.0,
                type="debit",
                category="Shopping",
                currency="LKR",
            ),
        ]
    )
    db.commit()
    db.close()
    yield
    engine.dispose()


@pytest.mark.parametrize("file_type", ("csv", "xlsx", "pdf"))
def test_member1_upload_extracts_and_saves_transactions_for_user(sqlite_database, file_type):
    filename = f"january_statement.{file_type}"
    if file_type == "csv":
        content = b"Date,Description,Amount,Balance\n2026-01-05,Market,-85.50,1414.50\n"
        media_type = "text/csv"
    elif file_type == "xlsx":
        content = create_test_xlsx(
            [
                ["Date", "Description", "Amount", "Balance"],
                ["2026-01-05", "Market", -85.50, 1414.50],
            ]
        )
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        content = create_test_table_pdf(
            ["Date", "Description", "Amount", "Balance"],
            [["2026-01-05", "Market", "-85.50", "1414.50"]],
        )
        media_type = "application/pdf"

    files = {
        "file": (
            filename,
            content,
            media_type,
        )
    }

    response = client.post("/upload-statement?user_id=alice", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["success"] is True
    assert body["data"]["total_transactions"] == 1
    assert body["data"]["transactions"][0]["description"] == "Market"

    alice_transactions = client.get("/transactions?user_id=alice").json()["data"]
    assert alice_transactions["count"] == 3
    saved_transaction = next(
        row for row in alice_transactions["transactions"] if row["description"] == "Market"
    )
    assert saved_transaction["user_id"] == "alice"
    assert saved_transaction["type"] == "debit"
    assert saved_transaction["amount"] == 85.5
    bob_transactions = client.get("/transactions?user_id=bob").json()["data"]
    assert bob_transactions["count"] == 1


def test_member1_image_upload_extracts_and_saves_transactions(
    sqlite_database, monkeypatch
):
    image = Image.new("RGB", (500, 250), color="white")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")

    def fake_run_extraction(
        image_bytes, filename, progress, progress_lock, file_type="pdf"
    ):
        transactions = [
            {
                "transaction_id": f"ocr-tx-{row_index}",
                "transaction_date": f"2026-01-{row_index + 5:02d}",
                "description": f"Market {row_index + 1}",
                "amount": 85.5,
                "currency": "USD",
                "transaction_type": "expense",
                "category": "Food",
                "source_dataset": filename,
            }
            for row_index in range(6)
        ]
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "filename": filename,
                "file_type": file_type,
                "total_transactions": len(transactions),
                "transactions": transactions,
                "warnings": [],
                "error": None,
            },
        )

    monkeypatch.setattr(
        integrations, "_run_isolated_extraction", fake_run_extraction
    )
    response = client.post(
        "/upload-statement?user_id=alice",
        files={"file": ("market.jpg", image_bytes.getvalue(), "image/jpeg")},
    )
    retry = client.post(
        "/upload-statement?user_id=alice",
        files={"file": ("market.jpg", image_bytes.getvalue(), "image/jpeg")},
    )

    assert response.status_code == 202
    assert retry.status_code == 202
    job_id = response.json()["data"]["job_id"]
    assert retry.json()["data"]["job_id"] == job_id
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status_response = client.get(
            f"/upload-status/{job_id}?user_id=alice"
        ).json()
        if status_response["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    assert status_response["status"] == "completed", status_response
    result = status_response["data"]["result"]
    assert result["total_transactions"] == 6
    assert all(
        transaction["currency"] == "USD"
        for transaction in result["transactions"]
    )

    alice_transactions = client.get("/transactions?user_id=alice").json()["data"]
    assert alice_transactions["count"] == 8
    saved = [
        row
        for row in alice_transactions["transactions"]
        if row["description"].startswith("Market ")
    ]
    assert len(saved) == 6
    assert all(row["type"] == "debit" for row in saved)
    assert all(row["amount"] == 85.5 for row in saved)
    assert all(row["currency"] == "USD" for row in saved)

    completed_retry = client.post(
        "/upload-statement?user_id=alice",
        files={"file": ("market.jpg", image_bytes.getvalue(), "image/jpeg")},
    )
    assert completed_retry.status_code == 200
    assert completed_retry.json()["data"]["transactions"] == result["transactions"]


def test_reupload_returns_corrected_dates_from_cached_image_result(
    sqlite_database, monkeypatch
):
    image = Image.new("RGB", (100, 60), color="white")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    content = image_bytes.getvalue()
    content_hash = hashlib.sha256(content).hexdigest()
    job_id = "cached-image-date-job"
    stale_dates = [
        "2021-01-06",
        "2021-01-06",
        "2021-03-06",
        "2021-04-06",
        "2021-07-06",
        "2021-08-06",
        "2021-09-06",
    ]
    expected_dates = [
        "2021-05-31",
        "2021-06-01",
        "2021-06-03",
        "2021-06-04",
        "2021-06-07",
        "2021-06-08",
        "2021-06-09",
    ]
    result = {
        "success": True,
        "filename": "output.png",
        "file_type": "png",
        "total_transactions": len(expected_dates),
        "transactions": [
            {
                "transaction_id": f"ocr-{index}",
                "transaction_date": stale_date,
                "description": f"{description_date} Purchase {index}",
                "amount": 10.0,
                "currency": None,
                "transaction_type": "expense",
                "category": "Other",
                "source_dataset": "output.png",
            }
            for index, (stale_date, description_date) in enumerate(
                zip(stale_dates, (
                    "05-31-2021",
                    "06-01-2021",
                    "06-03-2021",
                    "06-04-2021",
                    "06-07-2021",
                    "06-08-2021",
                    "06-09-2021",
                ))
            )
        ],
        "warnings": [],
        "error": None,
    }
    db = integrations.SessionLocal()
    db.add(
        UploadJob(
            id=job_id,
            user_id="alice",
            filename="output.png",
            content_hash=content_hash,
            status="completed",
            result_json=json.dumps(result),
        )
    )
    db.add(
        StatementUploadClaim(
            user_id="alice", content_hash=content_hash, job_id=job_id
        )
    )
    db.commit()
    db.close()
    monkeypatch.setattr(
        integrations,
        "_run_isolated_extraction",
        lambda *args, **kwargs: pytest.fail("cached upload should not be parsed"),
    )

    response = client.post(
        "/upload-statement?user_id=alice",
        files={"file": ("output.png", content, "image/png")},
    )

    assert response.status_code == 200
    assert [
        transaction["transaction_date"]
        for transaction in response.json()["data"]["transactions"]
    ] == expected_dates
    assert client.get("/transactions?user_id=alice").json()["data"]["count"] == 2
    status_result = client.get(
        f"/upload-status/{job_id}?user_id=alice"
    ).json()["data"]["result"]
    assert [
        transaction["transaction_date"]
        for transaction in status_result["transactions"]
    ] == expected_dates


def test_retry_after_response_timeout_does_not_persist_twice(
    sqlite_database, monkeypatch
):
    content = b"Date,Description,Amount\n2026-03-01,Bus,-2.50\n"
    persist = integrations._extract_and_save
    first_call = True

    def persist_then_timeout(file, user_id, job_id=None):
        nonlocal first_call
        response = persist(file, user_id, job_id)
        if first_call:
            first_call = False
            from fastapi import HTTPException

            raise HTTPException(status_code=504, detail="Synthetic response timeout.")
        return response

    monkeypatch.setattr(integrations, "_extract_and_save", persist_then_timeout)
    first = client.post(
        "/upload-statement?user_id=alice",
        files={"file": ("timeout.csv", content, "text/csv")},
    )

    assert first.status_code == 504
    rows = client.get("/transactions?user_id=alice").json()["data"]["transactions"]
    assert sum(row["description"] == "Bus" for row in rows) == 1

    monkeypatch.setattr(integrations, "_extract_and_save", persist)
    retry = client.post(
        "/upload-statement?user_id=alice",
        files={"file": ("timeout.csv", content, "text/csv")},
    )

    assert retry.status_code == 200
    rows = client.get("/transactions?user_id=alice").json()["data"]["transactions"]
    assert sum(row["description"] == "Bus" for row in rows) == 1


@pytest.mark.parametrize(
    ("filename", "content", "expected_status"),
    [
        ("notes.txt", b"not a statement", 400),
        ("invalid.csv", b"heading,other\nvalue,another\n", 422),
    ],
)
def test_member1_upload_returns_parser_errors_without_saving(
    sqlite_database, filename, content, expected_status
):
    response = client.post(
        "/upload-statement?user_id=alice",
        files={"file": (filename, content, "application/octet-stream")},
    )

    assert response.status_code == expected_status
    assert response.json()["success"] is False
    assert response.json()["total_transactions"] == 0
    assert client.get("/transactions?user_id=alice").json()["data"]["count"] == 2


def test_scanned_pdf_upload_is_backgrounded_and_persisted(
    sqlite_database, monkeypatch
):
    extraction_started = threading.Event()
    continue_extraction = threading.Event()

    def fake_run_extraction(pdf_bytes, filename, progress, progress_lock):
        extraction_started.set()
        assert continue_extraction.wait(timeout=5)
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "filename": filename,
                "file_type": "pdf",
                "total_transactions": 1,
                "transactions": [
                    {
                        "transaction_id": "ocr-tx-1",
                        "transaction_date": "2026-01-05",
                        "description": "Scanned groceries",
                        "amount": 85.5,
                        "currency": "LKR",
                        "transaction_type": "expense",
                        "category": "Food",
                        "source_dataset": filename,
                    }
                ],
                "warnings": [],
                "error": None,
            },
        )

    monkeypatch.setattr(
        integrations, "_run_isolated_extraction", fake_run_extraction
    )
    try:
        response = client.post(
            "/upload-statement?user_id=alice",
            files={"file": ("scanned.pdf", create_test_scanned_pdf(), "application/pdf")},
        )

        assert response.status_code == 202
        job_id = response.json()["data"]["job_id"]
        assert extraction_started.wait(timeout=10)
        duplicate_response = client.post(
            "/upload-statement?user_id=alice",
            files={
                "file": (
                    "scanned.pdf",
                    create_test_scanned_pdf(),
                    "application/pdf",
                )
            },
        )
        assert duplicate_response.status_code == 202
        assert duplicate_response.json()["data"]["job_id"] == job_id
        in_progress = client.get(
            f"/upload-status/{job_id}?user_id=alice"
        ).json()
        assert in_progress["status"] == "processing"
        assert in_progress["data"]["job_id"] == job_id
    finally:
        continue_extraction.set()

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = client.get(
            f"/upload-status/{job_id}?user_id=alice"
        ).json()
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    assert result["status"] == "completed", result
    assert result["data"]["result"]["total_transactions"] == 1
    saved = client.get("/transactions?user_id=alice").json()["data"]
    assert saved["count"] == 3
    transaction = next(
        row for row in saved["transactions"]
        if row["description"] == "Scanned groceries"
    )
    assert transaction["type"] == "debit"
    assert transaction["user_id"] == "alice"


def test_scanned_pdf_background_failure_is_reported_by_status(
    sqlite_database, monkeypatch
):
    monkeypatch.setattr(
        integrations,
        "_run_isolated_extraction",
        lambda pdf_bytes, filename, progress, progress_lock: JSONResponse(
            status_code=422,
            content={
                "success": False,
                "filename": filename,
                "file_type": "pdf",
                "total_transactions": 0,
                "transactions": [],
                "warnings": [],
                "error": "OCR could not extract any valid transactions.",
            },
        ),
    )

    response = client.post(
        "/upload-statement?user_id=alice",
        files={"file": ("scanned.pdf", create_test_scanned_pdf(), "application/pdf")},
    )

    assert response.status_code == 202
    job_id = response.json()["data"]["job_id"]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        result = client.get(
            f"/upload-status/{job_id}?user_id=alice"
        ).json()
        if result["status"] in {"completed", "failed"}:
            break
        time.sleep(0.01)

    assert result["status"] == "failed"
    assert result["data"]["error"] == "OCR could not extract any valid transactions."
    assert client.get("/transactions?user_id=alice").json()["data"]["count"] == 2


def test_persistence_preserves_unknown_currency_and_repeated_rows(sqlite_database):
    row = {
        "transaction_id": "repeat-1",
        "transaction_date": "2026-01-05",
        "description": "Same-day transfer",
        "amount": 25.0,
        "currency": None,
        "transaction_type": "expense",
        "category": "Unknown",
    }
    credit_row = {
        **row,
        "transaction_id": "repeat-credit",
        "transaction_type": "income",
    }
    extraction = JSONResponse(
        status_code=200,
        content={
            "filename": "statement.pdf",
            "transactions": [
                row,
                {**row, "transaction_id": "repeat-2"},
                credit_row,
            ],
        },
    )

    integrations._persist_extraction(extraction, "alice")

    db = integrations.SessionLocal()
    try:
        rows = (
            db.query(Transaction)
            .filter(
                Transaction.user_id == "alice",
                Transaction.description == "Same-day transfer",
            )
            .all()
        )
        assert len(rows) == 3
        assert {transaction.currency for transaction in rows} == {None}
        assert {transaction.type for transaction in rows} == {"debit", "credit"}
    finally:
        db.close()

    integrations._persist_extraction(extraction, "alice")
    db = integrations.SessionLocal()
    try:
        repeated_rows = (
            db.query(Transaction)
            .filter(
                Transaction.user_id == "alice",
                Transaction.description == "Same-day transfer",
            )
            .all()
        )
        assert len(repeated_rows) == 6
    finally:
        db.close()


def test_reuploading_same_pdf_is_idempotent_per_user(sqlite_database, monkeypatch):
    monkeypatch.setattr(integrations, "_is_scanned_pdf_isolated", lambda _bytes: False)
    pdf_content = create_test_table_pdf(
        ["Date", "Description", "Amount"],
        [["2026-02-01", "Coffee", "-4.50"]],
    )
    uploaded_file = ("same-statement.pdf", pdf_content, "application/pdf")
    renamed_retry = ("renamed-copy.pdf", pdf_content, "application/pdf")

    first = client.post(
        "/upload-statement?user_id=alice", files={"file": uploaded_file}
    )
    retry = client.post(
        "/upload-statement?user_id=alice", files={"file": renamed_retry}
    )
    other_user = client.post(
        "/upload-statement?user_id=bob", files={"file": uploaded_file}
    )

    assert first.status_code == retry.status_code == other_user.status_code == 200
    assert first.json()["data"]["transactions"] == retry.json()["data"]["transactions"]
    assert client.get("/transactions?user_id=alice").json()["data"]["count"] == 3
    assert client.get("/transactions?user_id=bob").json()["data"]["count"] == 2


def test_failed_upload_retry_can_be_reprocessed(sqlite_database, monkeypatch):
    content = b"Date,Description,Amount\n2026-03-01,Bus,-2.50\n"
    failure = JSONResponse(
        status_code=422,
        content={
            "success": False,
            "filename": "retry.csv",
            "file_type": "csv",
            "total_transactions": 0,
            "transactions": [],
            "warnings": [],
            "error": "Temporary extraction failure.",
        },
    )
    success = JSONResponse(
        status_code=200,
        content={
            "success": True,
            "filename": "retry.csv",
            "file_type": "csv",
            "total_transactions": 1,
            "transactions": [
                {
                    "transaction_id": "retry-transaction",
                    "transaction_date": "2026-03-01",
                    "description": "Bus",
                    "amount": 2.5,
                    "currency": None,
                    "transaction_type": "expense",
                    "category": "Transport",
                }
            ],
            "warnings": [],
            "error": None,
        },
    )
    responses = iter((failure, success))
    monkeypatch.setattr(
        integrations, "extract_statement", lambda _file, **kwargs: next(responses)
    )

    first = client.post(
        "/upload-statement?user_id=alice",
        files={"file": ("retry.csv", content, "text/csv")},
    )
    retry = client.post(
        "/upload-statement?user_id=alice",
        files={"file": ("retry.csv", content, "text/csv")},
    )

    assert first.status_code == 422
    assert retry.status_code == 200
    rows = client.get("/transactions?user_id=alice").json()["data"]["transactions"]
    assert sum(row["description"] == "Bus" for row in rows) == 1
    assert next(row for row in rows if row["description"] == "Bus")["currency"] is None


def test_distinct_uploads_keep_legitimate_identical_transactions(sqlite_database):
    first_content = b"Date,Description,Amount\n2026-04-01,Bus,-2.50\n"
    second_content = first_content + b"\n"
    assert first_content != second_content

    for filename, content in (
        ("statement-one.csv", first_content),
        ("statement-two.csv", second_content),
    ):
        response = client.post(
            "/upload-statement?user_id=alice",
            files={"file": (filename, content, "text/csv")},
        )
        assert response.status_code == 200

    bus_rows = [
        row
        for row in client.get("/transactions?user_id=alice").json()["data"]["transactions"]
        if row["description"] == "Bus"
    ]
    assert len(bus_rows) == 2
    assert bus_rows[0]["date"] == bus_rows[1]["date"]
    assert bus_rows[0]["amount"] == bus_rows[1]["amount"]


def test_upload_idempotency_schema_upgrade_preserves_existing_jobs(
    monkeypatch, tmp_path
):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE statement_upload_jobs ("
            "id VARCHAR PRIMARY KEY, user_id VARCHAR NOT NULL, "
            "filename VARCHAR NOT NULL, status VARCHAR NOT NULL, "
            "result_json TEXT, error TEXT, created_at DATETIME NOT NULL, "
            "updated_at DATETIME NOT NULL)"
        )
        connection.exec_driver_sql(
            "INSERT INTO statement_upload_jobs "
            "(id, user_id, filename, status, created_at, updated_at) "
            "VALUES ('legacy-job', 'alice', 'legacy.pdf', 'failed', "
            "'2026-01-01', '2026-01-01')"
        )
    monkeypatch.setattr(database_module, "engine", engine)

    database_module.init_db()

    with engine.connect() as connection:
        row = connection.exec_driver_sql(
            "SELECT id, status, content_hash, is_async "
            "FROM statement_upload_jobs WHERE id = 'legacy-job'"
        ).one()
    assert row == ("legacy-job", "failed", None, 0)
    engine.dispose()


def test_upload_claim_migration_backfills_duplicate_hashes_without_deleting_jobs(
    monkeypatch, tmp_path
):
    engine = create_engine(f"sqlite:///{tmp_path / 'duplicate_jobs.db'}")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE statement_upload_jobs ("
            "id VARCHAR PRIMARY KEY, user_id VARCHAR NOT NULL, "
            "filename VARCHAR NOT NULL, status VARCHAR NOT NULL, "
            "result_json TEXT, error TEXT, created_at DATETIME NOT NULL, "
            "updated_at DATETIME NOT NULL, content_hash VARCHAR(64), "
            "is_async BOOLEAN NOT NULL DEFAULT 0)"
        )
        connection.exec_driver_sql(
            "INSERT INTO statement_upload_jobs "
            "(id, user_id, filename, status, created_at, updated_at, content_hash) "
            "VALUES "
            "('old-failed', 'alice', 'old.pdf', 'failed', '2026-01-01', "
            "'2026-01-01', 'same-hash'), "
            "('old-completed', 'alice', 'old-copy.pdf', 'completed', "
            "'2026-01-02', '2026-01-02', 'same-hash')"
        )
    monkeypatch.setattr(database_module, "engine", engine)

    database_module.init_db()

    with engine.connect() as connection:
        jobs = connection.exec_driver_sql(
            "SELECT id FROM statement_upload_jobs ORDER BY id"
        ).fetchall()
        claims = connection.exec_driver_sql(
            "SELECT user_id, content_hash, job_id FROM statement_upload_claims"
        ).fetchall()
    assert jobs == [("old-completed",), ("old-failed",)]
    assert claims == [("alice", "same-hash", "old-completed")]
    engine.dispose()


def test_scanned_pdf_jobs_are_processed_serially_within_queue_capacity(
    sqlite_database, monkeypatch
):
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()

    def fake_run_extraction(pdf_bytes, filename, progress, progress_lock):
        if filename == "first-scan.pdf":
            first_started.set()
            assert release_first.wait(timeout=5)
        else:
            second_started.set()
            assert release_second.wait(timeout=5)
        timed_out = filename == "first-scan.pdf"
        return JSONResponse(
            status_code=504 if timed_out else 422,
            content={
                "success": False,
                "filename": filename,
                "file_type": "pdf",
                "total_transactions": 0,
                "transactions": [],
                "warnings": [],
                "error": (
                    "Synthetic OCR timeout."
                    if timed_out
                    else "Synthetic OCR failure."
                ),
            },
        )

    monkeypatch.setattr(
        integrations, "_run_isolated_extraction", fake_run_extraction
    )
    job_ids = []
    try:
        for filename in ("first-scan.pdf", "second-scan.pdf"):
            pdf_content = create_test_scanned_pdf()
            if filename == "second-scan.pdf":
                pdf_content += b"\n% distinct synthetic upload"
            response = client.post(
                "/upload-statement?user_id=alice",
                files={
                    "file": (
                        filename,
                        pdf_content,
                        "application/pdf",
                    )
                },
            )
            assert response.status_code == 202
            job_ids.append(response.json()["data"]["job_id"])

        assert first_started.wait(timeout=10)
        assert not second_started.wait(timeout=0.1)
        release_first.set()
        second_job_started = second_started.wait(timeout=10)
    finally:
        release_first.set()
        release_second.set()

    assert second_job_started is True
    job_results = []
    for job_id in job_ids:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = client.get(
                f"/upload-status/{job_id}?user_id=alice"
            ).json()
            if result["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert result["status"] == "failed", result
        job_results.append(result)
    assert job_results[0]["data"]["error"] == "Synthetic OCR timeout."


def test_stale_background_jobs_are_marked_failed(sqlite_database):
    stale_time = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=10)
    db = integrations.SessionLocal()
    db.add_all(
        [
            UploadJob(
                id="stale-queued",
                user_id="alice",
                filename="queued.pdf",
                status="queued",
                updated_at=stale_time,
            ),
            UploadJob(
                id="stale-processing",
                user_id="alice",
                filename="processing.pdf",
                status="processing",
                updated_at=stale_time,
            ),
            UploadJob(
                id="recent-processing",
                user_id="alice",
                filename="recent.pdf",
                status="processing",
            ),
        ]
    )
    db.commit()
    db.close()

    assert integrations.expire_stale_ocr_jobs() == 2

    db = integrations.SessionLocal()
    try:
        jobs = {
            job.id: job
            for job in db.query(UploadJob)
            .filter(UploadJob.id.in_(
                ("stale-queued", "stale-processing", "recent-processing")
            ))
            .all()
        }
        assert jobs["stale-queued"].status == "failed"
        assert jobs["stale-processing"].status == "failed"
        assert jobs["recent-processing"].status == "processing"
        assert jobs["stale-processing"].error
    finally:
        db.close()


def test_timed_out_extraction_process_is_killed_and_worker_can_continue():
    stages = []

    def record_stage(stage, page_number, elapsed_seconds):
        stages.append((stage, page_number, elapsed_seconds))

    assert integrations._run_isolated_process(
        _blocked_child_process,
        (),
        timeout_seconds=10,
        on_stage=record_stage,
    ) is None
    assert stages == [("ocr_inference_started", 2, None)]

    next_result = integrations._run_isolated_process(
        _successful_child_process,
        (),
        timeout_seconds=10,
        on_stage=record_stage,
    )
    assert next_result == {"type": "result", "status_code": 200, "body": b"done"}


def test_process_termination_failure_blocks_later_process_start(monkeypatch):
    process = _UnstoppableProcess()
    monkeypatch.setattr(integrations, "_unreaped_process", process)

    assert integrations._terminate_process(process) is False
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    with pytest.raises(integrations.OCRProcessTerminationError):
        integrations._run_isolated_process(
            _successful_child_process,
            (),
            timeout_seconds=1,
            on_stage=lambda _stage, _page, _elapsed: None,
        )


def test_scanned_pdf_detection_has_a_process_timeout(monkeypatch):
    captured = {}

    def fake_run_isolated_process(target, args, timeout_seconds, on_stage):
        captured["target"] = target
        captured["args"] = args
        captured["timeout_seconds"] = timeout_seconds
        return None

    monkeypatch.setattr(
        integrations, "_run_isolated_process", fake_run_isolated_process
    )

    with pytest.raises(TimeoutError):
        integrations._is_scanned_pdf_isolated(b"%PDF-synthetic")

    assert captured["timeout_seconds"] == integrations.OCR_SCAN_DETECTION_TIMEOUT


def test_ocr_persistence_timeout_fails_the_job_response(sqlite_database, monkeypatch):
    monkeypatch.setattr(
        integrations, "_run_isolated_process", lambda *args, **kwargs: None
    )

    response = integrations._persist_ocr_job_isolated(
        {"filename": "statement.pdf", "transactions": []},
        "alice",
        "job-timeout",
    )

    assert response.status_code == 504
    assert "persistence" in response.body.decode().lower()


def test_extraction_stage_logs_include_job_page_and_elapsed_time(monkeypatch, caplog):
    caplog.set_level("INFO", logger="app.api.integrations")

    def fake_run_isolated_process(target, args, timeout_seconds, on_stage):
        on_stage("ocr_inference_started", 2, None)
        on_stage("ocr_inference_completed", 2, 0.25)
        return {
            "type": "result",
            "status_code": 200,
            "body": b'{"success": true}',
        }

    monkeypatch.setattr(
        integrations, "_run_isolated_process", fake_run_isolated_process
    )
    progress = {
        "job_id": "job-id",
        "source_filename": "statement.pdf",
        "stage": "job_queued",
        "page_number": None,
    }

    result = integrations._run_isolated_extraction(
        b"%PDF-test",
        "statement.pdf",
        progress,
        threading.Lock(),
    )

    stage_logs = [
        record
        for record in caplog.records
        if record.message == "Scanned statement extraction stage"
    ]
    assert result.status_code == 200
    assert progress["stage"] == "ocr_inference_completed"
    assert progress["page_number"] == 2
    assert [record.stage for record in stage_logs] == [
        "ocr_inference_started",
        "ocr_inference_completed",
    ]
    assert all(record.job_id == "job-id" for record in stage_logs)
    assert all(record.source_filename == "statement.pdf" for record in stage_logs)
    assert all(record.page_number == 2 for record in stage_logs)
    assert stage_logs[1].operation_elapsed_seconds == 0.25


def test_scanned_pdf_queue_returns_service_unavailable_when_full(sqlite_database):
    acquired_slots = []
    try:
        for _ in range(2):
            assert integrations._ocr_job_slots.acquire(blocking=False)
            acquired_slots.append(True)

        response = client.post(
            "/upload-statement?user_id=alice",
            files={"file": ("scanned.pdf", create_test_scanned_pdf(), "application/pdf")},
        )

        assert response.status_code == 503
        assert "busy" in response.json()["message"].lower()
        assert response.headers["retry-after"] == "30"
    finally:
        for _ in acquired_slots:
            integrations._ocr_job_slots.release()


def test_stale_background_upload_is_reported_as_failed(sqlite_database):
    db = integrations.SessionLocal()
    db.add(
        UploadJob(
            id="stale-job",
            user_id="alice",
            filename="scanned.pdf",
            status="processing",
            updated_at=datetime.now(timezone.utc).replace(tzinfo=None)
            - timedelta(hours=1),
        )
    )
    db.commit()
    db.close()

    response = client.get("/upload-status/stale-job?user_id=alice")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "stopped" in response.json()["data"]["error"].lower()


def test_member2_transactions_successfully_retrieves_database_rows(sqlite_database):
    response = client.get("/transactions?user_id=alice")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["count"] == 2
    assert {row["description"] for row in body["data"]["transactions"]} == {
        "Monthly salary",
        "Grocery store",
    }


def test_member3_integrated_endpoints_return_successful_responses(sqlite_database):
    endpoints = (
        "/analytics/summary",
        "/analytics/categories",
        "/recommendations/budget",
        "/recommendations/savings",
        "/detection/overspending",
        "/detection/anomalies",
    )
    for endpoint in endpoints:
        response = client.get(f"{endpoint}?user_id=alice")
        assert response.status_code == 200, response.text
        assert response.json()["status"] == "success"
        assert response.json()["data"] is not None


def test_member3_analytics_summary_uses_user_database_transactions(sqlite_database):
    response = client.get("/analytics/summary?user_id=alice")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total_income"] == 4000.0
    assert data["total_expenses"] == 500.0
    assert data["net_savings"] == 3500.0
    assert data["income_transaction_count"] == 1
    assert data["expense_transaction_count"] == 1


def test_member_integration_pluggable_handler():
    """Verify that when a member attaches their handler, the endpoint responds with 200."""
    async def sample_summary_handler(user_id: str):
        return {"total_income": 4000.0, "total_expenses": 1500.0, "net": 2500.0}

    registry.member3_summary_handler = sample_summary_handler
    response = client.get("/analytics/summary?user_id=alice")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["net"] == 2500.0
