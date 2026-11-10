from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Index, String, Text, text

from app.database.database import Base


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class UploadJob(Base):
    __tablename__ = "statement_upload_jobs"
    __table_args__ = (
        Index(
            "uq_statement_upload_user_content_hash",
            "user_id",
            "content_hash",
            unique=True,
        ),
    )

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    filename = Column(String, nullable=False)
    status = Column(String, nullable=False, default="queued")
    content_hash = Column(String(64), nullable=True)
    is_async = Column(
        Boolean, nullable=False, default=False, server_default=text("0")
    )
    result_json = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=_utcnow, onupdate=_utcnow
    )
