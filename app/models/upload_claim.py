from sqlalchemy import Column, ForeignKey, String

from app.database.database import Base


class StatementUploadClaim(Base):
    """Database-enforced per-user claim for a statement's exact file contents."""

    __tablename__ = "statement_upload_claims"

    user_id = Column(String, primary_key=True)
    content_hash = Column(String(64), primary_key=True)
    job_id = Column(
        String,
        ForeignKey("statement_upload_jobs.id"),
        nullable=False,
        unique=True,
    )
