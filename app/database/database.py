from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./billfold.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 5.0},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    """Create all tables. Call once when the app starts."""
    from app.models import transaction  # noqa: F401  (registers the model)
    from app.models import upload_job  # noqa: F401  (registers the model)

    inspector = inspect(engine)
    if "statement_upload_jobs" in inspector.get_table_names():
        columns = {
            column["name"]
            for column in inspector.get_columns("statement_upload_jobs")
        }
        with engine.begin() as connection:
            if "content_hash" not in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE statement_upload_jobs ADD COLUMN content_hash VARCHAR(64)"
                )
            if "is_async" not in columns:
                connection.exec_driver_sql(
                    "ALTER TABLE statement_upload_jobs "
                    "ADD COLUMN is_async BOOLEAN NOT NULL DEFAULT 0"
                )
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()