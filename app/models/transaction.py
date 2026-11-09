from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Date, DateTime
from app.database.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True, default="demo_user")
    date = Column(Date, nullable=False)
    description = Column(String, nullable=False)
    amount = Column(Float, nullable=False)           # always positive
    type = Column(String, nullable=False)            # "debit" or "credit"
    category = Column(String, default="Uncategorized")
    balance = Column(Float, nullable=True)
    currency = Column(String, default="LKR")
    source_file = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)