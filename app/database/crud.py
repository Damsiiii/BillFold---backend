from datetime import date
import time
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from app.models.transaction import Transaction


def save_transactions(
    db: Session,
    user_id: str,
    transactions: List[Dict[str, Any]],
    source_file: Optional[str] = None,
    deadline: Optional[float] = None,
    commit: bool = True,
) -> int:
    """Save every extracted transaction; upload-level idempotency is handled separately."""
    for tx in transactions:
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Transaction persistence exceeded its time limit.")
        db.add(Transaction(user_id=user_id, source_file=source_file, **tx))
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Transaction persistence exceeded its time limit.")
    if commit:
        db.commit()
    return len(transactions)


def get_transactions(
    db: Session,
    user_id: str,
    limit: Optional[int] = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    category: Optional[str] = None,
    tx_type: Optional[str] = None,
) -> List[Transaction]:
    query = db.query(Transaction).filter(Transaction.user_id == user_id)
    if start_date:
        query = query.filter(Transaction.date >= start_date)
    if end_date:
        query = query.filter(Transaction.date <= end_date)
    if category:
        query = query.filter(Transaction.category == category)
    if tx_type:
        query = query.filter(Transaction.type == tx_type)
    query = query.order_by(Transaction.date.desc(), Transaction.id.desc())
    if limit is not None:
        query = query.limit(limit)
    return query.all()


def delete_user_transactions(db: Session, user_id: str) -> int:
    count = db.query(Transaction).filter(Transaction.user_id == user_id).delete()
    db.commit()
    return count


def to_dict(row: Transaction) -> Dict[str, Any]:
    return {
        "id": row.id,
        "user_id": row.user_id,
        "date": row.date.isoformat(),
        "description": row.description,
        "amount": row.amount,
        "type": row.type,
        "category": row.category,
        "balance": row.balance,
        "currency": row.currency,
    }