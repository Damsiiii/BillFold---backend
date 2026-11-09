from datetime import date
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session
from app.models.transaction import Transaction


def save_transactions(
    db: Session,
    user_id: str,
    transactions: List[Dict[str, Any]],
    source_file: Optional[str] = None,
) -> int:
    """Saves clean transactions. Skips rows already stored (re-uploads). Returns number saved."""
    saved = 0
    for tx in transactions:
        exists = (
            db.query(Transaction.id)
            .filter(
                Transaction.user_id == user_id,
                Transaction.date == tx["date"],
                Transaction.description == tx["description"],
                Transaction.amount == tx["amount"],
                Transaction.type == tx["type"],
                Transaction.balance == tx.get("balance"),
            )
            .first()
        )
        if exists:
            continue
        db.add(Transaction(user_id=user_id, source_file=source_file, **tx))
        saved += 1
    db.commit()
    return saved


def get_transactions(
    db: Session,
    user_id: str,
    limit: int = 50,
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
    return query.order_by(Transaction.date.desc(), Transaction.id.desc()).limit(limit).all()


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