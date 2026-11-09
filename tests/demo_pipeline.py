from app.database.database import SessionLocal, init_db
from app.database import crud
from app.services.preprocessing import clean_transactions
from app.services.categorization import categorize

raw = [
    {"date": "05/11/2026", "description": "  KEELLS   SUPER  COLOMBO ", "amount": "Rs. 2,450.00 Dr", "balance": "97,550.00"},
    {"date": "05/11/2026", "description": "  KEELLS   SUPER  COLOMBO ", "amount": "Rs. 2,450.00 Dr", "balance": "97,550.00"},  # duplicate
    {"date": "2026-11-01", "description": "SALARY NOV", "credit": "150,000.00", "balance": "250,000.00"},
    {"date": "not a date", "description": "broken row", "amount": "100"},
    {"date": "03 Nov 2026", "description": "PickMe ride", "debit": "850", "balance": "96,700"},
]

init_db()
clean, rejected = clean_transactions(raw)
for row in clean:
    row["category"] = categorize(row["description"], row["type"])

db = SessionLocal()
saved = crud.save_transactions(db, "demo_user", clean, source_file="demo.csv")
print(f"clean={len(clean)} rejected={len(rejected)} saved={saved}")
for r in crud.get_transactions(db, "demo_user"):
    print(crud.to_dict(r))
db.close()