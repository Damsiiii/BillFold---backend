"""
Canonical transaction data models adhering to the BillFold project specification.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """
    Canonical Transaction Schema as defined in the repository README.md:

    | Field            | Description |
    |---|---|
    | transaction_id   | Source or generated record identifier |
    | transaction_date | Standardized transaction date (YYYY-MM-DD) |
    | description      | Normalized transaction description |
    | amount           | Numeric transaction amount (positive float) |
    | currency         | Currency code, where known (e.g. USD, LKR) |
    | transaction_type | Income, expense, transfer, or unknown |
    | category         | Standardized spending category |
    | source_dataset   | Dataset/file from which the record originated |
    """
    transaction_id: str = Field(..., description="Unique transaction record identifier")
    transaction_date: str = Field(..., description="Standardized transaction date (YYYY-MM-DD)")
    description: str = Field(..., description="Normalized transaction description")
    amount: float = Field(..., ge=0.0, description="Normalized numeric transaction amount (positive float)")
    currency: Optional[str] = Field(default=None, description="ISO currency code, where known (e.g. USD, LKR, INR)")
    transaction_type: str = Field(default="unknown", description="Transaction type: income, expense, transfer, or unknown")
    category: Optional[str] = Field(default="Unknown", description="Spending category if identified, or Unknown")
    source_dataset: Optional[str] = Field(default=None, description="Dataset or filename from which record originated")
    balance: Optional[float] = Field(
        default=None,
        exclude=True,
        description="Account balance after transaction (retained in model, excluded from standard canonical export)"
    )

    def to_canonical_dict(self) -> dict:
        """
        Returns strictly the 8 canonical fields defined in the README.md specification:
        transaction_id, transaction_date, description, amount, currency,
        transaction_type, category, and source_dataset.
        """
        return {
            "transaction_id": self.transaction_id,
            "transaction_date": self.transaction_date,
            "description": self.description,
            "amount": self.amount,
            "currency": self.currency,
            "transaction_type": self.transaction_type,
            "category": self.category,
            "source_dataset": self.source_dataset,
        }


class ExtractionResult(BaseModel):
    """
    Standard output structure returned by all statement parsers and upload endpoints.
    """
    success: bool = Field(..., description="Indicates whether statement processing succeeded")
    filename: str = Field(..., description="Name of the parsed statement file")
    file_type: str = Field(..., description="File format: csv, excel, or pdf")
    total_transactions: int = Field(default=0, description="Total number of valid transactions extracted")
    transactions: List[Transaction] = Field(default_factory=list, description="Extracted canonical transactions")
    warnings: List[str] = Field(default_factory=list, description="List of any extraction warnings or anomalies")
    error: Optional[str] = Field(default=None, description="Error message if parsing or validation failed")

