from pydantic import BaseModel
from typing import Optional

class ParsedBill(BaseModel):
    vendor: Optional[str]
    bill_date: Optional[str]
    total_amount: Optional[float]
    currency: Optional[str]

    category: Optional[str]          # Medical, Rent, Insurance, Donation, Education, Other
    tax_section: Optional[str]       # 80D, 80G, 80GG, etc.
    tax_eligible: Optional[bool]

    confidence: Optional[float]

from typing import Optional
from datetime import datetime

class BillRecord(BaseModel):
    bill_id: str
    vendor: Optional[str]
    bill_date: Optional[str]
    total_amount: Optional[float]
    currency: Optional[str]

    category: Optional[str]
    tax_section: Optional[str]
    tax_eligible: Optional[bool]

    confidence: Optional[float]

    source: str = "ocr"
    created_at: datetime
