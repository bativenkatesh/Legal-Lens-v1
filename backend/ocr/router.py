from fastapi import APIRouter
from fastapi import UploadFile, File, HTTPException
from ocr.ocr_engine import extract_text_from_file
from ocr.parser import parse_bill_text
from uuid import uuid4
import os
from db.bills import get_bills_collection
from utils.date_utils import get_financial_year
from ocr.summary import compute_yearly_summary


router = APIRouter(
    prefix="/ocr",
    tags=["OCR"]
)
UPLOAD_DIR = "uploads/bills"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_bill(file: UploadFile = File(...)):
    # Basic validation
    if file.content_type not in [
        "image/png",
        "image/jpeg",
        "image/jpg",
        "application/pdf"
    ]:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    bill_id = str(uuid4())
    extension = os.path.splitext(file.filename)[1]
    filename = f"{bill_id}{extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save file")

    return {
        "bill_id": bill_id,
        "filename": filename,
        "status": "uploaded"
    }
@router.get("/extract/{bill_id}")
async def extract_bill_text(bill_id: str):
    bills_dir = "uploads/bills"

    # Find file by bill_id
    matches = [
        f for f in os.listdir(bills_dir)
        if f.startswith(bill_id)
    ]

    if not matches:
        raise HTTPException(status_code=404, detail="Bill not found")

    file_path = os.path.join(bills_dir, matches[0])

    try:
        text = extract_text_from_file(file_path)
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="OCR failed")

    return {
        "bill_id": bill_id,
        "raw_text": text[:3000]  # safety limit
    }

@router.post("/parse/{bill_id}")
async def parse_bill(bill_id: str):
    bills_dir = "uploads/bills"

    matches = [
        f for f in os.listdir(bills_dir)
        if f.startswith(bill_id)
    ]

    if not matches:
        raise HTTPException(status_code=404, detail="Bill not found")

    file_path = os.path.join(bills_dir, matches[0])

    raw_text = extract_text_from_file(file_path)

    from ocr.schemas import BillRecord

    from datetime import datetime

    parsed = parse_bill_text(raw_text)

    financial_year = get_financial_year(parsed.bill_date)

    record = {
        "bill_id": bill_id,
        "user_id": "user_123",  # TEMP, replace later
        "financial_year": financial_year,

        "vendor": parsed.vendor,
        "bill_date": parsed.bill_date,
        "total_amount": parsed.total_amount,
        "currency": parsed.currency,

        "category": parsed.category,
        "tax_section": parsed.tax_section,
        "tax_eligible": parsed.tax_eligible,

        "confidence": parsed.confidence,
        "source": "ocr",
        "created_at": datetime.utcnow()
    }

    bills = get_bills_collection()

    # Prevent duplicate insert
    existing = await bills.find_one({"bill_id": bill_id})
    if not existing:
        await bills.insert_one(record)

    record.pop("_id", None)

    return {
        "bill_record": record
    }
@router.get("/summary/{financial_year}")
async def yearly_summary(financial_year: str):
    # TEMP user_id until auth
    user_id = "user_123"

    summary = await compute_yearly_summary(user_id, financial_year)
    return summary
@router.get("/bills")
async def list_bills():
    user_id = "user_123"  # TEMP until auth

    bills = get_bills_collection()

    cursor = bills.find(
        {"user_id": user_id},
        {
            "_id": 0,  # hide Mongo internal id
            "bill_id": 1,
            "vendor": 1,
            "total_amount": 1,
            "currency": 1,
            "financial_year": 1,
            "tax_eligible": 1,
            "tax_section": 1,
        }
    ).sort("created_at", -1)

    results = []
    async for bill in cursor:
        results.append(bill)

    return results
@router.patch("/bill/{bill_id}")
async def update_bill(bill_id: str, payload: dict):
    bills = get_bills_collection()

    update_fields = {}

    if "tax_eligible" in payload:
        update_fields["tax_eligible"] = payload["tax_eligible"]

    if "tax_section" in payload:
        update_fields["tax_section"] = payload["tax_section"]

    if not update_fields:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    result = await bills.update_one(
        {"bill_id": bill_id},
        {"$set": update_fields}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Bill not found")

    return {"status": "updated"}
