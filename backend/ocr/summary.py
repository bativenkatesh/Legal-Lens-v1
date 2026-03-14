from db.bills import get_bills_collection
from ocr.tax_rules import SECTION_LIMITS

async def compute_yearly_summary(user_id: str, financial_year: str):
    bills = get_bills_collection()

    query = {
        "user_id": user_id,
        "tax_eligible": True
    }
    # If a specific FY is selected (not "all"), filter by it
    if financial_year != "all":
        query["financial_year"] = financial_year

    cursor = bills.find(query)

    section_totals = {}
    total_allowed = 0
    count = 0

    async for bill in cursor:
        count += 1
        section = bill.get("tax_section")
        amount = bill.get("total_amount", 0)

        if not section:
            continue

        section_totals.setdefault(section, 0)
        section_totals[section] += amount

    summary = {}

    for section, claimed in section_totals.items():
        limit = SECTION_LIMITS.get(section)

        allowed = claimed
        if limit is not None:
            allowed = min(claimed, limit)

        summary[section] = {
            "claimed": claimed,
            "allowed": allowed,
            "limit": limit
        }

        total_allowed += allowed

    return {
        "financial_year": financial_year,
        "total_bills": count,
        "section_wise": summary,
        "total_deduction_allowed": total_allowed
    }
