from db.bills import get_bills_collection

async def get_user_financial_context(user_id: str):
    bills = get_bills_collection()

    pipeline = [
        {"$match": {"user_id": user_id, "tax_eligible": True}},
        {
            "$group": {
                "_id": "$financial_year",
                "total_deduction": {"$sum": "$total_amount"},
            }
        }
    ]

    results = await bills.aggregate(pipeline).to_list(length=None)

    if not results:
        return "No tax eligible bills found."

    summary_text = "User Financial Summary:\n"

    for r in results:
        summary_text += (
            f"- Financial Year {r['_id']}: "
            f"Total Eligible Deduction = ₹{r['total_deduction']}\n"
        )

    return summary_text