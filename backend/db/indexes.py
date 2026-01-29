from db.mongo import get_db

async def create_indexes():
    db = get_db()
    bills = db["bills"]

    await bills.create_index("bill_id", unique=True)
    await bills.create_index("user_id")
    await bills.create_index("financial_year")
    await bills.create_index(
        [("user_id", 1), ("financial_year", 1)]
    )
