from db.mongo import get_db
def get_bills_collection():
    db = get_db()
    return db["bills"]

