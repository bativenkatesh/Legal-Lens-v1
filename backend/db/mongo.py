from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "tax_ocr")

client: AsyncIOMotorClient | None = None
db = None

async def connect_to_mongo():
    global client, db
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]

async def close_mongo_connection():
    global client
    if client:
        client.close()
def get_db():
    if db is None:
        raise RuntimeError("MongoDB not initialized")
    return db
