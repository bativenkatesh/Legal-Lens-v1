# db/conversation.py

from pymongo import MongoClient
from datetime import datetime

# ⚠️ Use same DB as your articles (Dataset1)
client = MongoClient("mongodb://localhost:27017/")
db = client["Dataset1"]
coll_messages = db["chat_messages"]

def get_messages(user_id: str, conversation_id: str, limit: int = 20):
    """Retrieve last N messages for a conversation, sorted by timestamp."""
    cursor = coll_messages.find({
        "user_id": user_id,
        "conversation_id": conversation_id
    }).sort("timestamp", 1).limit(limit)
    
    return list(cursor)

def save_message(user_id: str, conversation_id: str, role: str, content: str):
    """Save a single message as a document."""
    coll_messages.insert_one({
        "user_id": user_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow()
    })

def delete_conversation(user_id: str, conversation_id: str):
    """Delete all messages for a conversation."""
    coll_messages.delete_many({
        "user_id": user_id,
        "conversation_id": conversation_id
    })