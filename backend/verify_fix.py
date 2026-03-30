import os
import sys
import uuid
import pymongo
from langchain_core.messages import HumanMessage, AIMessage

# Ensure backend directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.conversation import get_messages, save_message
from agent_core import orchestrator

def test_full_flow():
    user_id = "test_user_999"
    conversation_id = f"test_session_{uuid.uuid4().hex[:8]}"
    
    print(f"--- [TEST] Starting Verification Flow ---")
    print(f"User ID: {user_id}")
    print(f"Session ID: {conversation_id}")

    # 1. First Message: What is Section 80C?
    query1 = "What is Section 80C and what are its limits?"
    print(f"\n[QUERY 1]: {query1}")
    
    result1 = orchestrator.invoke({
        "input": query1,
        "chat_history": []
    })
    
    answer1 = result1.get("output", "")
    print(f"\n[RESPONSE 1]:\n{answer1[:200]}...")
    
    # Save to history (simulating main.py logic)
    save_message(user_id, conversation_id, "user", query1)
    save_message(user_id, conversation_id, "assistant", answer1)
    print("✅ Saved first interaction to MongoDB.")

    # 2. Second Message: Can you give me a real world example of this?
    # This tests context (what is 'this'?) and the news/example tool.
    query2 = "Can you give me a real world example of this from news or practical cases?"
    print(f"\n[QUERY 2]: {query2}")
    
    # Reload history from DB
    history_docs = get_messages(user_id, conversation_id)
    history_msgs = []
    for m in history_docs:
        if m["role"] == "user":
            history_msgs.append(HumanMessage(content=m["content"]))
        else:
            history_msgs.append(AIMessage(content=m["content"]))
            
    result2 = orchestrator.invoke({
        "input": query2,
        "chat_history": history_msgs
    })
    
    answer2 = result2.get("output", "")
    print(f"\n[RESPONSE 2]:\n{answer2[:500]}...")

    # 3. Final Check: Does history have 4 messages now?
    save_message(user_id, conversation_id, "user", query2)
    save_message(user_id, conversation_id, "assistant", answer2)
    
    final_docs = get_messages(user_id, conversation_id)
    msg_count = len(final_docs)
    print(f"\n[VERIFICATION]: Total messages in history: {msg_count}")
    
    if msg_count == 4:
        print("✅ SUCCESS: Chat history is persisting correctly.")
    else:
        print(f"❌ FAILURE: History count is {msg_count}, expected 4.")

    if "Example" in answer2 or "EXAMPLE" in answer2:
        print("✅ SUCCESS: Real-world examples found in response.")
    else:
        print("⚠️  WARNING: Could not explicitly find 'Example' in response (check output).")

if __name__ == "__main__":
    try:
        test_full_flow()
    except Exception as e:
        print(f"❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
