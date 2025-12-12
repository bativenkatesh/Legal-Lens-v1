import os
import pymongo
import numpy as np
import certifi
from langchain.tools import tool
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
# 1. ATLAS CONFIG (Rules - 1024 dimensions)
# Replace with your actual connection string if different
ATLAS_URI = "mongodb+srv://venky27274_db_user:eaOlsGwJxuygDSOw@cluster0.3ue3woh.mongodb.net/"
ATLAS_DB_NAME = "Dataset1"
RULES_COLLECTION = "rules"
ATLAS_VECTOR_INDEX = "vector_index"

# 2. LOCAL CONFIG (Articles - 384 dimensions)
LOCAL_MONGO_URI = "mongodb://localhost:27017/"
LOCAL_DB_NAME = "DB1"   # <--- VERIFY THIS NAME (Check your local mongo setup)
ARTICLES_COLLECTION = "articles"

print("--- [INIT] Loading Embedding Models... ---")

# Model 1: For Rules (Atlas uses BAAI/bge-large-en-v1.5 -> 1024 dims)
try:
    print("   Loading Rules Model (1024 dims)...")
    model_rules = SentenceTransformer('BAAI/bge-large-en-v1.5')
except Exception as e:
    print(f"❌ Failed to load Rules model: {e}")
    model_rules = None

# Model 2: For Articles (Local uses all-MiniLM-L6-v2 -> 384 dims)
try:
    print("   Loading Articles Model (384 dims)...")
    model_articles = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
except Exception as e:
    print(f"❌ Failed to load Articles model: {e}")
    model_articles = None

print("--- [INIT] Connecting to Databases... ---")

# Connection 1: Atlas
try:
    client_atlas = pymongo.MongoClient(ATLAS_URI, tlsCAFile=certifi.where())
    coll_rules = client_atlas[ATLAS_DB_NAME][RULES_COLLECTION]
    # Ping to verify
    client_atlas.admin.command('ping')
    print("✅ Connected to Atlas (Rules)")
except Exception as e:
    print(f"❌ Atlas Connection Failed: {e}")
    coll_rules = None

# Connection 2: Local Mongo
try:
    client_local = pymongo.MongoClient(LOCAL_MONGO_URI)
    coll_articles = client_local[LOCAL_DB_NAME][ARTICLES_COLLECTION]
    client_local.admin.command('ping')
    print("✅ Connected to Local Mongo (Articles)")
except Exception as e:
    print(f"❌ Local Mongo Connection Failed: {e}")
    coll_articles = None


@tool
def search_legal_rules(query: str) -> str:
    """
    Search MongoDB Atlas for Indian Income Tax Rules/Sections (Official Law).
    Always use this FIRST to find the section numbers and legal basis.
    """
    if not coll_rules or not model_rules:
        return "Error: Atlas DB or Rules Model not loaded."

    print(f"🔍 [TOOL] Searching Rules for: {query}")
    
    try:
        # 1. Encode (1024 dims)
        query_vec = model_rules.encode(query).tolist()
        
        # 2. Atlas Vector Search
        results = coll_rules.aggregate([
            {
                "$vectorSearch": {
                    "index": ATLAS_VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": query_vec,
                    "numCandidates": 50,
                    "limit": 3
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "section": 1,
                    "title": 1,
                    "full_content": 1
                }
            }
        ])
        
        output = []
        for doc in results:
            sec = doc.get('section', 'N/A')
            title = doc.get('title', 'N/A')
            text = doc.get('full_content', '')[:1000]
            output.append(f"SECTION: {sec} - {title}\nTEXT: {text}...")
            
        return "\n\n".join(output) if output else "No relevant rules found."
        
    except Exception as e:
        return f"Error in Atlas search: {e}"

@tool
def search_practical_articles(query: str) -> str:
    """
    Search Local Database for articles, case laws, and practical examples.
    Use this SECOND to interpret the law.
    """
    if not coll_articles or not model_articles:
        return "Error: Local DB or Articles Model not loaded."

    print(f"🔍 [TOOL] Searching Articles for: {query}")

    try:
        # 1. Encode (384 dims)
        query_vec = model_articles.encode(query) # numpy array
        
        # 2. Manual Vector Search (Cosine Similarity)
        # Fetch minimal fields + embedding
        candidates = list(coll_articles.find({}, {"embedding": 1, "title": 1, "full_text": 1}))
        
        if not candidates:
            return "No articles found in local database."

        # Extract embeddings
        cand_vecs = [c['embedding'] for c in candidates]
        cand_vecs = np.array(cand_vecs)
        
        # Calculate Cosine Similarity
        # reshape query_vec to (1, 384) for sklearn
        similarities = cosine_similarity([query_vec], cand_vecs)[0]
        
        # Get Top 3 Indices
        top_indices = similarities.argsort()[-3:][::-1]
        
        output = []
        for idx in top_indices:
            score = similarities[idx]
            if score < 0.35: # Filter noise
                continue
            
            doc = candidates[idx]
            output.append(f"ARTICLE: {doc.get('title')}\nTEXT: {doc.get('full_text', '')[:600]}...")
            
        return "\n\n".join(output) if output else "No relevant articles found."

    except Exception as e:
        return f"Error in Local search: {e}"