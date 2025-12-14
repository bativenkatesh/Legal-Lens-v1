# --- MAC OS SQLITE FIX (MUST BE TOP) ---
import sqlite3
import sys

if sqlite3.sqlite_version_info < (3, 35, 0):
    sqlite3.sqlite_version_info = (3, 35, 0)
    sqlite3.sqlite_version = "3.35.0"
    sys.modules['sqlite3'] = sqlite3
# ---------------------------------------

import numpy as np
import pymongo
from langchain.tools import tool
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from flashrank import Ranker, RerankRequest


# --- CONFIG ---
CHROMA_PATH = "./chroma_db_smart"
LOCAL_MONGO_URI = "mongodb://localhost:27017/"
LOCAL_DB_NAME = "DB1"
ARTICLES_COLLECTION = "articles"

print("--- [INIT] Loading Search Engines... ---")


# --- ENGINE 1: LEGAL RULES (Smart RAG) ---
print(" Loading Legal Rules (ChromaDB + Flashrank)...")

try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Using the STABLE model (Avoids token_type_ids and invalid rank errors)
    ranker = Ranker(
        model_name="ms-marco-MiniLM-L-12-v2",
        cache_dir="./opt"
    )

    print("✅ Legal Engine Ready.")

except Exception as e:
    print(f"❌ Failed to load Legal Engine: {e}")
    vectorstore = None
    ranker = None


# --- ENGINE 2: PRACTICAL ARTICLES ---
print(" Loading Articles Engine...")

try:
    model_articles = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    client_local = pymongo.MongoClient(LOCAL_MONGO_URI)
    coll_articles = client_local[LOCAL_DB_NAME][ARTICLES_COLLECTION]

    print("✅ Articles Engine Ready.")

except Exception as e:
    print(f"❌ Failed to load Articles Engine: {e}")
    coll_articles = None
    model_articles = None


@tool
def search_legal_rules(query: str) -> str:
    """
    Search for Indian Income Tax Rules.
    Returns the FULL text of the most relevant sections.
    Use this tool to find the specific legal statutes and conditions.
    """
    if vectorstore is None:
        return "Error: Legal DB not loaded."

    try:
        # 1. Vector Search
        results = vectorstore.similarity_search(query, k=15)

        # 2. Extract Candidates
        unique_parents = {}

        for doc in results:
            sec_id = doc.metadata.get("section")
            full_text = doc.metadata.get("parent_content")

            if sec_id and sec_id not in unique_parents:
                unique_parents[sec_id] = {
                    "text": full_text,
                    "title": doc.metadata.get("title"),
                    "meta": doc.metadata
                }

        # 3. Re-rank
        passages = [
            {"id": k, "text": v["text"], "meta": v["meta"]}
            for k, v in unique_parents.items()
        ]

        rerank_request = RerankRequest(
            query=query,
            passages=passages
        )

        ranked_results = ranker.rerank(rerank_request)

        # 4. Return Top 3
        output = []

        print("\n===== DEBUG RETRIEVAL =====")
        for i, hit in enumerate(ranked_results[:3]):
            sec_id = hit["id"]
            full_text = unique_parents[sec_id]["text"]
            title = unique_parents[sec_id]["title"]
            score = hit["score"]

            print(
                f"Rank {i + 1}: Section {sec_id} "
                f"(Score: {score:.4f})"
            )

            output.append(
                f"SECTION: {sec_id}\n"
                f"TITLE: {title}\n"
                f"FULL CONTENT:\n{full_text}\n---"
            )

        print("==========================\n")

        return "\n".join(output) if output else "No relevant sections found."

    except Exception as e:
        return f"Error in search: {e}"


@tool
def search_practical_articles(query: str) -> str:
    """
    Search for practical articles, case laws, and real-world examples.
    Use this tool when the user asks for examples or interpretations.
    """
    if coll_articles is None:
        return "Error: Article DB not loaded"

    try:
        query_vec = model_articles.encode(query)

        candidates = list(
            coll_articles.find(
                {},
                {"embedding": 1, "title": 1, "full_text": 1}
            ).limit(200)
        )

        if not candidates:
            return "No articles found."

        cand_vecs = np.array([c["embedding"] for c in candidates])
        sims = cosine_similarity([query_vec], cand_vecs)[0]

        output = []
        for idx in sims.argsort()[-3:][::-1]:
            if sims[idx] < 0.35:
                continue

            doc = candidates[idx]
            output.append(
                f"ARTICLE: {doc.get('title')}\n"
                f"TEXT: {doc.get('full_text', '')[:800]}..."
            )

        return "\n\n".join(output) if output else "No articles found."

    except Exception as e:
        return f"Error: {e}"
