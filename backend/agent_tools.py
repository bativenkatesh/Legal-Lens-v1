# --- MAC OS SQLITE FIX (MUST BE TOP) ---
import sqlite3
import sys
import re  # Added regex for smart detection

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
LOCAL_DB_NAME = "Dataset1"
ARTICLES_COLLECTION = "articles"
NEWS_COLLECTION = "articles" # Using articles collection as news for now, or change if a specific news collection exists

print("--- [INIT] Loading Search Engines... ---")

# --- ENGINE 1: LEGAL RULES (ChromaDB) ---
print(" Loading Legal Rules (ChromaDB Only)...")

try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    print("✅ Legal Engine Ready.")

except Exception as e:
    print(f"❌ Failed to load Legal Engine: {e}")
    vectorstore = None
print(" Loading FlashRank Reranker...")

try:
    # Try with a persistent cache directory to avoid /tmp issues
    import os
    cache_dir = os.path.expanduser("~/.cache/flashrank")
    os.makedirs(cache_dir, exist_ok=True)
    reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=cache_dir)
    print(f"✅ FlashRank Ready (Cache: {cache_dir}).")
except Exception as e:
    print(f"⚠️ FlashRank failed to load, falling back to standard search: {e}")
    reranker = None

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
        print(f"🔍 Searching VectorDB for: {query}")
        
        results = []
        
        # --- SMART LOOKUP: Detect "Section X" or just "80C" ---
        # Regex looks for patterns like: "Section 80C", "sec 80c", or just "80c" (if short)
        # It captures the alphanumeric code (e.g., 80C, 10A)
        section_match = re.search(r"(?:section\s+|sec\s+|s\.\s*)?([0-9]+[a-zA-Z]{0,3})", query, re.IGNORECASE)
        
        if section_match:
            # Normalize to Upper Case (e.g., "80c" -> "80C")
            target_section = section_match.group(1).upper()
            
            print(f"   🎯 Detected specific section request: {target_section}")
            
            # 1. Try Metadata Filter Search (High Precision)
            # This looks ONLY for documents where metadata['section'] == "80C"
            try:
                results = vectorstore.similarity_search(
                    query, 
                    k=10, 
                    filter={"section": target_section}
                )
                if results:
                    print(f"   ✅ Found {len(results)} exact matches via filter.")
            except Exception as filter_err:
                print(f"   ⚠️ Filter search failed (maybe metadata is different): {filter_err}")
                results = []

        # 2. Fallback: Standard Semantic Search (Broad Recall)
        # If the filter found nothing (or user didn't ask for a specific section), run normal search
        if not results:
            print("   🌐 Running standard semantic search...")
            if not results:
                print("   🌐 Running semantic search (Top-20 for reranking)...")
                results = vectorstore.similarity_search(query, k=20)
            if reranker and results:
                print("   🔁 Reranking with FlashRank...")

                passages = [
                    {
                        "id": str(i),
                        "text": doc.page_content
                    }
                    for i, doc in enumerate(results)
                ]

                rerank_request = RerankRequest(
                    query=query,
                    passages=passages
                )

                reranked = reranker.rerank(rerank_request)

                # Reorder Chroma docs according to FlashRank scores
                reranked_docs = []
                for item in reranked[:5]:  # Top-5 after reranking
                    reranked_docs.append(results[int(item["id"])])

                results = reranked_docs

        # 3. Process & Deduplicate Results
        seen_sections = set()
        output = []

        for doc in results:
            sec_id = doc.metadata.get("section")
            full_text = doc.metadata.get("parent_content")
            title = doc.metadata.get("title")

            # Deduplicate by section ID
            if sec_id in seen_sections:
                continue
            
            seen_sections.add(sec_id)

            output.append(
                f"SECTION: {sec_id}\n"
                f"TITLE: {title}\n"
                f"FULL CONTENT:\n{full_text}\n---"
            )
            
            # Stop after 3 unique sections to keep context window clean
            if len(output) >= 3:
                break

        final_response = "\n".join(output) if output else "No relevant sections found."
        return final_response

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
@tool
def search_news_articles(query: str) -> str:
    """
    Search for real-world news articles and practical examples related to tax rules.
    Use this tool to provide 'Real World Examples' after finding a legal rule.
    """
    if coll_articles is None:
        return "Error: News/Article DB not loaded"

    try:
        # Standardize query
        query_vec = model_articles.encode(query)

        # Basic vector search in the articles collection
        candidates = list(
            coll_articles.find(
                {},
                {"embedding": 1, "title": 1, "full_text": 1}
            ).limit(100)
        )

        if not candidates:
            return "No relevant news examples found."

        cand_vecs = np.array([c["embedding"] for c in candidates])
        sims = cosine_similarity([query_vec], cand_vecs)[0]

        output = []
        # Get top 2 results for examples
        for idx in sims.argsort()[-2:][::-1]:
            if sims[idx] < 0.3: 
                continue

            doc = candidates[idx]
            output.append(
                f"EXAMPLE FROM NEWS/ARTICLE: {doc.get('title')}\n"
                f"SUMMARY: {doc.get('full_text', '')[:500]}..."
            )

        return "\n\n".join(output) if output else "No specific real-world examples found."

    except Exception as e:
        return f"Error searching news: {e}"
