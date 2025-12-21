import sys
import time
import numpy as np
import pymongo
import certifi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from flashrank import Ranker, RerankRequest
from tabulate import tabulate

# --- CONFIGURATION ---
MONGO_URI = "mongodb+srv://venky27274_db_user:eaOlsGwJxuygDSOw@cluster0.3ue3woh.mongodb.net/"
DB_NAME = "LegalLens"
COLLECTION_NAME = "acts"
CHROMA_PATH_SMART = "./chroma_db_smart"

print("--- [BENCHMARK] Initializing Ultimate Comparison Suite ---")

# 1. SETUP MONGODB (KEYWORD SEARCH)
print("🔌 Connecting to MongoDB (Keyword Engine)...")
try:
    client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
    mongo_col = client[DB_NAME][COLLECTION_NAME]
    # Create Text Index if it doesn't exist
    mongo_col.create_index([
        ("full_content", "text"), 
        ("title", "text"), 
        ("section", "text")
    ], name="legal_text_search")
    print("✅ MongoDB Text Index Ready.")
except Exception as e:
    print(f"⚠️ MongoDB Connection Failed: {e}")
    mongo_col = None

# 2. SETUP VECTOR STORE
print("🔌 Loading ChromaDB (Vector Engine)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=CHROMA_PATH_SMART, embedding_function=embeddings)

# 3. SETUP RERANKER
print("🔁 Loading FlashRank (Reranker)...")
try:
    reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./opt")
    print("✅ FlashRank Ready.")
except Exception as e:
    print(f"❌ FlashRank failed: {e}")
    reranker = None

# --- EXPANDED DATASET (60 Queries) ---
test_set = [

# =====================================================
# CLUSTER 1: EXPORT & SEZ (Section 10A / 10AA / 10B / 10BA)
# =====================================================
{"query": "deductions for 100% export-oriented undertakings", "expected": "10B"},
{"query": "tax holiday available for SEZ units", "expected": "10AA"},
{"query": "conditions for software technology park units", "expected": "10A"},
{"query": "export of handmade wooden articles deduction", "expected": "10BA"},
{"query": "audit requirements for claiming export deduction", "expected": "10B"},
{"query": "deduction for online export of computer software", "expected": "10B"},
{"query": "definition of computer programme for export benefits", "expected": "10BB"},
{"query": "tax benefits for SEZ developers", "expected": "10AA"},

# =====================================================
# CLUSTER 2: STARTUPS & BUSINESS INCENTIVES (80 SERIES)
# =====================================================
{"query": "tax deduction for eligible startups incorporated after 2016", "expected": "80-IAC"},
{"query": "profit linked deduction for infrastructure development undertakings", "expected": "80-IA"},
{"query": "deduction for employment of new employees", "expected": "80JJAA"},
{"query": "tax benefits for cooperative societies", "expected": "80P"},
{"query": "deduction for profits from export business", "expected": "80HHC"},
{"query": "deduction for power generation companies", "expected": "80-IA"},
{"query": "tax benefits for housing projects", "expected": "80-IBA"},
{"query": "deduction for research and development expenditure", "expected": "80-IB"},

# =====================================================
# CLUSTER 3: CAPITAL GAINS (SECTION 54 FAMILY)
# =====================================================
{"query": "capital gains exemption on sale of residential house", "expected": "54"},
{"query": "exemption for transfer of agricultural land", "expected": "54B"},
{"query": "capital gains exemption for investment in NHAI bonds", "expected": "54EC"},
{"query": "capital gains exemption when only one residential house is owned", "expected": "54F"},
{"query": "time limit for reinvestment to claim capital gains exemption", "expected": "54"},
{"query": "capital gains relief on compulsory acquisition of land", "expected": "54B"},

# =====================================================
# CLUSTER 4: PERSONAL DEDUCTIONS (INDIVIDUAL TAXPAYERS)
# =====================================================
{"query": "deduction for interest paid on education loan", "expected": "80E"},
{"query": "medical insurance premium deduction limits", "expected": "80D"},
{"query": "deduction for maintenance of disabled dependent", "expected": "80DD"},
{"query": "deduction for treatment of specified diseases", "expected": "80DDB"},
{"query": "interest income deduction for senior citizens", "expected": "80TTB"},
{"query": "savings account interest deduction", "expected": "80TTA"},
{"query": "deduction for rent paid if HRA not received", "expected": "80GG"},
{"query": "donation deduction for PM relief fund", "expected": "80G"},

# =====================================================
# CLUSTER 5: SALARY & RETIREMENT (LIMITED SECTION 10)
# =====================================================
{"query": "house rent allowance exemption rules", "expected": "10(13A)"},
{"query": "gratuity exemption limit on retirement", "expected": "10(10)"},
{"query": "voluntary retirement compensation tax exemption", "expected": "10(10C)"},
{"query": "leave encashment exemption at retirement", "expected": "10(10AA)"},
{"query": "commuted pension exemption rules", "expected": "10(10A)"},
{"query": "agricultural income tax exemption", "expected": "10(1)"},

# =====================================================
# CLUSTER 6: SEMANTIC / NATURAL LANGUAGE (REAL USER QUERIES)
# =====================================================
{"query": "how can a startup save tax legally in india", "expected": "80-IAC"},
{"query": "how to reduce capital gains tax after selling land", "expected": "54"},
{"query": "tax benefits for senior citizens earning interest income", "expected": "80TTB"},
{"query": "tax saving options for salaried employees without hra", "expected": "80GG"},
{"query": "how to save tax on profits from exports", "expected": "10B"},
{"query": "is income from farming taxable in india", "expected": "10(1)"},
{"query": "tax benefit for parents paying insurance for children", "expected": "80D"},
{"query": "tax deduction available for donations to charity", "expected": "80G"},
{"query": "tax benefit for investment in infrastructure bonds", "expected": "54EC"},
{"query": "deduction for interest on education loan taken abroad", "expected": "80E"}
]

def check_match(doc_content, doc_meta, expected_id):
    """Universal matcher for Mongo and Chroma docs"""
    # 1. Metadata Check
    if isinstance(doc_meta, dict):
        section_val = str(doc_meta.get("section", ""))
        if expected_id == section_val: return True
    
    # 2. Content Check (Smart Stamp)
    # Checks if "Section 10B" appears in the first 200 chars
    if f"Section {expected_id}" in str(doc_content)[:200]:
        return True
    
    return False

def run_benchmark():
    print(f"\n🚀 Running Tri-Fold Benchmark on {len(test_set)} Queries...")
    
    table_data = []
    stats = {
        "mongo": {"hits": 0, "mrr": 0},
        "vector": {"hits": 0, "mrr": 0},
        "rerank": {"hits": 0, "mrr": 0}
    }

    for case in test_set:
        query = case["query"]
        expected = case["expected"]
        
        # --- 1. MONGO SEARCH (Keyword) ---
        m_rank = "-"
        if mongo_col is not None: # <--- FIXED CRASH HERE
            try:
                # Text search sorted by relevance score
                cursor = mongo_col.find(
                    {"$text": {"$search": query}},
                    {"score": {"$meta": "textScore"}, "section": 1, "full_content": 1}
                ).sort([("score", {"$meta": "textScore"})]).limit(15)
                
                for i, doc in enumerate(cursor):
                    if check_match(doc.get("full_content"), doc, expected):
                        m_rank = i + 1
                        stats["mongo"]["hits"] += 1
                        stats["mongo"]["mrr"] += 1/m_rank
                        break
            except Exception: pass

        # --- 2. VECTOR SEARCH (Standard RAG) ---
        v_rank = "-"
        v_results = []
        try:
            # Fetch top 15 directly
            v_results = vectorstore.similarity_search(query, k=15)
            for i, doc in enumerate(v_results):
                if check_match(doc.page_content, doc.metadata, expected):
                    v_rank = i + 1
                    stats["vector"]["hits"] += 1
                    stats["vector"]["mrr"] += 1/v_rank
                    break
        except Exception: pass

        # --- 3. RERANK SEARCH (LegalLens) ---
        r_rank = "-"
        try:
            # Step A: Fetch Deep (Recall) - Increased to 60
            deep_results = vectorstore.similarity_search(query, k=60)
            
            # Step B: Rerank (Precision)
            if reranker is not None:
                passages = [{"id": str(i), "text": d.page_content} for i, d in enumerate(deep_results)]
                rerank_req = RerankRequest(query=query, passages=passages)
                ranked = reranker.rerank(rerank_req)
                
                # Step C: Top 15
                final_docs = [deep_results[int(r['id'])] for r in ranked[:15]]
                
                for i, doc in enumerate(final_docs):
                    if check_match(doc.page_content, doc.metadata, expected):
                        r_rank = i + 1
                        stats["rerank"]["hits"] += 1
                        stats["rerank"]["mrr"] += 1/r_rank
                        break
        except Exception: pass

        # Log Row
        # Format: Query | Exp | Mongo | Vector | Rerank
        row = [
            query[:30] + "...",
            expected,
            f"✅ {m_rank}" if m_rank != "-" else "❌",
            f"✅ {v_rank}" if v_rank != "-" else "❌",
            f"✅ {r_rank}" if r_rank != "-" else "❌"
        ]
        table_data.append(row)
        sys.stdout.write(".")
        sys.stdout.flush()

    print("\n\n")
    headers = ["Query", "Exp", "Mongo (Key)", "Vector (Std)", "Rerank (Adv)"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))

    # --- FINAL SCORES ---
    total = len(test_set)
    print("\n" + "="*60)
    print("📊 FINAL SCIENTIFIC COMPARISON")
    print("="*60)
    print(f"{'METRIC':<15} | {'MONGO (Baseline)':<18} | {'VECTOR (RAG)':<18} | {'LEGALLENS (Ours)':<18}")
    print("-" * 80)
    
    m_rec = stats['mongo']['hits']/total
    v_rec = stats['vector']['hits']/total
    r_rec = stats['rerank']['hits']/total
    
    m_mrr = stats['mongo']['mrr']/total
    v_mrr = stats['vector']['mrr']/total
    r_mrr = stats['rerank']['mrr']/total

    print(f"{'Recall@15':<15} | {m_rec:.2%}             | {v_rec:.2%}             | {r_rec:.2%}")
    print(f"{'MRR Score':<15} | {m_mrr:.4f}             | {v_mrr:.4f}             | {r_mrr:.4f}")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()