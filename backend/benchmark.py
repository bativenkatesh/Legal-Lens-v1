# backend/benchmark.py
import sys
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from flashrank import Ranker, RerankRequest


# --- CONFIG ---
CHROMA_PATH_SMART = "./chroma_db_smart"

print("--- [BENCHMARK] Initializing LegalLens Evaluation ---")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory=CHROMA_PATH_SMART, embedding_function=embeddings)
print("🔁 Initializing FlashRank...")

try:
    reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    print("✅ FlashRank Ready.")
except Exception as e:
    print(f"❌ FlashRank failed to load: {e}")
    reranker = None

# --- THE GOLDEN DATASET (20 Complex Queries) ---
# --- THE GOLDEN DATASET (50 Complex Queries) ---
test_set = [

    # ======================================================
    # CLUSTER 1: EXPORT & SEZ INCENTIVES (Section 10)
    # ======================================================
    {"query": "What deductions are available for 100% export-oriented undertakings?", "expected_section": "10B"},
    {"query": "tax holiday for special economic zones (SEZ)", "expected_section": "10A"},
    {"query": "conditions for newly established undertakings in free trade zones", "expected_section": "10A"},
    {"query": "deduction for export of wood based handmade articles", "expected_section": "10BA"},
    {"query": "consequences of amalgamation or demerger for export undertakings", "expected_section": "10B"},
    {"query": "profit deduction for software technology park units", "expected_section": "10A"},
    {"query": "tax exemption for export profits of EOUs", "expected_section": "10B"},
    {"query": "eligibility conditions for SEZ unit tax exemption", "expected_section": "10AA"},

    # ======================================================
    # CLUSTER 3: STARTUPS & BUSINESS DEDUCTIONS (Chapter VI-A)
    # ======================================================
    {"query": "tax deduction for eligible startups incorporated after 2016", "expected_section": "80-IAC"},
    {"query": "deduction in respect of employment of new employees", "expected_section": "80JJAA"},
    {"query": "royalty income deduction for authors of books", "expected_section": "80QQB"},
    {"query": "deduction for royalty on patents", "expected_section": "80RRB"},
    {"query": "profits deduction for infrastructure development undertakings", "expected_section": "80-IA"},
    {"query": "deduction for power generation companies", "expected_section": "80-IA"},
    {"query": "tax benefits for cooperative societies", "expected_section": "80P"},
    {"query": "deduction for export profits of certain undertakings", "expected_section": "80HHC"},

    # ======================================================
    # CLUSTER 4: CAPITAL GAINS EXEMPTIONS (Section 54 Series)
    # ======================================================
    {"query": "capital gains exemption on sale of residential house property", "expected_section": "54"},
    {"query": "exemption for transfer of agricultural land", "expected_section": "54B"},
    {"query": "capital gain exemption for investment in NHAI or REC bonds", "expected_section": "54EC"},
    {"query": "capital gains exemption on sale of multiple residential houses", "expected_section": "54F"},
    {"query": "exemption on compulsory acquisition of land", "expected_section": "54B"},
    {"query": "time limit for reinvestment to claim capital gain exemption", "expected_section": "54"},

    # ======================================================
    # CLUSTER 5: PERSONAL DEDUCTIONS (Individual Taxpayers)
    # ======================================================
    {"query": "deduction for interest paid on loan for higher education", "expected_section": "80E"},
    {"query": "deduction for interest on loan taken for residential house property", "expected_section": "80EE"},
    {"query": "medical insurance premium deduction limits", "expected_section": "80D"},
    {"query": "deduction for rent paid if HRA is not received", "expected_section": "80GG"},
    {"query": "tax deduction for donations to charitable institutions", "expected_section": "80G"},
    {"query": "deduction for maintenance of disabled dependent", "expected_section": "80DD"},
    {"query": "deduction for medical treatment of specified diseases", "expected_section": "80DDB"},
    {"query": "interest income deduction for senior citizens", "expected_section": "80TTB"},
    {"query": "savings account interest deduction", "expected_section": "80TTA"},

        # ======================================================
    # CLUSTER 6: EDGE CASES & HIGH-CONFUSION QUERIES
    # ======================================================

    # --- Similar wording, different sections (ranking stress) ---
    {"query": "tax exemption on profits of newly established industrial undertaking", "expected_section": "10A"},
    {"query": "profit linked deduction for infrastructure facilities", "expected_section": "80-IA"},
    {"query": "income tax benefit for SEZ developers", "expected_section": "10AA"},
    {"query": "tax exemption for export of computer software", "expected_section": "10A"},

    # --- Capital gains corner cases ---
    {"query": "capital gains exemption when land is compulsorily acquired by government", "expected_section": "54B"},
    {"query": "investment in bonds to save long term capital gains tax", "expected_section": "54EC"},
    {"query": "capital gains exemption if only one residential house is owned", "expected_section": "54F"},
    {"query": "capital gains tax relief for farmers selling agricultural land", "expected_section": "54B"},

    # --- Senior citizen & individual relief ---
    {"query": "tax deduction for interest income of senior citizens", "expected_section": "80TTB"},
    {"query": "medical expense deduction for dependent with disability", "expected_section": "80DD"},
    {"query": "tax deduction for treatment of cancer or chronic diseases", "expected_section": "80DDB"},
    {"query": "deduction for donations made to PM relief fund", "expected_section": "80G"}

]

def calculate_metrics(k=15):
    print(f"\n🚀 Starting Stress Test (Top-k={k})...")
    print(f"{'QUERY':<60} | {'EXPECTED':<10} | {'STATUS':<10} | {'RANK':<5}")
    print("-" * 100)
    
    hits = 0
    reciprocal_ranks = []
    
    for case in test_set:
        query = case["query"]
        expected = case["expected_section"]
        
        # 1. Run Retrieval
        try:
            # 1. Recall Stage (Vector Search)
            results = vectorstore.similarity_search(query, k=k)

            # 2. Precision Stage (FlashRank)
            if reranker and results:
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

                # Reorder docs based on reranker score
                reranked_docs = []
                for item in reranked:
                    reranked_docs.append(results[int(item["id"])])

                results = reranked_docs

        except Exception as e:
            print(f"Error searching '{query}': {e}")
            continue
        
        found = False
        rank = 0
        
        # 2. Check Results
        for i, doc in enumerate(results):
            # We check both the metadata field and the stamped content
            # This handles cases where metadata might be formatted differently
            meta_id = str(doc.metadata.get("section", ""))
            content_preview = doc.page_content[:100] # Peek at the "Smart Stamp"
            
            # Match Logic: Does the ID appear in the metadata OR the stamped text?
            if (expected in meta_id) or (f"Section {expected}" in content_preview):
                found = True
                rank = i + 1
                break
        
        # 3. Log Output
        if found:
            hits += 1
            reciprocal_ranks.append(1 / rank)
            print(f"{query[:58]:<60} | {expected:<10} | {'✅ PASS':<10} | {rank:<5}")
        else:
            reciprocal_ranks.append(0)
            print(f"{query[:58]:<60} | {expected:<10} | {'❌ FAIL':<10} | {'-':<5}")

    # --- STATISTICS ---
    recall = hits / len(test_set)
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    
    return recall, mrr

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # IMPORTANT: We use k=15 to solve the "Crowding" problem
    recall_score, mrr_score = calculate_metrics(k=15)

    print("\n" + "="*40)
    print("📊 LEGAL LENS BENCHMARK REPORT")
    print("="*40)
    print(f"Total Queries: {len(test_set)}")
    print(f"Recall@15:     {recall_score:.2f} ({recall_score*100}%)")
    print(f"MRR Score:     {mrr_score:.2f}")
    print("-" * 40)
    
    if recall_score > 0.85:
        print("🏆 STATUS: PUBLICATION READY")
        print("Great job! Your system is robust enough for a Scopus paper.")
    elif recall_score > 0.6:
        print("⚠️ STATUS: GOOD BUT NEEDS TUNING")
        print("Check if your MongoDB is missing chapters (e.g., Section 80 or 54).")
    else:
        print("❌ STATUS: CRITICAL FAILURE")
        print("Did you re-run 'ingest_smart.py'? Your DB might be empty or old.")