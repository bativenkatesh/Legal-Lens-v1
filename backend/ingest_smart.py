import pymongo
import certifi
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm
import re

# --- CONFIGURATION ---
MONGO_URI = "mongodb+srv://venky27274_db_user:eaOlsGwJxuygDSOw@cluster0.3ue3woh.mongodb.net/"
DB_NAME = "LegalLens"
COLLECTION_NAME = "acts"
CHROMA_PATH = "./chroma_db_smart"

# --- 1. CONNECT TO DATA ---
print("🔌 Connecting to MongoDB...")
client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
collection = client[DB_NAME][COLLECTION_NAME]
cursor = collection.find({}, {"full_content": 1, "section": 1, "title": 1})
total_docs = collection.count_documents({})

# --- 2. PREPARE EMBEDDING ---
print("🧠 Loading Embedding Model (nomic-embed-text)...")
# increase timeout to avoid "read timeout" errors on larger chunks
embeddings = OllamaEmbeddings(model="nomic-embed-text") 

# --- 3. OPTIMIZED CHUNKING LOGIC ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=300,
    separators=[
        # --- 1. Logical Legal Boundaries (High Priority) ---
        "\nNotwithstanding anything contained",
        "\nProvided that",
        "\nSubject to the provisions of",
        "\nSave as otherwise provided",
        "\nExplanation ",
        "\nExplanation.—",
        "\nExplanation:",
        "\nFor the purposes of this section",
        "\nSub-section",
        "\nClause",
        "\nProvided further that",
        
        # --- 2. Structural Fallbacks (Medium Priority) ---
        "\n\n",   # Standard Paragraph break
        "\n",     # Line break
        ". ",     # Sentence break
        " ",      # Word break
        ""        # Char break (Absolute last resort)
    ]
)

docs_to_index = []
print("⚙️  Smart-Processing Documents...")

for mongo_doc in tqdm(cursor, total=total_docs):
    full_text = mongo_doc.get("full_content", "")
    section_id = mongo_doc.get("section", "Unknown Section")
    title = mongo_doc.get("title", "") or "General Provision"
    
    if not full_text: continue

    chunks = splitter.split_text(full_text)
    
    for i, chunk in enumerate(chunks):
        # IMPROVEMENT: Extract Subsection if present (e.g., "(1)", "(2A)")
        subsection_match = re.search(r"\((\d+[A-Z]{0,2})\)", chunk)
        subsection = subsection_match.group(1) if subsection_match else None
        
        # Smart Context Injection
        semantic_stamp = (
            f"This legal provision is from Section {section_id}"
            f"{f'({subsection})' if subsection else ''} "
            f"of the Indian Income Tax Act, titled '{title}'. "
            f"It explains eligibility conditions, definitions, "
            f"time limits, procedures, forms, exceptions, "
            f"and consequences applicable under this section.\n\n"
        )
        
        new_doc = Document(
            page_content=semantic_stamp + chunk,
            metadata={
                "section": section_id,
                "subsection": subsection if subsection else "Main",
                "title": title,
                "chunk_id": i,
                # Store parent content only if needed (saves DB space if skipped)
                # "parent_content": full_text[:100] + "..." 
            }
        )
        docs_to_index.append(new_doc)

print(f"🧩 Created {len(docs_to_index)} smart-context chunks.")

# --- 4. SAVE TO CHROMA (BATCHED) ---
print(f"💾 Saving to {CHROMA_PATH}...")

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

# REDUCED BATCH SIZE: 256 -> 32
# Large chunks + Metadata = Heavy Payload. 32 is safer for Local LLMs.
BATCH_SIZE = 32
total_batches = (len(docs_to_index) // BATCH_SIZE) + 1

for i in range(0, len(docs_to_index), BATCH_SIZE):
    batch = docs_to_index[i : i + BATCH_SIZE]
    if not batch: continue
    
    print(f"   Writing batch {i//BATCH_SIZE + 1}/{total_batches} ({len(batch)} docs)...")
    
    try:
        vectorstore.add_documents(documents=batch)
        # Sleep to let Ollama/GPU cool down slightly
        time.sleep(0.5) 
    except Exception as e:
        print(f"❌ Error in batch {i//BATCH_SIZE + 1}: {e}")
        # Optional: Retry logic could go here
        continue

print("✅ Smart Indexing Complete!")