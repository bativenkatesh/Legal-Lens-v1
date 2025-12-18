import pymongo
import certifi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

# --- CONFIGURATION ---
MONGO_URI = "mongodb+srv://venky27274_db_user:eaOlsGwJxuygDSOw@cluster0.3ue3woh.mongodb.net/"
DB_NAME = "LegalLens"
COLLECTION_NAME = "acts"
CHROMA_PATH = "./chroma_db_smart"

# --- 1. CONNECT TO DATA ---
print("🔌 Connecting to MongoDB...")
client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
collection = client[DB_NAME][COLLECTION_NAME]
# Only fetch necessary fields to save memory
cursor = collection.find({}, {"full_content": 1, "section": 1, "title": 1})
total_docs = collection.count_documents({})

# --- 2. PREPARE EMBEDDING ---
print("🧠 Loading Embedding Model (nomic-embed-text)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- 3. OPTIMIZED CHUNKING LOGIC ---
# IMPROVEMENT 1: Increased Chunk Size
# 'nomic' works better with larger context. 1200 chars ensures we capture
# the full "Proviso" (condition) and its context in one vector.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,      # Increased from 600
    chunk_overlap=300,    # Increased overlap to prevent cutting sentences
    separators=["\n\n", "\n", ". ", " ", ""] # Prioritize paragraph breaks
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
        # IMPROVEMENT 2: Natural Language Context Injection
        # Instead of "Section: X", we make a sentence. This matches natural queries better.
        smart_content = (
            f"This text is from Section {section_id} of the Indian Income Tax Act "
            f"regarding '{title}'. \n\n{chunk}"
        )
        
        new_doc = Document(
            page_content=smart_content,
            metadata={
                "section": section_id,
                "title": title,
                "original_chunk": chunk,
                "parent_content": full_text
            }
        )
        docs_to_index.append(new_doc)

print(f"🧩 Created {len(docs_to_index)} smart-context chunks.")

# --- 4. SAVE TO CHROMA (BATCHED) ---
print(f"💾 Saving to {CHROMA_PATH}...")

# Initialize (or reset) the vectorstore
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

BATCH_SIZE = 2000 
total_batches = (len(docs_to_index) // BATCH_SIZE) + 1

for i in range(0, len(docs_to_index), BATCH_SIZE):
    batch = docs_to_index[i : i + BATCH_SIZE]
    if not batch: continue
    
    print(f"   Writing batch {i//BATCH_SIZE + 1}/{total_batches} ({len(batch)} docs)...")
    vectorstore.add_documents(documents=batch)

print("✅ Smart Indexing Complete!")