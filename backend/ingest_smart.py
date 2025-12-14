import pymongo
import certifi
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

# --- CONFIGURATION ---
MONGO_URI = "mongodb+srv://venky27274_db_user:eaOlsGwJxuygDSOw@cluster0.3ue3woh.mongodb.net/" # Or your Atlas URI
DB_NAME = "LegalLens"
COLLECTION_NAME = "acts"
CHROMA_PATH = "./chroma_db_smart"  # New folder for the smart index

# --- 1. CONNECT TO DATA ---
print("🔌 Connecting to MongoDB...")
client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())# If using Atlas, uncomment below:
# client = pymongo.MongoClient("YOUR_ATLAS_URI", tlsCAFile=certifi.where())

collection = client[DB_NAME][COLLECTION_NAME]
cursor = collection.find({})
total_docs = collection.count_documents({})

# --- 2. PREPARE EMBEDDING ---
print("🧠 Loading Embedding Model...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- 3. SMART CHUNKING LOGIC ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150
)

docs_to_index = []
print("⚙️  Smart-Processing Documents...")

for mongo_doc in tqdm(cursor, total=total_docs):
    full_text = mongo_doc.get("full_content", "")
    # Try to find the specific section number from the doc
    # (Assuming your Mongo docs have a 'section' field, or we parse it)
    section_id = mongo_doc.get("section", "Unknown Section")
    title = mongo_doc.get("title", "")
    
    # Clean up empty text
    if not full_text: continue

    # Create chunks
    chunks = splitter.split_text(full_text)
    
    for i, chunk in enumerate(chunks):
        # --- THE MAGIC TRICK ---
        # We PREPEND the Section ID to the content. 
        # Now, even a chunk in the middle KNOWS it belongs to Section 10B.
        
        smart_content = f"Section {section_id}: {title}\nContent: {chunk}"
        
        new_doc = Document(
            page_content=smart_content,  # <--- We search this ENRICHED text
            metadata={
                "section": section_id,
                "title": title,
                "original_chunk": chunk,     # Keep original for display
                "parent_content": full_text  # Keep full law for context
            }
        )
        docs_to_index.append(new_doc)

print(f"🧩 Created {len(docs_to_index)} smart-context chunks.")

# --- 4. SAVE TO CHROMA ---
print(f"💾 Saving to {CHROMA_PATH}...")
vectorstore = Chroma.from_documents(
    documents=docs_to_index, 
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)
print("✅ Smart Indexing Complete!")