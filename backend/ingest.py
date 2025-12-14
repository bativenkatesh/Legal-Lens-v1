import pymongo
import certifi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm  # Progress bar

# --- 1. CONFIGURATION ---
MONGO_URI = "mongodb+srv://venky27274_db_user:eaOlsGwJxuygDSOw@cluster0.3ue3woh.mongodb.net/"
DB_NAME = "LegalLens"       # Check your actual DB name in Mongo Atlas
COLLECTION_NAME = "acts"  # Check your actual collection name
CHROMA_PATH = "./chroma_db_new" # We use a new folder name to be safe

# --- 2. CONNECT TO MONGO ---
print("🔌 Connecting to MongoDB...")
client = pymongo.MongoClient(MONGO_URI, tlsCAFile=certifi.where())
collection = client[DB_NAME][COLLECTION_NAME]

# Fetch all documents
cursor = collection.find({})
total_docs = collection.count_documents({})
print(f"📚 Found {total_docs} legal documents in MongoDB.")

# --- 3. PREPARE EMBEDDING MODEL ---
# Using 'nomic-embed-text' for high-quality, long-context support
print("🧠 Loading Embedding Model (nomic-embed-text)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- 4. PROCESSING LOGIC (The "Small-to-Big" Magic) ---
# We use a SMALL chunk size for the search index (accuracy)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=[
        "\nSection ",
        "\nExplanation",
        "\nProvided that",
        "\nSub-section",
        "\n("
    ]
)

docs_to_index = []

print("⚙️ Processing and splitting documents...")
for mongo_doc in tqdm(cursor, total=total_docs):
    # Get the "Parent" content (The Full Law)
    full_text = mongo_doc.get("full_content", "")
    section = mongo_doc.get("section", "Unknown")
    title = mongo_doc.get("title", "No Title")
    
    if not full_text:
        continue

    # Create "Child" chunks from the Parent
    child_chunks = child_splitter.split_text(full_text)
    
    for i, chunk_text in enumerate(child_chunks):
        # Create a Document for the Vector Store
        # CRITICAL: We store the 'full_text' in metadata so we can retrieve it later!
        new_doc = Document(
            page_content=chunk_text, # We SEARCH on this small text
            metadata={
                "section": section,
                "title": title,
                "chunk_id": i,
                "type": "child_fragment",
                "parent_content": full_text  # <--- The "Big" Context stored here
            }
        )
        docs_to_index.append(new_doc)

print(f"🧩 Created {len(docs_to_index)} searchable fragments.")

# --- 5. SAVE TO CHROMA ---
print("Floppy Disk Save: Indexing to ChromaDB (This may take a minute)...")
vectorstore = Chroma.from_documents(
    documents=docs_to_index,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)

print(f"✅ Success! Database saved to '{CHROMA_PATH}'.")
print("You can now delete the old 'chroma_db' folder if you want.")