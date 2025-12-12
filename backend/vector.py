import pymongo
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import certifi

# 1. Connect to your existing MongoDB
client = pymongo.MongoClient("mongodb+srv://venky27274_db_user:eaOlsGwJxuygDSOw@cluster0.3ue3woh.mongodb.net/",tlsCAFile=certifi.where())
db = client["LegalLens"]
collection = db["acts"]

# 2. Fetch the documents (The raw data you already imported)
cursor = collection.find({}) # Fetches all docs

all_chunks = []
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    separators=["\n\n", "\n", "(", "."]
)

print("Processing documents from MongoDB...")

for doc in cursor:
    # Handle missing fields gracefully
    full_text = doc.get("full_content", "")
    section = doc.get("section", "Unknown")
    title = doc.get("title", "")
    
    # Skip empty documents
    if not full_text:
        continue

    # --- THE "CHUNKING" LOGIC I GAVE YOU ---
    
    # 3. Create the chunks
    split_texts = text_splitter.split_text(full_text)

    for i, text in enumerate(split_texts):
        # We create a LangChain 'Document' object for each chunk
        new_doc = Document(
            page_content=f"Section: {section}. Title: {title}. Content: {text}",
            metadata={
                "section": section,
                "title": title,
                "original_mongo_id": str(doc["_id"]), # Link back to Mongo if needed
                "chunk_index": i
            }
        )
        all_chunks.append(new_doc)

# 4. Save these NEW chunks into your Vector Store (Chroma)
# This creates the "Smart Index" separate from your Mongo data
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Note: This might take time depending on how many docs you have
vectorstore = Chroma.from_documents(
    documents=all_chunks, 
    embedding=embeddings,
    persist_directory="./chroma_db" # Save to disk so you don't have to run this every time
)

print(f"Finished! Created {len(all_chunks)} searchable chunks from your MongoDB data.")