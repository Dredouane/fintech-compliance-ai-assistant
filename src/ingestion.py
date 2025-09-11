import os
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, StorageContext
from config import get_config


# Get the shared configuration
client, vector_store, collection_name = get_config()

# Set up the storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- Data Ingestion Pipeline ---
# Load documents from the 'data' directory
print("Loading documents from the 'data/' directory...")
documents = SimpleDirectoryReader(
    input_dir="data",
    recursive=True  
).load_data()
print(f"Found {len(documents)} documents.")

# Create the vector index, which ingests, embeds, and stores the documents
# This is the core of our RAG pipeline's ingestion step
print("Creating the vector index and storing embeddings in Qdrant...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
print("Ingestion complete. Documents are now indexed.")

print("Setup complete. You are ready to start querying.")

