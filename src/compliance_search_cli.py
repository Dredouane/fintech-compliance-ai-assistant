import sys
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from config import get_config

# Get the shared configuration
client, vector_store, collection_name = get_config()

# --- Query Engine Setup ---
print("Setting up the query engine...")

# Load the existing index from the vector store
index = VectorStoreIndex.from_vector_store(vector_store)

# Create a query engine from the index
query_engine = index.as_query_engine()

# --- CLI Tool Loop ---
print("CLI tool is ready. Type your query or 'exit' to quit.")
while True:
    query_text = input("Query: ")
    if query_text.lower() == "exit":
        break
    
    try:
        # Query the engine and print the response
        response = query_engine.query(query_text)
        print("\n" + "="*50)
        print("Response:")
        print(response.response)
        print("="*50 + "\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the RAG database is running properly.")