import os
import sys
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenerativeAI
from llama_index.embeddings.google_genai import GoogleEmbedding
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Load environment variables from .env file
load_dotenv()

# --- Shared Configuration ---
api_key = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "financial_compliance"

if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
    sys.exit(1)

def get_config():
    """
    Sets up and returns all necessary configurations for the application.
    """
    # Configure LlamaIndex global settings
    Settings.llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-latest", api_key=api_key)
    Settings.embed_model = GoogleEmbedding(model_name="models/embedding-001", api_key=api_key)

    # Set up the Qdrant client
    qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)
    
    # Set up the Qdrant vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME
    )
    
    return qdrant_client, vector_store, COLLECTION_NAME