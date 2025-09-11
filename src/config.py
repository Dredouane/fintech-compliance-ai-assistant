import os
import sys
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Load environment variables from .env file
load_dotenv()

# --- Shared Configuration ---
# NOTE: No API key is needed for Ollama
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
COLLECTION_NAME = "financial_compliance"


def get_config():
    """
    Sets up and returns all necessary configurations for the application.
    """
    # Configure LlamaIndex to use local Ollama models
    Settings.llm = Ollama(model="llama3:8b", base_url=OLLAMA_URL)

    # The embedding model is used to convert documents into vectors
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=OLLAMA_URL
    )

    # Set up the Qdrant client
    qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)

    # Set up the Qdrant vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME
    )

    return qdrant_client, vector_store, COLLECTION_NAME