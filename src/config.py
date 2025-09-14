import os
import sys
from dotenv import load_dotenv
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import psycopg2

# Load environment variables from .env file
load_dotenv()

# --- Shared Configuration ---
QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# --- PostgreSQL Configuration from .env file ---
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432))
}


def get_rag_config(collection_name: str):
    """
    Sets up and returns the LlamaIndex and Qdrant configurations for RAG.
    """
    # Configure LlamaIndex to use local Mistral models
    Settings.llm = MistralAI(model="mistral-medium", api_key=MISTRAL_API_KEY)
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url=OLLAMA_URL
    )
    
    # Set up the Qdrant client and vector store
    qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name
    )
    
    return qdrant_client, vector_store


def get_sql_config():
    """
    Connects to the database, creates tables from SQL scripts, and returns the connection.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        with open("scripts/create_table.sql", "r") as f:
            cur.execute(f.read())
        
        conn.commit()
        print("Database tables checked/created successfully.")
        
        return conn
        
    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error setting up database: {error}")
        sys.exit(1)


def get_config(collection_name: str = "financial_compliance"):
    """
    Combines configurations and returns all necessary services.
    This is the main entry point for the application to get all services.
    """
    qdrant_client, vector_store = get_rag_config(collection_name)
    db_connection = get_sql_config()
    
    return qdrant_client, vector_store, db_connection