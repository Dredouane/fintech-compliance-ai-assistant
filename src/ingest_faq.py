import os
import json
import logging
from pathlib import Path
from llama_index.core import Document, VectorStoreIndex, StorageContext
from qdrant_client.http.models import Distance, VectorParams
from config import get_rag_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_faq_data(collection_name: str, file_path: str):
    """
    Ingests FAQ data into a dedicated Qdrant collection.
    
    Args:
        collection_name (str): The name of the Qdrant collection to use.
        file_path (str): The path to the JSON file containing the FAQ data.
    """
    logger.info(f"Setting up Qdrant for FAQ collection: {collection_name}...")
    
    # We use get_rag_config to get the Qdrant client and vector store
    # and we pass the new collection name
    qdrant_client, vector_store = get_rag_config(collection_name)
    
    # Create the collection if it doesn't exist
    if not qdrant_client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' not found. Creating it now...")
        # Note: You can retrieve vector size dynamically from your model if needed.
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        logger.info("Collection created.")
    else:
        logger.info(f"Collection '{collection_name}' already exists.")
    
    # Use pathlib for more robust path handling
    faq_file = Path(file_path)
    
    # Load data from the JSON file with robust error handling
    try:
        with faq_file.open('r', encoding='utf-8') as f:
            faq_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: FAQ file not found at {file_path}")
        return
    except json.JSONDecodeError:
        logger.error(f"Error: Failed to decode JSON from file at {file_path}. Check for syntax errors.")
        return
    
    if not isinstance(faq_data, list):
        logger.error(f"Error: Expected JSON data to be a list of objects, but got {type(faq_data).__name__}")
        return
    
    faq_documents = [
        Document(
            text=f"Question: {item['question']}\nAnswer: {item['answer']}",
            metadata={"type": "faq", "question": item['question']}
        )
        for item in faq_data
    ]
    
    logger.info(f"Preparing to ingest {len(faq_documents)} FAQ documents...")
    
    # Create a StorageContext with the vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create the index with the storage context
    # This will embed and store the documents in Qdrant
    index = VectorStoreIndex.from_documents(
        faq_documents,
        storage_context=storage_context,
        show_progress=True
    )
    
    # Verify ingestion by checking collection info
    collection_info = qdrant_client.get_collection(collection_name)
    logger.info(f"FAQ ingestion complete. Collection now has {collection_info.points_count} points.")
    
    return index


if __name__ == '__main__':
    # Define collection name and file path for the entry point
    FAQ_COLLECTION_NAME = os.getenv("FAQ_COLLECTION_NAME", "faq_compliance")
    FAQ_FILE_PATH = os.getenv("FAQ_FILE_PATH", "data/faq/faqs.json")
    
    ingest_faq_data(
        collection_name=FAQ_COLLECTION_NAME,
        file_path=FAQ_FILE_PATH
    )