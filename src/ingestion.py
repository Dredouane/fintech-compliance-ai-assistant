import logging
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from qdrant_client.http.models import Distance, VectorParams
from config import get_config

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ingestion():
    """
    Loads documents file by file, creates nodes, and builds the vector index in Qdrant.
    """
    logger.info("Starting file-by-file ingestion process...")

    qdrant_client, vector_store, collection_name = get_config()

    # Create the collection if it doesn't exist
    if not qdrant_client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' not found. Creating it now...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        logger.info("Collection created.")
    else:
        logger.info(f"Collection '{collection_name}' already exists.")

    # Get a recursive list of all documents
    doc_paths = []
    for dirpath, dirnames, filenames in os.walk("data"):
        for filename in filenames:
            if filename.endswith(".pdf"):
                doc_paths.append(os.path.join(dirpath, filename))
    
    logger.info(f"Found {len(doc_paths)} documents to process.")
    
    # Create the StorageContext, linking the index to the Qdrant vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Loop through each document and process it individually
    for doc_path in doc_paths:
        try:
            logger.info(f"Processing file: {doc_path}")
            
            # Load only the current document
            reader = SimpleDirectoryReader(input_files=[doc_path])
            documents = reader.load_data()
            
            if not documents:
                logger.warning(f"No documents loaded from {doc_path}. Skipping.")
                continue

            # Create the index for the current document. This handles chunking, embedding, and storage.
            # Using from_documents on a single document ensures it's processed atomically.
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            
            logger.info(f"Successfully processed and indexed {doc_path}")

        except Exception as e:
            logger.error(f"Failed to process file {doc_path}. Error: {e}", exc_info=True)
            continue
            
    logger.info("File-by-file ingestion complete.")


if __name__ == "__main__":
    run_ingestion()