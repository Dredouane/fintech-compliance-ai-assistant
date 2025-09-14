import logging
import os
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from qdrant_client.http.models import Distance, VectorParams
from config import get_rag_config

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


compliance_rag_collection = os.getenv("COMPLIANCE_RAG_COLLECTION", "financial_compliance")

def run_ingestion():
    """
    Loads documents file by file, creates nodes, and builds the vector index in Qdrant.
    """
    logger.info("Starting file-by-file ingestion process...")

    qdrant_client, vector_store = get_rag_config(compliance_rag_collection)

    # Create the collection if it doesn't exist
    if not qdrant_client.collection_exists(compliance_rag_collection):
        logger.info(f"Collection '{compliance_rag_collection}' not found. Creating it now...")
        qdrant_client.create_collection(
            collection_name=compliance_rag_collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        logger.info("Collection created.")
    else:
        logger.info(f"Collection '{compliance_rag_collection}' already exists.")
        # Get initial point count
        collection_info = qdrant_client.get_collection(compliance_rag_collection)
        initial_points = collection_info.points_count
        logger.info(f"Collection currently has {initial_points} points.")

    # Get a recursive list of all documents using pathlib for better path handling
    data_dir = Path("data")
    doc_paths = list(data_dir.rglob("*.pdf"))
    
    logger.info(f"Found {len(doc_paths)} PDF documents to process.")
    
    if not doc_paths:
        logger.warning("No PDF documents found in the data directory.")
        return
    
    # Create the StorageContext, linking the index to the Qdrant vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Track processing statistics
    successful_docs = 0
    failed_docs = 0
    total_chunks = 0

    # Loop through each document and process it individually
    for doc_path in doc_paths:
        try:
            logger.info(f"Processing file: {doc_path}")
            
            # Load only the current document
            reader = SimpleDirectoryReader(input_files=[str(doc_path)])
            documents = reader.load_data()
            
            if not documents:
                logger.warning(f"No documents loaded from {doc_path}. Skipping.")
                failed_docs += 1
                continue
            
            # Add metadata to each document
            for doc in documents:
                doc.metadata.update({
                    "source": str(doc_path),
                    "filename": doc_path.name,
                    "type": "compliance_document"
                })
            
            logger.info(f"Loaded {len(documents)} document chunks from {doc_path.name}")

            # Create the index for the current document. This handles chunking, embedding, and storage.
            # Using from_documents ensures documents are processed and stored in Qdrant.
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            
            successful_docs += 1
            total_chunks += len(documents)
            logger.info(f"Successfully processed and indexed {doc_path.name}")

        except Exception as e:
            logger.error(f"Failed to process file {doc_path}. Error: {e}", exc_info=True)
            failed_docs += 1
            continue
    
    # Final verification
    collection_info = qdrant_client.get_collection(compliance_rag_collection)
    final_points = collection_info.points_count
    
    logger.info("=" * 50)
    logger.info("File-by-file ingestion complete.")
    logger.info(f"Successfully processed: {successful_docs}/{len(doc_paths)} documents")
    logger.info(f"Failed: {failed_docs} documents")
    logger.info(f"Total document chunks indexed: {total_chunks}")
    logger.info(f"Total points in collection: {final_points}")
    logger.info("=" * 50)
    
    return {
        "successful_docs": successful_docs,
        "failed_docs": failed_docs,
        "total_chunks": total_chunks,
        "total_points": final_points
    }


if __name__ == "__main__":
    run_ingestion()