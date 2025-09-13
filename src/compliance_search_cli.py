import logging
import sys
from llama_index.core import VectorStoreIndex, StorageContext
from config import get_config

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_query_engine():
    """
    Sets up the query engine to answer questions based on the vector store.
    """
    logger.info("Setting up the query engine...")
    
    # Get the Qdrant vector store from config
    _, vector_store, collection_name = get_config()

    # Create the StorageContext, linking to the Qdrant vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store, 
        storage_context=storage_context
    )
    
    # Create the query engine
    query_engine = index.as_query_engine()

    logger.info("Query engine is ready. You can start asking questions.")
    
    # Enter a loop to ask questions
    while True:
        prompt = input("Ask a question about your documents (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            logger.info("Exiting.")
            break
        
        try:
            response = query_engine.query(prompt)
            print(f"Response: {response}")
            
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            continue
            
if __name__ == "__main__":
    run_query_engine()