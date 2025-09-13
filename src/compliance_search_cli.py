import logging
import os
import sys
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from config import get_config

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_query_engine_with_scores(query_text: str):
    """
    Queries the Qdrant vector store and returns the response with source documents and their confidence scores.
    """
    logger.info("Setting up the query engine...")
    
    # Use the centralized config file to get all necessary components
    
    _, vector_store, _ = get_config()
    
    # Load the index from the vector store
    index = VectorStoreIndex.from_vector_store(vector_store)


    # Number of top similar documents to retrieve
    TOP_K = int(os.getenv("TOP_K", 5))  

    # Configure a retriever to get documents and their scores
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=TOP_K  # Retrieve the top K most similar documents
    )

    # Retrieve nodes with scores
    retrieved_nodes_with_scores = retriever.retrieve(query_text)

    # Assemble the query engine
    response_synthesizer = get_response_synthesizer()
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )

    # Get the final response
    response = query_engine.query(query_text)

    # Print the confidence scores for debugging
    print("\n--- Retrieval Confidence Scores ---")
    for node_with_score in retrieved_nodes_with_scores:
        print(f"File: {node_with_score.metadata.get('file_name', 'N/A')}, Score: {node_with_score.score:.4f}")
    
    return response

if __name__ == '__main__':
    query = "What is the primary purpose of the Liquidity Coverage Ratio (LCR)?"
    result = run_query_engine_with_scores(query)
    print("\n--- LLM Response ---")
    print(result)