import logging
import os
from enum import Enum
from llama_index.core import VectorStoreIndex, get_response_synthesizer, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.cleanlab import CleanlabTLM
from config import get_config
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define an Enum to easily switch scoring methods
class ConfidenceMethod(Enum):
    RAG_SCORE = "rag_score"
    LLM_SCORE = "llm_score"
    CLEANLAB_TLM = "cleanlab_tlm"

def run_query_engine_with_scores(query_text: str):
    """
    Queries the vector store and returns a comprehensive result object with scores and sources.
    """
    # Get the scoring method from the environment variable with RAG_SCORE as default
    scoring_method_str = os.getenv("SCORING_METHOD", "rag_score").lower()
    try:
        scoring_method = ConfidenceMethod(scoring_method_str)
    except ValueError:
        logger.warning(f"Invalid scoring method '{scoring_method_str}' in .env. Defaulting to 'rag_score'.")
        scoring_method = ConfidenceMethod.RAG_SCORE
        
    logger.info(f"Setting up the query engine with scoring method: {scoring_method.value}...")
    
    _, vector_store, _ = get_config()
    mistral_llm = Settings.llm
    TOP_K = int(os.getenv("TOP_K", 5))

    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)
    retrieved_nodes_with_scores = retriever.retrieve(query_text)

    response_synthesizer = get_response_synthesizer(llm=mistral_llm)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )

    response = query_engine.query(query_text)
    
    result = {
        "response": str(response),
        "score": None,
        "score_type": scoring_method.value,
        "source_nodes": retrieved_nodes_with_scores # Store the source nodes here
    }

    if scoring_method == ConfidenceMethod.RAG_SCORE:
        if retrieved_nodes_with_scores:
            result["score"] = max(node.score for node in retrieved_nodes_with_scores)
    elif scoring_method == ConfidenceMethod.CLEANLAB_TLM:
        llm = CleanlabTLM(
            api_key=os.environ.get("CLEANLAB_API_KEY")
        )
        # Use CleanlabTLM to generate the response and get the score
        cleanlab_response = llm.complete(query_text)
        
        # Check if the score is present in the additional_kwargs
        if 'trustworthiness_score' in cleanlab_response.additional_kwargs:
            result["score"] = cleanlab_response.additional_kwargs['trustworthiness_score']
            result["response"] = cleanlab_response.text
        else:
            result["score"] = None
            result["response"] = cleanlab_response.text

    return result

def run_cli():
    """
    Runs the command-line interface for the compliance search agent.
    """
    while True:
        prompt = input("Ask a question about your documents (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            logger.info("Exiting.")
            break
        
        try:
            result = run_query_engine_with_scores(prompt)
            print("\n--- Final Result ---")
            print(f"Confidence Score ({result['score_type']}): {result['score']:.4f}" if result['score'] is not None else "Confidence Score: N/A")
            print("\n--- LLM Response ---")
            print(result["response"])
            print("\n--- Sources (Citations) ---")
            for node in result['source_nodes']:
                filename = node.metadata.get('file_name', 'N/A')
                print(f"File: {filename}, Score: {node.score:.4f}")
            print("\n" + "-"*50 + "\n")
            
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)
            continue
            
if __name__ == '__main__':
    run_cli()