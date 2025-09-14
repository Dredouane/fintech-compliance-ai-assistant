import logging
import os
from enum import Enum
from llama_index.core import VectorStoreIndex, get_response_synthesizer, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.cleanlab import CleanlabTLM
from config import get_sql_config, get_rag_config
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql

# Load environment variables from .env file
load_dotenv()

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

compliance_rag_collection = os.getenv("COMPLIANCE_RAG_COLLECTION", "financial_compliance")

# Define an Enum to easily switch scoring methods
class ConfidenceMethod(Enum):
    RAG_SCORE = "rag_score"
    CLEANLAB_TLM = "cleanlab_tlm"

# --- CORE SEARCH FUNCTION (NO GUARDRails) ---
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
    
    _, vector_store = get_rag_config(compliance_rag_collection)
    mistral_llm = Settings.llm
    TOP_K = int(os.getenv("TOP_K", 5))

    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)
    retrieved_nodes_with_scores = retriever.retrieve(query_text)

    # --- Generate Initial Score ---
    initial_score = None
    if scoring_method == ConfidenceMethod.RAG_SCORE:
        if retrieved_nodes_with_scores:
            initial_score = max(node.score for node in retrieved_nodes_with_scores)
    elif scoring_method == ConfidenceMethod.CLEANLAB_TLM:
        llm = CleanlabTLM(
            api_key=os.environ.get("CLEANLAB_API_KEY")
        )
        cleanlab_response = llm.complete(query_text)
        if 'trustworthiness_score' in cleanlab_response.additional_kwargs:
            initial_score = cleanlab_response.additional_kwargs['trustworthiness_score']

    # If the score is high enough, proceed to synthesize the response
    response_synthesizer = get_response_synthesizer(llm=mistral_llm)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )
    final_response = query_engine.query(query_text)

    # Return the full, confident result
    return {
        "response": str(final_response),
        "score": initial_score,
        "score_type": scoring_method.value,
        "source_nodes": retrieved_nodes_with_scores
    }


def log_for_human_review(data: dict):
    """
    Logs a low-confidence query and its details to a PostgreSQL database.
    """
    try:
        conn = get_sql_config()
        cur = conn.cursor()

        # Prepare the data to be inserted
        original_query = data.get("original_query", "")
        llm_response = data.get("response", "")
        confidence_score = data.get("score", None)
        score_type = data.get("score_type", "")
        
        # Extract filenames from the source nodes
        source_filenames = [
            node.metadata.get('file_name', 'N/A')
            for node in data.get("source_nodes", [])
        ]
        
        # Use sql.Identifier and sql.Literal for safe queries
        insert_query = sql.SQL("""
            INSERT INTO human_review_queue (
                original_query, llm_response, confidence_score, score_type, source_filenames, timestamp
            ) VALUES (
                {original_query}, {llm_response}, {confidence_score}, {score_type}, {source_filenames}, NOW()
            );
        """).format(
            original_query=sql.Literal(original_query),
            llm_response=sql.Literal(llm_response),
            confidence_score=sql.Literal(confidence_score),
            score_type=sql.Literal(score_type),
            source_filenames=sql.Literal(source_filenames)
        )
        
        cur.execute(insert_query)
        conn.commit()
        logger.info("Successfully logged low-confidence query for human review.")

    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(f"Error logging to database: {error}")
    finally:
        if conn:
            cur.close()
            conn.close()

# --- GUARDRail WRAPPER FUNCTION (updated) ---
def get_guarded_response(query_text: str):
    """
    A wrapper function that applies guardrails and returns a final response.
    """
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.60))
    
    # Get the core response and score
    core_result = run_query_engine_with_scores(query_text)

    # Apply Guardrail
    if core_result["score"] is not None and core_result["score"] < CONFIDENCE_THRESHOLD:
        logger.warning(f"Confidence score {core_result['score']:.4f} is below threshold {CONFIDENCE_THRESHOLD}.")
        
        # Log the full details for human review before we modify the response
        log_for_human_review(core_result)
        
        # Append the warning message to the original response
        warning_message = "\n\n⚠️ **WARNING: Confidence is below the threshold. Please consult a human expert.**"
        
        return {
            "response": core_result['response'] + warning_message,
            "score": core_result["score"],
            "score_type": core_result["score_type"],
            "source_nodes": core_result['source_nodes']
        }
    else:
        return core_result

    
# --- MAIN CLI LOOP ---
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
            # Call the new guarded function
            result = get_guarded_response(prompt)

            print("\n--- Final Result ---")
            print(f"Confidence Score ({result['score_type']}): {result['score']:.4f}" if result['score'] is not None else "Confidence Score: N/A")
            print("\n--- LLM Response ---")
            print(result["response"])
            if result['source_nodes']:
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