import logging
import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from query_router import run_query_engine_with_scores # Import the new function
from config import get_sql_config # Keep this for database logging

# Load environment variables from .env file
load_dotenv()

# Set up logging for better visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


compliance_rag_collection = os.getenv("COMPLIANCE_RAG_COLLECTION", "financial_compliance")

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


def get_guarded_response(query_text: str):
    """
    A wrapper function that applies guardrails and returns a final response.
    """
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.60))
    
    # Get the core response and score from the new function in query_router
    core_result = run_query_engine_with_scores(query_text, compliance_rag_collection)

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
