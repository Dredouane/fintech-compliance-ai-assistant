import os
import logging
from enum import Enum
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.cleanlab import CleanlabTLM
from typing import Dict, Any
from config import get_rag_config, setup_llms, get_sql_config
import psycopg2
from psycopg2 import sql

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfidenceMethod(str, Enum):
    RAG_SCORE = "rag_score"
    CLEANLAB_TLM = "cleanlab_tlm"

# Load LLMs once at startup
try:
    SMALL_LLM, MEDIUM_LLM = setup_llms()
except ValueError as e:
    logger.error(e)
    # Re-raise the exception to stop execution if the API key is missing
    raise e

def _log_fallback_data(original_query: str, faq_score: float, fallback_score: float, response: str, score_type: str):
    """
    Logs data to the compliance_search_fallbacks_data table for future model training.
    """
    try:
        conn = get_sql_config()
        cur = conn.cursor()

        # Use sql.Identifier and sql.Literal for safe queries
        insert_query = sql.SQL("""
            INSERT INTO compliance_search_fallbacks_data (
                original_query, faq_confidence, fallback_confidence, llm_response, score_type, timestamp
            ) VALUES (
                {original_query}, {faq_confidence}, {fallback_confidence}, {llm_response}, {score_type}, NOW()
            );
        """).format(
            original_query=sql.Literal(original_query),
            faq_confidence=sql.Literal(faq_score),
            fallback_confidence=sql.Literal(fallback_score),
            llm_response=sql.Literal(response),
            score_type=sql.Literal(score_type)
        )
        
        cur.execute(insert_query)
        conn.commit()
        logger.info("Successfully logged fallback data for human review.")

    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(f"Error logging to database: {error}")
    finally:
        if conn:
            cur.close()
            conn.close()

def run_query_engine_with_scores(query_text: str, compliance_rag_collection: str):
    """
    Implements a fallback RAG strategy. Queries first against a small, fast LLM and an
    FAQ-specific vector store. If confidence is low, it falls back to a larger LLM
    and a comprehensive compliance vector store.
    """
    # Get the scoring method from the environment variable with RAG_SCORE as default
    scoring_method_str = os.getenv("SCORING_METHOD", "rag_score").lower()
    try:
        scoring_method = ConfidenceMethod(scoring_method_str)
    except ValueError:
        logger.warning(f"Invalid scoring method '{scoring_method_str}' in .env. Defaulting to 'rag_score'.")
        scoring_method = ConfidenceMethod.RAG_SCORE
        
    FAQ_RAG_COLLECTION = os.getenv("FAQ_COLLECTION_NAME", "faq_compliance")
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.60))
    TOP_K = int(os.getenv("TOP_K", 5))

    # --- Step 1: Attempt with Lightweight LLM and FAQ RAG ---
    logger.info("Attempting primary query with small Mistral LLM and FAQ RAG...")
    Settings.llm = SMALL_LLM
    _, faq_vector_store = get_rag_config(FAQ_RAG_COLLECTION)
    faq_index = VectorStoreIndex.from_vector_store(faq_vector_store)
    faq_retriever = VectorIndexRetriever(index=faq_index, similarity_top_k=TOP_K)
    faq_nodes_with_scores = faq_retriever.retrieve(query_text)
    
    faq_score = None
    if scoring_method == ConfidenceMethod.RAG_SCORE:
        faq_score = max(node.score for node in faq_nodes_with_scores) if faq_nodes_with_scores else None
    elif scoring_method == ConfidenceMethod.CLEANLAB_TLM:
        llm = CleanlabTLM(api_key=os.environ.get("CLEANLAB_API_KEY"))
        cleanlab_response = llm.complete(query_text)
        if 'trustworthiness_score' in cleanlab_response.additional_kwargs:
            faq_score = cleanlab_response.additional_kwargs['trustworthiness_score']
    
    # Check if the FAQ answer is confident enough
    if faq_score and faq_score >= CONFIDENCE_THRESHOLD:
        logger.info(f"FAQ RAG confidence score {faq_score:.4f} is above threshold. Using FAQ response.")
        response_synthesizer = get_response_synthesizer(llm=SMALL_LLM)
        faq_query_engine = RetrieverQueryEngine(
            retriever=faq_retriever,
            response_synthesizer=response_synthesizer
        )
        final_response = faq_query_engine.query(query_text)
        
        _log_fallback_data(query_text, faq_score, None, str(final_response), scoring_method.value)
        
        return {
            "response": str(final_response),
            "score": faq_score,
            "score_type": scoring_method.value,
            "source_nodes": faq_nodes_with_scores
        }
    else:
        # --- Step 2: Fallback to Complex LLM and Compliance RAG ---
        logger.warning(f"FAQ RAG confidence score {faq_score:.4f} is below threshold. Falling back to complex RAG.")
        
        # Switch to the medium LLM for the fallback
        Settings.llm = MEDIUM_LLM
        
        _, compliance_vector_store = get_rag_config(compliance_rag_collection)
        compliance_index = VectorStoreIndex.from_vector_store(compliance_vector_store)
        compliance_retriever = VectorIndexRetriever(index=compliance_index, similarity_top_k=TOP_K)
        compliance_nodes_with_scores = compliance_retriever.retrieve(query_text)

        compliance_score = None
        if scoring_method == ConfidenceMethod.RAG_SCORE:
            compliance_score = max(node.score for node in compliance_nodes_with_scores) if compliance_nodes_with_scores else None
        elif scoring_method == ConfidenceMethod.CLEANLAB_TLM:
            llm = CleanlabTLM(api_key=os.environ.get("CLEANLAB_API_KEY"))
            cleanlab_response = llm.complete(query_text)
            if 'trustworthiness_score' in cleanlab_response.additional_kwargs:
                compliance_score = cleanlab_response.additional_kwargs['trustworthiness_score']

        response_synthesizer = get_response_synthesizer(llm=MEDIUM_LLM)
        compliance_query_engine = RetrieverQueryEngine(
            retriever=compliance_retriever,
            response_synthesizer=response_synthesizer
        )
        final_response = compliance_query_engine.query(query_text)
        
        # Log the full fallback event
        _log_fallback_data(query_text, faq_score, compliance_score, str(final_response), scoring_method.value)

        return {
            "response": str(final_response),
            "score": compliance_score,
            "score_type": scoring_method.value,
            "source_nodes": compliance_nodes_with_scores
        }
