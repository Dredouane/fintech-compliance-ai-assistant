import os
import logging
from enum import Enum
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.llms.mistralai import MistralAI
from llama_index.core.llms import ChatMessage
from typing import Dict, Any
from config import get_rag_config, setup_llms

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

def classify_query_complexity(query_text: str, classifier_llm: MistralAI) -> str:
    """
    Uses a small, fast LLM to classify the query as 'lightweight' or 'complex'.
    """
    system_prompt = """
    You are a query classifier. Your task is to determine the complexity of a user's query.
    A 'lightweight' query is a simple, factual question that can be answered quickly (e.g., "What is a 401k?").
    A 'complex' query requires detailed reasoning, synthesis of multiple points, or is lengthy (e.g., "Compare and contrast the benefits of a 401k vs an IRA for a high-income earner saving for retirement.").
    Respond with a single word: 'lightweight' or 'complex'. Do not provide any other text.
    """
    
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=query_text)
    ]
    
    response = classifier_llm.chat(messages)
    classification = response.message.content.strip().lower()

    if classification not in ['lightweight', 'complex']:
        logger.warning(f"Unexpected classification from LLM: {classification}. Defaulting to 'complex'.")
        return 'complex'

    return classification


def run_query_engine_with_scores(query_text: str, compliance_rag_collection: str):
    """
    Queries the vector store, selects the appropriate LLM based on complexity,
    and returns a comprehensive result object with scores and sources.
    """
    
    # 1. Classify the query and select the appropriate LLM
    query_type = classify_query_complexity(query_text, SMALL_LLM)
    
    if query_type == 'lightweight':
        selected_llm = SMALL_LLM
        logger.info("Query classified as lightweight. Using small Mistral LLM.")
    else:
        selected_llm = MEDIUM_LLM
        logger.info("Query classified as complex. Using medium Mistral LLM.")
        
    Settings.llm = selected_llm
    
    # 2. Proceed with the rest of the RAG pipeline
    scoring_method_str = os.getenv("SCORING_METHOD", "rag_score").lower()
    try:
        scoring_method = ConfidenceMethod(scoring_method_str)
    except ValueError:
        logger.warning(f"Invalid scoring method '{scoring_method_str}' in .env. Defaulting to 'rag_score'.")
        scoring_method = ConfidenceMethod.RAG_SCORE
        
    logger.info(f"Setting up the query engine with scoring method: {scoring_method.value}...")
    
    _, vector_store = get_rag_config(compliance_rag_collection)
    TOP_K = int(os.getenv("TOP_K", 5))

    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)
    retrieved_nodes_with_scores = retriever.retrieve(query_text)

    # --- Generate Initial Score ---
    initial_score = None
    if scoring_method == ConfidenceMethod.RAG_SCORE:
        if retrieved_nodes_with_scores:
            initial_score = max(node.score for node in retrieved_nodes_with_scores)
    # The Cleanlab TLM logic is omitted for brevity but would go here
    # as in your original implementation.

    response_synthesizer = get_response_synthesizer(llm=selected_llm)
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
