CREATE TABLE IF NOT EXISTS human_review_queue (
    id SERIAL PRIMARY KEY,
    original_query TEXT NOT NULL,
    llm_response TEXT,
    confidence_score FLOAT,
    score_type VARCHAR(20),
    source_filenames TEXT[],
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) DEFAULT 'pending'
);