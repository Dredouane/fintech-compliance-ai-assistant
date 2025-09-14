# 🌐 AI-Powered Financial Compliance & Research Assistant

## Overview
This is a financial compliance and research assistant that leverages an AI-powered RAG (Retrieval-Augmented Generation) pipeline. The system helps manage large document stores of financial regulations, enabling users to perform semantic search and generate human-ready reports.

## Sprint 1 – Data Foundations
This sprint focuses on building the core RAG pipeline by indexing financial regulations, contracts, and reports.

### Stack
* **LlamaIndex**: For data ingestion and retrieval.
* **Embeddings**: nomic-embed-text.
* **LLM**: mistral-medium.
* **Vector DB**: Qdrant (local first, via Docker).

### Prerequisites
* Python 3.8 or higher
* Docker & Docker Compose

---

### Getting Started

#### 1. Repository Setup
Clone the repository and navigate into the project directory:

```bash
git clone [https://github.com/Dredouane/fintech-compliance-ai-assistant.git](https://github.com/Dredouane/fintech-compliance-ai-assistant.git)
cd fintech-compliance-ai-assistant
```

#### 2. Local Environment Setup

## a. Virtual Environment & Dependencies
Create a Python virtual environment to manage dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:
```bash
pip install qdrant-client  python-dotenv  llama-index llama-index-vector-stores-qdrant llama-index-llms-ollama llama-index-embeddings-ollama llama-index-llms-mistralai langchain langchain-mistralai langchain-qdrant langchainhub presidio-analyzer presidio-anonymizer llama-index-llms-cleanlab llama-index-llms-cleanlab streamlit
```

## d. Qdrant Setup
Start the Qdrant vector database using Docker Compose:

```bash

docker compose up -d
```


## c. Ollama models
Here are the commands onces the docker compose is up

```bash
ollama pull nomic-embed-text
```

#### 3. Data Ingestion
Place your PDF documents into the source_documents directory. To avoid confusion, you should rename the files to match the new professional names listed in the "References" section below.

To embed and index your documents into Qdrant, run the ingestion script:

```bash

python3 src/ingestion.py
```
#### 4. Semantic Search CLI Tool
To query your documents, run the interactive CLI tool:

```bash

python3 src/compliance_search_cli.py
```
Type your questions and press Enter to get answers powered by your indexed documents.

#### 📚 References
The documents used in this project are publicly available and sourced from official financial regulatory bodies.

From the Bank for International Settlements (BIS):

Basel III: A global regulatory framework for more resilient banks and banking systems.

BCBS189: International framework for liquidity risk measurement, standards and monitoring.

BCBS238: The Liquidity Coverage Ratio and liquidity risk monitoring tools.

d295: The G-SIB framework and methodology.

d424: Finalizing the post-crisis reforms.

d457: The BCBS governance and implementation report.

From EUR-Lex (European Union law):

Directive 2014/65/EU (MiFID II): On markets in financial instruments.

Regulation (EU) 2022/2554 (DORA): On digital operational resilience for the financial sector.