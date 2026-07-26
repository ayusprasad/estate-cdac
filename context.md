# DocuRAG — Project Context & System Architecture

> **Purpose of this File**: This document serves as the persistent single source of truth for the **DocuRAG** project. It contains hardware specifications, system architecture, tech stack details, complete directory structures, development workflows, phase-by-phase status, and current active task guidelines to ensure AI coding assistants retain full context across sessions.

---

## 1. System Overview & Core Philosophy

**DocuRAG** is an enterprise-grade Document Intelligence and Retrieval-Augmented Generation (RAG) system built to ingest heterogeneous document formats (text/scanned PDFs, Word, Excel, CSV, images, SQL databases), extract structured knowledge, and deliver citation-grounded, hallucination-minimised responses.

### Core Principles
- **Accuracy & Grounding over Speed**: Every answer must map back to explicit evidence (Document ID, Page Number, Section, Bounding Box / Table Row).
- **CPU-First & Memory-Efficient**: Designed specifically to run on consumer-grade laptop hardware without discrete GPUs.
- **Local & Open Source**: No cloud API dependencies. Local LLM runtime powered by `llama.cpp` using quantized GGUF models.
- **Unified Vector + Relational Storage**: PostgreSQL + `pgvector` for combined relational metadata and vector embeddings.
- **Modular & Agentic**: Micro-agent architecture for document classification, parsing, chunking, hybrid retrieval, SQL routing, and citation verification.

---

## 2. Target Hardware Specifications

All architectural and model selection decisions are optimized for the following host machine:

| Component | Specification |
|---|---|
| **CPU** | Intel Core i5-12500H (12 Cores: 4P + 8E, 16 Threads) |
| **RAM** | 16 GB DDR4/DDR5 |
| **GPU / iGPU** | Intel Iris Xe Graphics (Shared memory, CPU-first inference) |
| **OS** | Windows 11 (PowerShell environment) |
| **Storage** | NVMe SSD (Local storage constraints apply) |
| **LLM Engine** | `llama.cpp` (GGUF quantized models — Q4_K_M / Q5_K_M) |

---

## 3. Tech Stack & Environment

| Component Layer | Technology Selected |
|---|---|
| **Programming Language** | Python 3.11+ |
| **API Framework** | FastAPI + Uvicorn |
| **Database & Vectors** | PostgreSQL 15+ with `pgvector` extension |
| **ORM & Migrations** | SQLAlchemy 2.0 (Async) + Alembic |
| **Document Ingestion** | OpenDataLoader, `pdfplumber`, `pypdf`, `python-docx`, `openpyxl` |
| **OCR Pipeline** | Tesseract OCR (`pytesseract`), EasyOCR |
| **Embeddings Model** | `BAAI/bge-small-en-v1.5` |
| **Hybrid Search** | PostgreSQL Vector Cosine Similarity + BM25 Lexical Keyword Search |
| **Knowledge Graph** | Phase 11 (Planned) Entity-Relation extraction for GraphRAG |
| **Reranking** | Cross-Encoder Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| **LLM Runtime** | `llama.cpp` Python bindings (Qwen2.5-14B-Instruct Q4_K_M GGUF) |
| **Task Queue / Cache** | Redis 7+ / Celery (Optional async workers) |
| **Testing & Quality** | `pytest`, `pytest-asyncio`, `ruff`, `mypy` |

---

## 4. Project Directory Structure

```
estate/ (DocuRAG Root)
├── .env                              # Environment variables configuration
├── .env.example                      # Sample configuration reference
├── alembic.ini                       # Database migration settings
├── pyproject.toml                    # Tooling and dependency declarations
├── requirements.txt                  # Python dependencies
├── Makefile                          # Helper build & run commands
├── README.md                         # Project introduction & quickstart
├── context.md                        # THIS CONTEXT FILE (AI Assistant Context)
├── alembic/                          # DB migrations scripts
│   └── versions/
├── application_configuration/        # Central configuration & logging
│   ├── environment_settings.py       # Pydantic Settings management
│   └── logger_setup.py               # Structlog / standard logging setup
├── data/                             # Storage directory (Git-ignored)
│   ├── raw/                          # Raw uploaded original files
│   ├── processed/                    # Extracted markdown/JSON outputs
│   └── temp/                         # Temporary upload buffers
├── frontend/                         # Web User Interface
│   ├── index.html
│   ├── static/css/index.css
│   └── static/js/app.js
├── models/                           # GGUF LLM weights (Git-ignored)
├── logs/                             # Application runtime logs (Git-ignored)
├── scripts/                          # Setup & administrative scripts
│   ├── setup_dirs.py                 # Directory initialization
│   └── setup_postgres.py             # Database & pgvector setup
├── src/                              # Main Application Source Code
│   ├── main_application.py           # FastAPI entrypoint app
│   ├── api/                          # REST Web API layer
│   │   ├── v1_router.py              # API v1 Router aggregator
│   │   └── routes/                   # Route handlers
│   │       ├── document_routes.py    # Ingestion & document upload endpoints
│   │       ├── search_routes.py      # Search & retrieval endpoints
│   │       ├── chat_routes.py        # Conversational RAG endpoints
│   │       ├── sql_routes.py         # Phase 7: SQL database management & NL query
│   │       └── health_routes.py      # System diagnostic endpoints
│   ├── database_models/              # SQLAlchemy ORM Data Models
│   │   ├── database_connection.py    # Async engine & session makers
│   │   ├── shared_enums.py           # Document & Job Status Enums
│   │   ├── document_model.py         # Parent Document entity
│   │   ├── page_model.py             # Page-level metadata & extracted layout
│   │   ├── chunk_model.py            # Text chunk & vector embedding entity
│   │   └── processing_job_model.py   # Async job execution tracking
│   ├── document_processing/          # Data Ingestion & Extraction Pipeline
│   │   ├── ingestion_pipeline.py     # Orchestrator for document flow
│   │   ├── document_classifier.py    # Digital vs Scanned vs Mixed classifier
│   │   ├── data_extractor.py         # OpenDataLoader extraction engine
│   │   ├── semantic_chunker.py       # Layout & boundary-aware chunking
│   │   ├── vector_embedder.py        # SentenceTransformers embedding generator
│   │   └── processing_schemas.py     # Pydantic schemas for processing pipeline
│   ├── retrieval/                    # Hybrid Retrieval Engine
│   │   ├── engine.py                 # Core hybrid retrieval engine
│   │   ├── retriever.py              # Vector & BM25 retrieval components
│   │   ├── reranker.py               # Cross-Encoder score reranker
│   │   ├── search_router.py          # Phase 6: Search route intent classifier
│   │   ├── query_planner.py          # Phase 6: Multi-step query decomposition
│   │   ├── search_service.py         # Phase 5: High-level search orchestration
│   │   └── sql_agent.py              # Phase 7: NL→SQL agent with schema grounding
│   ├── llm/                          # Local LLM Runtime Integration
│   │   └── generator.py              # llama.cpp prompt runner & streaming
│   └── shared_utilities/             # Shared helpers
│       ├── file_operations.py        # Safe file saving & hashing
│       └── text_cleaning.py          # Text normalization utilities
└── tests/                            # Pytest test suite
```

---

## 5. Phase-by-Phase Progress & Roadmap

| Phase | Description | Status | Evidence / Notes |
|---|---|---|---|
| **Phase 0** | **Project Setup & Base Architecture** | ✅ Complete | FastAPI, Postgres, logging, configuration, async DB connection verified. |
| **Phase 1** | **Ingestion & Document Classification** | ✅ Complete | Document upload `/api/v1/documents/upload`, classification into `digital`, `scanned`, or `mixed` pages working. |
| **Phase 2** | **Document Extraction & Layout Parsing** | ✅ Complete | OpenDataLoader integration active; extracts page structures, layout, tables, text. |
| **Phase 3** | **Semantic Chunking** | ✅ Complete | Dynamic chunker breaks pages into retrieval units (verified: 41 chunks on test PDF). |
| **Phase 4** | **Vector Embedding & Indexing** | ✅ Complete | `all-MiniLM-L6-v2` embeds chunks and stores vectors into PostgreSQL via `pgvector`. |
| **Phase 5** | **Hybrid Retrieval Engine & Reranker** | ✅ Complete | Vector Cosine + BM25 keyword search + cross-encoder reranking via `cross-encoder/ms-marco-MiniLM-L-6-v2`. |
| **Phase 6** | **Query Planner & Intent Router** | ✅ Complete | 9 intents: FACTUAL/ANALYTICAL/NUMERICAL/TABULAR/SQL_DATA/MULTILINGUAL/SUMMARISATION/IMAGE/GENERAL. |
| **Phase 7** | **SQL Database Integration Agent** | ✅ Complete | NL→SQL agent (`sql_agent.py`). Schema inspection, safe SELECT-only, citation results. `/api/v1/sql/*`. |
| **Phase 8** | **llama.cpp Local RAG Generation** | ✅ Complete | Dual-mode: llama.cpp GGUF (Qwen2.5-14B) or extractive fallback. `generator.py`. |
| **Phase 9** | **Citation & Verification Engine** | ✅ Complete | Lexical overlap faithfulness scorer (`citation_verifier.py`). Returns 0.0–1.0 + per-sentence grounding map. |
| **Phase 10**| **Evaluation & Quality Benchmarks** | ✅ Complete | Precision@K, Recall@K, MRR, NDCG@K, P95 latency, faithfulness distribution. `evaluator.py` + `/api/v1/eval/*`. |
| **Phase 11**| **Knowledge Graph & GraphRAG** | 🔜 Planned | Extract entities/relations to build a Knowledge Graph (NetworkX/Neo4j) to support multi-hop reasoning over the dataset. |

---

## 6. Daily Execution & PowerShell Commands

### Virtual Environment Setup & Activation
```powershell
# Navigate to project directory
cd C:\Users\kumar\Desktop\estate

# Activate Virtual Environment (PowerShell)
.venv\Scripts\Activate.ps1

# If script execution is restricted:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.venv\Scripts\Activate.ps1
```

### Running the Application Server
```powershell
# Start Uvicorn backend with hot reloading
uvicorn src.main_application:app --reload --host 0.0.0.0 --port 8000
```
- **Web Interface**: `http://localhost:8000/`
- **Swagger API Docs**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`

---

## 7. Current Focus: Phase 7 (SQL Database Integration Agent)

### Phase 7 Architecture

When the QueryPlanner detects `SQL_DATA` intent (keywords: sql, database, query, select, schema...), it routes to the SQLAgent instead of the vector search pipeline:

```
NL Query
    │
    ▼  QueryPlanner detects SQL_DATA intent
    ▼
 SQLAgent.query()
    │
    ├── SchemaInspector.inspect()   ← Reads table/column names (cached)
    │
    ├── SQLQueryBuilder.build()     ← NL → safe SELECT SQL
    │
    ├── SafeQueryExecutor.validate()← Blocks INSERT/UPDATE/DELETE/DROP etc.
    │
    ├── SafeQueryExecutor.execute() ← Runs SELECT against target DB
    │
    └── SQLResultFormatter.format() ← Rows → CitedChunk dicts (same as vector search)
```

### New API Endpoints (Phase 7)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/sql/connect` | Register a SQLite or PostgreSQL database |
| `DELETE` | `/api/v1/sql/connect/{label}` | Deregister a database |
| `GET` | `/api/v1/sql/connections` | List all registered databases |
| `GET` | `/api/v1/sql/schema/{label}` | Inspect schema (tables + columns) |
| `POST` | `/api/v1/sql/query` | Natural language query → SQL results |

### Quick Test Workflow (Phase 7)
```powershell
# 1. Register a SQLite database
curl -X POST http://localhost:8000/api/v1/sql/connect `
  -H "Content-Type: application/json" `
  -d '{"label": "mydb", "connection_url": "sqlite+aiosqlite:///./data/mydb.sqlite3"}'

# 2. Query it in natural language
curl -X POST http://localhost:8000/api/v1/sql/query `
  -H "Content-Type: application/json" `
  -d '{"query": "how many records are in orders?", "db_label": "mydb"}'
```

---
*Created and maintained automatically for continuous context retention.*
