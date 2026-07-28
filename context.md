---
docurag-context: v3.27-07-2026
---

# DocuRAG — Project Context & System Architecture

> **Purpose of this File**: This document serves as the persistent single source of truth for the **DocuRAG** project. It contains hardware specifications, system architecture, tech stack details, complete directory structures, development workflows, phase-by-phase status, and current active task guidelines to ensure AI coding assistants retain full context across sessions.

## 1. System Overview & Core Philosophy

**DocuRAG** is an enterprise-grade Document Intelligence and Retrieval-Augmented Generation (RAG) system built to ingest heterogeneous document formats (text/scanned PDFs, Word, Excel, CSV, images, SQL databases), extract structured knowledge, and deliver citation-grounded, hallucination-minimised responses.

### Core Principles
- **Accuracy & Grounding over Speed**: Every answer must map back to explicit evidence (Document ID, Page Number, Section, Bounding Box / Table Row).
- **CPU-First and Memory-Efficient**: Designed specifically to run on consumer-grade laptop hardware without discrete GPUs.
- **Local & Open Source**: No cloud API dependencies. Local LLM runtime powered by `llama.cpp` using quantized GGUF models.
- **Unified Vector + Relational Storage**: PostgreSQL + `pgvector` for combined relational metadata and vector embeddings.
- **Modular & Agentic**: Micro-agent architecture for document classification, parsing, chunking, hybrid retrieval, SQL routing, and citation verification.

## 2. Target Hardware Specifications

All architectural and model selection decisions are optimized for the following host machine:

| Component | Specification |
|---|---|
| **CPU** | Intel Core i5-12500H (12 Cores: 4P + 8E, 16 Threads) |
| **RAM** | 16 GB DDR4/DDR5 |
| **GPU / iGPU** | Intel Iris Xe Graphics (Shared memory, CPU-first inference) |
| **OS** | Windows 11 (PowerShell environment) |
| **Storage** | NVMe SSD |
| **LLM Engine** | `llama.cpp` (GGUF quantized models — Q4_K_M / Q5_K_M) |

## 3. Tech Stack & Environment

| Component | Technology |
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
│   ├── api/
│   │   ├── __init__.py               # API package initializer
│   │   └── routes/
│   │       ├── __init__.py           # API routes package initializer
│   │       ├── health_routes.py      # System diagnostic endpoints
│   │       ├── document_routes.py    # Ingestion & document upload endpoints
│   │       ├── search_routes.py      # Hybrid retrieval engine & intent router
│   │       ├── sql_routes.py         # NL→SQL agent endpoints
│   │       ├── chat_routes.py        # Full RAG pipeline (retrieval+generate+verify)
│   │       └── eval_routes.py        # Evaluation metrics endpoints
│   ├── database_models/
│   │   ├── __init__.py               # ORM imports initializer
│   │   ├── database_connection.py    # Async engine & session makers
│   │   ├── shared_enums.py           # Document & Job Status Enums
│   │   ├── document_model.py         # Parent Document entity
│   │   ├── page_model.py             # Page-level metadata & extracted layout
│   │   ├── chunk_model.py            # Text chunk & vector embedding entity
│   │   └── processing_job_model.py   # Async job execution tracking
│   ├── document_processing/
│   │   ├── __init__.py               # Data processing package initializer
│   │   ├── canonical_json_model.py   # Structured document output format
│   │   ├── ingestion_pipeline.py     # Document flow orchestrator
│   │   ├── unified_parser.py         # Multi-format parser
│   │   ├── semantic_chunker.py       # Layout-aware chunking
│   │   ├── vector_embedder.py        # GGUF embedding generator
│   │   ├── data_extractor.py         # Extraction pipeline
│   │   ├── document_classifier.py    # Digital vs Scanned classifier
│   │   └── processing_schemas.py     # Schemas for processing pipeline
│   ├── retrieval/
│   │   ├── __init__.py               # Retrieval package initializer
│   │   ├── engine.py                 # Core hybrid retrieval engine
│   │   ├── retriever.py              # Retrieval components
│   │   ├── agent_orchestrator.py     # Retrieval agent coordination
│   │   ├── query_expander.py         # Query expansion
│   │   ├── reranker.py               # Cross-encoder reranking
│   │   ├── search_router.py          # Intent routing engine
│   │   ├── query_planner.py          # Query intent classification
│   │   └── sql_agent.py              # SQL agent implementation
│   ├── llm/
│   │   ├── __init__.py               # LLM package initializer
│   │   ├── generator.py              # RAG generation engine (llama.cpp or extractive)
│   │   ├── citation_verifier.py      # Faithfulness scoring
│   │   └── evaluator.py              # Quality metrics evaluation
│   └── shared_utilities/
│       ├── __init__.py               # Shared utilities package initializer
│       ├── text_cleaning.py          # Text normalization utilities
│       └── file_operations.py        # Safe file operations
│   └── shared_utilities/chat_history_logger.py  # Chat history persistence
├── tests/                            # Pytest test suite
│   ├── __init__.py
│   └── test_chat_history.py  # Example test file
└── .gitignore                        # Git ignore file
```

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
| **Phase 7** | **SQL Database Integration Agent** | ✅ Complete | NL→SQL agent with schema inspection, safe SELECT-only, citation results. `/api/v1/sql/*`. |
| **Phase 8** | **llama.cpp Local RAG Generation** | ✅ Complete | Dual-mode: llama.cpp GGUF (Qwen2.5-14B-Instruct) or extractive fallback. |
| **Phase 9** | **Citation & Verification Engine** | ✅ Complete | Lexical overlap faithfulness scorer (`citation_verifier.py`). Returns 0.0–1.0 + per-sentence grounding map. |
| **Phase 10**| **Evaluation & Quality Benchmarks** | ✅ Complete | Precision@K, Recall@K, MRR, NDCG@K, P95 latency, faithfulness distribution. |
| **Phase 11**| **Knowledge Graph & GraphRAG** | 🔜 Planned | Extract entities/relations to build Knowledge Graph for multi-hop reasoning. |

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

## 7. Current Focus: Phase 7 (SQL Database Agent) and Phase 8-10 Integration

### Phase 7 Architecture
When the QueryPlanner detects `SQL_DATA` intent (keywords: sql, database, query, select, schema...), it routes queries through the SQLAgent instead of the vector search pipeline.

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

## 8. Recent Commit History (Summary)

```
0b50188 Implement Phase 5-10 features and fix RAG LLM pollution bugs
e9a0725 Initial commit
73b7e70 aaaaaaaaaaaaaaaaaa
```

The repository contains multiple recently modified files including:

- `src/api/routes/search_routes.py` (Phase 5/6 search endpoints)
- `src/api/routes/chat_routes.py` (Full RAG pipeline)
- `src/api/routes/sql_routes.py` (SQL integration)
- `src/api/routes/eval_routes.py` (Evaluation endpoints)
- `src/llm/generator.py` (RAG generation engine)
- `src/llm/citation_verifier.py` (Faithfulness scoring)
- `src/llm/evaluator.py` (Evaluation metrics)

Git status shows modified files including:
- src/api/routes/chat_routes.py
- src/api/routes/search_routes.py
- src/document_processing/semantic_chunker.py
- src/document_processing/vector_embedder.py
- ... and others indicating active development

## 9. Daily Commands Archive

```powershell
# Directory initialization scripts
python scripts/setup_dirs.py
python scripts/setup_postgres.py

# Database schema management
alembic upgrade head
alembic revision --autogenerate -m "Describe changes" && alembic upgrade
```

---

*Created and maintained automatically for continuous context retention.*