# DocuRAG — Enterprise AI Document Intelligence & Retrieval-Augmented Generation System

> **CPU‑optimised, open‑source, production‑grade Document Intelligence platform** with Retrieval‑Augmented Generation, citation‑aware answers, and a fully modular agent architecture.

---

## 📑 Table of Contents
1. [Overview](#overview)  
2. [Core Principles](#core-principles)  
3. [Tech Stack](#tech-stack)  
4. [Project Structure](#project-structure)  
5. [Phase Roadmap](#phase-roadmap)  
6. [Quick Start (Native Windows)](#quick-start-native-windows)  
7. [Configuration](#configuration)  
8. [Recent Changes & Active Development](#recent-changes--active-development)  
9. [API Reference](#api-reference)  
10. [Contributing](#contributing)  

---

## Overview
DocuRAG ingests heterogeneous documents (PDFs — text and scanned — Word, Excel, CSV, images, SQL databases), extracts structured knowledge, stores it in a **vector + relational** database, and answers queries with **citation‑grounded, hallucination‑minimised responses**.  

Design pillars:
- **Accuracy over speed** – every answer references the exact source (document, page, section, table row).  
- **CPU‑first** – runs on an Intel i5‑12500H with 16 GB RAM using `llama.cpp` GGUF models.  
- **Modular agents** – each processing stage is an independent, swappable component.  
- **Single database** – PostgreSQL + `pgvector` eliminates a separate vector store.  
- **Open‑source** – no proprietary cloud APIs required.

---

## Core Principles
- **Grounded Generation** – answers always cite explicit evidence (Document ID, page, section, table row).  
- **CPU‑First Architecture** – optimised for Intel i5‑12500H + 16 GB RAM; no GPU required.  
- **Local & Open Source** – all components are self‑hosted; no proprietary cloud dependencies.  
- **Unified Storage** – PostgreSQL + `pgvector` stores both relational metadata and vector embeddings.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.11+ |
| **API Framework** | FastAPI + Uvicorn |
| **Database & Vectors** | PostgreSQL 15 + `pgvector` |
| **Task Queue / Cache** | Redis 7 + Celery (optional) |
| **ORM & Migrations** | SQLAlchemy 2.0 (async) + Alembic |
| **Embeddings** | `sentence‑transformers/all‑MiniLM‑L6‑v2` + Cross‑Encoder reranker |
| **LLM Runtime** | `llama.cpp` (GGUF Q4_K_M / Q5_K_M, CPU‑only) – with extractive fallback |
| **OCR** | Tesseract (`pytesseract`), EasyOCR |
| **Document Ingestion** | OpenDataLoader, `pdfplumber`, `pypdf`, `python-docx`, `openpyxl` |
| **Testing / Quality** | `pytest`, `pytest-asyncio`, `ruff`, `mypy` |

---

## Project Structure

```
estate/ (DocuRAG Root)
├── .env                              # Environment‑variable template
├── .env.example                      # Sample configuration
├── alembic.ini                       # Alembic migration settings
├── Makefile                          # Helper build/run commands
├── README.md                         # THIS FILE (project overview)
├── context.md                        # AI‑assistant persistent context
├── pyproject.toml                    # Build & dev dependencies
├── requirements.txt                  # Production dependencies
├── scripts/                          # Setup & administrative scripts
│   ├── setup_dirs.py                 # Create project directories
│   └── setup_postgres.py             # Initialise PostgreSQL + pgvector
├── data/                             # Runtime storage (git‑ignored)
│   ├── raw/                          # Original uploads
│   ├── processed/                    # Extracted markdown/JSON
│   └── temp/                         # Temporary buffers
├── frontend/                         # Web UI (HTML/CSS/JS)
│   ├── index.html
│   ├── static/css/index.css
│   └── static/js/app.js
├── models/                           # GGUF LLM weights (git‑ignored)
├── logs/                             # Application logs (git‑ignored)
├── src/                              # All source code
│   ├── main_application.py           # FastAPI entry point
│   ├── api/
│   │   ├── __init__.py               # API package init
│   │   └── routes/
│   │       ├── __init__.py           # Routes package init
│   │       ├── health_routes.py      # System diagnostics
│   │       ├── document_routes.py    # Document upload & status
│   │       ├── search_routes.py      # Hybrid search & intent routing
│   │       ├── sql_routes.py         # SQL Agent (NL→SQL)
│   │       ├── chat_routes.py        # Full RAG pipeline (retrieval + generate + verify)
│   │       └── eval_routes.py        # Evaluation endpoints (metrics, reports)
│   ├── database_models/
│   │   ├── __init__.py               # ORM imports
│   │   ├── database_connection.py    # Async engine/session factory
│   │   ├── shared_enums.py           # Document & job status enums
│   │   ├── document_model.py         # Document entity
│   │   ├── page_model.py             # Page metadata
│   │   ├── chunk_model.py            # Chunk & vector entity
│   │   └── processing_job_model.py   # Async job tracking
│   ├── document_processing/
│   │   ├── __init__.py               # Package init
│   │   ├── canonical_json_model.py   # Structured JSON output format
│   │   ├── ingestion_pipeline.py     # Orchestrates ingestion flow
│   │   ├── unified_parser.py         # Multi‑format parser
│   │   ├── semantic_chunker.py       # Layout‑aware chunking
│   │   ├── vector_embedder.py        # Embedding generation
│   │   ├── data_extractor.py         # Extraction pipeline
│   │   ├── document_classifier.py    # Digital / scanned / mixed classifier
│   │   └── processing_schemas.py     # Pydantic validation schemas
│   ├── retrieval/
│   │   ├── __init__.py               # Package init
│   │   ├── engine.py                 # Core hybrid retrieval engine
│   │   ├── retriever.py              # Retrieval components
│   │   ├── agent_orchestrator.py     # Agent coordination
│   │   ├── query_expander.py         # Query expansion utilities
│   │   ├── reranker.py               # Cross‑encoder reranking
│   │   ├── search_router.py          # Intent routing engine
│   │   ├── query_planner.py          # Query‑intent classifier (9 intents)
│   │   └── sql_agent.py              # SQL‑agent implementation
│   ├── llm/
│   │   ├── __init__.py               # Package init
│   │   ├── generator.py              # RAG generator (llama.cpp or extractive)
│   │   ├── citation_verifier.py      # Faithfulness scoring
│   │   └── evaluator.py              # Metrics (precision, recall, NDCG, etc.)
│   └── shared_utilities/
│       ├── __init__.py               # Package init
│       ├── text_cleaning.py          # Text normalisation
│       └── file_operations.py        # Safe file handling
│   └── shared_utilities/chat_history_logger.py  # Persistent chat log storage
├── tests/                            # Pytest suite
│   ├── __init__.py
│   └── test_chat_history.py
└── .gitignore                        # Git ignore rules
```

> **Note:** Every sub‑directory contains an `__init__.py` file, making the project fully **package‑importable**.

---

## Phase Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| **0** | Project setup & base architecture (FastAPI, Postgres, logging, async DB) | ✅ Complete |
| **1** | Document ingestion & classification pipeline | ✅ Complete |
| **2** | OCR pipeline, table/formula extraction | ✅ Complete |
| **3** | Semantic chunking (layout‑aware) | ✅ Complete |
| **4** | Vector embedding & indexing (pgvector) | ✅ Complete |
| **5** | Hybrid retrieval engine & reranker | ✅ Complete |
| **6** | Query planner & intent router (9 intents) | ✅ Complete |
| **7** | **SQL Database Agent** (NL → SQL, schema inspection, safe SELECT) | ✅ Complete |
| **8** | LLaMA GGUF generation **or** extractive fallback (dual‑mode) | ✅ Complete |
| **9** | Citation & verification engine (sentence‑level grounding) | ✅ Complete |
| **10**| Evaluation & quality benchmarks (precision@k, recall@k, MRR, NDCG, latency, faithfulness) | ✅ Complete |
| **11**| Knowledge‑graph integration & GraphRAG (planned) | 🔜 Planned |

All phases **0 – 10** are production‑ready; Phase 11 is slated for the next release cycle.

---

## Quick Start (Native Windows)

```powershell
# 1️⃣ Clone & navigate
git clone <repo-url>
cd estate

# 2️⃣ Create virtual environment & activate
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Set up environment variables
cp .env.example .env
# Edit .env with your DB credentials and file paths

# 5️⃣ Initialise storage directories & DB schema
python scripts\setup_dirs.py
python scripts\setup_postgres.py

# 6️⃣ Run migrations
alembic upgrade head

# 7️⃣ Start the API server
uvicorn src.main_application:app --reload --host 0.0.0.0 --port 8000
```

Open a browser at **http://localhost:8000**:  
- **Swagger UI** → `/docs`  
- **ReDoc** → `/redoc`  
- **Health checks** → `/health`

---

## Configuration

All configuration is driven by environment variables (`.env`).  
See `.env.example` for a full reference of supported keys (DB URL, Redis URL, storage paths, LLM settings, OCR options, etc.).  

**Tip:** When adding a new document, the system automatically updates internal enums and triggers re‑indexing.

---

## Recent Changes & Active Development
The repository is actively maintained. Recent commits (as of **v3.27‑07‑2026**) include:

```
0b50188 Implement Phase 5-10 features and fix RAG LLM pollution bugs
e9a0725 Initial commit
73b7e70 aaaaaaaaaaaaaaaaaa
```

Key files modified recently:
- `src/api/routes/chat_routes.py` – Full RAG pipeline (Phase 8 + 9)  
- `src/api/routes/search_routes.py` – Hybrid search & intent routing (Phase 5/6)  
- `src/api/routes/sql_routes.py` – SQL Agent endpoints  
- `src/api/routes/eval_routes.py` – Evaluation and metrics endpoints  
- `src/llm/generator.py` – RAG generator (llama.cpp or extractive)  
- `src/llm/citation_verifier.py` – Faithfulness scoring  
- `src/llm/evaluator.py` – Metrics evaluation  

Git status also shows updates to:
- `src/document_processing/semantic_chunker.py`  
- `src/document_processing/vector_embedder.py`  
- Various files under `src/retrieval/agents/`  

These changes indicate ongoing work on **Phase 7‑10 integration** and bug‑fixes for RAG LLM pollution.

### Daily Commands Archive (for developers)
```powershell
# Initialise project directories and DB schema
python scripts/setup_dirs.py
python scripts/setup_postgres.py

# Apply DB migrations
alembic upgrade head

# Run the test suite
pytest tests/
```

---

## API Reference
The full REST API is automatically documented via **Swagger UI** (`/docs`) and **ReDoc** (`/redoc`). Key endpoint groups:

| Category | Representative Endpoints |
|----------|--------------------------|
| **Health** | `/health/*` |
| **Document** | `/api/v1/documents/upload` |
| **Search** | `/api/v1/search` |
| **SQL** | `/api/v1/sql/*` |
| **Chat (RAG)** | `/api/v1/chat/query`, `/api/v1/chat/history`, `/api/v1/chat/status` |
| **Eval** | `/api/v1/eval/*` |

Each route returns structured JSON with citations, grounding scores, and performance metrics.

---

## Contributing
See `CONTRIBUTING.md` for coding standards, branch strategy, and pull‑request workflow.  
All contributions are welcome; please keep feature branches short and ensure tests pass.

---

*Created and maintained automatically for continuous context retention.*