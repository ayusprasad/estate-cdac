# DocuRAG — Enterprise AI Document Intelligence & RAG System

> CPU-optimised, open-source, production-grade Document Intelligence platform with Retrieval-Augmented Generation, citation-aware answers, and a fully modular agent architecture.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start (Native Windows)](#quick-start-native-windows)
- [Phase Roadmap](#phase-roadmap)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)

---

## Overview

DocuRAG is an enterprise-grade Document Intelligence platform that ingests heterogeneous document types (PDFs — text and scanned — Word, Excel, CSV, images, SQL databases), extracts structured knowledge, stores it in a vector+relational database, and answers queries with **citation-grounded, hallucination-minimised responses**.

Key design principles:
- **Accuracy over speed** — every answer references the exact source (document, page, section, table row)
- **CPU-first** — runs on an Intel i5-12500H with 16 GB RAM using `llama.cpp` GGUF models
- **Modular agents** — each processing stage is an independent agent that can be swapped or extended
- **Single database** — PostgreSQL + pgvector eliminates the need for a separate vector store
- **Open-source** — no proprietary cloud APIs required

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DocuRAG Platform                         │
├────────────┬────────────┬──────────────┬──────────────────────┤
│  Ingestion │ Extraction │  Retrieval   │   Generation (RAG)   │
│  Pipeline  │   Agents   │   Engine     │   + Citation Layer   │
├────────────┴────────────┴──────────────┴──────────────────────┤
│              FastAPI REST / WebSocket Layer                      │
├────────────────────────────────────────────────────────────────┤
│  PostgreSQL + pgvector  │  Redis (queue+cache)  │  File Store  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| API Framework | FastAPI + Uvicorn |
| Database | PostgreSQL 15 + pgvector |
| Task Queue | Celery + Redis |
| ORM | SQLAlchemy 2.0 (async) + Alembic |
| LLM Runtime | llama.cpp (GGUF, CPU) |
| Embeddings | sentence-transformers (CPU) |
| PDF Parsing | pdfplumber, pypdf |
| OCR | Tesseract + pytesseract, easyocr |
| Word/Excel | python-docx, openpyxl |
| Language Detection | langdetect |
| Testing | pytest + pytest-asyncio |
| Linting | ruff, mypy |

---

## Project Structure

```
docurag/
├── app/
│   ├── api/                 # FastAPI routers & request/response schemas
│   ├── agents/              # Specialised processing agents
│   ├── ingestion/           # Document ingestion & classification pipeline
│   ├── extraction/          # Text, table, image, formula extractors
│   ├── ocr/                 # OCR pipeline (Tesseract / EasyOCR)
│   ├── embeddings/          # Embedding generation & management
│   ├── retrieval/           # Vector + hybrid search
│   ├── llm/                 # LLM runtime wrapper (llama.cpp)
│   ├── sql_integration/     # SQL DB connectors and query agents
│   ├── metadata/            # Metadata management & traceability
│   ├── evaluation/          # RAG quality evaluation
│   ├── models/              # SQLAlchemy ORM models
│   ├── services/            # Business logic services
│   ├── tasks/               # Celery async tasks
│   └── utils/               # Shared utilities
├── config/                  # Configuration management
├── alembic/                 # Database migrations
├── tests/                   # Test suites
├── frontend/                # Web UI
├── scripts/                 # Dev & ops helper scripts
├── data/                    # Runtime data (gitignored)
├── models/                  # LLM model files (gitignored)
└── logs/                    # Application logs (gitignored)
```

---

## Quick Start (Native Windows)

### Prerequisites
- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- Redis 7+
- Tesseract OCR
- Git

### Setup

```powershell
# 1. Clone repository
git clone <repo-url>
cd docurag

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Edit .env with your database credentials

# 5. Create data directories
python scripts\setup_dirs.py

# 6. Run database migrations
alembic upgrade head

# 7. Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Phase Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Project foundation, config, logging, DB schema | ✅ Complete |
| 1 | Document ingestion & classification pipeline | ✅ Complete |
| 2 | OCR pipeline, table/formula extraction | 🔜 Planned |
| 3 | Semantic chunking, embedding generation | 🔜 Planned |
| 4 | Retrieval engine, hybrid search, reranking | 🔜 Planned |
| 5 | RAG generation with citation grounding | 🔜 Planned |
| 6 | SQL integration agent | 🔜 Planned |
| 7 | Evaluation framework & quality metrics | 🔜 Planned |
| 8 | Frontend UI | 🔜 Planned |

---

## Configuration

All configuration is driven by environment variables (`.env` file). See `.env.example` for a full reference.

---

## Contributing

See `CONTRIBUTING.md` for coding standards, branch strategy, and PR guidelines.
