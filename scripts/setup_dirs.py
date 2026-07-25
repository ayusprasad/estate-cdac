#!/usr/bin/env python3
"""
DocuRAG — Directory Setup Script

Creates all required runtime directories with proper structure.
Safe to run multiple times (idempotent).

Usage (Windows):
    python scripts/setup_dirs.py

Usage (Make):
    make setup-dirs
"""
from __future__ import annotations

from pathlib import Path


REQUIRED_DIRECTORIES = [
    # Raw document storage — original uploaded files (gitignored)
    "data/raw",
    # Processed document storage — OCR output, extracted text (gitignored)
    "data/processed",
    # Temporary upload staging area (gitignored)
    "data/temp",
    # Application logs (gitignored)
    "logs",
    # LLM model files — place GGUF models here (gitignored)
    "models/llm",
    # Embedding model cache
    "models/embeddings",
    # Alembic migration versions directory
    "alembic/versions",
]

# Placeholder files to preserve empty directories in git
GITKEEP_DIRS = [
    "data/raw",
    "data/processed",
    "data/temp",
    "logs",
    "models/llm",
    "models/embeddings",
]


def setup_directories(base_path: Path | None = None) -> None:
    """Create all required directories relative to base_path (default: cwd)."""
    root = base_path or Path.cwd()

    print(f"Setting up DocuRAG directories in: {root}")
    print("-" * 50)

    for dir_path in REQUIRED_DIRECTORIES:
        full_path = root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {dir_path}")

    # Add .gitkeep to preserve empty dirs
    for dir_path in GITKEEP_DIRS:
        gitkeep = root / dir_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    print("-" * 50)
    print("[DONE] Directory setup complete.")
    print()
    print("Next steps:")
    print("  1. Copy .env.example to .env and configure your database credentials")
    print("  2. Run: alembic upgrade head")
    print("  3. Run: uvicorn app.main:app --reload")


if __name__ == "__main__":
    setup_directories()
