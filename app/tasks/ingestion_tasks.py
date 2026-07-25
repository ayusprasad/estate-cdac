"""
DocuRAG — Async Ingestion Celery Tasks

Phase 1 provides the task scaffolding for async processing.
The tasks themselves trigger downstream pipeline stages that will be
implemented in Phase 2+ (OCR, extraction, embedding).

All tasks follow the pattern:
  1. Load the document from DB
  2. Update status to PROCESSING
  3. Execute the pipeline stage
  4. Update status on success/failure
  5. Chain to the next stage task
"""
from __future__ import annotations

import logging
from uuid import UUID

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="app.tasks.ingestion_tasks.process_document",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
)
def process_document(self: object, document_id: str) -> dict:
    """
    Main async document processing task.

    Orchestrates the full pipeline for a document after initial ingestion:
      Phase 2: OCR + text extraction
      Phase 3: Semantic chunking
      Phase 4: Embedding generation

    Currently a stub — will be implemented in Phase 2.

    Args:
        document_id: String UUID of the document to process.

    Returns:
        Dict with task result summary.
    """
    logger.info("Processing document task started", extra={"document_id": document_id})

    # TODO Phase 2: Implement OCR + extraction
    # TODO Phase 3: Implement chunking
    # TODO Phase 4: Implement embedding

    return {
        "document_id": document_id,
        "status": "queued_for_phase2",
        "message": "Extraction pipeline will be implemented in Phase 2",
    }


@celery_app.task(
    name="app.tasks.ingestion_tasks.classify_document",
    bind=True,
    max_retries=2,
)
def classify_document_async(self: object, document_id: str) -> dict:
    """
    Async wrapper for the synchronous page classification stage.

    Phase 1 runs classification synchronously in the API request.
    This task exists for future use when classification is offloaded.

    Args:
        document_id: String UUID of the document.

    Returns:
        Classification result summary dict.
    """
    logger.info("Async classification task started", extra={"document_id": document_id})
    return {"document_id": document_id, "status": "classification_handled_synchronously"}
