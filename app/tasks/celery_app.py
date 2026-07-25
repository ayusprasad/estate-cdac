"""
DocuRAG — Celery Application Instance

Configures and exports the Celery application used for async task processing.

Task routing:
  - ingestion.*  → ingestion queue (low concurrency, I/O heavy)
  - ocr.*        → ocr queue (CPU heavy)
  - embedding.*  → embedding queue (CPU heavy, batched)

All queues are backed by Redis. The number of workers and concurrency
is tuned for the target hardware (Intel i5-12500H, 16 GB RAM):
- ingestion: concurrency=2 (I/O bound)
- ocr: concurrency=1 (CPU bound, single Tesseract process)
- embedding: concurrency=1 (CPU bound, PyTorch)
"""
from __future__ import annotations

from celery import Celery  # type: ignore[import]

from config.settings import get_settings

settings = get_settings()

# Create the Celery app
celery_app = Celery(
    "docurag",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend,
    include=[
        "app.tasks.ingestion_tasks",
        # Future phases will add:
        # "app.tasks.ocr_tasks",
        # "app.tasks.extraction_tasks",
        # "app.tasks.embedding_tasks",
    ],
)

celery_app.conf.update(
    task_serializer=settings.celery.task_serializer,
    accept_content=[settings.celery.task_serializer],
    result_serializer=settings.celery.task_serializer,
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,  # Re-queue on worker crash
    worker_prefetch_multiplier=1,  # One task at a time per worker (CPU-safe)
    task_max_retries=settings.celery.max_retries,
    task_default_retry_delay=30,  # 30 seconds between retries
    # Queue routing
    task_routes={
        "app.tasks.ingestion_tasks.*": {"queue": "ingestion"},
        "app.tasks.ocr_tasks.*": {"queue": "ocr"},
        "app.tasks.extraction_tasks.*": {"queue": "extraction"},
        "app.tasks.embedding_tasks.*": {"queue": "embedding"},
    },
    # Prevent tasks from running indefinitely
    task_soft_time_limit=300,   # 5 minutes soft limit
    task_time_limit=600,        # 10 minutes hard limit
)
