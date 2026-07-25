"""
DocuRAG — ProcessingJob ORM Model

Tracks each async Celery task submitted for a document.
Provides full audit trail: when each pipeline stage ran, how long it took,
and any errors that occurred. Essential for debugging and re-processing.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database import Base
from app.models.enums import JobStatus


class ProcessingJob(Base):
    """
    Celery task record for a single pipeline stage on a document.

    One document may have multiple jobs (classification, extraction,
    embedding, etc.) — one per pipeline stage.
    """

    __tablename__ = "processing_jobs"
    __table_args__ = (
        Index("ix_jobs_document_id", "document_id"),
        Index("ix_jobs_status", "status"),
        Index("ix_jobs_celery_task_id", "celery_task_id"),
        {"schema": "docurag"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("docurag.documents.id", ondelete="CASCADE"),
        nullable=False,
    )

    celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(256),
        nullable=True,
        comment="Celery task UUID for status polling",
    )

    job_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="Pipeline stage name (classify, extract, embed, etc.)",
    )

    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, name="job_status", schema="docurag"),
        nullable=False,
        default=JobStatus.QUEUED,
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    retry_count: Mapped[int] = mapped_column(
        default=0,
        nullable=False,
    )

    result_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Stage-specific output summary (page count, chunk count, etc.)",
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document", back_populates="processing_jobs"
    )

    def __repr__(self) -> str:
        return (
            f"<ProcessingJob id={self.id} type={self.job_type!r} "
            f"status={self.status.value!r}>"
        )
