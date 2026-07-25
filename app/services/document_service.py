"""
DocuRAG — Document Service

Business logic layer between API endpoints and the database.
Encapsulates complex queries and business rules that don't belong
in route handlers or ORM models.

Future phases will add:
- get_document_chunks() for retrieval
- search_documents() for metadata-based filtering
- reprocess_document() for re-ingestion
"""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.page import Page
from app.models.enums import DocumentStatus
from app.utils.exceptions import DocumentNotFoundError
from config.logging_config import get_logger

logger = get_logger(__name__)


class DocumentService:
    """
    Service class for document-related business operations.

    Accepts an injected AsyncSession — does not create its own sessions.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def get_by_id(self, document_id: uuid.UUID) -> Document:
        """
        Retrieve a document by its UUID.

        Args:
            document_id: Document UUID.

        Returns:
            Document ORM instance.

        Raises:
            DocumentNotFoundError: If no document with that ID exists.
        """
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        if document is None:
            raise DocumentNotFoundError(str(document_id))
        return document

    async def get_by_checksum(self, checksum: str) -> Optional[Document]:
        """
        Find an existing document by SHA-256 checksum.

        Used for deduplication during ingestion.

        Args:
            checksum: SHA-256 hex digest string.

        Returns:
            Document instance if found, None otherwise.
        """
        result = await self.db.execute(
            select(Document).where(Document.checksum == checksum)
        )
        return result.scalar_one_or_none()

    async def update_status(
        self,
        document_id: uuid.UUID,
        new_status: DocumentStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update the processing status of a document.

        Args:
            document_id: Document UUID.
            new_status: Target status.
            error_message: Optional error detail (for FAILED status).
        """
        await self.db.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                status=new_status,
                error_message=error_message,
            )
        )
        logger.info(
            "Document status updated",
            doc_id=str(document_id),
            new_status=new_status.value,
        )

    async def get_pages(
        self,
        document_id: uuid.UUID,
    ) -> list[Page]:
        """
        Retrieve all pages for a document ordered by page number.

        Args:
            document_id: Document UUID.

        Returns:
            Ordered list of Page ORM instances.
        """
        result = await self.db.execute(
            select(Page)
            .where(Page.document_id == document_id)
            .order_by(Page.page_number)
        )
        return list(result.scalars().all())

    async def count_by_status(self) -> dict[str, int]:
        """
        Return counts of documents grouped by status.

        Useful for dashboard and monitoring endpoints.

        Returns:
            Dict mapping status string to count.
        """
        from sqlalchemy import func
        result = await self.db.execute(
            select(Document.status, func.count(Document.id))
            .group_by(Document.status)
        )
        return {status.value: count for status, count in result.all()}
