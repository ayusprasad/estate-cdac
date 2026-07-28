"""
DocuRAG — Chat History Logging Utility

Saves every question asked through the UI/API and its generated answer
(along with metadata like timestamp, mode, faithfulness, and citations)
to persistent files:
  1. `data/chat_history.jsonl` — Machine-readable JSON Lines format
  2. `data/chat_history.md`    — Human-readable Markdown log format
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from application_configuration.environment_settings import get_settings
from application_configuration.logger_setup import get_logger

logger = get_logger(__name__)
settings = get_settings()

CHAT_HISTORY_JSONL = Path(settings.storage.base_dir) / "chat_history.jsonl"
CHAT_HISTORY_MD = Path(settings.storage.base_dir) / "chat_history.md"


def log_chat_interaction(
    query: str,
    answer: str,
    mode: str,
    intent: str,
    faithfulness: float,
    citations: List[Dict[str, Any]],
    latency_ms: float,
    document_ids: Optional[List[str]] = None,
) -> None:
    """
    Append a question & answer pair with full metadata to persistent history files.
    
    Creates target directory if missing. Fails gracefully without breaking request handling.
    """
    try:
        CHAT_HISTORY_JSONL.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Build structured record
        record = {
            "timestamp": timestamp,
            "question": query,
            "answer": answer,
            "mode": mode,
            "intent": intent,
            "faithfulness": round(faithfulness, 4),
            "citations": citations,
            "latency_ms": latency_ms,
            "document_ids": document_ids,
        }

        # 1. Append to JSONL file
        with open(CHAT_HISTORY_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 2. Append to Markdown log file
        md_entry = (
            f"### [{timestamp}] Question & Answer\n\n"
            f"**User Question:** {query}\n\n"
            f"**Answer ({mode.upper()} mode | Grounding: {faithfulness*100:.0f}%):**\n"
            f"{answer}\n\n"
        )
        if citations:
            md_entry += "**Sources Cited:**\n"
            for c in citations:
                doc = c.get("document_name") or "Unknown Document"
                page = f"Page {c.get('page_number')}" if c.get("page_number") else "Page —"
                sec = f"§ {c.get('section_title')}" if c.get("section_title") else ""
                md_entry += f"- [{c.get('rank', '?')}] {doc} ({page} {sec})\n"
            md_entry += "\n"

        md_entry += "---\n\n"

        with open(CHAT_HISTORY_MD, "a", encoding="utf-8") as f:
            f.write(md_entry)

        logger.info(
            "Chat interaction logged",
            jsonl=str(CHAT_HISTORY_JSONL),
            md=str(CHAT_HISTORY_MD),
        )

    except Exception as exc:
        logger.error("Failed to log chat interaction", error=str(exc))
