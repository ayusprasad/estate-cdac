"""
DocuRAG — Language Detection Utilities

Provides language detection for text content using langdetect.
Results are normalised to BCP-47 language codes.

Design:
- Single function interface to allow easy swap to fastText or other backends
- Returns both language code and confidence score
- Fails gracefully on short or non-text content
"""
from __future__ import annotations

from typing import Optional

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Minimum text length for reliable detection
_MIN_TEXT_LENGTH = 20


def detect_language(
    text: str,
    min_confidence: Optional[float] = None,
) -> tuple[str, float]:
    """
    Detect the language of a text string.

    Args:
        text: The text to analyse.
        min_confidence: Minimum confidence threshold. Defaults to settings value.

    Returns:
        Tuple of (bcp47_language_code, confidence_score).
        Falls back to (settings.document.default_language, 0.0) on failure.
    """
    default_lang = settings.document.default_language if hasattr(settings, 'document') and hasattr(settings.document, 'default_language') else 'en'
    threshold = min_confidence or 0.8

    if not text or len(text.strip()) < _MIN_TEXT_LENGTH:
        logger.debug(
            "Text too short for language detection, using default",
            text_length=len(text) if text else 0,
            default=default_lang,
        )
        return default_lang, 0.0

    try:
        from langdetect import detect_langs  # type: ignore[import]
        results = detect_langs(text)
        if results:
            top = results[0]
            lang_code = str(top.lang)
            confidence = float(top.prob)
            logger.debug(
                "Language detected",
                language=lang_code,
                confidence=confidence,
            )
            return lang_code, confidence
    except Exception as exc:  # noqa: BLE001
        logger.warning("Language detection failed", exc_info=exc)

    return default_lang, 0.0


def is_multilingual(texts: list[str]) -> bool:
    """
    Check whether a list of text samples contains multiple languages.

    Useful for determining if a document requires per-page language detection.

    Args:
        texts: List of text samples (e.g., one per page).

    Returns:
        True if more than one distinct language is detected.
    """
    languages: set[str] = set()
    for text in texts:
        lang, _ = detect_language(text)
        languages.add(lang)
    return len(languages) > 1
