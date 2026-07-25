"""
DocuRAG — Structured Logging Configuration

Provides a configure_logging() factory that sets up structlog with:
- JSON output in production (LOG_FORMAT=json)
- Human-readable coloured output in development (LOG_FORMAT=console)
- Automatic inclusion of timestamp, log level, logger name, and request_id
- File rotation via standard logging handlers
- Consistent log record format across all modules

Usage:
    from application_configuration.logger_setup import configure_logging, get_logger
    configure_logging()  # call once at startup
    logger = get_logger(__name__)
    logger.info("Document uploaded", doc_id=str(doc_id), filename=filename)
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger


def _add_app_context(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Processor: inject application name and environment into every log record."""
    try:
        from application_configuration.environment_settings import get_settings
        settings = get_settings()
        event_dict["app"] = settings.app.name
        event_dict["env"] = settings.app.env
    except Exception:  # noqa: BLE001
        pass
    return event_dict


def configure_logging(log_level: str = "INFO", log_format: str = "console") -> None:
    """
    Initialise structlog and the standard logging system.

    Args:
        log_level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
        log_format: Output format — "json" for production, "console" for development.
    """
    # ---- Standard library logging setup ------------------------------------
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # File handler with rotation (lazy — only when log dir exists)
    try:
        from application_configuration.environment_settings import get_settings
        settings = get_settings()
        log_path = Path(settings.logging.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=settings.logging.retention,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    except Exception:  # noqa: BLE001
        pass

    # ---- structlog processor chain ----------------------------------------
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_app_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a bound structlog logger for the given module name.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A structlog BoundLogger that supports keyword arguments as context.
    """
    return structlog.get_logger(name)
