"""
DocuRAG — Centralised Application Settings

All configuration is driven by environment variables with sensible defaults.
Uses Pydantic BaseSettings for validation, type coercion, and documentation.
A single shared instance is obtained via `get_settings()` (cached singleton).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Core application metadata and runtime mode."""

    name: str = Field(default="DocuRAG", alias="APP_NAME")
    env: str = Field(default="development", alias="APP_ENV")
    version: str = Field(default="0.1.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    secret_key: str = Field(default="change-me", alias="SECRET_KEY")

    model_config = SettingsConfigDict(populate_by_name=True)


class APISettings(BaseSettings):
    """FastAPI server configuration."""

    host: str = Field(default="0.0.0.0", alias="API_HOST")
    port: int = Field(default=8000, alias="API_PORT")
    reload: bool = Field(default=True, alias="API_RELOAD")
    workers: int = Field(default=1, alias="API_WORKERS")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        alias="CORS_ORIGINS",
    )
    cors_allow_credentials: bool = Field(default=True, alias="CORS_ALLOW_CREDENTIALS")

    model_config = SettingsConfigDict(populate_by_name=True)


class DatabaseSettings(BaseSettings):
    """PostgreSQL + pgvector connection settings."""

    host: str = Field(default="localhost", alias="POSTGRES_HOST")
    port: int = Field(default=5432, alias="POSTGRES_PORT")
    db: str = Field(default="docurag", alias="POSTGRES_DB")
    user: str = Field(default="docurag_user", alias="POSTGRES_USER")
    password: str = Field(default="password", alias="POSTGRES_PASSWORD")
    pool_size: int = Field(default=5, alias="POSTGRES_POOL_SIZE")
    max_overflow: int = Field(default=10, alias="POSTGRES_MAX_OVERFLOW")

    model_config = SettingsConfigDict(populate_by_name=True)

    # Allow setting full URL directly from .env
    url: str = Field(default="sqlite+aiosqlite:///./docurag.db", alias="DATABASE_URL")

    model_config = SettingsConfigDict(populate_by_name=True)

    @property
    def async_url(self) -> str:
        return self.url

    @property
    def sync_url(self) -> str:
        # Just swap aiosqlite with sqlite if needed, but since we are fully async, 
        # sync_url might not be used outside alembic.
        return self.url.replace('+aiosqlite', '') if 'sqlite' in self.url else self.url


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    host: str = Field(default="localhost", alias="REDIS_HOST")
    port: int = Field(default=6379, alias="REDIS_PORT")
    db: int = Field(default=0, alias="REDIS_DB")

    model_config = SettingsConfigDict(populate_by_name=True)

    @property
    def url(self) -> str:
        """Redis connection URL."""
        return f"redis://{self.host}:{self.port}/{self.db}"


class StorageSettings(BaseSettings):
    """File storage path configuration."""

    base_dir: Path = Field(default=Path("./data"), alias="STORAGE_BASE_DIR")
    raw_docs_dir: Path = Field(default=Path("./data/raw"), alias="RAW_DOCS_DIR")
    processed_docs_dir: Path = Field(
        default=Path("./data/processed"), alias="PROCESSED_DOCS_DIR"
    )
    temp_dir: Path = Field(default=Path("./data/temp"), alias="TEMP_DIR")
    logs_dir: Path = Field(default=Path("./logs"), alias="LOGS_DIR")

    model_config = SettingsConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def resolve_paths(self) -> "StorageSettings":
        """Resolve all paths to absolute and create them if missing."""
        for field_name in ("base_dir", "raw_docs_dir", "processed_docs_dir", "temp_dir", "logs_dir"):
            path: Path = getattr(self, field_name)
            resolved = path.resolve()
            setattr(self, field_name, resolved)
            resolved.mkdir(parents=True, exist_ok=True)
        return self


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration (CPU-optimised)."""

    model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="EMBEDDING_MODEL",
    )
    device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    dim: int = Field(default=384, alias="EMBEDDING_DIM")

    model_config = SettingsConfigDict(populate_by_name=True)


class LLMSettings(BaseSettings):
    """llama.cpp GGUF model runtime settings."""

    model_path: Path = Field(
        default=Path("./models/llm/model.gguf"), alias="LLM_MODEL_PATH"
    )
    context_size: int = Field(default=4096, alias="LLM_CONTEXT_SIZE")
    max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")
    temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    n_threads: int = Field(default=8, alias="LLM_N_THREADS")
    n_gpu_layers: int = Field(default=0, alias="LLM_N_GPU_LAYERS")  # 0 = CPU only

    model_config = SettingsConfigDict(populate_by_name=True)


class OCRSettings(BaseSettings):
    """OCR pipeline configuration."""

    engine: str = Field(default="tesseract", alias="OCR_ENGINE")
    tesseract_cmd: str = Field(default="tesseract", alias="TESSERACT_CMD")
    dpi: int = Field(default=300, alias="OCR_DPI")
    confidence_threshold: float = Field(default=60.0, alias="OCR_CONFIDENCE_THRESHOLD")

    model_config = SettingsConfigDict(populate_by_name=True)


class DocumentSettings(BaseSettings):
    """Document ingestion & processing settings."""

    max_file_size_mb: int = Field(default=100, alias="MAX_FILE_SIZE_MB")
    supported_extensions: list[str] = Field(
        default=[".pdf", ".docx", ".xlsx", ".csv", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
        alias="SUPPORTED_EXTENSIONS",
    )
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    min_chunk_size: int = Field(default=50, alias="MIN_CHUNK_SIZE")

    # Threshold for classifying a PDF page as scanned vs digital
    scanned_page_text_threshold: int = Field(
        default=50,
        description="Min chars to consider a page digital",
    )

    # Language detection
    default_language: str = Field(default="en", alias="DEFAULT_LANGUAGE")
    langdetect_min_confidence: float = Field(
        default=0.8, alias="LANGDETECT_MIN_CONFIDENCE"
    )

    model_config = SettingsConfigDict(populate_by_name=True)

    @property
    def max_file_size_bytes(self) -> int:
        """Max allowed upload size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


class LoggingSettings(BaseSettings):
    """Structured logging configuration."""

    level: str = Field(default="INFO", alias="LOG_LEVEL")
    format: str = Field(default="console", alias="LOG_FORMAT")  # "json" | "console"
    file: str = Field(default="./logs/docurag.log", alias="LOG_FILE")
    rotation: str = Field(default="10MB", alias="LOG_ROTATION")
    retention: int = Field(default=7, alias="LOG_RETENTION")  # days

    model_config = SettingsConfigDict(populate_by_name=True)


class CelerySettings(BaseSettings):
    """Celery task queue settings."""

    broker_url: str = Field(default="redis://localhost:6379/0", alias="CELERY_BROKER_URL")
    result_backend: str = Field(default="redis://localhost:6379/1", alias="CELERY_RESULT_BACKEND")
    task_serializer: str = Field(default="json", alias="CELERY_TASK_SERIALIZER")
    max_retries: int = Field(default=3, alias="CELERY_MAX_RETRIES")

    model_config = SettingsConfigDict(populate_by_name=True)


class Settings(BaseSettings):
    """
    Unified top-level settings container.

    Composes all sub-settings groups into a single object.
    Loaded once at startup and cached via `get_settings()`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # Sub-settings (each reads its own env vars)
    app: AppSettings = Field(default_factory=AppSettings)
    api: APISettings = Field(default_factory=APISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    ocr: OCRSettings = Field(default_factory=OCRSettings)
    document: DocumentSettings = Field(default_factory=DocumentSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached application settings singleton.

    Uses lru_cache so settings are loaded and validated exactly once
    for the lifetime of the process. In tests, call `get_settings.cache_clear()`
    before patching environment variables.
    """
    return Settings()
