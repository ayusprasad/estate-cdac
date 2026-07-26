"""
DocuRAG — pytest conftest.py

Shared fixtures available to all test modules.

Fixtures:
- settings_override: Patches settings for test isolation
- anyio_backend: Configures asyncio for pytest-asyncio
- temp_upload_dir: Isolated temporary directory per test
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import src.llm.generator as _generator_module


@pytest.fixture(autouse=True)
def isolate_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Override storage paths and DB URL for every test.

    Prevents tests from writing to real data directories or databases.
    """
    # Override storage paths to use pytest's tmp_path
    monkeypatch.setenv("STORAGE_BASE_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAW_DOCS_DIR", str(tmp_path / "data" / "raw"))
    monkeypatch.setenv("PROCESSED_DOCS_DIR", str(tmp_path / "data" / "processed"))
    monkeypatch.setenv("TEMP_DIR", str(tmp_path / "data" / "temp"))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "logs" / "test.log"))

    # Use a test database (won't actually connect in unit tests)
    monkeypatch.setenv("POSTGRES_DB", "docurag_test")
    monkeypatch.setenv("DEBUG", "false")

    # Clear the cached settings singleton so new env vars are picked up
    from application_configuration.environment_settings import get_settings
    get_settings.cache_clear()

    yield

    # Re-clear after test
    get_settings.cache_clear()


@pytest.fixture
def anyio_backend() -> str:
    """Use asyncio as the async backend for all async tests."""
    return "asyncio"


@pytest.fixture(autouse=True)
def reset_generator_state() -> None:
    """
    Reset generator.py module-level singletons before and after every test.

    generator.py loads _llm_available / _llm at import time (module level).
    If any test patches these or causes _try_load_llama() to set them to True,
    the state bleeds into subsequent tests.

    IMPORTANT: We must *force* the module globals to the known-safe baseline
    (False / None) BEFORE yielding — not just save and restore.  If we only
    save-and-restore, a prior test that left _llm_available=True would cause us
    to save True, yield True (failing the next test), and restore True forever.
    """
    # Save originals so we can restore them after the test (belt-and-suspenders).
    orig_available = _generator_module._llm_available
    orig_llm = _generator_module._llm
    orig_generator = _generator_module._generator

    # ── Force safe baseline BEFORE the test runs ──────────────────────────────
    # No GGUF model exists in CI / unit-test environments.
    _generator_module._llm_available = False
    _generator_module._llm = None
    _generator_module._generator = None

    yield

    # ── Restore originals AFTER the test ──────────────────────────────────────
    _generator_module._llm_available = orig_available
    _generator_module._llm = orig_llm
    _generator_module._generator = orig_generator
