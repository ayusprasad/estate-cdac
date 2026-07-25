#!/usr/bin/env python3
"""
DocuRAG — PostgreSQL Setup Script

Creates the database, user, and required extensions for development.
Run this once when setting up a fresh local environment.

Prerequisites:
  - PostgreSQL 15+ installed and running
  - psql available in PATH
  - Sufficient privileges (superuser or pg_createdb role)

Usage:
    python scripts/setup_postgres.py
    python scripts/setup_postgres.py --host localhost --port 5432

"""
from __future__ import annotations

import argparse
import subprocess
import sys


def run_psql(command: str, host: str, port: int, superuser: str) -> None:
    """Execute a psql command against the PostgreSQL server."""
    result = subprocess.run(
        [
            "psql",
            f"--host={host}",
            f"--port={port}",
            f"--username={superuser}",
            "--no-password",
            f"--command={command}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 and "already exists" not in result.stderr:
        print(f"WARNING: {result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"  ✓  {command[:60]}...")


def setup_postgres(
    host: str = "localhost",
    port: int = 5432,
    superuser: str = "postgres",
    db_name: str = "docurag",
    db_user: str = "docurag_user",
    db_password: str = "change-this-password",
) -> None:
    """Create database, user, and install required extensions."""

    print("DocuRAG PostgreSQL Setup")
    print("=" * 50)

    commands = [
        f"CREATE USER {db_user} WITH PASSWORD '{db_password}';",
        f"CREATE DATABASE {db_name} OWNER {db_user};",
        f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {db_user};",
    ]

    # Connect to postgres database to create user and DB
    for cmd in commands:
        run_psql(cmd, host, port, superuser)

    # Connect to the new database to install extensions
    extension_commands = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";',
        "CREATE EXTENSION IF NOT EXISTS pg_trgm;",
        f"CREATE SCHEMA IF NOT EXISTS docurag AUTHORIZATION {db_user};",
        f"CREATE SCHEMA IF NOT EXISTS audit AUTHORIZATION {db_user};",
    ]

    for cmd in extension_commands:
        result = subprocess.run(
            [
                "psql",
                f"--host={host}",
                f"--port={port}",
                f"--username={superuser}",
                f"--dbname={db_name}",
                "--no-password",
                f"--command={cmd}",
            ],
            capture_output=True,
            text=True,
        )
        print(f"  ✓  {cmd[:60]}...")

    print("=" * 50)
    print("✅ PostgreSQL setup complete.")
    print()
    print("Update your .env file:")
    print(f"  POSTGRES_HOST={host}")
    print(f"  POSTGRES_PORT={port}")
    print(f"  POSTGRES_DB={db_name}")
    print(f"  POSTGRES_USER={db_user}")
    print(f"  POSTGRES_PASSWORD={db_password}")
    print()
    print("Then run: alembic upgrade head")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup DocuRAG PostgreSQL database")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--superuser", default="postgres")
    parser.add_argument("--db-name", default="docurag")
    parser.add_argument("--db-user", default="docurag_user")
    parser.add_argument("--db-password", default="change-this-password")
    args = parser.parse_args()

    setup_postgres(
        host=args.host,
        port=args.port,
        superuser=args.superuser,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
    )
