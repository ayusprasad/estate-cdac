-- DocuRAG PostgreSQL Initialisation Script
-- Creates pgvector extension and required schemas

-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable pg_trgm for full-text fuzzy search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create schemas for logical separation
CREATE SCHEMA IF NOT EXISTS docurag;
CREATE SCHEMA IF NOT EXISTS audit;

-- Grant permissions to application user
GRANT ALL ON SCHEMA docurag TO docurag_user;
GRANT ALL ON SCHEMA audit TO docurag_user;
GRANT ALL ON SCHEMA public TO docurag_user;
