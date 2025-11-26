-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Table to store ingested documents (CVs)
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    content_hash TEXT,
    last_ingested_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Table to store document chunks with embeddings
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create index on document URL for fast lookups
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);

-- Create index on document content hash for change detection
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);

-- Create index on embeddings document_id for faster joins
CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON embeddings(document_id);

-- Create IVFFlat index for vector similarity search using cosine distance
-- Lists parameter set to 100, suitable for datasets up to 1M vectors
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
ON embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create GIN index on metadata for faster JSON queries
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_embeddings_metadata ON embeddings USING gin(metadata);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update updated_at on documents table
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions to the RAG user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rag_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rag_user;
