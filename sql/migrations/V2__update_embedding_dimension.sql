ALTER TABLE confluence_chunks ALTER COLUMN embedding TYPE VECTOR(1024);
DROP INDEX IF EXISTS confluence_chunks_embedding_hnsw_cosine;
CREATE INDEX IF NOT EXISTS confluence_chunks_embedding_hnsw_cosine
  ON confluence_chunks
  USING hnsw (embedding vector_cosine_ops);