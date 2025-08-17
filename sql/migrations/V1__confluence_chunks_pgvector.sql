-- V1__confluence_chunks_pgvector.sql

-- 1) Расширение pgvector (иногда требует суперправа; на RDS/Aurora проверь политику)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) Основная таблица для чанков Confluence
CREATE TABLE IF NOT EXISTS confluence_chunks (
  id BIGSERIAL PRIMARY KEY,
  page_id TEXT NOT NULL,
  page_title TEXT NOT NULL,
  space_key TEXT,
  url TEXT,
  chunk_idx INT NOT NULL,
  content TEXT NOT NULL,
  embedding VECTOR(1536),               -- размерность под text-embedding-3-small
  created_at TIMESTAMPTZ DEFAULT now(),
  -- уникальность пары страница+номер чанка, чтобы ON CONFLICT работал
  CONSTRAINT confluence_chunks_page_chunk_uniq UNIQUE (page_id, chunk_idx)
);

-- 3) Индекс HNSW под косинусное расстояние (лучше для текстовых эмбеддингов)
CREATE INDEX IF NOT EXISTS confluence_chunks_embedding_hnsw_cosine
  ON confluence_chunks
  USING hnsw (embedding vector_cosine_ops);

-- Примечание:
-- Для exact KNN можно обойтись без индекса, но будет медленнее (ORDER BY embedding <=> $1 LIMIT k).
