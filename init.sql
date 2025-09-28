CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS docs (
  id SERIAL PRIMARY KEY,
  path_origem TEXT NOT NULL,
  num_pagina INTEGER NULL,
  indice_chunk INTEGER,
  conteudo TEXT,
  embedding VECTOR(1024), -- Bedrock embedding possui 1536 dimensoes.
  modtempo TIMESTAMPTZ
);

ALTER TABLE docs ADD CONSTRAINT unique_chunk 
UNIQUE (path_origem, num_pagina, indice_chunk);

CREATE INDEX IF NOT EXISTS docs_embeddings_id ON docs USING hnsw (embedding vector_cosine_ops);
